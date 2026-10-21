"""Grouped enclosures retain unique atom-pair contributions in every scorer."""

from copy import deepcopy
from dataclasses import asdict
import json
from math import dist

import pytest

from chemur.cosplif import build_cosplif_tokens, build_ligand_blocks
from chemur.ifm_config import IFMConfig
from chemur.ifm_matcher import (
    _protein_score_params,
    _protein_scoring_record,
    _score_indexed_interaction_pair,
    analysis_result_to_ifm_payloads,
)
from chemur.ifm_models import InteractionRecord
from chemur.ifm_scoring import score_interaction_pair
from chemur.interaction_projection import project_interactions_for_scoring
from chemur.models import AnalysisResult, AssignedInteraction, AtomRecord, ComponentRecord, FeatureRecord


def _result():
    specifications = [
        ("L:LIG:1", "ligand", (0., 0., 0.)),
        ("A:VAL:2", "protein", (3., 0., 0.)),
        ("B:LEU:3", "protein", (0., 3., 0.)),
        ("C:ILE:4", "protein", (0., 0., 3.)),
        ("D:PHE:5", "protein", (-2., -2., -2.)),
        ("L:LIG:1", "ligand", (3., 2., 0.)),
    ]
    atoms = []
    for atom_id, (component_id, kind, coord) in enumerate(specifications):
        chain, name, residue_id = component_id.split(":")
        atoms.append(AtomRecord(atom_id, f"C{atom_id}", "C", coord, name, residue_id,
                                chain, component_id, kind, is_hydrophobe=True))
    components = []
    for component_id in dict.fromkeys(atom.component_id for atom in atoms):
        members = [atom for atom in atoms if atom.component_id == component_id]
        atom = members[0]
        components.append(ComponentRecord(component_id, atom.residue_name, atom.chain_id,
                                          atom.residue_id, atom.molecule_type,
                                          tuple(member.atom_id for member in members)))
    features = [FeatureRecord(atom.atom_id + 100, "hydrophobe", atom.component_id,
                              (atom.atom_id,), atom.coord) for atom in atoms]
    return AnalysisResult(atoms, components, features, [])


def _contact(result, first, second, *, order=1):
    atoms = {atom.atom_id: atom for atom in result.atoms}
    return AssignedInteraction(
        "hydrophobic", (first + 100, second + 100), tuple(sorted((first, second))),
        (atoms[first].component_id, atoms[second].component_id),
        dist(atoms[first].coord, atoms[second].coord), 140, "inter", order,
    )


def _enclosure(result, center, neighbors, *, order=1):
    contacts = [_contact(result, center, neighbor) for neighbor in sorted(neighbors)]
    atom_ids = tuple(sorted({center, *neighbors}))
    return AssignedInteraction(
        "hydrophobic_enclosure", tuple(atom_id + 100 for atom_id in atom_ids), atom_ids,
        tuple(sorted({component_id for contact in contacts for component_id in contact.component_ids})),
        max(contact.distance for contact in contacts), 135, "inter", order,
        metadata={
            "center_atom_id": center,
            "neighbor_atom_ids": sorted(neighbors),
            "enclosure_margin": 0.25,
            "interaction_subtype": "hydrophobic_enclosure",
            "legs": [{"atom_ids": list(contact.atom_ids), "feature_ids": list(contact.feature_ids),
                      "component_ids": list(contact.component_ids), "distance": contact.distance}
                     for contact in contacts],
        },
    )


def _ifm_records(result):
    payload, = analysis_result_to_ifm_payloads(result)
    assert not payload["warnings"]
    return payload["interaction_records"]


def _normalize(value):
    return json.loads(json.dumps(value))


def test_projection_deduplicates_overlapping_centers_and_preserves_both_groups():
    result = _result()
    result.interactions = [_enclosure(result, 0, [1, 2, 3, 4]),
                           _enclosure(result, 1, [0, 2, 3, 5], order=2)]
    before = deepcopy(result)
    projected = project_interactions_for_scoring(result.interactions)
    assert len(projected) == 7
    shared, = [interaction for interaction in projected if interaction.atom_ids == (0, 1)]
    assert shared.metadata["enclosure_center_atom_ids"] == [0, 1]
    assert [group["center_atom_id"] for group in shared.metadata["enclosure_groups"]] == [0, 1]
    assert all(group["enclosure_margin"] == 0.25 for group in shared.metadata["enclosure_groups"])
    assert len(shared.feature_ids) == 2
    assert shared.distance == 3.0
    assert result == before

    payload_projection = project_interactions_for_scoring(result.to_dict()["interactions"])
    assert _normalize(payload_projection) == _normalize([asdict(interaction) for interaction in projected])
    assert project_interactions_for_scoring(reversed(result.interactions)) == projected


def test_ifm_preserves_all_targets_actual_legs_and_unique_ids():
    result = _result()
    result.interactions = [_enclosure(result, 0, [1, 2, 3, 4]),
                           _enclosure(result, 1, [0, 2, 3, 5], order=2)]
    records = _ifm_records(result)
    assert {(tuple(record["ligand_atom_ids"]), tuple(record["target_atom_ids"])) for record in records} == {
        ((0,), (1,)), ((0,), (2,)), ((0,), (3,)), ((0,), (4,)), ((5,), (1,)),
    }
    assert len({record["interaction_id"] for record in records}) == 5
    assert {record["target_entity_id"] for record in records} == {"A:VAL:2", "B:LEU:3", "C:ILE:4", "D:PHE:5"}
    assert all(record["interaction_subtype"] == "hydrophobic_enclosure" for record in records)
    assert all(record["source_interaction_type"] == "hydrophobic_enclosure" for record in records)
    assert all(record["interaction_family"] == "hydrophobic" and record["strength_weight"] == 1.0
               for record in records)
    shared, = [record for record in records if record["ligand_atom_ids"] == (0,)
               and record["target_atom_ids"] == (1,)]
    assert shared["metadata"]["enclosure_center_atom_ids"] == [0, 1]
    terminal, = [record for record in records if record["ligand_atom_ids"] == (5,)]
    assert terminal["geometry"]["distance"] == 2.0


def test_receptor_center_does_not_create_ligand_to_other_shell_contacts():
    result = _result()
    result.interactions = [_enclosure(result, 1, [0, 2, 3, 5])]
    records = _ifm_records(result)
    assert len(records) == 2
    assert {record["target_entity_id"] for record in records} == {"A:VAL:2"}
    assert sum(build_cosplif_tokens(result, "L:LIG:1").values()) == 4
    blocks = build_ligand_blocks(result, "L:LIG:1")["blocks"]
    assert sum(block["interaction_counts"].get("hydrophobic", 0) for block in blocks) == 2


def test_cosplif_distinguishes_enclosure_but_preserves_unique_pair_contributions():
    result = _result()
    result.interactions = [_enclosure(result, 0, [1, 2, 3, 4]),
                           _enclosure(result, 1, [0, 2, 3, 5], order=2)]
    enclosed_tokens = build_cosplif_tokens(result, "L:LIG:1")
    enclosed_blocks = build_ligand_blocks(result, "L:LIG:1")["blocks"]
    assert sum(enclosed_tokens.values()) == 10  # Two resolutions per unique ligand-target leg.
    assert sum(block["interaction_counts"]["hydrophobic"] for block in enclosed_blocks) == 5

    plain = deepcopy(result)
    plain.interactions = [_contact(result, first, second, order=index)
                          for index, (first, second) in enumerate([(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)], 1)]
    plain_tokens = build_cosplif_tokens(plain, "L:LIG:1")
    assert sum(plain_tokens.values()) == sum(enclosed_tokens.values())
    assert not plain_tokens.keys() & enclosed_tokens.keys()
    assert build_ligand_blocks(plain, "L:LIG:1")["blocks"] == enclosed_blocks
    assert all(record["interaction_subtype"] is None for record in _ifm_records(plain))
    assert project_interactions_for_scoring(plain.interactions) == plain.interactions


def test_imported_payload_with_both_enclosure_and_plain_leg_scores_pair_once():
    result = _result()
    result.interactions = [_enclosure(result, 0, [1, 2, 3, 4]), _contact(result, 0, 1, order=2)]
    assert len(_ifm_records(result)) == 4
    assert sum(build_cosplif_tokens(result, "L:LIG:1").values()) == 8


def test_ordinary_contact_retains_existing_cosplif_token_hashes():
    result = _result()
    result.interactions = [_contact(result, 0, 1)]
    assert build_cosplif_tokens(result, "L:LIG:1") == {
        "6d8073e2a693149f032779b90c0fea2f3ebd2d2c": 1,
        "932723dc1ccca330b926ab8c15f9aaa816a5865f": 1,
    }


def _record(subtype, family="hydrophobic"):
    return InteractionRecord(
        "record", "L:LIG:1", "target", family, subtype, "100", (0,),
        "residue", "A:VAL:2", "sidechain", (1,), geometry={"distance": 3.0},
    )


@pytest.mark.parametrize("reference_subtype,query_subtype,family,expected", [
    (None, "hydrophobic_enclosure", "hydrophobic", 0.75),
    ("hydrophobic_enclosure", None, "hydrophobic", 0.75),
    ("hydrophobic_enclosure", "hydrophobic_enclosure", "hydrophobic", 1.0),
    (None, None, "hydrophobic", 1.0),
    (None, "sidechain_hbond", "hydrogen_bond", 1.0),
])
def test_direct_and_indexed_ifm_subtype_credit_agree(reference_subtype, query_subtype, family, expected):
    config = IFMConfig()
    reference = _record(reference_subtype, family)
    query = _record(query_subtype, family)
    expected *= config.family_weights[family]
    assert score_interaction_pair(reference, query, config, strict=True) == expected
    codebook = {}
    indexed_reference = _protein_scoring_record(reference, config, codebook, 0, {})
    indexed_query = _protein_scoring_record(query, config, codebook, 1, {})
    assert _score_indexed_interaction_pair(indexed_reference, indexed_query,
                                         _protein_score_params(config), strict=True) == expected
    assert reference.to_dict()["interaction_subtype"] == reference_subtype
    assert query.to_dict()["interaction_subtype"] == query_subtype


def test_enclosure_subtype_credit_uses_existing_configuration():
    config = IFMConfig.from_dict({"matching": {"subtype_partial_credit": 0.4}})
    assert score_interaction_pair(_record(None), _record("hydrophobic_enclosure"),
                                  config, strict=True) == 0.4


def test_malformed_enclosure_does_not_silently_turn_into_cross_shell_pairs():
    result = _result()
    enclosure = _enclosure(result, 0, [1, 2, 3, 4])
    enclosure.metadata["legs"][0]["atom_ids"] = [1, 2]
    with pytest.raises(ValueError, match="center to one neighbor"):
        project_interactions_for_scoring([enclosure])
