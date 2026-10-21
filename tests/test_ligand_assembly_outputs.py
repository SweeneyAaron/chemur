"""Assembly-aware outputs retain residue identity and score the whole molecule."""
from copy import deepcopy
import json

import pytest

from chemur.cosplif import build_cosplif_payload, build_cosplif_tokens, build_ligand_blocks
from chemur.ifm_matcher import analysis_result_to_ifm_payloads
from chemur.ligand_selection import ligand_selections, resolve_ligand
from chemur.models import (AnalysisResult, AssignedInteraction, AtomRecord,
                           ComponentRecord, FeatureRecord, ResourceDemand)
from chemur.scoring import build_scoring_batch, score_pose_batch


FIRST = "A:L1:1"
SECOND = "B:L2:2"
ASSEMBLY = f"assembly:{FIRST}"
TARGET1 = "P:ASN:7"
TARGET2 = "P:SER:8"


def _result(*, registry=True):
    # The chemical graph is O-C-O, deliberately split across deposited residues.
    atoms = [
        AtomRecord(10, "O1", "O", (0., 0., 0.), "L1", "1", "A", FIRST, "ligand",
                   acceptor_capacity=2, donor_capacity=1, bonds=(20,)),
        AtomRecord(20, "C1", "C", (1.4, 0., 0.), "L1", "1", "A", FIRST, "ligand",
                   bonds=(10, 30)),
        AtomRecord(30, "O2", "O", (2.8, 0., 0.), "L2", "2", "B", SECOND, "ligand",
                   acceptor_capacity=2, donor_capacity=1, bonds=(20,)),
        AtomRecord(90, "ND2", "N", (0., 2.8, 0.), "ASN", "7", "P", TARGET1, "protein",
                   donor_capacity=1),
        AtomRecord(91, "OG", "O", (2.8, 2.8, 0.), "SER", "8", "P", TARGET2, "protein",
                   donor_capacity=1),
    ]
    assembly = {
        "assembly_id": ASSEMBLY, "component_ids": [FIRST, SECOND],
        "atom_ids": [10, 20, 30], "smiles": "OCO",
        "template_mapping": {"0": 10, "1": 20, "2": 30},
        "template_rings": [], "template_carbonyls": [], "template_amides": [],
        "bond_orders": [{"atom_id_1": 10, "atom_id_2": 20, "order": 1.0},
                        {"atom_id_1": 20, "atom_id_2": 30, "order": 1.0}],
        "links": [], "diagnostics": [],
    }
    components = [
        ComponentRecord(FIRST, "L1", "A", "1", "ligand", (10, 20),
                        {"ligand_assembly_id": ASSEMBLY, "ligand_assembly": assembly}),
        ComponentRecord(SECOND, "L2", "B", "2", "ligand", (30,),
                        {"ligand_assembly_id": ASSEMBLY}),
        ComponentRecord(TARGET1, "ASN", "P", "7", "protein", (90,)),
        ComponentRecord(TARGET2, "SER", "P", "8", "protein", (91,)),
    ]
    for atom in atoms[:3]:
        atom.metadata["ligand_assembly_id"] = ASSEMBLY
    features = [
        FeatureRecord(0, "acceptor", FIRST, (10,), atoms[0].coord),
        # Feature ownership is intentionally the assembly owner, not O2's residue.
        FeatureRecord(1, "acceptor", FIRST, (30,), atoms[2].coord),
        FeatureRecord(2, "amide", FIRST, (10, 20, 30), (1.4, 0., 0.),
                      metadata={"component_ids": [FIRST, SECOND], "ligand_assembly_id": ASSEMBLY}),
        FeatureRecord(3, "amide", TARGET1, (20, 30, 90), (1.4, 1., 0.),
                      metadata={"crosses_attachment_boundary": True}),
        FeatureRecord(4, "donor", TARGET1, (90,), atoms[3].coord),
        FeatureRecord(5, "donor", TARGET2, (91,), atoms[4].coord),
    ]
    interactions = [
        AssignedInteraction("hbond", (0, 4), (10, 90), (FIRST, TARGET1), 2.8, 30, "inter", 1,
                            resource_demands=(ResourceDemand("donor", 90),)),
        AssignedInteraction("hbond", (1, 5), (30, 91), (SECOND, TARGET2), 2.8, 30, "inter", 2,
                            resource_demands=(ResourceDemand("donor", 91),)),
        AssignedInteraction("hbond", (0, 1), (10, 30), (FIRST, SECOND), 2.8, 30, "intra", 3),
        AssignedInteraction("hbond", (3, 5), (20, 30, 90, 91),
                            (FIRST, SECOND, TARGET1, TARGET2), 2.8, 30, "inter", 4),
    ]
    return AnalysisResult(atoms, components, features, interactions,
                          metadata={"ligand_assemblies": {ASSEMBLY: assembly}} if registry else {})


@pytest.mark.parametrize("registry", [False, True])
def test_selection_reads_owner_metadata_and_result_registry(registry):
    result = _result(registry=registry)
    original_labels = [atom.label for atom in result.atoms]
    selections = ligand_selections(result)
    assert len(selections) == 1
    selection = selections[0]
    assert selection.ligand_id == ASSEMBLY
    assert selection.atom_ids == (10, 20, 30)
    for alias in (FIRST, SECOND, ASSEMBLY):
        assert resolve_ligand(result, alias) == selection
        assert resolve_ligand(result.to_dict(), alias) == selection
    assert [atom.label for atom in result.atoms] == original_labels
    with pytest.raises(ValueError, match="Ligand not found"):
        result.select_ligand("missing")


def test_split_outputs_keep_both_source_residues_and_all_ligand_features(tmp_path):
    result = _result()
    split = result.split_by_ligand()
    assert set(split) == {ASSEMBLY}
    selected = split[ASSEMBLY]
    assert selected.to_dict() == result.select_ligand(SECOND).to_dict()
    assert selected.metadata["ligand_component_id"] == FIRST
    assert selected.metadata["ligand_component_ids"] == [FIRST, SECOND]
    assert {atom.atom_id for atom in selected.atoms} == {10, 20, 30, 90, 91}
    assert {feature.feature_id for feature in selected.features} >= {0, 1, 2}
    assert [component.component_id for component in selected.components] == [FIRST, SECOND, TARGET1, TARGET2]
    written = result.write_ligand_outputs(tmp_path, output_format="both")
    assert len(written) == 2
    payload = json.loads(next(path for path in written if path.suffix == ".json").read_text())
    assert payload["metadata"]["ligand_assembly_id"] == ASSEMBLY
    csv = next(path for path in written if path.suffix == ".csv").read_text()
    assert "A:L1:1:O1:10" in csv and "B:L2:2:O2:30" in csv


def test_ifm_uses_entire_assembly_and_excludes_internal_and_boundary_groups():
    result = _result()
    payloads = analysis_result_to_ifm_payloads(result)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["ligand_id"] == ASSEMBLY
    records = payload["interaction_records"]
    assert len(records) == 2
    assert {tuple(record["ligand_atom_ids"]) for record in records} == {(10,), (30,)}
    assert {record["target_entity_id"] for record in records} == {TARGET1, TARGET2}
    assert {feature["feature_id"] for feature in payload["ligand_features"]} == {"0", "1", "2"}
    reloaded = json.loads(result.select_ligand(FIRST).to_json())
    assert analysis_result_to_ifm_payloads(reloaded) == payloads


def test_cosplif_aliases_have_identical_tokens_and_complete_blocks():
    result = _result()
    canonical = build_cosplif_payload(result, ASSEMBLY)
    assert canonical == build_cosplif_payload(result, FIRST)
    assert canonical == build_cosplif_payload(result, SECOND)
    assert sum(canonical["tokens"].values()) == 4  # Two external contacts, two resolutions.
    blocks = canonical["ligand_blocks_v1"]
    assert blocks["heavy_atom_count"] == 3
    assert {atom_id for block in blocks["blocks"] for atom_id in block["atom_ids"]} == {10, 20, 30}
    assert sum(block["interaction_count"] for block in blocks["blocks"]) == 2
    assert not blocks["warnings"]


def test_ligand_environments_do_not_walk_into_attached_protein():
    result = _result()
    atom = AtomRecord(95, "SG", "S", (1.4, -1.4, 0.), "CYS", "9", "P", "P:CYS:9", "protein",
                      bonds=(20,))
    result.atoms.append(atom)
    result.components.append(ComponentRecord("P:CYS:9", "CYS", "P", "9", "protein", (95,)))
    result.atoms[1].bonds = (10, 30, 95)
    before = build_cosplif_payload(result, ASSEMBLY)
    atom.element = "Se"
    assert build_cosplif_payload(result, ASSEMBLY) == before


def test_scoring_batch_has_one_entry_for_a_multi_residue_ligand():
    result = _result()
    batch = build_scoring_batch({"reference": result, "pose": deepcopy(result)})
    assert set(batch["results"]) == {"reference", "pose"}
    for payload in batch["results"].values():
        assert payload["source_assembly_id"] == ASSEMBLY
        assert payload["source_component_ids"] == [FIRST, SECOND]
        assert payload["source_component_id"] == FIRST
    rows = score_pose_batch(batch, "reference")
    assert len(rows) == 2
    assert rows[0]["cosplif_score"] == rows[1]["cosplif_score"]


def test_single_residue_without_assembly_metadata_keeps_legacy_identity():
    result = _result()
    for atom in result.atoms[:3]:
        atom.component_id = FIRST
        atom.metadata.clear()
    result.components[0].atom_ids = (10, 20, 30)
    result.components[0].metadata = {"smiles": "OCO", "template_mapping": {"0": 10, "1": 20, "2": 30}}
    del result.components[1]
    result.metadata.clear()
    result.interactions = result.interactions[:2]
    result.interactions[1].component_ids = (FIRST, TARGET2)
    assert set(result.split_by_ligand()) == {FIRST}
    assert analysis_result_to_ifm_payloads(result)[0]["ligand_id"] == FIRST
    assert sum(build_cosplif_tokens(result, FIRST).values()) == 4
    assert build_ligand_blocks(result, FIRST)["heavy_atom_count"] == 3


def _attached_product(context_element):
    """A deposited acetyl ligand joined to explicitly mapped polymer context."""
    from chemur.assemblies import prepare_assemblies

    specs = [(10, "CM", "C", FIRST, "ligand"), (20, "CO", "C", FIRST, "ligand"),
             (30, "O", "O", SECOND, "ligand"), (90, "N", "N", TARGET1, "protein"),
             (91, "X1", context_element, TARGET1, "protein"),
             (92, "C2", "C", TARGET1, "protein"), (93, "C3", "C", TARGET1, "protein"),
             (94, "C4", "C", TARGET1, "protein")]
    bonds = {10: (20,), 20: (10, 30), 30: (20,), 90: (91,), 91: (90, 92),
             92: (91, 93), 93: (92, 94), 94: (93,)}
    atoms = [AtomRecord(aid, name, element, (float(position), 0., 0.),
                        cid.split(":")[1], cid.split(":")[2], cid.split(":")[0], cid, kind,
                        bonds=bonds[aid])
             for position, (aid, name, element, cid, kind) in enumerate(specs)]
    components = [ComponentRecord(FIRST, "L1", "A", "1", "ligand", (10, 20)),
                  ComponentRecord(SECOND, "L2", "B", "2", "ligand", (30,)),
                  ComponentRecord(TARGET1, "ASN", "P", "7", "protein", (90, 91, 92, 93, 94))]
    atom_map = {str(number): {"component": atom.component_id, "atom": atom.name}
                for number, atom in enumerate(atoms, 1)}
    context = "[CH2:5]" if context_element == "C" else "[O:5]"
    override = {"version": 1, "products": [{
        "components": [FIRST, SECOND],
        "product_smiles": f"[CH3:1][C:2](=[O:3])[NH:4]{context}[CH2:6][CH2:7][CH3:8]",
        "atom_map": atom_map, "update_polymer_atoms": [4],
    }]}
    assert prepare_assemblies(atoms, components, assembly_overrides=override,
                              fetch_ccd=False, strict=True) == {FIRST, SECOND}
    return AnalysisResult(atoms, components, [], [])


def test_prepared_product_serialized_indices_match_source_atoms_and_bonds():
    from rdkit import Chem

    result = _attached_product("C")
    selection = resolve_ligand(result, ASSEMBLY)
    molecule = Chem.MolFromSmiles(selection.chemistry["smiles"])
    mapping = {int(index): atom_id for index, atom_id in selection.chemistry["template_mapping"].items()}
    by_id = {atom.atom_id: atom for atom in result.atoms}
    assert set(mapping.values()) == set(by_id)
    assert selection.atom_ids == (10, 20, 30)
    for index, atom_id in mapping.items():
        assert molecule.GetAtomWithIdx(index).GetSymbol() == by_id[atom_id].element
    orders = {frozenset((record["atom_id_1"], record["atom_id_2"])): record["order"]
              for record in selection.chemistry["bond_orders"]}
    for bond in molecule.GetBonds():
        assert orders[frozenset((mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]))] == bond.GetBondTypeAsDouble()


def test_prepared_attached_ligand_fingerprint_excludes_polymer_chemistry():
    first = build_ligand_blocks(_attached_product("C"), ASSEMBLY)
    second = build_ligand_blocks(_attached_product("O"), ASSEMBLY)
    assert first["heavy_atom_count"] == second["heavy_atom_count"] == 3
    assert first == second
    assert all(bit.startswith("rdkit-morgan:") for block in first["blocks"] for bit in block["fingerprint"])
