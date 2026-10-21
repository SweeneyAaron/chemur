"""Enclosure derives from valid contacts and replaces exactly its supporting pairs."""

from copy import deepcopy
from math import sqrt

import pytest

from chemur.api import assign_for_atoms
from chemur.assignment import assign_interactions
from chemur.features import perceive_features
from chemur.interactions import generate_candidates
from chemur.models import AtomRecord, ComponentRecord, ResourceDemand
from chemur.profile import load_profile


TETRAHEDRON = [(2., 2., 2.), (2., -2., -2.), (-2., 2., -2.), (-2., -2., 2.)]
OCTAHEDRON = [(2.5, 0., 0.), (-2.5, 0., 0.), (0., 2.5, 0.),
              (0., -2.5, 0.), (0., 0., 2.5), (0., 0., -2.5)]


def _atom(atom_id, coord, molecule_type="protein", hydrophobic=True):
    name = "LIG" if molecule_type == "ligand" else "ALA"
    return AtomRecord(atom_id, f"C{atom_id}", "C", coord, name, str(atom_id), "A",
                      f"A:{name}:{atom_id}", molecule_type, is_hydrophobe=hydrophobic)


def _shell(points=TETRAHEDRON, center_type="ligand"):
    atoms = [_atom(10, (0., 0., 0.), center_type)]
    atoms.extend(_atom(20 + 10 * i, point) for i, point in enumerate(points))
    return atoms


def _components(atoms):
    return [ComponentRecord(atom.component_id, atom.residue_name, atom.chain_id,
                            atom.residue_id, atom.molecule_type, (atom.atom_id,),
                            metadata={"ligand_assembly_id": atom.metadata["ligand_assembly_id"]}
                            if "ligand_assembly_id" in atom.metadata else {})
            for atom in atoms]


def _enclosures(interactions):
    return [item for item in interactions if item.interaction_type == "hydrophobic_enclosure"]


def _analyze(atoms, profile=None):
    return assign_for_atoms(atoms, _components(atoms), load_profile(profile or "default"))


@pytest.mark.parametrize("center_type", ["ligand", "protein", "nucleotide"])
def test_default_enclosure_replaces_pairs_and_preserves_geometry(center_type):
    atoms = _shell(center_type=center_type)
    features, candidates, assigned = _analyze(atoms)
    if center_type != "ligand":
        assert not _enclosures(assigned)
        assert len(assigned) == 4
        assert all(item.interaction_type == "hydrophobic" for item in assigned)
        return
    enclosure, = assigned
    assert enclosure.interaction_type == "hydrophobic_enclosure"
    assert enclosure.atom_ids == (10, 20, 30, 40, 50)
    assert enclosure.component_ids == tuple(sorted(atom.component_id for atom in atoms))
    assert set(enclosure.feature_ids) == {feature.feature_id for feature in features}
    assert enclosure.priority == 135
    assert enclosure.stage == "inter"
    assert enclosure.resource_demands == ()
    assert enclosure.angle is enclosure.offset is None
    assert enclosure.distance == pytest.approx(sqrt(12.))
    assert enclosure.metadata["center_atom_id"] == 10
    assert enclosure.metadata["neighbor_atom_ids"] == [20, 30, 40, 50]
    assert [leg["atom_ids"] for leg in enclosure.metadata["legs"]] == [
        [10, 20], [10, 30], [10, 40], [10, 50],
    ]
    pairs = [candidate for candidate in candidates if candidate.interaction_type == "hydrophobic"]
    assert len(pairs) == 4
    assert all(pair.rejection_reason == "subsumed_by_hydrophobic_enclosure" for pair in pairs)
    assert all(pair.metadata["enclosure_center_atom_ids"] == [10] for pair in pairs)


def test_replacement_preserves_shell_shell_and_unrelated_contacts():
    atoms = _shell(OCTAHEDRON) + [_atom(100, (20., 0., 0.)), _atom(110, (23.5, 0., 0.))]
    _, candidates, assigned = _analyze(atoms)
    assert len(_enclosures(assigned)) == 1
    ordinary = [item for item in assigned if item.interaction_type == "hydrophobic"]
    assert len(ordinary) == 13  # Twelve octahedral edges and one remote contact.
    assert (100, 110) in [item.atom_ids for item in ordinary]
    assert all(10 not in item.atom_ids for item in ordinary)
    assert sum(c.rejection_reason == "subsumed_by_hydrophobic_enclosure" for c in candidates) == 6
    assert [item.assignment_order for item in assigned] == list(range(1, len(assigned) + 1))


@pytest.mark.parametrize("rule,expected_pairs", [("hydrophobic_enclosure", 4), ("hydrophobic", 0)])
def test_disabled_rules_leave_expected_pair_results(rule, expected_pairs):
    _, candidates, assigned = _analyze(_shell(), {"rules": {rule: {"enabled": False}}})
    assert not _enclosures(candidates)
    assert len(assigned) == expected_pairs
    assert all(item.interaction_type == "hydrophobic" for item in assigned)


@pytest.mark.parametrize("distance,expected", [(3.4, 0), (3.5, 1)])
def test_enclosure_inherits_ordinary_contact_distance(distance, expected):
    _, _, assigned = _analyze(_shell(), {"rules": {"hydrophobic": {"distance": distance}}})
    assert len(_enclosures(assigned)) == expected


@pytest.mark.parametrize("points", [TETRAHEDRON[:3], OCTAHEDRON[:-1], OCTAHEDRON[:4]])
def test_incomplete_or_planar_shell_does_not_replace_pairs(points):
    _, candidates, assigned = _analyze(_shell(points))
    assert not _enclosures(candidates)
    assert all(item.interaction_type == "hydrophobic" for item in assigned)
    assert not any(c.rejection_reason == "subsumed_by_hydrophobic_enclosure" for c in candidates)


def test_occluded_contact_cannot_complete_enclosure():
    atoms = _shell()
    atoms.append(_atom(90, (1., 1., 1.), hydrophobic=False))
    _, candidates, assigned = _analyze(atoms)
    assert not _enclosures(assigned)
    blocked, = [candidate for candidate in candidates if candidate.atom_ids == (10, 20)]
    assert blocked.rejection_reason == "occluded"
    assert len(assigned) == 3


def test_imported_covalent_link_cannot_complete_enclosure():
    atoms = _shell()
    atoms[0].bonds = (20,)
    atoms[1].bonds = (10,)
    atoms[0].metadata["assembly_link_neighbors"] = [20]
    atoms[1].metadata["assembly_link_neighbors"] = [10]
    _, candidates, assigned = _analyze(atoms)
    assert not _enclosures(assigned)
    excluded, = [candidate for candidate in candidates if candidate.atom_ids == (10, 20)]
    assert excluded.rejection_reason == "covalent_neighborhood"


def test_same_ligand_assembly_cannot_supply_enclosing_contact():
    atoms = _shell()
    atoms[1].molecule_type = "ligand"
    for atom in atoms[:2]:
        atom.metadata["ligand_assembly_id"] = "assembly:connected"
    _, candidates, assigned = _analyze(atoms)
    assert not _enclosures(assigned)
    assert not any(candidate.atom_ids == (10, 20) for candidate in candidates)
    assert len(assigned) == 3


def test_only_accepted_enclosures_replace_contacts_regardless_of_priority():
    atoms = _shell()
    profile = load_profile({"rules": {"hydrophobic_enclosure": {"priority": 999}}})
    candidates = generate_candidates(atoms, perceive_features(atoms, _components(atoms)), profile)
    accepted = assign_interactions(atoms, deepcopy(candidates), profile)
    assert len(accepted) == len(_enclosures(accepted)) == 1

    enclosure, = _enclosures(candidates)
    enclosure.resource_demands = (ResourceDemand("donor", 10),)
    rejected = assign_interactions(atoms, candidates, profile)
    assert not _enclosures(rejected)
    assert len(rejected) == 4
    assert enclosure.rejection_reason == "resource_exhausted:donor:10"
    assert all(not item.rejection_reason for item in candidates if item.interaction_type == "hydrophobic")


def test_overlapping_enclosures_share_contacts_without_order_dependence():
    # The inner two atoms enclose one another using opposite tetrahedral shells.
    # A small occlusion radius keeps this fixture focused on graph replacement.
    atoms = [_atom(10, (-0.75, 0., 0.), "ligand"), _atom(20, (0.75, 0., 0.), "ligand")]
    atoms.extend(_atom(30 + 10 * i, point) for i, point in enumerate([
        (-2., 2., 2.), (-2., -2., 2.), (-2., 0., -2.),
        (2., 2., 2.), (2., -2., 2.), (2., 0., -2.),
    ]))
    profile = load_profile({"occlusion_radius": 0.1})
    candidates = generate_candidates(atoms, perceive_features(atoms, _components(atoms)), profile)
    forward = assign_interactions(atoms, deepcopy(candidates), profile)
    backward_candidates = deepcopy(list(reversed(candidates)))
    backward = assign_interactions(atoms, backward_candidates, profile)
    assert forward == backward
    assert {item.metadata["center_atom_id"] for item in _enclosures(forward)} == {10, 20}
    shared, = [candidate for candidate in backward_candidates
               if candidate.interaction_type == "hydrophobic" and candidate.atom_ids == (10, 20)]
    assert shared.rejection_reason == "subsumed_by_hydrophobic_enclosure"
    assert shared.metadata["enclosure_center_atom_ids"] == [10, 20]


def test_extra_valid_neighbor_is_preserved_with_its_individual_distance():
    atoms = _shell() + [_atom(60, (0., 0., 3.9))]
    _, _, assigned = _analyze(atoms)
    enclosure, = _enclosures(assigned)
    assert enclosure.metadata["neighbor_atom_ids"] == [20, 30, 40, 50, 60]
    assert enclosure.metadata["legs"][-1]["distance"] == pytest.approx(3.9)
    assert enclosure.distance == pytest.approx(3.9)
