"""A hydrophobic endpoint's own hydrogen is not a third-party obstruction."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from math import sqrt

import pytest

from chemur.api import assign_for_atoms
from chemur.models import AtomRecord, ComponentRecord
from chemur.parser import normalize_element
from chemur.profile import load_profile


def _atom(atom_id, coord, *, component=1, element="C", hydrophobic=False):
    molecule_type = "ligand" if component == 1 else "protein"
    residue_name = "LIG" if component == 1 else "ALA"
    return AtomRecord(
        atom_id, f"{element}{atom_id}", element, coord, residue_name,
        str(component), "A", f"A:{residue_name}:{component}", molecule_type,
        is_hydrophobe=hydrophobic,
    )


def _bond(first, second):
    first.bonds += (second.atom_id,)
    second.bonds += (first.atom_id,)


def _analyze(atoms, *, enclosure=True):
    grouped = defaultdict(list)
    for atom in atoms:
        grouped[atom.component_id].append(atom)
    components = [
        ComponentRecord(
            component_id, members[0].residue_name, members[0].chain_id,
            members[0].residue_id, members[0].molecule_type,
            tuple(atom.atom_id for atom in members),
        )
        for component_id, members in grouped.items()
    ]
    profile = load_profile({
        "rules": {"hydrophobic_enclosure": {"enabled": enclosure}},
    })
    return assign_for_atoms(atoms, components, profile)


def _pair():
    return [
        _atom(10, (0., 0., 0.), hydrophobic=True),
        _atom(20, (3.5, 0., 0.), component=2, hydrophobic=True),
    ]


@pytest.mark.parametrize("endpoint", [0, 1])
@pytest.mark.parametrize("reverse_atoms", [False, True])
@pytest.mark.parametrize("isotope", ["H", "D", "T"])
def test_endpoint_hydrogen_does_not_block_hydrophobic_contact(
    endpoint, reverse_atoms, isotope,
):
    atoms = _pair()
    hydrogen = _atom(
        90, (1. if endpoint == 0 else 2.5, 0., 0.),
        component=endpoint + 1, element=normalize_element(isotope),
    )
    if isotope != "H":
        hydrogen.metadata["isotope"] = isotope
    _bond(atoms[endpoint], hydrogen)
    atoms.append(hydrogen)
    if reverse_atoms:
        atoms.reverse()

    _, candidates, assigned = _analyze(atoms)

    candidate, = candidates
    contact, = assigned
    assert candidate.rejection_reason is None
    assert candidate.atom_ids == contact.atom_ids == (10, 20)
    assert candidate.interaction_type == contact.interaction_type == "hydrophobic"
    assert candidate.distance == contact.distance == pytest.approx(3.5)


def test_unrelated_hydrogen_still_blocks_hydrophobic_contact():
    atoms = _pair()
    atoms.append(_atom(90, (1.5, 0., 0.), component=3, element="H"))

    _, candidates, assigned = _analyze(atoms)

    candidate, = candidates
    assert candidate.atom_ids == (10, 20)
    assert candidate.distance == pytest.approx(3.5)
    assert candidate.rejection_reason == "occluded"
    assert assigned == []


def test_endpoint_bonded_heavy_atom_still_blocks_hydrophobic_contact():
    atoms = _pair()
    substituent = _atom(90, (1.5, 0., 0.))
    _bond(atoms[0], substituent)
    atoms.append(substituent)

    _, candidates, assigned = _analyze(atoms)

    candidate, = candidates
    assert candidate.rejection_reason == "occluded"
    assert assigned == []


def test_hydrogen_two_bonds_from_endpoint_still_blocks_contact():
    atoms = _pair()
    # The substituent itself is outside the 1 A occlusion cylinder. Only its
    # hydrogen enters the segment, two bonds from the participating carbon.
    substituent = _atom(80, (1., 1.2, 0.))
    hydrogen = _atom(90, (1.5, 0.3, 0.), element="H")
    _bond(atoms[0], substituent)
    _bond(substituent, hydrogen)
    atoms.extend([substituent, hydrogen])

    _, candidates, assigned = _analyze(atoms)

    candidate, = candidates
    assert candidate.rejection_reason == "occluded"
    assert assigned == []


def test_ignored_template_hydrogen_remains_exempt_without_endpoint_bond():
    atoms = _pair()
    hydrogen = _atom(90, (1.5, 0., 0.), component=3, element="H")
    hydrogen.metadata["ignored_by_template"] = True
    atoms.append(hydrogen)

    _, candidates, assigned = _analyze(atoms)

    candidate, = candidates
    contact, = assigned
    assert candidate.rejection_reason is None
    assert contact.atom_ids == (10, 20)


def test_endpoint_hydrogen_exemption_does_not_change_salt_bridge_occlusion():
    atoms = [
        _atom(10, (0., 0., 0.), element="N"),
        _atom(20, (3.5, 0., 0.), component=2, element="O"),
    ]
    atoms[0].formal_charge = 1
    atoms[1].formal_charge = -1
    hydrogen = _atom(90, (1., 0., 0.), element="H")
    _bond(atoms[0], hydrogen)
    atoms.append(hydrogen)

    _, candidates, assigned = _analyze(atoms)

    candidate, = candidates
    assert candidate.interaction_type == "salt_bridge"
    assert candidate.rejection_reason == "occluded"
    assert assigned == []


@pytest.mark.parametrize("endpoint", [0, 1])
@pytest.mark.parametrize("enclosure", [False, True])
def test_endpoint_hydrogen_preserves_tetrahedral_enclosure_and_pair_distances(
    endpoint, enclosure,
):
    coords = [(0., 0., 0.), (2., 2., 2.), (2., -2., -2.),
              (-2., 2., -2.), (-2., -2., 2.)]
    atoms = [
        _atom(10 * (i + 1), coord, component=i + 1, hydrophobic=True)
        for i, coord in enumerate(coords)
    ]
    _, _, baseline = _analyze(deepcopy(atoms), enclosure=enclosure)
    offset = .6 if endpoint == 0 else 1.4
    hydrogen = _atom(
        90, (offset, offset, offset), component=endpoint + 1, element="H",
    )
    _bond(atoms[endpoint], hydrogen)
    atoms.append(hydrogen)

    _, candidates, assigned = _analyze(atoms, enclosure=enclosure)

    assert [asdict(contact) for contact in assigned] == [
        asdict(contact) for contact in baseline
    ]
    pairs = [candidate for candidate in candidates
             if candidate.interaction_type == "hydrophobic"]
    assert {pair.atom_ids for pair in pairs} == {
        (10, 20), (10, 30), (10, 40), (10, 50),
    }
    assert all(pair.distance == pytest.approx(sqrt(12.)) for pair in pairs)
    if enclosure:
        group, = assigned
        assert group.interaction_type == "hydrophobic_enclosure"
        assert group.metadata["center_atom_id"] == 10
        assert group.metadata["neighbor_atom_ids"] == [20, 30, 40, 50]
        assert all(pair.rejection_reason == "subsumed_by_hydrophobic_enclosure"
                   for pair in pairs)
    else:
        assert len(assigned) == 4
        assert all(contact.interaction_type == "hydrophobic" for contact in assigned)
        assert all(pair.rejection_reason is None for pair in pairs)
