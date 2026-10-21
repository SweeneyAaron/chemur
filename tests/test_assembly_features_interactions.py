"""Linked ligand residues share chemistry without losing source ownership."""

from math import cos, pi, sin

import pytest

from chemur.api import assign_for_atoms
from chemur.features import perceive_features
from chemur.interactions import generate_candidates
from chemur.models import AtomRecord, ComponentRecord, FeatureRecord
from chemur.profile import load_profile


LIGAND_A = "L:AAA:1"
LIGAND_B = "L:BBB:2"
PROTEIN = "P:ASN:3"
ASSEMBLY = "assembly:linked-ligand"


def _atom(atom_id, element, coord, component_id=LIGAND_A, molecule_type="ligand", **kwargs):
    chain, residue, number = component_id.split(":")
    return AtomRecord(atom_id, element + str(atom_id), element, coord, residue, number,
                      chain, component_id, molecule_type, **kwargs)


def _components(atoms):
    components = []
    for component_id in dict.fromkeys(atom.component_id for atom in atoms):
        members = [atom for atom in atoms if atom.component_id == component_id]
        first = members[0]
        components.append(ComponentRecord(component_id, first.residue_name, first.chain_id,
                                          first.residue_id, first.molecule_type,
                                          tuple(atom.atom_id for atom in members)))
    return components


def _assemble(atoms, components, *, rings=(), carbonyls=(), amides=()):
    ligand_atoms = [atom for atom in atoms if atom.molecule_type == "ligand"]
    member_ids = sorted({atom.component_id for atom in ligand_atoms})
    for atom in ligand_atoms:
        atom.metadata["ligand_assembly_id"] = ASSEMBLY
    for component in components:
        if component.component_id in member_ids:
            component.metadata["ligand_assembly_id"] = ASSEMBLY
    owner = next(component for component in components if component.component_id == member_ids[0])
    owner.metadata["ligand_assembly"] = {
        "assembly_id": ASSEMBLY,
        "component_ids": member_ids,
        "atom_ids": [atom.atom_id for atom in ligand_atoms],
        "template_rings": list(rings),
        "template_carbonyls": list(carbonyls),
        "template_amides": list(amides),
        "links": [],
    }


def _link(atoms, first_id, second_id):
    by_id = {atom.atom_id: atom for atom in atoms}
    for source, target in ((first_id, second_id), (second_id, first_id)):
        atom = by_id[source]
        atom.bonds = tuple(sorted(set(atom.bonds) | {target}))
        atom.metadata.setdefault("assembly_link_neighbors", []).append(target)


def test_split_ring_is_perceived_once_with_all_source_residues():
    atoms = [
        _atom(i, "C", (1.4 * cos(i * pi / 3), 1.4 * sin(i * pi / 3), 0.),
              LIGAND_A if i < 3 else LIGAND_B, is_aromatic=True, is_hydrophobe=True)
        for i in range(6)
    ]
    components = _components(atoms)
    _assemble(atoms, components, rings=[tuple(range(6))])
    features = perceive_features(atoms, components)

    ring, = [feature for feature in features if feature.feature_type == "ring"]
    assert ring.atom_ids == tuple(range(6))
    assert ring.metadata["component_ids"] == [LIGAND_A, LIGAND_B]
    assert ring.metadata["ligand_assembly_id"] == ASSEMBLY
    assert ring.metadata["aromatic"] is True
    assert not ring.metadata.get("crosses_attachment_boundary")
    assert all(feature.metadata["ligand_assembly_id"] == ASSEMBLY for feature in features)


@pytest.mark.parametrize("attached_to_polymer", [False, True])
def test_assembly_carbonyl_and_amide_keep_boundary_ownership(attached_to_polymer):
    nitrogen_component = PROTEIN if attached_to_polymer else LIGAND_B
    atoms = [
        _atom(0, "C", (0., 0., 0.), bonds=(1, 2, 3)),
        _atom(1, "O", (0., 1.2, 0.), bonds=(0,)),
        _atom(2, "N", (1.3, -0.5, 0.), nitrogen_component,
              "protein" if attached_to_polymer else "ligand", bonds=(0,)),
        _atom(3, "C", (-1.3, -0.5, 0.), bonds=(0,)),
    ]
    components = _components(atoms)
    _assemble(atoms, components, carbonyls=[(0, 1)], amides=[(0, 1, 2)])
    grouped = [feature for feature in perceive_features(atoms, components)
               if feature.feature_type in {"carbonyl", "amide"}]

    assert [feature.feature_type for feature in grouped] == ["carbonyl", "amide"]
    for feature in grouped:
        assert feature.metadata["component_ids"] == sorted([LIGAND_A, nitrogen_component])
        if attached_to_polymer:
            assert feature.component_id == PROTEIN
            assert feature.metadata["crosses_attachment_boundary"] is True
            assert "ligand_assembly_id" not in feature.metadata
        else:
            assert feature.metadata["ligand_assembly_id"] == ASSEMBLY
            assert not feature.metadata.get("crosses_attachment_boundary")


def test_unresolved_polymer_atom_suppresses_point_and_grouped_chemistry():
    atoms = [
        _atom(0, "C", (0., 0., 0.), bonds=(1, 2)),
        _atom(1, "O", (0., 1.2, 0.), bonds=(0,), acceptor_capacity=2),
        _atom(2, "N", (1.3, -0.5, 0.), PROTEIN, "protein", bonds=(0,),
              donor_capacity=1, acceptor_capacity=1, formal_charge=1, is_hydrophobe=True,
              metadata={"assembly_chemistry_unresolved": True}),
    ]
    components = _components(atoms)
    _assemble(atoms, components, carbonyls=[(0, 1)], amides=[(0, 1, 2)])
    features = perceive_features(atoms, components)

    assert [(feature.feature_type, feature.atom_ids) for feature in features] == [("acceptor", (1,))]


@pytest.mark.parametrize("modified", [False, True])
def test_product_context_only_overrides_modified_polymer_groups(modified):
    atoms = [_atom(0, "C", (8., 0., 0.))]
    for index, name in enumerate(("CG", "CD1", "CE1", "CZ", "CE2", "CD2"), start=1):
        theta = (index - 1) * pi / 3
        atom = _atom(index, "C", (1.4 * cos(theta), 1.4 * sin(theta), 0.),
                     "P:PHE:3", "protein", is_aromatic=not modified)
        atom.name = name
        atoms.append(atom)
    components = _components(atoms)
    _assemble(atoms, components, rings=[tuple(range(1, 7))])
    components[0].metadata["ligand_assembly"]["updated_polymer_atom_ids"] = list(range(1, 7)) if modified else []
    rings = [feature for feature in perceive_features(atoms, components) if feature.feature_type == "ring"]

    ring, = rings
    assert ring.component_id == "P:PHE:3"
    assert ring.metadata["component_ids"] == ["P:PHE:3"]
    assert ring.metadata["aromatic"] is (not modified)
    assert ring.metadata["source"] == ("smiles" if modified else "protein_template")
    assert "ligand_assembly_id" not in ring.metadata


def test_carbonyl_product_override_includes_updated_plane_substituent():
    atoms = [
        _atom(0, "C", (8., 0., 0.)),
        _atom(1, "C", (0., 0., 0.), PROTEIN, "protein", bonds=(2, 3)),
        _atom(2, "O", (0., 1.2, 0.), PROTEIN, "protein", bonds=(1,)),
        _atom(3, "N", (1.3, -.5, 0.), PROTEIN, "protein", bonds=(1,)),
    ]
    atoms[1].name, atoms[2].name = "C", "O"
    components = _components(atoms)
    _assemble(atoms, components, carbonyls=[(1, 2)])
    components[0].metadata["ligand_assembly"]["updated_polymer_atom_ids"] = [3]
    carbonyl, = [feature for feature in perceive_features(atoms, components) if feature.feature_type == "carbonyl"]

    assert carbonyl.component_id == PROTEIN
    assert carbonyl.metadata["source"] == "smiles"
    assert set(carbonyl.atom_ids) == {1, 2, 3}


def test_unresolved_atom_still_blocks_contact_geometry():
    atoms = [
        _atom(0, "C", (0., 0., 0.), is_hydrophobe=True),
        _atom(1, "C", (3., 0., 0.), PROTEIN, "protein", is_hydrophobe=True),
        _atom(2, "C", (1.5, 0., 0.), PROTEIN, "protein", is_hydrophobe=True,
              metadata={"assembly_chemistry_unresolved": True}),
    ]
    features, candidates, assigned = assign_for_atoms(atoms, _components(atoms), load_profile())
    contact, = [candidate for candidate in candidates if candidate.interaction_type == "hydrophobic"]

    assert not any(2 in feature.atom_ids for feature in features)
    assert contact.rejection_reason == "occluded"
    assert assigned == []


def test_hbond_between_residues_of_same_assembly_is_intra_with_actual_components():
    atoms = [
        _atom(0, "N", (0., 0., 0.), donor_capacity=1, bonds=(1,)),
        _atom(1, "H", (1., 0., 0.), bonds=(0,)),
        _atom(2, "O", (2.8, 0., 0.), LIGAND_B, acceptor_capacity=1, bonds=(3,)),
        _atom(3, "C", (4.0, 0., 0.), LIGAND_B, bonds=(2,)),
    ]
    components = _components(atoms)
    _assemble(atoms, components)
    _, _, assigned = assign_for_atoms(atoms, components, load_profile())
    contact, = assigned

    assert contact.interaction_type == "hbond"
    assert contact.stage == "intra"
    assert contact.component_ids == (LIGAND_A, LIGAND_B)


@pytest.mark.parametrize("interaction_type", ["hydrophobic", "salt_bridge", "pipi_stack"])
def test_same_assembly_uses_existing_same_ligand_exclusions(interaction_type):
    atoms = [_atom(0, "C", (0., 0., 0.)), _atom(1, "C", (0., 0., 3.), LIGAND_B)]
    feature_types = {
        "hydrophobic": ("hydrophobe", "hydrophobe"),
        "salt_bridge": ("cation", "anion"),
        "pipi_stack": ("ring", "ring"),
    }[interaction_type]
    features = [
        FeatureRecord(i, feature_type, atom.component_id, (atom.atom_id,), atom.coord,
                      normal=(0., 0., 1.), metadata={"ligand_assembly_id": ASSEMBLY})
        for i, (atom, feature_type) in enumerate(zip(atoms, feature_types))
    ]
    assert not any(candidate.interaction_type == interaction_type
                   for candidate in generate_candidates(atoms, features, load_profile()))

    for feature in features:
        feature.metadata.clear()
    assert any(candidate.interaction_type == interaction_type
               for candidate in generate_candidates(atoms, features, load_profile()))


@pytest.mark.parametrize("link_position", [0, 1])
def test_two_bond_contact_across_imported_link_is_raw_covalent_neighborhood(link_position):
    atoms = [
        _atom(0, "C", (0., 0., 0.), is_hydrophobe=True, bonds=(1,)),
        _atom(1, "C", (1.5, 0., 0.), PROTEIN if link_position == 0 else LIGAND_A,
              "protein" if link_position == 0 else "ligand", bonds=(0, 2)),
        _atom(2, "C", (3., 0., 0.), PROTEIN, "protein", is_hydrophobe=True, bonds=(1,)),
    ]
    _link(atoms, link_position, link_position + 1)
    _, candidates, assigned = assign_for_atoms(atoms, _components(atoms), load_profile())
    contact, = [candidate for candidate in candidates if candidate.interaction_type == "hydrophobic"]

    # The middle atom would also occlude this segment; covalent topology wins.
    assert contact.rejection_reason == "covalent_neighborhood"
    assert assigned == []


@pytest.mark.parametrize("imported", [False, True])
def test_direct_bond_exclusion_requires_imported_link_provenance(imported):
    atoms = [
        _atom(0, "C", (0., 0., 0.), is_hydrophobe=True, bonds=(1,)),
        _atom(1, "C", (1.5, 0., 0.), PROTEIN, "protein", is_hydrophobe=True, bonds=(0,)),
    ]
    if imported:
        _link(atoms, 0, 1)
    _, candidates, assigned = assign_for_atoms(atoms, _components(atoms), load_profile())

    assert candidates[0].rejection_reason == ("covalent_neighborhood" if imported else None)
    assert len(assigned) == int(not imported)


def test_three_bond_path_across_attachment_does_not_exclude_longer_range_contact():
    atoms = [
        _atom(0, "C", (0., 0., 0.), is_hydrophobe=True, bonds=(1,)),
        _atom(1, "C", (1., 1.4, 0.), bonds=(0, 2)),
        _atom(2, "C", (2., 1.4, 0.), PROTEIN, "protein", bonds=(1, 3)),
        _atom(3, "C", (3., 0., 0.), PROTEIN, "protein", is_hydrophobe=True, bonds=(2,)),
    ]
    _link(atoms, 1, 2)
    _, _, assigned = assign_for_atoms(atoms, _components(atoms), load_profile())

    assert [(contact.interaction_type, contact.atom_ids) for contact in assigned] == [("hydrophobic", (0, 3))]


def test_link_within_ring_does_not_reject_external_contact_or_lose_component_ids():
    atoms = [
        _atom(i, "C", (1.4 * cos(i * pi / 3), 1.4 * sin(i * pi / 3), 0.),
              LIGAND_A if i < 3 else LIGAND_B, is_aromatic=True)
        for i in range(6)
    ]
    atoms.append(_atom(6, "N", (0., 0., 3.), PROTEIN, "protein", formal_charge=1))
    _link(atoms, 2, 3)
    components = _components(atoms)
    _assemble(atoms, components, rings=[tuple(range(6))])
    _, _, assigned = assign_for_atoms(atoms, components, load_profile())
    contact, = [contact for contact in assigned if contact.interaction_type == "cation_pi"]

    assert contact.stage == "inter"
    assert set(contact.component_ids) == {LIGAND_A, LIGAND_B, PROTEIN}


def test_water_cannot_bridge_two_residues_of_the_same_ligand_assembly():
    atoms = [
        _atom(0, "N", (-2.8, 0., 0.), donor_capacity=1, bonds=(1,)),
        _atom(1, "H", (-1.8, 0., 0.), bonds=(0,)),
        _atom(2, "O", (0., 0., 0.), "W:HOH:2", "solvent", acceptor_capacity=2),
        _atom(3, "N", (2.8, 0., 0.), LIGAND_B, donor_capacity=1, bonds=(4,)),
        _atom(4, "H", (1.8, 0., 0.), LIGAND_B, bonds=(3,)),
    ]
    components = _components(atoms)
    _assemble(atoms, components)
    _, candidates, assigned = assign_for_atoms(atoms, components, load_profile())

    assert [contact.interaction_type for contact in assigned] == ["hbond", "hbond"]
    assert not any(candidate.interaction_type == "solvent_bridge" for candidate in candidates)
