"""Activation typing is separate from the availability of a donor hydrogen."""

import pytest

from chemur.api import analyze
from chemur.chemistry import (
    apply_ligand_smiles_templates,
    apply_rdkit_ligand_template,
    mark_untemplated_ligands_ignored,
    rdkit_atom_annotations,
)
from chemur.models import AtomRecord, ComponentRecord
from chemur.features import perceive_features
from chemur.parser import _assign_default_atom_chemistry, parse_structure

Chem = pytest.importorskip("rdkit.Chem")


def _mapped_activation(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    donor = next(atom for atom in molecule.GetAtoms() if atom.GetAtomMapNum() == 1)
    return rdkit_atom_annotations(molecule)[donor.GetIdx()].weak_hbond_activated


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("[CH3:1]N", True),
        ("[CH3:1]O", True),
        ("[CH3:1]F", True),
        ("[CH3:1]Cl", True),
        ("[CH3:1]Br", True),
        ("[CH3:1]I", True),
        ("[CH3:1]C=O", True),
        ("[CH3:1]C#N", True),
        ("[CH3:1]S(=O)(=O)C", True),
        ("[CH:1]#CC", True),
        ("[CH3:1]CC", False),
        ("[cH:1]1ccccc1", False),
        ("[cH:1]1ccoc1", False),
        ("[CH3:1]SC", False),
        ("[CH3:1]S(=O)C", False),
        ("[CH3:1]P", False),
        ("[C:1](#CC)C", False),
        ("[CH2-:1]N", False),
        ("[CH3:1][O-]", False),
        ("[CH3:1][NH-]", False),
        ("[CH3:1][C-]=O", False),
        ("[CH:1]#[C-]", False),
        ("[CH3:1]OC(=O)[O-]", True),
        ("[CH3:1]C(=O)[O-]", True),
        # An alkoxide must not be mistaken for a carbonyl on the adjacent carbon.
        ("[CH3:1]C[O-]", False),
        ("[CH3:1]CC[O-]", False),
    ],
)
def test_rdkit_weak_hbond_activation_motifs_and_local_charges(smiles, expected):
    assert _mapped_activation(smiles) is expected


@pytest.mark.parametrize(
    "smiles",
    [
        "[CH3:1]S(=O)(=O)C",
        "[CH3:1][S+](=O)([O-])C",
        "[CH3:1][S+2]([O-])([O-])C",
    ],
)
def test_sulfonyl_activation_is_resonance_independent(smiles):
    assert _mapped_activation(smiles) is True


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("[cH:1]1ccccc1", False),
        ("[cH:1]1ccoc1", False),
        ("O=c1[cH:1]c[nH]c(=O)[nH]1", True),
        ("[cH:1]1ncc[nH]1", True),
    ],
)
def test_aromatic_and_kekule_representations_agree(smiles, expected):
    aromatic = Chem.MolFromSmiles(smiles)
    kekule = Chem.Mol(aromatic)
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    aromatic_flags = rdkit_atom_annotations(aromatic)
    kekule_flags = rdkit_atom_annotations(kekule)
    assert {
        idx: annotation.weak_hbond_activated for idx, annotation in aromatic_flags.items()
    } == {
        idx: annotation.weak_hbond_activated for idx, annotation in kekule_flags.items()
    }
    donor = next(atom for atom in aromatic.GetAtoms() if atom.GetAtomMapNum() == 1)
    assert aromatic_flags[donor.GetIdx()].weak_hbond_activated is expected


def _atom(index, name, element, *, bonds=(), residue="ALA", molecule_type="protein", charge=0):
    return AtomRecord(
        index, name, element, (1.4 * index, 0.0, 0.0), residue, "1", "A",
        f"A:{residue}:1", molecule_type, bonds=bonds, formal_charge=charge,
    )


@pytest.mark.parametrize(
    "residue, donor_name, neighbor_name, neighbor_element, expected",
    [
        ("ALA", "CA", "N", "N", True),
        ("ALA", "CB", "CA", "C", False),
        ("HIS", "CE1", "ND1", "N", True),
        ("HIS", "CD2", "NE2", "N", True),
        ("PHE", "CD1", "CG", "C", False),
        ("MET", "CE", "SD", "S", False),
        ("CYS", "CB", "SG", "S", False),
        ("SER", "CB", "OG", "O", True),
    ],
)
def test_standard_residue_direct_activation(
    residue, donor_name, neighbor_name, neighbor_element, expected
):
    atoms = [
        _atom(0, donor_name, "C", bonds=(1,), residue=residue),
        _atom(1, neighbor_name, neighbor_element, bonds=(0,), residue=residue),
    ]
    _assign_default_atom_chemistry(atoms)
    assert atoms[0].weak_hbond_activated is expected
    # Atom order and an absent hydrogen do not change electronic activation.
    _assign_default_atom_chemistry(list(reversed(atoms)))
    assert atoms[0].weak_hbond_activated is expected


@pytest.mark.parametrize(
    "residue, donor_name, carbon_name, oxygen_name, molecule_type",
    [
        ("ALA", "CA", "C", "O", "protein"),
        ("ASN", "CB", "CG", "OD1", "protein"),
        ("ASP", "CB", "CG", "OD1", "protein"),
        ("ASP", "CB", "CG", "OD2", "protein"),
        ("GLN", "CG", "CD", "OE1", "protein"),
        ("GLU", "CG", "CD", "OE1", "protein"),
        ("GLU", "CG", "CD", "OE2", "protein"),
        *[(name, "C5", "C4", "O4", "nucleotide") for name in ("U", "DU", "URA", "T", "DT", "THY")],
        *[(name, "C5", "C6", "O6", "nucleotide") for name in ("G", "DG", "GUA")],
        # A cytosine C2 carbonyl is also recognized. The minimal graph isolates
        # the carbonyl rule from the direct-nitrogen rule elsewhere in the base.
        *[(name, "CX", "C2", "O2", "nucleotide") for name in ("C", "DC", "CYT", "U", "DU", "URA", "T", "DT", "THY")],
    ],
)
def test_standard_residue_carbonyl_adjacency(
    residue, donor_name, carbon_name, oxygen_name, molecule_type
):
    atoms = [
        _atom(0, donor_name, "C", bonds=(1,), residue=residue, molecule_type=molecule_type),
        _atom(1, carbon_name, "C", bonds=(0, 2), residue=residue, molecule_type=molecule_type),
        _atom(2, oxygen_name, "O", bonds=(1,), residue=residue, molecule_type=molecule_type),
    ]
    _assign_default_atom_chemistry(atoms)
    assert atoms[0].weak_hbond_activated is True
    _assign_default_atom_chemistry(list(reversed(atoms)))
    assert atoms[0].weak_hbond_activated is True

    atoms[1].bonds = (0,)
    atoms[2].bonds = ()
    _assign_default_atom_chemistry(atoms)
    assert atoms[0].weak_hbond_activated is False
    atoms[1].bonds = (0, 2)
    _assign_default_atom_chemistry(atoms[:2])
    assert atoms[0].weak_hbond_activated is False


@pytest.mark.parametrize("charge_position", [0, 1])
def test_structure_activation_excludes_local_negative_charges(charge_position):
    atoms = [
        _atom(0, "CB", "C", bonds=(1,), residue="SER"),
        _atom(1, "OG", "O", bonds=(0,), residue="SER"),
    ]
    atoms[charge_position].formal_charge = -1
    _assign_default_atom_chemistry(atoms)
    assert atoms[0].weak_hbond_activated is False


def test_structure_carbonyl_needs_standard_identity_and_same_component():
    atoms = [
        _atom(0, "CA", "C", bonds=(1,)),
        _atom(1, "C", "C", bonds=(0, 2)),
        _atom(2, "O", "O", bonds=(1,)),
    ]
    atoms[2].component_id = "B:ALA:2"
    _assign_default_atom_chemistry(atoms)
    assert atoms[0].weak_hbond_activated is False
    atoms[2].component_id = atoms[1].component_id
    for atom in atoms:
        atom.residue_name = "LIG"
        atom.molecule_type = "ligand"
    _assign_default_atom_chemistry(atoms)
    assert atoms[0].weak_hbond_activated is False


def _ethanol_records():
    atoms = [
        _atom(0, "C1", "C", bonds=(1,), residue="EOH", molecule_type="ligand"),
        _atom(1, "C2", "C", bonds=(0, 2), residue="EOH", molecule_type="ligand"),
        _atom(2, "O1", "O", bonds=(1,), residue="EOH", molecule_type="ligand"),
    ]
    component = ComponentRecord("A:EOH:1", "EOH", "A", "1", "ligand", (0, 1, 2))
    return atoms, component


@pytest.mark.parametrize("source", ["smiles", "ccd", "sdf"])
def test_template_application_transfers_activation_and_ignore_clears_it(source):
    atoms, component = _ethanol_records()
    # Seed opposite flags to check both true and false are transferred.
    atoms[0].weak_hbond_activated = True
    molecule = Chem.MolFromSmiles("CCO")
    apply_rdkit_ligand_template(
        atoms, component, molecule, {0: 0, 1: 1, 2: 2}, smiles="CCO", source=source,
    )
    assert [atom.weak_hbond_activated for atom in atoms] == [False, True, False]
    assert [atom.hydrogen_count for atom in atoms] == [3, 2, 1]
    assert atoms[0].carbon_donor_capacity == 3

    mark_untemplated_ligands_ignored(atoms, [component], {}, reason="test skip")
    assert all(not atom.weak_hbond_activated for atom in atoms)


def test_smiles_mapping_transfers_activation_to_the_correct_carbon():
    atoms, component = _ethanol_records()
    apply_ligand_smiles_templates(atoms, [component], {"EOH": "OCC"})
    assert [atom.weak_hbond_activated for atom in atoms] == [False, True, False]


@pytest.mark.parametrize(
    "residue_name, donor_name, donor_charge, neighbor_name, neighbor_element, neighbor_charge, activated, charge_features",
    [
        ("LIG", "C1", -1, "O1", "O", 0, False, []),
        ("LIG", "C1", 0, "O1", "O", -1, False, []),
        ("LYS", "CE", 0, "NZ", "N", 1, True, ["cation"]),
        ("ASP", "CG", 0, "OD1", "O", -1, False, ["anion"]),
    ],
)
def test_deposited_charge_affects_activation_without_duplicating_charge_features(
    tmp_path, residue_name, donor_name, donor_charge, neighbor_name,
    neighbor_element, neighbor_charge, activated, charge_features,
):
    gemmi = pytest.importorskip("gemmi")
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = residue_name
    residue.seqid = gemmi.SeqId(1, " ")
    for name, element, x, charge in [
        (donor_name, "C", 0.0, donor_charge),
        (neighbor_name, neighbor_element, 1.4, neighbor_charge),
    ]:
        atom = gemmi.Atom()
        atom.name = name
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(x, 0.0, 0.0)
        atom.charge = charge
        residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    path = tmp_path / "charged.pdb"
    structure.write_pdb(str(path))
    atoms, components = parse_structure(path)
    assert [atom.metadata.get("structure_formal_charge", 0) for atom in atoms] == [
        donor_charge, neighbor_charge,
    ]
    assert all(atom.formal_charge == 0 for atom in atoms)
    assert atoms[0].weak_hbond_activated is activated
    assert [
        feature.feature_type
        for feature in perceive_features(atoms, components)
        if feature.feature_type in {"cation", "anion"}
    ] == charge_features


def test_deposited_protein_charges_do_not_duplicate_salt_bridges(tmp_path):
    gemmi = pytest.importorskip("gemmi")
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for index, (name, atom_name, element, x, charge) in enumerate(
        [("LYS", "NZ", "N", 0.0, 1), ("ASP", "OD1", "O", 3.0, -1)], start=1
    ):
        residue = gemmi.Residue()
        residue.name = name
        residue.seqid = gemmi.SeqId(index, " ")
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(x, 0.0, 0.0)
        atom.charge = charge
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    path = tmp_path / "charged_protein.pdb"
    structure.write_pdb(str(path))

    result = analyze(path, fetch_ccd_smiles=False)
    salt_bridges = [
        interaction for interaction in result.interactions
        if interaction.interaction_type == "salt_bridge"
    ]
    assert len(salt_bridges) == 1
    assert salt_bridges[0].atom_ids == (0, 1)
