import warnings

import pytest

from chemur.chemistry import apply_ligand_smiles_templates, mark_untemplated_ligands_ignored
from chemur.errors import LigandTemplateError
from chemur.models import AtomRecord, ComponentRecord
from chemur.parser import parse_embedded_ligand_smiles

rdkit = pytest.importorskip("rdkit")


def _three_atom_ethanol_ligand():
    atoms = [
        AtomRecord(0, "C1", "C", (0.0, 0.0, 0.0), "EOH", "1", "A", "A:EOH:1", "ligand", bonds=(1,)),
        AtomRecord(1, "C2", "C", (1.5, 0.0, 0.0), "EOH", "1", "A", "A:EOH:1", "ligand", bonds=(0, 2)),
        AtomRecord(2, "O1", "O", (2.9, 0.0, 0.0), "EOH", "1", "A", "A:EOH:1", "ligand", bonds=(1,)),
    ]
    components = [ComponentRecord("A:EOH:1", "EOH", "A", "1", "ligand", (0, 1, 2))]
    return atoms, components


def test_smiles_template_transfers_basic_ligand_chemistry():
    atoms = [
        AtomRecord(0, "C1", "C", (0.0, 0.0, 0.0), "EOH", "1", "A", "A:EOH:1", "ligand", bonds=(1,)),
        AtomRecord(1, "C2", "C", (1.5, 0.0, 0.0), "EOH", "1", "A", "A:EOH:1", "ligand", bonds=(0, 2)),
        AtomRecord(2, "O1", "O", (2.9, 0.0, 0.0), "EOH", "1", "A", "A:EOH:1", "ligand", bonds=(1,)),
    ]
    components = [
        ComponentRecord("A:EOH:1", "EOH", "A", "1", "ligand", (0, 1, 2))
    ]

    apply_ligand_smiles_templates(atoms, components, {"EOH": "CCO"})

    oxygen = atoms[2]
    assert oxygen.acceptor_capacity == 2
    assert oxygen.donor_capacity == 1
    assert components[0].metadata["smiles"] == "CCO"


def test_deprotonated_template_ignores_extra_structure_hydrogen():
    atoms = [
        AtomRecord(0, "C1", "C", (0.0, 0.0, 0.0), "ACE", "1", "A", "A:ACE:1", "ligand", bonds=(1,)),
        AtomRecord(1, "C2", "C", (1.5, 0.0, 0.0), "ACE", "1", "A", "A:ACE:1", "ligand", bonds=(0, 2)),
        AtomRecord(2, "O1", "O", (2.9, 0.0, 0.0), "ACE", "1", "A", "A:ACE:1", "ligand", bonds=(1, 3)),
        AtomRecord(3, "H1", "H", (3.8, 0.0, 0.0), "ACE", "1", "A", "A:ACE:1", "ligand", bonds=(2,)),
    ]
    components = [
        ComponentRecord("A:ACE:1", "ACE", "A", "1", "ligand", (0, 1, 2, 3))
    ]

    apply_ligand_smiles_templates(atoms, components, {"ACE": "CC[O-]"})

    oxygen = atoms[2]
    hydrogen = atoms[3]
    assert oxygen.formal_charge == -1
    assert oxygen.donor_capacity == 0
    assert oxygen.metadata["ignored_hydrogen_atom_ids"] == [3]
    assert hydrogen.metadata["ignored_by_template"] is True


def test_untemplated_ligand_is_ignored_after_failed_ccd_lookup():
    atoms = [
        AtomRecord(0, "C1", "C", (0.0, 0.0, 0.0), "UNK", "1", "A", "A:UNK:1", "ligand", donor_capacity=1, is_hydrophobe=True),
    ]
    components = [
        ComponentRecord("A:UNK:1", "UNK", "A", "1", "ligand", (0,))
    ]

    mark_untemplated_ligands_ignored(
        atoms,
        components,
        {},
        reason="No CCD SMILES template was available",
    )

    assert components[0].metadata["ignored_by_template"] is True
    assert atoms[0].metadata["ignored_by_template"] is True
    assert atoms[0].donor_capacity == 0
    assert atoms[0].is_hydrophobe is False


def test_unmappable_ligand_is_skipped_and_warns_by_default():
    atoms, components = _three_atom_ethanol_ligand()

    # SMILES has one heavy atom but the structure has three -> unmappable.
    with pytest.warns(RuntimeWarning):
        apply_ligand_smiles_templates(atoms, components, {"EOH": "C"})

    assert components[0].metadata["ignored_by_template"] is True
    assert "ignored_reason" in components[0].metadata
    for atom in atoms:
        assert atom.metadata["ignored_by_template"] is True
        assert atom.acceptor_capacity == 0
        assert atom.donor_capacity == 0


def test_unmappable_ligand_raises_when_strict():
    atoms, components = _three_atom_ethanol_ligand()

    with pytest.raises(LigandTemplateError):
        apply_ligand_smiles_templates(atoms, components, {"EOH": "C"}, strict=True)


def _phosphate_ligand():
    """A phosphate P bonded to four interchangeable terminal oxygens."""
    atoms = [
        AtomRecord(0, "P", "P", (0.0, 0.0, 0.0), "PO4", "1", "A", "A:PO4:1", "ligand", bonds=(1, 2, 3, 4)),
        AtomRecord(1, "O1", "O", (1.5, 0.0, 0.0), "PO4", "1", "A", "A:PO4:1", "ligand", bonds=(0,)),
        AtomRecord(2, "O2", "O", (-1.5, 0.0, 0.0), "PO4", "1", "A", "A:PO4:1", "ligand", bonds=(0,)),
        AtomRecord(3, "O3", "O", (0.0, 1.5, 0.0), "PO4", "1", "A", "A:PO4:1", "ligand", bonds=(0,)),
        AtomRecord(4, "O4", "O", (0.0, -1.5, 0.0), "PO4", "1", "A", "A:PO4:1", "ligand", bonds=(0,)),
    ]
    components = [ComponentRecord("A:PO4:1", "PO4", "A", "1", "ligand", (0, 1, 2, 3, 4))]
    return atoms, components


def test_symmetric_phosphate_maps_without_false_ambiguity():
    # The phosphate's four terminal oxygens are topologically identical (each a
    # degree-1 O on P) but chemically distinct in the SMILES (one =O, three -OH).
    # This used to be rejected as an "ambiguous mapping"; it must now map cleanly.
    atoms, components = _phosphate_ligand()

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning (a skip) fails the test
        apply_ligand_smiles_templates(atoms, components, {"PO4": "OP(=O)(O)O"})

    assert "ignored_by_template" not in components[0].metadata
    assert components[0].metadata["smiles"] == "OP(=O)(O)O"
    oxygens = [atom for atom in atoms if atom.element == "O"]
    # Group-level chemistry is mapping-independent: four neutral acceptor oxygens.
    assert sum(atom.acceptor_capacity for atom in oxygens) == 8
    assert sum(atom.formal_charge for atom in oxygens) == 0
    assert sum(atom.donor_capacity for atom in oxygens) == 3  # the three -OH oxygens


def test_symmetric_carboxylate_maps_and_preserves_net_charge():
    atoms = [
        AtomRecord(0, "C1", "C", (0.0, 0.0, 0.0), "ACT", "1", "A", "A:ACT:1", "ligand", bonds=(1,)),
        AtomRecord(1, "C2", "C", (1.5, 0.0, 0.0), "ACT", "1", "A", "A:ACT:1", "ligand", bonds=(0, 2, 3)),
        AtomRecord(2, "O1", "O", (2.2, 1.0, 0.0), "ACT", "1", "A", "A:ACT:1", "ligand", bonds=(1,)),
        AtomRecord(3, "O2", "O", (2.2, -1.0, 0.0), "ACT", "1", "A", "A:ACT:1", "ligand", bonds=(1,)),
    ]
    components = [ComponentRecord("A:ACT:1", "ACT", "A", "1", "ligand", (0, 1, 2, 3))]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_ligand_smiles_templates(atoms, components, {"ACT": "CC(=O)[O-]"})

    assert "ignored_by_template" not in components[0].metadata
    # The -1 lands on one of the two equivalent carboxylate oxygens; the net is -1.
    assert sum(atom.formal_charge for atom in atoms) == -1


def test_parse_embedded_ligand_smiles_alphafold_form(tmp_path):
    cif = tmp_path / "model.cif"
    cif.write_text(
        "data_test\n"
        "loop_\n"
        "_chem_comp.id\n"
        "_chem_comp.pdbx_smiles\n"
        "LIG CCO\n"
        'ADP "OP(=O)(O)O"\n'
        "ALA ?\n"
    )

    embedded = parse_embedded_ligand_smiles(cif)
    assert embedded["LIG"] == "CCO"
    assert embedded["ADP"] == "OP(=O)(O)O"
    assert "ALA" not in embedded  # '?' is a CIF null, not a SMILES


def test_parse_embedded_ligand_smiles_descriptor_loop(tmp_path):
    cif = tmp_path / "wwpdb.cif"
    cif.write_text(
        "data_test\n"
        "loop_\n"
        "_pdbx_chem_comp_descriptor.comp_id\n"
        "_pdbx_chem_comp_descriptor.type\n"
        "_pdbx_chem_comp_descriptor.program\n"
        "_pdbx_chem_comp_descriptor.descriptor\n"
        'BAR "Canonical SMILES" "OpenEye OEToolkits" CCO\n'
    )

    embedded = parse_embedded_ligand_smiles(cif)
    assert embedded["BAR"] == "CCO"


def test_parse_embedded_ligand_smiles_pdb_returns_empty(tmp_path):
    pdb = tmp_path / "model.pdb"
    pdb.write_text("HETATM    1  C1  LIG A   1       0.000   0.000   0.000\n")
    assert parse_embedded_ligand_smiles(pdb) == {}
