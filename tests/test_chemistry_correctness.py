"""Scientific regression cases for acceptor typing and ligand correspondence."""
from copy import deepcopy
from itertools import permutations

import pytest

from chemur.api import analyze, assign_for_atoms
from chemur.chemistry import apply_ligand_smiles_templates, rdkit_atom_annotations
from chemur.errors import LigandTemplateError
from chemur.models import AtomRecord, ComponentRecord
from chemur.parser import _assign_default_atom_chemistry
from chemur.profile import load_profile

Chem = pytest.importorskip("rdkit.Chem")


@pytest.mark.parametrize("smiles, expected", [
    ("C[NH2:1]", 1), ("C[NH:1]C", 1), ("C[N:1](C)C", 1),
    ("CC(=O)[NH2:1]", 0), ("CC(=O)[N:1](C)C", 0),
    ("CS(=O)(=O)[NH2:1]", 0), ("c1cc[nH:1]c1", 0),
    ("c1cc[n:1]cc1", 1), ("C[NH3+:1]", 0),
    ("C[N+:1](C)(C)C", 0), ("CC(=O)[OH:1]", 0),
    ("CC(=O)[O-:1]", 2), ("C[OH:1]", 2), ("C[S:1]C", 2),
])
def test_ligand_acceptors_are_hydrogen_representation_independent(smiles, expected):
    implicit = Chem.MolFromSmiles(smiles)
    explicit = Chem.AddHs(implicit)
    isotope = Chem.Mol(explicit)
    for atom in isotope.GetAtoms():
        if atom.GetAtomicNum() == 1:
            atom.SetIsotope(2)
    # Explicit H can precede heavy atoms in an SDF, so test index preservation too.
    reordered = Chem.RenumberAtoms(explicit, list(reversed(range(explicit.GetNumAtoms()))))
    for molecule in (implicit, explicit, isotope, reordered):
        target = next(atom for atom in molecule.GetAtoms() if atom.GetAtomMapNum() == 1)
        before = Chem.MolToMolBlock(molecule)
        assert rdkit_atom_annotations(molecule)[target.GetIdx()].acceptor_capacity == expected
        assert Chem.MolToMolBlock(molecule) == before


def _site(name, element, residue, *, molecule_type="protein", index=0, bonds=()):
    return AtomRecord(index, name, element, (float(index), 0.0, 0.0), residue, "1", "A",
                      f"A:{residue}:1", molecule_type, bonds=bonds)


@pytest.mark.parametrize("residue,name,element,expected", [
    ("ALA", "N", "N", 0), ("PRO", "N", "N", 0),
    ("ASN", "ND2", "N", 0), ("GLN", "NE2", "N", 0),
    ("ARG", "NE", "N", 0), ("ARG", "NH1", "N", 0),
    ("LYS", "NZ", "N", 0), ("TRP", "NE1", "N", 0),
    ("ALA", "O", "O", 2), ("ALA", "OXT", "O", 2),
    ("ASN", "OD1", "O", 2), ("GLN", "OE1", "O", 2),
    ("ASP", "OD2", "O", 2), ("GLU", "OE2", "O", 2),
    ("SER", "OG", "O", 2), ("THR", "OG1", "O", 2),
    ("TYR", "OH", "O", 2), ("MET", "SD", "S", 2), ("CYS", "SG", "S", 2),
])
def test_protein_acceptors_use_residue_chemistry(residue, name, element, expected):
    atom = _site(name, element, residue)
    _assign_default_atom_chemistry([atom])
    assert atom.acceptor_capacity == expected


@pytest.mark.parametrize("residue,name,expected", [
    ("A", "N1", 1), ("DA", "N3", 1), ("ADE", "N7", 1),
    ("A", "N6", 0), ("A", "N9", 0),
    ("G", "N1", 0), ("DG", "N2", 0), ("GUA", "N3", 1),
    ("G", "N7", 1), ("G", "O6", 2),
    ("C", "N3", 1), ("DC", "N4", 0), ("CYT", "O2", 2),
    ("U", "N3", 0), ("DU", "O2", 2), ("URA", "O4", 2),
    ("T", "N1", 0), ("DT", "O4", 2), ("THY", "O2", 2),
    ("A", "O2'", 2), ("DA", "O4*", 2), ("G", "OP1", 2),
])
def test_nucleotide_acceptors_exclude_exocyclic_and_pyrrole_nitrogens(residue, name, expected):
    atom = _site(name, name[0], residue, molecule_type="nucleotide")
    _assign_default_atom_chemistry([atom])
    assert atom.acceptor_capacity == expected


def test_histidine_requires_tautomer_evidence_and_honors_positive_charge():
    atoms = [_site("ND1", "N", "HIS"), _site("NE2", "N", "HIS", index=1)]
    _assign_default_atom_chemistry(atoms)
    assert [atom.acceptor_capacity for atom in atoms] == [0, 0]
    assert all(atom.metadata["acceptor_unresolved_reason"] == "histidine_tautomer_unknown" for atom in atoms)
    atoms.append(_site("HD1", "H", "HIS", index=2, bonds=(0,)))
    atoms[0].bonds = (2,)
    _assign_default_atom_chemistry(list(reversed(atoms)))
    assert [atom.acceptor_capacity for atom in atoms[:2]] == [0, 1]
    atoms[0].metadata["structure_formal_charge"] = 1
    _assign_default_atom_chemistry(atoms)
    assert [atom.acceptor_capacity for atom in atoms[:2]] == [0, 0]
    atoms[0].metadata.pop("structure_formal_charge")
    atoms.append(_site("HE2", "H", "HIS", index=3, bonds=(1,)))
    atoms[1].bonds = (3,)
    _assign_default_atom_chemistry(atoms)
    assert [atom.acceptor_capacity for atom in atoms[:2]] == [0, 0]


def test_explicit_acid_proton_and_unknown_site_do_not_accept():
    atoms = [_site("OD2", "O", "ASP", bonds=(1,)), _site("HD2", "H", "ASP", index=1, bonds=(0,))]
    _assign_default_atom_chemistry(atoms)
    assert atoms[0].acceptor_capacity == 0
    unknown = _site("OX", "O", "ASN")
    _assign_default_atom_chemistry([unknown])
    assert unknown.acceptor_capacity == 0
    assert unknown.metadata["acceptor_unresolved_reason"] == "unsupported_residue_site"


@pytest.mark.parametrize("acceptor_name,expected", [("ND2", False), ("OD1", True)])
def test_public_analysis_does_not_hydrogen_bond_to_asparagine_amide(tmp_path, acceptor_name, expected):
    records = [("OG", "SER", 1, "O", 0.0), ("HG", "SER", 1, "H", 1.0),
               (acceptor_name, "ASN", 2, acceptor_name[0], 2.8),
               ("CG", "ASN", 2, "C", 4.0)]
    path = tmp_path / "amide.pdb"
    path.write_text("".join(
        f"ATOM  {idx:5d} {name:>4s} {residue:3s} A{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}          {element:>2s}\n"
        for idx, (name, residue, number, element, x) in enumerate(records, 1)
    ) + "END\n")
    result = analyze(path, fetch_ccd_smiles=False)
    assert any(interaction.interaction_type == "hbond"
               for interaction in result.interactions) is expected


@pytest.mark.parametrize("smiles,expected", [
    ("[CH2:1]=O", 0), ("C[CH:1]=O", 0), ("C[C:1]#N", 0),
    ("C[C:1](=O)C", 0), ("[c:1]1ccccc1O", 0), ("[CH:1](O)=CC", 0),
    ("[CH3:1]O", 1), ("[CH2:1](O)N", 2), ("[CH3:1]Cl", 0), ("[CH3:1]C", 1),
])
def test_only_sp3_ligand_carbons_have_tetrel_capacity(smiles, expected):
    molecule = Chem.MolFromSmiles(smiles)
    for representation in (molecule, Chem.AddHs(molecule)):
        atom = next(atom for atom in representation.GetAtoms() if atom.GetAtomMapNum() == 1)
        assert rdkit_atom_annotations(representation)[atom.GetIdx()].tetrel_donor_capacity == expected


def test_unspecified_hybridization_is_not_treated_as_sp3():
    molecule = Chem.MolFromSmiles("CO")
    molecule.GetAtomWithIdx(0).SetHybridization(Chem.HybridizationType.UNSPECIFIED)
    assert rdkit_atom_annotations(molecule)[0].tetrel_donor_capacity == 0


def _acid_fixture(order=(0, 1, 2, 3, 4)):
    atoms = [
        AtomRecord(0, "C1", "C", (0.0, 0.0, 0.0), "ACE", "1", "A", "A:ACE:1", "ligand", bonds=(1,)),
        AtomRecord(1, "C2", "C", (1.5, 0.0, 0.0), "ACE", "1", "A", "A:ACE:1", "ligand", bonds=(0, 2, 3)),
        AtomRecord(2, "OC", "O", (2.1, 1.0, 0.0), "ACE", "1", "A", "A:ACE:1", "ligand", bonds=(1,)),
        AtomRecord(3, "OH", "O", (2.1, -1.0, 0.0), "ACE", "1", "A", "A:ACE:1", "ligand", bonds=(1, 4)),
        AtomRecord(4, "H", "H", (3.0, -1.0, 0.0), "ACE", "1", "A", "A:ACE:1", "ligand", bonds=(3,)),
    ]
    ids = {old: new for new, old in enumerate(order)}
    result = [deepcopy(atoms[old]) for old in order]
    for atom in result:
        atom.atom_id = ids[atom.atom_id]
        atom.bonds = tuple(ids[idx] for idx in atom.bonds if idx in ids)
    component = ComponentRecord("A:ACE:1", "ACE", "A", "1", "ligand", tuple(ids.values()))
    return result, component


def _site_result(atoms):
    return {atom.name: (atom.acceptor_capacity, atom.donor_capacity, atom.formal_charge,
                        bool(atom.metadata.get("ignored_by_template"))) for atom in atoms}


def test_acetic_acid_mapping_and_hydrogen_bond_are_invariant_to_atom_order():
    expected = None
    for heavy_order in permutations(range(4)):
        atoms, component = _acid_fixture((*heavy_order, 4))
        apply_ligand_smiles_templates(atoms, [component], {"ACE": "CC(=O)O"}, strict=True)
        current = _site_result(atoms)
        expected = current if expected is None else expected
        assert current == expected
        assert current["H"][3] is False
        assert current["OH"][:2] == (0, 1)
        assert current["OC"][:2] == (2, 0)
        atoms.append(AtomRecord(5, "O", "O", (4.8, -1.0, 0.0), "ALA", "2", "B", "B:ALA:2", "protein", acceptor_capacity=2, bonds=(6,)))
        atoms.append(AtomRecord(6, "C", "C", (6.0, -1.0, 0.0), "ALA", "2", "B", "B:ALA:2", "protein", bonds=(5,)))
        receptor = ComponentRecord("B:ALA:2", "ALA", "B", "2", "protein", (5, 6))
        _, _, interactions = assign_for_atoms(atoms, [component, receptor], load_profile())
        contacts = [interaction for interaction in interactions if interaction.interaction_type == "hbond"]
        assert len(contacts) == 1
        by_id = {atom.atom_id: atom for atom in atoms}
        assert {by_id[idx].name for idx in contacts[0].atom_ids} == {"OH", "O"}


def test_absent_hydrogens_do_not_choose_an_acid_oxygen_arbitrarily():
    atoms, component = _acid_fixture((0, 1, 2, 3))
    with pytest.warns(RuntimeWarning, match="Ambiguous chemistry"):
        apply_ligand_smiles_templates(atoms, [component], {"ACE": "CC(=O)O"})
    assert component.metadata["template_mapping_status"] == "ambiguous"
    with pytest.raises(LigandTemplateError, match="Ambiguous chemistry"):
        apply_ligand_smiles_templates(atoms, [component], {"ACE": "CC(=O)O"}, strict=True)


def test_known_bond_orders_resolve_mapping_without_hydrogens():
    atoms, component = _acid_fixture((0, 1, 2, 3))
    component.metadata["bond_orders"] = [
        {"atom_id_1": 1, "atom_id_2": 2, "order": 2.0},
        {"atom_id_1": 1, "atom_id_2": 3, "order": 1.0},
    ]
    apply_ligand_smiles_templates(atoms, [component], {"ACE": "CC(=O)O"}, strict=True)
    assert atoms[2].acceptor_capacity == 2
    assert atoms[3].donor_capacity == 1


def test_template_reapplication_restores_previously_ignored_hydrogen():
    atoms, component = _acid_fixture()
    apply_ligand_smiles_templates(atoms, [component], {"ACE": "CC(=O)[O-]"}, strict=True)
    assert atoms[4].metadata["ignored_by_template"] is True
    apply_ligand_smiles_templates(atoms, [component], {"ACE": "CC(=O)O"}, strict=True)
    assert "ignored_by_template" not in atoms[4].metadata
    assert "ignored_hydrogen_atom_ids" not in atoms[3].metadata
    assert atoms[3].donor_capacity == 1


@pytest.mark.parametrize("constant", ["MAX_TEMPLATE_MAPPINGS", "MAX_TEMPLATE_SEARCH_STATES"])
def test_mapping_search_exhaustion_never_returns_a_partial_answer(monkeypatch, constant):
    import chemur.chemistry as chemistry
    monkeypatch.setattr(chemistry, constant, 1)
    atoms, component = _acid_fixture()
    with pytest.warns(RuntimeWarning, match="search_limit"):
        apply_ligand_smiles_templates(atoms, [component], {"ACE": "CC(=O)O"})
    assert component.metadata["template_mapping_status"] == "search_limit"
    assert component.metadata["ignored_by_template"] is True


def test_equivalent_smiles_preserve_site_chemistry_and_reconstructable_mapping():
    expected = None
    for smiles in ("CC(=O)[O-]", "[O-]C(C)=O", "O=C([O-])C"):
        for order in ((0, 1, 2, 3, 4), (3, 2, 4, 1, 0)):
            atoms, component = _acid_fixture(order)
            apply_ligand_smiles_templates(atoms, [component], {"ACE": smiles}, strict=True)
            result = _site_result(atoms)
            expected = result if expected is None else expected
            assert result == expected
            # CoSPLIF reconstructs the stored original SMILES, so its atom indices
            # must still address the recorded mapping after canonical search.
            rebuilt = rdkit_atom_annotations(Chem.MolFromSmiles(component.metadata["smiles"]))
            by_id = {atom.atom_id: atom for atom in atoms}
            for idx, atom_id in component.metadata["template_mapping"].items():
                assert by_id[atom_id].formal_charge == rebuilt[int(idx)].formal_charge
                assert by_id[atom_id].acceptor_capacity == rebuilt[int(idx)].acceptor_capacity


def test_phosphate_hydrogens_resolve_the_non_equivalent_oxygen_among_24_mappings():
    atoms = [_site("P", "P", "PO4", molecule_type="ligand", bonds=(1, 2, 3, 4))]
    for idx in range(1, 5):
        hydrogen = (idx + 4,) if idx < 4 else ()
        atoms.append(_site(f"O{idx}", "O", "PO4", molecule_type="ligand", index=idx, bonds=(0, *hydrogen)))
    atoms.extend(_site(f"H{idx}", "H", "PO4", molecule_type="ligand", index=idx + 4, bonds=(idx,))
                 for idx in range(1, 4))
    component = ComponentRecord("A:PO4:1", "PO4", "A", "1", "ligand", tuple(range(8)))
    apply_ligand_smiles_templates(atoms, [component], {"PO4": "OP(=O)(O)O"}, strict=True)
    assert [atom.donor_capacity for atom in atoms[1:5]] == [1, 1, 1, 0]
    assert [atom.acceptor_capacity for atom in atoms[1:5]] == [0, 0, 0, 2]
    assert all("ignored_by_template" not in atom.metadata for atom in atoms)
    assert component.metadata["template_mapping_evidence"]["equally_supported_mappings"] == 6


def test_parse_reports_uncertain_sites_once_per_component(tmp_path):
    from chemur.parser import parse_structure
    path = tmp_path / "histidines.pdb"
    path.write_text("".join(
        f"ATOM  {idx:5d} {name:>4s} HIS A{residue:4d}    "
        f"{float(idx * 3):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}           N\n"
        for idx, (residue, name) in enumerate(((1, "ND1"), (1, "NE2"), (2, "ND1"), (2, "NE2")), 1)
    ) + "END\n")
    with pytest.warns(RuntimeWarning, match="histidine_tautomer_unknown") as recorded:
        atoms, components = parse_structure(path)
    assert len(recorded) == 2
    assert len(components) == 2
    assert all(len(component.metadata["acceptor_unresolved_sites"]) == 2 for component in components)
    assert all(atom.acceptor_capacity == 0 for atom in atoms)


def test_sdf_overrides_preserve_source_charge_and_use_source_bond_orders(tmp_path):
    from chemur.sdf import SdfLigand, add_sdf_ligands
    molecule = Chem.MolFromSmiles("CC(=O)[O-]")
    molecule.AddConformer(Chem.Conformer(molecule.GetNumAtoms()))
    ligand = SdfLigand(tmp_path / "acid.sdf", 0, "ACE", "acid", "CC(=O)[O-]", molecule)
    atoms, components = [], []
    add_sdf_ligands(atoms, components, [ligand])
    original_anion = next(atom for atom in atoms if atom.formal_charge == -1)
    assert original_anion.metadata["structure_formal_charge"] == -1
    apply_ligand_smiles_templates(atoms, components, {"ACE": "OC(C)=O"}, strict=True)
    assert original_anion.formal_charge == 0
    assert original_anion.donor_capacity == 1
    assert original_anion.metadata["structure_formal_charge"] == -1
    apply_ligand_smiles_templates(atoms, components, {"ACE": "[O-]C(C)=O"}, strict=True)
    assert original_anion.formal_charge == -1
    assert original_anion.donor_capacity == 0
    assert components[0].metadata["template_mapping_evidence"]["bond_order_mismatches"] == 0
