"""Local motif typing and stable ring-plane eligibility."""
from math import cos, sin, pi

import pytest
from rdkit import Chem

from chemur.chemistry import apply_rdkit_ligand_template, rdkit_atom_annotations
from chemur.features import _unique_mean_plane, perceive_features
from chemur.models import AtomRecord, ComponentRecord


def prepared(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    atoms = [AtomRecord(a.GetIdx(), f"{a.GetSymbol()}{a.GetIdx()}", a.GetSymbol(),
                        (float(a.GetIdx()), 0., 0.), "LIG", "1", "L", "L:LIG:1", "ligand",
                        bonds=tuple(n.GetIdx() for n in a.GetNeighbors())) for a in molecule.GetAtoms()]
    component = ComponentRecord("L:LIG:1", "LIG", "L", "1", "ligand", tuple(a.atom_id for a in atoms))
    apply_rdkit_ligand_template(atoms, component, molecule, {a.atom_id: a.atom_id for a in atoms}, smiles=smiles, source="test")
    return atoms, component


def test_sulfoxide_resonance_has_identical_charge_capacity_and_features():
    neutral, neutral_component = prepared("CS(=O)C")
    separated, separated_component = prepared("C[S+]([O-])C")
    assert rdkit_atom_annotations(Chem.MolFromSmiles("CS(=O)C")) == rdkit_atom_annotations(Chem.MolFromSmiles("C[S+]([O-])C"))
    for atoms, component in ((neutral, neutral_component), (separated, separated_component)):
        sulfur = atoms[1]
        assert sulfur.formal_charge == 0 and sulfur.chalcogen_donor_capacity == 0
        assert sulfur.metadata["sulfur_pi_types"] == ["sulfoxide"]
        features = perceive_features(atoms, [component])
        assert any(f.feature_type == "sulfur_pi" for f in features)
        assert not any(f.feature_type in {"cation", "anion"} for f in features)
        encoded = Chem.MolFromSmiles(component.metadata["smiles"])
        assert all(a.GetFormalCharge() == 0 for a in encoded.GetAtoms())
        for index, aid in component.metadata["template_mapping"].items():
            assert encoded.GetAtomWithIdx(int(index)).GetSymbol() == atoms[aid].element


@pytest.mark.parametrize("smiles,expected", [("CSC", ["thioether"]), ("CSSC", ["disulfide"]),
                                            ("CS(=O)C", ["sulfoxide"]), ("c1ccsc1", []),
                                            ("CS(=O)(=O)C", []), ("C[S+](C)C", []), ("C[S-]", [])])
def test_sulfur_pi_chemical_scope_is_independent_of_sigma_hole_capacity(smiles, expected):
    atoms, _ = prepared(smiles)
    for atom in atoms:
        if atom.element == "S":
            assert atom.metadata.get("sulfur_pi_types", []) == expected


@pytest.mark.parametrize("smiles,capacity", [("CC", 1), ("CN", 1), ("CO", 1), ("CS", 1),
                                            ("CCl", 0), ("C=C", 0), ("C#C", 0), ("C=O", 0)])
def test_tetrel_scope_requires_sp3_carbon_and_cnos_root(smiles, capacity):
    annotations = rdkit_atom_annotations(Chem.MolFromSmiles(smiles))
    assert annotations[0].tetrel_donor_capacity == capacity


@pytest.mark.parametrize("smiles,expected", [("C[NH+](C)C", True), ("C=[N+](C)C", False),
                                            ("C[N+](C)(C)C", False), ("CN(C)C", False)])
def test_tertiary_amine_cation_is_a_validated_sp3_motif(smiles, expected):
    atoms, _ = prepared(smiles)
    nitrogen = next(a for a in atoms if a.element == "N")
    assert nitrogen.metadata["tertiary_amine_cation"] is expected


def test_coordinate_retyping_clears_obsolete_sulfur_motif_metadata():
    from chemur.parser import _assign_default_atom_chemistry
    atoms = [AtomRecord(0, "SD", "S", (0., 0., 0.), "MET", "1", "A", "A:MET:1", "protein", bonds=(1, 2)),
             AtomRecord(1, "CG", "C", (1.8, 0., 0.), "MET", "1", "A", "A:MET:1", "protein", bonds=(0,)),
             AtomRecord(2, "CE", "C", (0., 1.8, 0.), "MET", "1", "A", "A:MET:1", "protein", bonds=(0,))]
    _assign_default_atom_chemistry(atoms)
    assert atoms[0].metadata["sulfur_pi_types"] == ["thioether"]
    atoms[0].bonds = (1,)
    _assign_default_atom_chemistry(atoms)
    assert "sulfur_pi_types" not in atoms[0].metadata


@pytest.mark.parametrize("coords", [[(float(i), 0., 0.) for i in range(6)],
                                    [(0., 0., 0.)] * 6,
                                    [(1., 0., 0.), (-1., 0., 0.), (0., 1., 0.), (0., -1., 0.), (0., 0., 1.), (0., 0., -1.)],
                                    [(0., 0., 0.), (1., 0., 0.), (0., float("nan"), 0.)]])
def test_degenerate_or_nonfinite_ring_plane_is_unavailable(coords):
    assert _unique_mean_plane(coords) == (None, None)


def test_puckered_chair_retains_unique_mean_plane():
    coords = [(cos(i * pi / 3), sin(i * pi / 3), .3 * (-1) ** i) for i in range(6)]
    normal, planarity = _unique_mean_plane(coords)
    assert normal is not None and abs(normal[2]) == pytest.approx(1.)
    assert planarity is not None
