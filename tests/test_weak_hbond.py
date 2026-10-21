"""Weak C-H hydrogen bonds require activation as well as usable geometry."""

from math import cos, pi, sin

import pytest

from chemur.api import analyze, assign_for_atoms
from chemur.chemistry import apply_ligand_smiles_templates
from chemur.models import AtomRecord, ComponentRecord
from chemur.profile import load_profile


def _pair(*, activated=False):
    atoms = [
        AtomRecord(
            0, "C1", "C", (0.0, 0.0, 0.0), "LIG", "1", "A", "A:LIG:1", "ligand",
            carbon_donor_capacity=1, bonds=(1,), weak_hbond_activated=activated,
        ),
        AtomRecord(1, "H1", "H", (1.0, 0.0, 0.0), "LIG", "1", "A", "A:LIG:1", "ligand", bonds=(0,)),
        AtomRecord(2, "O", "O", (3.4, 0.0, 0.0), "ALA", "2", "B", "B:ALA:2", "protein", acceptor_capacity=1),
    ]
    components = [
        ComponentRecord("A:LIG:1", "LIG", "A", "1", "ligand", (0, 1)),
        ComponentRecord("B:ALA:2", "ALA", "B", "2", "protein", (2,)),
    ]
    return atoms, components


def _weak(candidates):
    return [candidate for candidate in candidates if candidate.interaction_type == "weak_hbond"]


@pytest.mark.parametrize("activated", [False, True])
def test_identical_geometry_requires_activated_carbon(activated):
    atoms, components = _pair(activated=activated)
    features, candidates, interactions = assign_for_atoms(atoms, components, load_profile())
    candidate, = _weak(candidates)

    assert candidate.angle == pytest.approx(180.0)
    assert candidate.distance == pytest.approx(3.4)
    assert candidate.metadata["weak_hbond_activated"] is activated
    assert candidate.metadata["hydrogen_atom_id"] == 1
    assert candidate.rejection_reason == (None if activated else "carbon_not_activated")
    assert bool(_weak(interactions)) is activated
    assert any(feature.feature_type == "carbon_donor" for feature in features)


@pytest.mark.parametrize("activated", [False, True])
@pytest.mark.parametrize("hydrogen_state", ["missing", "ignored", "poor_angle"])
def test_activation_cannot_replace_valid_hydrogen_geometry(activated, hydrogen_state):
    atoms, components = _pair(activated=activated)
    if hydrogen_state == "missing":
        atoms[0].bonds = ()
        atoms = [atoms[0], atoms[2]]
        components[0].atom_ids = (0,)
    elif hydrogen_state == "ignored":
        atoms[1].metadata["ignored_by_template"] = True
    else:
        atoms[1].coord = (-1.0, 0.0, 0.0)

    _, candidates, interactions = assign_for_atoms(atoms, components, load_profile())
    candidate, = _weak(candidates)
    if hydrogen_state == "poor_angle":
        assert candidate.rejection_reason.startswith("angle_below_cutoff:")
    else:
        assert candidate.rejection_reason == "missing_donor_hydrogen_geometry"
    assert _weak(interactions) == []


def test_activation_does_not_extend_distance_cutoff():
    atoms, components = _pair(activated=True)
    atoms[2].coord = (4.1, 0.0, 0.0)
    _, candidates, _ = assign_for_atoms(atoms, components, load_profile())
    assert _weak(candidates) == []


def test_rejected_weak_bond_preserves_both_capacities_for_other_interactions():
    atoms, components = _pair()
    # A ring on the C-H axis competes for carbon_donor capacity after weak_hbond.
    ring_ids = tuple(range(3, 9))
    ring_names = ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
    atoms.extend(
        AtomRecord(
            atom_id, name, "C", (3.6, 1.39 * cos(i * pi / 3), 1.39 * sin(i * pi / 3)),
            "PHE", "3", "C", "C:PHE:3", "protein", is_aromatic=True,
        )
        for i, (atom_id, name) in enumerate(zip(ring_ids, ring_names))
    )
    components.append(ComponentRecord("C:PHE:3", "PHE", "C", "3", "protein", ring_ids))
    # An activated sulfur competes for the same acceptor after weak_hbond.
    atoms.extend([
        AtomRecord(
            9, "S", "S", (3.4, 0.0, 3.2), "LIG", "4", "D", "D:LIG:4", "ligand",
            chalcogen_donor_capacity=1, sigma_hole_activated=True, bonds=(10,),
        ),
        AtomRecord(10, "C", "C", (3.4, 0.0, 4.9), "LIG", "4", "D", "D:LIG:4", "ligand", bonds=(9,)),
    ])
    components.append(ComponentRecord("D:LIG:4", "LIG", "D", "4", "ligand", (9, 10)))
    # Occlusion is irrelevant to the capacity regression; the two independent
    # competing contacts intentionally share a crowded idealized geometry.
    profile = load_profile({"occlusion_radius": 0.0})
    _, candidates, interactions = assign_for_atoms(atoms, components, profile)

    candidate, = _weak(candidates)
    assert candidate.rejection_reason == "carbon_not_activated"
    assert {interaction.interaction_type for interaction in interactions} >= {"ch_pi", "chalcogen_bond"}
    assert _weak(interactions) == []


def test_template_ignored_extra_hydrogen_cannot_supply_the_best_angle():
    pytest.importorskip("rdkit")
    atoms, components = _pair()
    atoms[1].coord = (-1.0, 0.0, 0.0)
    atoms.extend([
        AtomRecord(3, "O1", "O", (0.0, -1.4, 0.0), "LIG", "1", "A", "A:LIG:1", "ligand", bonds=(0,)),
        AtomRecord(4, "H2", "H", (0.0, 1.0, 0.0), "LIG", "1", "A", "A:LIG:1", "ligand", bonds=(0,)),
        AtomRecord(5, "H3", "H", (0.0, 0.0, 1.0), "LIG", "1", "A", "A:LIG:1", "ligand", bonds=(0,)),
        AtomRecord(6, "H4", "H", (1.0, 0.0, 0.0), "LIG", "1", "A", "A:LIG:1", "ligand", bonds=(0,)),
    ])
    atoms[0].bonds = (1, 3, 4, 5, 6)
    components[0].atom_ids = (0, 1, 3, 4, 5, 6)
    apply_ligand_smiles_templates(atoms, components, {"LIG": "CO"}, strict=True)
    assert atoms[-1].metadata["ignored_by_template"] is True

    _, candidates, interactions = assign_for_atoms(atoms, components, load_profile())
    candidate, = [candidate for candidate in _weak(candidates) if candidate.atom_ids == (0, 2)]
    assert candidate.metadata["hydrogen_atom_id"] != 6
    assert candidate.rejection_reason.startswith("angle_below_cutoff:")
    assert not any(interaction.atom_ids == (0, 2) for interaction in _weak(interactions))


def test_zero_hydrogen_template_overrides_deposited_carbon_hydrogen():
    pytest.importorskip("rdkit")
    atoms, components = _pair(activated=True)
    atoms.extend([
        AtomRecord(3, "O1", "O", (-1.2, 0.0, 0.0), "LIG", "1", "A", "A:LIG:1", "ligand", bonds=(0,)),
        AtomRecord(4, "C2", "C", (0.0, 1.5, 0.0), "LIG", "1", "A", "A:LIG:1", "ligand", bonds=(0,)),
        AtomRecord(5, "C3", "C", (0.0, -1.5, 0.0), "LIG", "1", "A", "A:LIG:1", "ligand", bonds=(0,)),
    ])
    atoms[0].bonds = (1, 3, 4, 5)
    components[0].atom_ids = (0, 1, 3, 4, 5)
    apply_ligand_smiles_templates(atoms, components, {"LIG": "C(=O)(C)C"}, strict=True)

    assert atoms[0].hydrogen_count == 0
    assert atoms[1].metadata["ignored_by_template"] is True
    _, candidates, interactions = assign_for_atoms(atoms, components, load_profile())
    candidate, = [candidate for candidate in _weak(candidates) if candidate.atom_ids == (0, 2)]
    assert candidate.rejection_reason == "missing_donor_hydrogen_geometry"
    assert not any(interaction.atom_ids[0] == 0 for interaction in _weak(interactions))


def _write_ligand_pdb(path, neighbor_element):
    path.write_text(
        "ATOM      1  O   ALA B   2       3.400   0.000   0.000  1.00  0.00           O\n"
        "HETATM    2  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        f"HETATM    3  X1  LIG A   1      -1.500   0.000   0.000  1.00  0.00           {neighbor_element}\n"
        "HETATM    4  H1  LIG A   1       1.000   0.000   0.000  1.00  0.00           H\n"
        "END\n"
    )


@pytest.mark.parametrize("source", ["smiles", "embedded", "ccd", "sdf"])
@pytest.mark.parametrize("smiles,activated", [("CO", True), ("CC", False), ("[CH2-]O", False), ("C[O-]", False)])
def test_public_pipeline_preserves_activation_and_raw_diagnostics(tmp_path, monkeypatch, source, smiles, activated):
    chem = pytest.importorskip("rdkit.Chem")
    molecule = chem.MolFromSmiles(smiles)
    pdb_path = tmp_path / "complex.pdb"
    _write_ligand_pdb(pdb_path, molecule.GetAtomWithIdx(1).GetSymbol())
    kwargs = {"include_raw": True, "fetch_ccd_smiles": False, "threads": 1, "strict_ligand_templates": True}
    if source == "smiles":
        kwargs["ligand_smiles"] = {"LIG": smiles}
    elif source == "embedded":
        gemmi = pytest.importorskip("gemmi")
        document = gemmi.read_structure(str(pdb_path)).make_mmcif_document()
        loop = document.sole_block().init_loop("_chem_comp.", ["id", "pdbx_smiles"])
        loop.add_row(["LIG", gemmi.cif.quote(smiles)])
        pdb_path = tmp_path / "complex.cif"
        document.write_file(str(pdb_path))
    elif source == "ccd":
        from chemur import ccd

        monkeypatch.setattr(ccd, "fetch_ccd_smiles", lambda component_id, **_: smiles)
        kwargs["fetch_ccd_smiles"] = True
    else:
        # Keep just one explicit H, as in the deposited PDB. Other hydrogens are
        # implicit chemistry and must not silently become usable coordinates.
        molecule = chem.RWMol(molecule)
        donor = molecule.GetAtomWithIdx(0)
        donor.SetNumExplicitHs(max(0, donor.GetNumExplicitHs() - 1))
        hydrogen_idx = molecule.AddAtom(chem.Atom("H"))
        molecule.AddBond(0, hydrogen_idx, chem.BondType.SINGLE)
        molecule = molecule.GetMol()
        chem.SanitizeMol(molecule)
        molecule.SetProp("_Name", "LIG")
        conformer = chem.Conformer(molecule.GetNumAtoms())
        for atom_idx, coord in enumerate([(0.0, 0.0, 0.0), (-1.5, 0.0, 0.0), (1.0, 0.0, 0.0)]):
            conformer.SetAtomPosition(atom_idx, coord)
        molecule.AddConformer(conformer)
        sdf_path = tmp_path / "ligand.sdf"
        with chem.SDWriter(str(sdf_path)) as writer:
            writer.write(molecule)
        kwargs["ligand_sdf"] = sdf_path

    result = analyze(pdb_path, **kwargs)
    donor = next(atom for atom in result.atoms if atom.element == "C" and atom.coord == (0.0, 0.0, 0.0))
    assert donor.weak_hbond_activated is activated
    candidate, = [candidate for candidate in _weak(result.raw_candidates) if donor.atom_id in candidate.atom_ids and candidate.component_ids[1] == "B:ALA:2"]
    assert candidate.metadata["weak_hbond_activated"] is activated
    assert candidate.angle == pytest.approx(180.0)
    assert candidate.rejection_reason == (None if activated else "carbon_not_activated")
    assert bool(_weak(result.interactions)) is activated

    payload = result.to_dict(include_raw=True)
    serialized_donor = next(atom for atom in payload["atoms"] if atom["atom_id"] == donor.atom_id)
    assert serialized_donor["weak_hbond_activated"] is activated
    assert any(row["rejection_reason"] == "carbon_not_activated" for row in result.rows(include_raw=True)) is not activated


@pytest.mark.parametrize("smiles,activated", [("CO", True), ("CC", False)])
def test_trajectory_retains_templated_activation_across_frames(tmp_path, smiles, activated):
    chem = pytest.importorskip("rdkit.Chem")
    np = pytest.importorskip("numpy")
    from chemur.parser import parse_structure
    from chemur.trajectory import ArrayFrameSource, analyze_trajectory

    pdb_path = tmp_path / "trajectory.pdb"
    molecule = chem.MolFromSmiles(smiles)
    _write_ligand_pdb(pdb_path, molecule.GetAtomWithIdx(1).GetSymbol())
    atoms, _ = parse_structure(pdb_path)
    base = np.array([atom.coord for atom in atoms], dtype=float)
    broken = base.copy()
    acceptor_id = next(atom.atom_id for atom in atoms if atom.molecule_type == "protein")
    broken[acceptor_id, 0] = 12.0
    coords = np.stack([base, broken, base])
    result = analyze_trajectory(
        ArrayFrameSource(str(pdb_path), coords), ligand_smiles={"LIG": smiles}, processes=1,
    )

    donor = next(atom for atom in result.atoms if atom.name == "C1")
    assert donor.weak_hbond_activated is activated
    assert [len(_weak(frame.interactions)) for frame in result.frames] == ([1, 0, 1] if activated else [0, 0, 0])
    if activated:
        key, = [key for key in result.occupancy() if key[0] == "weak_hbond"]
        assert result.occupancy()[key] == pytest.approx(2 / 3)
        assert all(interaction.metadata["weak_hbond_activated"] for frame in result.frames for interaction in _weak(frame.interactions))
