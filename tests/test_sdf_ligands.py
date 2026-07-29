import pytest

from chemur.sdf import add_sdf_ligands, load_sdf_ligands

rdkit = pytest.importorskip("rdkit")


def test_sdf_ligand_uses_sdf_coordinates_and_chemistry(tmp_path):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    sdf_path = tmp_path / "ethanol.sdf"
    molecule = Chem.MolFromSmiles("CCO")
    molecule.SetProp("_Name", "EOH")
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, Point3D(1.5, 0.0, 0.0))
    conformer.SetAtomPosition(2, Point3D(2.9, 0.0, 0.0))
    molecule.AddConformer(conformer)
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(molecule)
    writer.close()

    ligands = load_sdf_ligands(sdf_path)
    atoms = []
    components = []

    smiles = add_sdf_ligands(atoms, components, ligands)

    assert len(components) == 1
    assert components[0].component_id == "S:EOH:1"
    assert components[0].metadata["source"] == "sdf"
    assert smiles == {"S:EOH:1": "CCO"}
    assert atoms[2].coord == (2.9, 0.0, 0.0)
    assert atoms[1].bonds == (0, 2)
    assert atoms[2].donor_capacity == 1
    assert atoms[2].acceptor_capacity == 2


def test_sdf_directory_loads_all_sdf_files(tmp_path):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    for name, smiles in [("aaa", "CO"), ("bbb", "CCN")]:
        path = tmp_path / f"{name}.sdf"
        molecule = Chem.MolFromSmiles(smiles)
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        for atom_idx in range(molecule.GetNumAtoms()):
            conformer.SetAtomPosition(atom_idx, Point3D(float(atom_idx), 0.0, 0.0))
        molecule.AddConformer(conformer)
        writer = Chem.SDWriter(str(path))
        writer.write(molecule)
        writer.close()

    ligands = load_sdf_ligands(directories=tmp_path)

    assert [ligand.source_path.name for ligand in ligands] == ["aaa.sdf", "bbb.sdf"]
