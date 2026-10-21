"""MDAnalysis topology handoff preserves explicit source chemistry and indexing."""
from pathlib import Path
from types import SimpleNamespace

import gemmi
import numpy as np
import pytest

from chemur.trajectory import MDAnalysisFrameSource


def pdb_atom(serial, name, residue, number, x, element):
    return (f"HETATM{serial:5d} {name:^4s} {residue:3s} A{number:4d}    "
            f"{x:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{10.:6.2f}          {element:>2s}\n")


class FakeAtoms(list):
    def __init__(self, specs, positions):
        super().__init__(SimpleNamespace(name=name, resname=residue, resid=number,
                                       chainID="A", altLoc="", icode="", id=serial,
                                       element=element)
                         for serial, name, residue, number, element in specs)
        self.positions = np.asarray(positions, dtype=float)
        self.write_kwargs = None

    def write(self, path, **kwargs):
        self.write_kwargs = kwargs
        Path(path).write_text("".join(
            pdb_atom(atom.id, atom.name, atom.resname, atom.resid, xyz[0], atom.element)
            for atom, xyz in zip(self, self.positions)
        ) + "END\n")


class FakeTrajectory:
    def __getitem__(self, index):
        assert index == 0
        return self


def source(tmp_path, specs, bonds=(), *, topology=None):
    frame_source = MDAnalysisFrameSource(topology or tmp_path / "topology.gro", "trajectory.xtc")
    atoms = FakeAtoms(specs, [(10. * index, 2., 3.) for index in range(len(specs))])
    frame_source._universe = SimpleNamespace(atoms=atoms, trajectory=FakeTrajectory(), bonds=bonds)
    return frame_source


def bond(first, second, order=None, *, guessed=False):
    return SimpleNamespace(indices=(first, second), order=order, is_guessed=guessed)


@pytest.mark.parametrize("order", [1, 2.0, 1.5, "single", "doub", "arom"])
def test_explicit_mda_cross_residue_bond_order_survives_pdb_handoff(tmp_path, order):
    frame_source = source(tmp_path, [(1, "C1", "LIG", 1, "C"), (2, "O1", "TWO", 2, "O")],
                          [bond(0, 1, order)])
    atoms, components = frame_source.parse_topology()
    link = components[0].metadata["structure_links"][0]
    assert link["atom_ids"] == [0, 1]
    assert link["order"] == {"single": 1., "doub": 2., "arom": 1.5}.get(order, order)
    assert link["source"] == "mdanalysis_topology"
    assert atoms[0].bonds == (1,)
    assert atoms[1].metadata["assembly_link_neighbors"] == [0]
    assert frame_source._u().atoms.write_kwargs == {"bonds": None}


def test_unspecified_order_remains_unknown_instead_of_becoming_single(tmp_path):
    frame_source = source(tmp_path, [(1, "C1", "LIG", 1, "C"), (2, "N1", "TWO", 2, "N")],
                          [bond(0, 1)])
    atoms, components = frame_source.parse_topology()
    assert components[0].metadata["structure_links"][0]["order"] is None
    assert atoms[0].bonds == (1,)


def test_guessed_and_metal_coordination_bonds_are_not_covalent_links(tmp_path):
    frame_source = source(tmp_path, [(1, "N1", "LIG", 1, "N"), (2, "ZN", "TWO", 2, "Zn"),
                                    (3, "C1", "THR", 3, "C")],
                          [bond(0, 1, 1), bond(0, 2, 1, guessed=True)])
    atoms, components = frame_source.parse_topology()
    assert all("structure_links" not in component.metadata for component in components)
    assert not atoms[0].bonds


def test_intracomponent_explicit_order_becomes_mapping_evidence(tmp_path):
    frame_source = source(tmp_path, [(1, "C1", "LIG", 1, "C"), (2, "O1", "LIG", 1, "O")],
                          [bond(0, 1, 2.)])
    _, components = frame_source.parse_topology()
    assert components[0].metadata["bond_orders"] == [{"atom_id_1": 0, "atom_id_2": 1, "order": 2.}]


@pytest.fixture
def named_cif(tmp_path):
    text = (pdb_atom(1, "C1", "LIG", 1, 0., "C")
            + pdb_atom(2, "O1", "LIG", 1, 1.4, "O")
            + pdb_atom(3, "N1", "TWO", 2, 2.8, "N") + "END\n")
    structure = gemmi.read_pdb_string(text)
    connection = gemmi.Connection()
    connection.name = "link1"
    connection.type = gemmi.ConnectionType.Covale
    connection.asu = gemmi.Asu.Same
    connection.partner1 = gemmi.AtomAddress("A", gemmi.SeqId(1, " "), "LIG", "O1")
    connection.partner2 = gemmi.AtomAddress("A", gemmi.SeqId(2, " "), "TWO", "N1")
    structure.connections.append(connection)
    document = structure.make_mmcif_document()
    block = document.sole_block()
    connections = block.get_mmcif_category("_struct_conn.")
    connections["pdbx_value_order"] = ["sing"]
    block.set_mmcif_category("_struct_conn.", connections)
    loop = block.init_loop("_chem_comp_atom.", ["comp_id", "atom_id", "type_symbol", "charge"])
    for row in [("LIG", "C1", "C", "0"), ("LIG", "O1", "O", "0"), ("TWO", "N1", "N", "0")]:
        loop.add_row(row)
    loop = block.init_loop("_chem_comp_bond.", ["comp_id", "atom_id_1", "atom_id_2", "value_order"])
    loop.add_row(["LIG", "C1", "O1", "sing"])
    path = tmp_path / "topology.cif"
    document.write_file(str(path))
    return path


def test_original_dictionary_and_links_survive_reordered_mda_topology(named_cif, tmp_path):
    frame_source = source(tmp_path, [(3, "N1", "TWO", 2, "N"), (1, "C1", "LIG", 1, "C"),
                                    (2, "O1", "LIG", 1, "O")],
                          [bond(0, 2, 1.)], topology=named_cif)
    atoms, components = frame_source.parse_topology()
    assert [atom.atom_id for atom in atoms] == [2, 0, 1]
    assert [atom.coord for atom in atoms] == [(0., 2., 3.), (10., 2., 3.), (20., 2., 3.)]
    ligand = next(component for component in components if component.name == "LIG")
    assert set(ligand.metadata["chemical_template"]["atoms"]) == {"C1", "O1"}
    link = ligand.metadata["structure_links"][0]
    assert set(link["atom_ids"]) == {1, 2}
    assert link["order"] == 1.
    assert {item["source"] for item in link["provenance"]} == {"struct_conn", "mdanalysis_topology"}
    assert frame_source._u().atoms.write_kwargs is None


def test_conflicting_source_and_mda_orders_are_diagnostic(named_cif, tmp_path):
    frame_source = source(tmp_path, [(1, "C1", "LIG", 1, "C"), (2, "O1", "LIG", 1, "O"),
                                    (3, "N1", "TWO", 2, "N")],
                          [bond(1, 2, 2.)], topology=named_cif)
    _, components = frame_source.parse_topology()
    assert "Conflicting" in components[0].metadata["structure_links"][0]["error"]


def test_original_source_identity_mismatch_is_not_silently_discarded(named_cif, tmp_path):
    frame_source = source(tmp_path, [(1, "WRONG", "LIG", 1, "C"), (2, "O1", "LIG", 1, "O"),
                                    (3, "N1", "TWO", 2, "N")], topology=named_cif)
    with pytest.raises(ValueError, match="Cannot uniquely align"):
        frame_source.parse_topology()


def test_mda_without_bond_attribute_uses_source_chemistry(named_cif, tmp_path):
    frame_source = source(tmp_path, [(1, "C1", "LIG", 1, "C"), (2, "O1", "LIG", 1, "O"),
                                    (3, "N1", "TWO", 2, "N")], topology=named_cif)
    del frame_source._universe.bonds
    _, components = frame_source.parse_topology()
    assert components[0].metadata["structure_links"][0]["order"] == 1.


def test_unsupported_mda_bond_order_produces_diagnostic(tmp_path):
    frame_source = source(tmp_path, [(1, "C1", "LIG", 1, "C"), (2, "N1", "TWO", 2, "N")],
                          [bond(0, 1, "force_field_type_7")])
    _, components = frame_source.parse_topology()
    link = components[0].metadata["structure_links"][0]
    assert link["order"] is None
    assert "Unsupported MDAnalysis" in link["error"]
