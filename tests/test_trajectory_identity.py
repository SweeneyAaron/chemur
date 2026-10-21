"""Filtering preserves topology identity, chemistry references, and frame columns."""

from copy import deepcopy
import json
from math import cos, pi, sin

import numpy as np
import pytest

from chemur.models import AtomRecord, ComponentRecord
from chemur.parser import parse_structure
from chemur.trajectory import ArrayFrameSource, analyze_trajectory


class SparseFrameSource:
    """Coordinates are in topology order, although atom IDs are deliberately sparse."""

    def __init__(self):
        self.atoms = [
            AtomRecord(99, "O", "O", (30., 0., 0.), "HOH", "1", "A", "A:HOH:1", "solvent"),
            AtomRecord(5, "N", "N", (0., 0., 0.), "ALA", "2", "A", "A:ALA:2", "protein",
                       donor_capacity=1, bonds=(42, 99)),
            AtomRecord(42, "H", "H", (1., 0., 0.), "ALA", "2", "A", "A:ALA:2", "protein",
                       bonds=(5,)),
            AtomRecord(80, "O", "O", (2.8, 0., 0.), "ALA", "3", "B", "B:ALA:3", "protein",
                       acceptor_capacity=2, bonds=(81,)),
            AtomRecord(81, "C", "C", (4., 0., 0.), "ALA", "3", "B", "B:ALA:3", "protein",
                       bonds=(80,)),
        ]
        self.components = [
            ComponentRecord("A:HOH:1", "HOH", "A", "1", "solvent", (99,)),
            ComponentRecord("A:ALA:2", "ALA", "A", "2", "protein", (5, 42)),
            ComponentRecord("B:ALA:3", "ALA", "B", "3", "protein", (80, 81)),
        ]
        self.coords = np.array([[a.coord for a in self.atoms]] * 2)
        self.coords[1, 3, 0] = 12.0

    def parse_topology(self):
        return deepcopy((self.atoms, self.components))

    @property
    def n_atoms(self):
        return len(self.atoms)

    @property
    def n_frames(self):
        return len(self.coords)

    def coords_for(self, index):
        return self.coords[index]


@pytest.mark.parametrize("exclude_solvent", [False, True])
def test_frame_columns_are_independent_of_atom_ids(exclude_solvent):
    result = analyze_trajectory(SparseFrameSource(), exclude_solvent=exclude_solvent)
    assert [a.atom_id for a in result.atoms] == ([5, 42, 80, 81] if exclude_solvent else [99, 5, 42, 80, 81])
    donor = next(a for a in result.atoms if a.atom_id == 5)
    assert donor.bonds == ((42,) if exclude_solvent else (42, 99))
    hbonds = [i for i in result.frames[0].interactions if i.interaction_type == "hbond"]
    assert len(hbonds) == 1
    assert set(hbonds[0].atom_ids) == {5, 80}
    assert not any(i.interaction_type == "hbond" for i in result.frames[1].interactions)
    assert next(a for a in result.atoms if a.atom_id == 80).coord == (12., 0., 0.)


@pytest.fixture
def templated_trajectory(tmp_path):
    # Water precedes the ring; an ion separates it from an amide with an explicit H.
    specs = [("O", "O", "HOH", 1, (50., 0., 0.))]
    specs.extend(
        (f"C{i + 1}", "C", "BEN", 2, (15. + 1.4 * cos(i * pi / 3), 1.4 * sin(i * pi / 3), 0.))
        for i in range(6)
    )
    specs.extend([
        ("NA", "Na", "NA", 3, (40., 0., 0.)),
        ("C1", "C", "LAC", 4, (0., 0., 0.)),
        ("C2", "C", "LAC", 4, (1.5, 0., 0.)),
        ("O1", "O", "LAC", 4, (2.1, 1.05, 0.)),
        ("N1", "N", "LAC", 4, (2.1, -1.15, 0.)),
        ("H1", "H", "LAC", 4, (3.1, -1.15, 0.)),
        ("O", "O", "ALA", 5, (4.9, -1.15, 0.)),
        ("C", "C", "ALA", 5, (6.1, -1.15, 0.)),
        ("O", "O", "HOH", 6, (60., 0., 0.)),
    ])
    lines = [
        f"HETATM{i + 1:5d} {name:^4s} {residue:3s} A{number:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}"
        for i, (name, element, residue, number, xyz) in enumerate(specs)
    ]
    path = tmp_path / "templated.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    atoms, _ = parse_structure(path)
    coords = np.array([[a.coord for a in atoms]] * 2)
    coords[1, 13, 0] += 12.0
    coords[1, 1:7] += (1., 2., 0.)
    return path, coords


@pytest.mark.parametrize("processes", [1, 2])
@pytest.mark.parametrize("exclude_solvent,exclude_ions", [(True, False), (False, True), (True, True)])
def test_filtered_templates_keep_references_and_frame_interactions(
    templated_trajectory, tmp_path, processes, exclude_solvent, exclude_ions
):
    path, coords = templated_trajectory
    smiles = {"BEN": "c1ccccc1", "LAC": "CC(=O)N"}
    full = analyze_trajectory(ArrayFrameSource(path, coords), ligand_smiles=smiles)
    result = analyze_trajectory(
        ArrayFrameSource(path, coords), ligand_smiles=smiles,
        exclude_solvent=exclude_solvent, exclude_ions=exclude_ions, processes=processes,
    )
    dropped = ({0, 15} if exclude_solvent else set()) | ({7} if exclude_ions else set())
    expected_ids = set(range(16)) - dropped
    assert {a.atom_id for a in result.atoms} == expected_ids
    assert all(set(a.bonds) <= expected_ids for a in result.atoms)
    original_components = {c.component_id: c for c in full.components}
    for component in result.components:
        assert component.atom_ids == original_components[component.component_id].atom_ids
        if component.molecule_type != "ligand":
            continue
        for key in ("template_mapping", "template_rings", "template_carbonyls", "template_amides"):
            assert component.metadata[key] == original_components[component.component_id].metadata[key]
        assert set(component.metadata["template_mapping"].values()) <= expected_ids
        for key in ("template_rings", "template_carbonyls", "template_amides"):
            assert all(set(ids) <= expected_ids for ids in component.metadata[key])

    key = ("hbond", frozenset({11, 13}))
    assert result.occupancy()[key] == pytest.approx(0.5)
    series = result.timeseries(key)
    assert [row["present"] for row in series] == [True, False]
    assert series[0]["distance"] == pytest.approx(2.8)
    assert any("LAC:4:N1:11" in row["atom_labels"] for row in result.rows())
    assert json.loads(result.to_json())["rows"] == result.rows()
    csv_path = tmp_path / "trajectory.csv"
    result.to_csv(csv_path)
    assert "LAC:4:N1:11" in csv_path.read_text()
    by_id = {a.atom_id: a for a in result.atoms}
    assert by_id[1].coord == pytest.approx(coords[-1, 1])
    assert by_id[13].coord == pytest.approx(coords[-1, 13])
