import numpy as np
import pytest

from chemeleonx.trajectory import ArrayFrameSource, analyze_trajectory


# A backbone donor (N + its H) and an acceptor (O) on two residues, plus a water.
MINI_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H   ALA A   1       0.900   0.000   0.000  1.00  0.00           H
ATOM      3  O   ALA A   2       2.800   0.000   0.000  1.00  0.00           O
HETATM    4  O   HOH A   3      10.000  10.000  10.000  1.00  0.00           O
END
"""


@pytest.fixture()
def mini_structure(tmp_path):
    pdb = tmp_path / "mini.pdb"
    pdb.write_text(MINI_PDB)
    from chemeleonx.parser import parse_structure

    atoms, _ = parse_structure(str(pdb))
    base = np.array([[a.coord for a in atoms]], dtype=np.float32)
    # Frame 0: acceptor in H-bond range. Frame 1: acceptor pushed away (contact broken).
    broken = base.copy()
    broken[0, 2, 0] = 12.0
    coords = np.concatenate([base, broken], axis=0)
    return str(pdb), coords


def _hbond_keys(result):
    return {k for k in result.occupancy() if k[0] == "hbond"}


def test_occupancy_tracks_contact_across_frames(mini_structure):
    pdb, coords = mini_structure
    result = analyze_trajectory(ArrayFrameSource(pdb, coords), processes=1)

    assert result.n_frames == 2
    hbonds = _hbond_keys(result)
    assert len(hbonds) == 1
    key = next(iter(hbonds))
    # Present in frame 0, absent in frame 1 -> 50% occupancy.
    assert result.occupancy()[key] == pytest.approx(0.5)

    series = result.timeseries(key)
    assert series[0]["present"] is True
    assert series[1]["present"] is False
    assert series[0]["distance"] == pytest.approx(2.8, abs=1e-3)
    assert series[1]["distance"] is None


def test_counts_per_frame_and_rows(mini_structure):
    pdb, coords = mini_structure
    result = analyze_trajectory(ArrayFrameSource(pdb, coords), processes=1)

    counts = result.counts_per_frame()
    assert counts["frames"] == [0, 1]
    assert counts["total"][0] >= 1
    assert counts["total"][1] < counts["total"][0]

    rows = result.rows()
    assert all("frame" in row for row in rows)
    assert {row["frame"] for row in rows} <= {0, 1}


def test_exclude_solvent_drops_water_atoms(mini_structure):
    pdb, coords = mini_structure
    full = analyze_trajectory(ArrayFrameSource(pdb, coords), processes=1)
    no_solvent = analyze_trajectory(
        ArrayFrameSource(pdb, coords), exclude_solvent=True, processes=1
    )
    assert no_solvent.metadata["n_atoms"] == full.metadata["n_atoms"] - 1
    assert all(a.molecule_type != "solvent" for a in no_solvent.atoms)


def test_exclude_solvent_preserves_donor_hydrogen_bonds(mini_structure):
    """Regression: excluding solvent renumbers atoms; the donor heavy atom must
    keep its H neighbour so the N-H...O hbond is still detected (was dropped when
    the bond remap was single-pass)."""
    pdb, coords = mini_structure
    result = analyze_trajectory(
        ArrayFrameSource(pdb, coords), exclude_solvent=True, processes=1
    )
    # The hbond present in frame 0 must survive the exclusion renumbering.
    assert any(k[0] == "hbond" for k in result.occupancy())
    # The donor heavy atom (ALA:1 N) still has a bonded hydrogen after renumber.
    by_id = {a.atom_id: a for a in result.atoms}
    donor = next(a for a in result.atoms if a.element == "N")
    assert any(by_id[b].element == "H" for b in donor.bonds)


def test_frame_slicing(mini_structure):
    pdb, coords = mini_structure
    coords3 = np.concatenate([coords, coords[:1]], axis=0)  # 3 frames
    result = analyze_trajectory(
        ArrayFrameSource(pdb, coords3), frame_start=0, frame_stop=3, frame_stride=2
    )
    assert result.frame_indices == [0, 2]


def test_array_source_npy_path(mini_structure, tmp_path):
    pdb, coords = mini_structure
    npy = tmp_path / "coords.npy"
    np.save(npy, coords)
    result = analyze_trajectory(ArrayFrameSource(pdb, str(npy)), processes=1)
    assert result.n_frames == 2
    assert len(_hbond_keys(result)) == 1
