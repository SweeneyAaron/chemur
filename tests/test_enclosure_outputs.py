"""Enclosure exports retain grouped legs and trajectory centre identity."""

from copy import deepcopy
import csv
import json

import pytest

from chemur.cli import _batch_payload, _write_batch_csv, _write_batch_json
from chemur.models import (
    AnalysisResult,
    AssignedInteraction,
    AtomRecord,
    CandidateInteraction,
    ComponentRecord,
    FeatureRecord,
)
from chemur.trajectory import FrameInteractions, TrajectoryResult


@pytest.fixture
def result():
    atoms = [
        AtomRecord(
            atom_id, f"C{atom_id}", "C", (float(atom_id), 0.0, 0.0),
            "LIG" if atom_id == 1 else "ALA", str(atom_id), "A",
            f"A:{'LIG' if atom_id == 1 else 'ALA'}:{atom_id}",
            "ligand" if atom_id == 1 else "protein", is_hydrophobe=True,
        )
        for atom_id in range(6)
    ]
    components = [
        ComponentRecord(
            atom.component_id, atom.residue_name, atom.chain_id,
            atom.residue_id, atom.molecule_type, (atom.atom_id,),
        )
        for atom in atoms
    ]
    features = [
        FeatureRecord(atom.atom_id, "hydrophobe", atom.component_id,
                      (atom.atom_id,), atom.coord)
        for atom in atoms
    ]
    analysis = AnalysisResult(atoms, components, features, [])
    analysis.interactions = [_enclosure(analysis, 0, [1, 2, 3, 4])]
    leg = analysis.interactions[0].metadata["legs"][0]
    analysis.raw_candidates = [CandidateInteraction(
        "hydrophobic", tuple(leg["feature_ids"]), tuple(leg["atom_ids"]),
        tuple(leg["component_ids"]), leg["distance"], 140, "inter",
        metadata={"source": "contact"},
        rejection_reason="subsumed_by_hydrophobic_enclosure",
    )]
    return analysis


def _enclosure(result, center, neighbors):
    by_id = {atom.atom_id: atom for atom in result.atoms}
    atom_ids = tuple(sorted([center, *neighbors]))
    legs = [
        {
            "atom_ids": sorted([center, neighbor]),
            "feature_ids": sorted([center, neighbor]),
            "component_ids": [by_id[aid].component_id for aid in sorted([center, neighbor])],
            "distance": 3.0 + index / 10,
        }
        for index, neighbor in enumerate(sorted(neighbors))
    ]
    return AssignedInteraction(
        "hydrophobic_enclosure", atom_ids, atom_ids,
        tuple(by_id[aid].component_id for aid in atom_ids),
        max(leg["distance"] for leg in legs), 135, "inter", 1,
        metadata={
            "center_atom_id": center,
            "neighbor_atom_ids": sorted(neighbors),
            "interaction_subtype": "hydrophobic_enclosure",
            "legs": legs,
        },
    )


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_static_rows_and_json_keep_metadata_dicts_and_csv_encodes_json(result, tmp_path):
    original = deepcopy(result.to_dict(include_raw=True))
    rows = result.rows(include_raw=True)
    metadata = result.interactions[0].metadata
    assert rows[0]["metadata"] == metadata
    assert rows[1]["metadata"] == {"source": "contact"}
    assert rows[1]["rejection_reason"] == "subsumed_by_hydrophobic_enclosure"
    payload = json.loads(result.to_json(include_raw=True))
    assert payload["interactions"][0]["metadata"] == metadata
    path = tmp_path / "static.csv"
    result.to_csv(path, include_raw=True)
    csv_rows = _read_csv(path)
    assert json.loads(csv_rows[0]["metadata"]) == metadata
    assert json.loads(csv_rows[1]["metadata"]) == {"source": "contact"}
    assert result.to_dict(include_raw=True) == original


def test_batch_json_and_csv_retain_enclosure_metadata(result, tmp_path):
    metadata = result.interactions[0].metadata
    results = {"pose": result}
    assert _batch_payload(results, include_raw=True)["results"]["pose"]["interactions"][0]["metadata"] == metadata
    json_path = tmp_path / "batch.json"
    _write_batch_json(results, json_path, include_raw=True)
    assert json.loads(json_path.read_text())["results"]["pose"]["interactions"][0]["metadata"] == metadata
    csv_path = tmp_path / "batch.csv"
    _write_batch_csv(results, csv_path, include_raw=True)
    rows = _read_csv(csv_path)
    assert rows[0]["batch_id"] == "pose"
    assert json.loads(rows[0]["metadata"]) == metadata
    assert json.loads(rows[1]["metadata"]) == {"source": "contact"}
    assert isinstance(result.rows()[0]["metadata"], dict)


def test_ligand_selection_keeps_complete_enclosure_and_legs(result, tmp_path):
    # The ligand is a surrounding atom, so a split must retain the protein centre
    # and all other members of its group for later pair projection in scoring.
    ligand_id = result.atoms[1].component_id
    selected = result.split_by_ligand(include_raw=True)[ligand_id]
    expected = result.interactions[0]
    assert selected.interactions == [expected]
    assert {atom.atom_id for atom in selected.atoms} == set(expected.atom_ids)
    assert {feature.feature_id for feature in selected.features} == set(expected.feature_ids)
    assert selected.raw_candidates == result.raw_candidates
    written = selected.write_ligand_outputs(tmp_path, include_raw=True)
    json_path = next(path for path in written if path.suffix == ".json")
    assert json.loads(json_path.read_text())["interactions"][0]["metadata"] == expected.metadata


def test_trajectory_tracks_center_with_changing_partners_and_separate_centers(result, tmp_path):
    first = result.interactions[0]
    changed_shell = _enclosure(result, 0, [1, 2, 3, 5])
    other_center = _enclosure(result, 1, [0, 2, 3, 4])
    assert first.atom_ids == other_center.atom_ids
    trajectory = TrajectoryResult(
        result.atoms, result.components, [0, 2, 4],
        [FrameInteractions(0, [first, other_center]),
         FrameInteractions(2, [changed_shell]), FrameInteractions(4, [])],
    )
    first_key = ("hydrophobic_enclosure", frozenset({0}))
    second_key = ("hydrophobic_enclosure", frozenset({1}))
    assert trajectory.interaction_key(changed_shell) == first_key
    assert trajectory.interaction_key(other_center) == second_key
    assert trajectory.occupancy() == pytest.approx({first_key: 2 / 3, second_key: 1 / 3})
    assert trajectory.interaction_labels() == {
        first_key: result.atoms[0].label, second_key: result.atoms[1].label,
    }
    series = trajectory.timeseries(first_key)
    assert [row["frame"] for row in series] == [0, 2, 4]
    assert [row["present"] for row in series] == [True, True, False]
    assert series[0]["metadata"] == first.metadata
    assert series[1]["metadata"] == changed_shell.metadata
    assert series[2]["metadata"] == {}
    assert series[2]["distance"] is None
    rows = trajectory.rows()
    assert [row["metadata"] for row in rows] == [first.metadata, other_center.metadata, changed_shell.metadata]
    assert json.loads(trajectory.to_json())["rows"] == rows
    csv_path = tmp_path / "trajectory.csv"
    trajectory.to_csv(csv_path)
    assert [json.loads(row["metadata"]) for row in _read_csv(csv_path)] == [
        row["metadata"] for row in rows
    ]


def test_ordinary_trajectory_key_and_timeseries_are_unchanged(result):
    contact = AssignedInteraction(
        "hydrophobic", (0, 1), (0, 1),
        tuple(atom.component_id for atom in result.atoms[:2]), 3.0, 140, "inter", 1,
    )
    trajectory = TrajectoryResult(
        result.atoms, result.components, [0], [FrameInteractions(0, [contact])],
    )
    key = ("hydrophobic", frozenset({0, 1}))
    assert trajectory.interaction_key(contact) == key
    assert trajectory.timeseries(key) == [
        {"frame": 0, "present": True, "distance": 3.0, "angle": None, "offset": None}
    ]
    assert trajectory.rows()[0]["metadata"] == {}
