"""CLI reports separate final assignments from raw candidates without changing APIs."""

from copy import deepcopy
import csv
import json

import pytest

from chemur import cli
from chemur.models import (
    AnalysisResult, AssignedInteraction, AtomRecord, CandidateInteraction,
    ComponentRecord, FeatureRecord,
)


@pytest.fixture
def result():
    atoms = [
        AtomRecord(
            aid, f"C{aid}", "C", (float(aid), 0., 0.),
            "LIG" if aid in {0, 3} else "ALA", str(aid), "A",
            f"A:{'LIG' if aid in {0, 3} else 'ALA'}:{aid}",
            "ligand" if aid in {0, 3} else "protein",
        )
        for aid in range(4)
    ]
    components = [
        ComponentRecord(atom.component_id, atom.residue_name, atom.chain_id,
                        atom.residue_id, atom.molecule_type, (atom.atom_id,))
        for atom in atoms
    ]
    features = [
        FeatureRecord(atom.atom_id, "hydrophobe", atom.component_id,
                      (atom.atom_id,), atom.coord)
        for atom in atoms
    ]
    candidates = [
        CandidateInteraction(
            "hydrophobic", ids, ids,
            tuple(atoms[aid].component_id for aid in ids), 3.5, 140, "inter",
            metadata={"contact_atoms": list(ids), "source": "fixture"},
            rejection_reason=None if ids == (0, 1) else "occluded",
        )
        for ids in ((0, 1), (0, 2), (2, 3))
    ]
    return AnalysisResult(
        atoms, components, features,
        [AssignedInteraction.from_candidate(candidates[0], 1)],
        raw_candidates=candidates,
    )


def _stub_analysis(monkeypatch, result, batch):
    captured = {}

    def analyze(*args, **kwargs):
        captured.update(kwargs)
        return {"pose/one": result} if batch else result

    monkeypatch.setattr(cli, "analyze_batch" if batch else "analyze", analyze)
    return captured


def _args(batch):
    args = ["analyze", "structure.cif"]
    if batch:
        args += ["--batch", "--ligand-sdf", "ligand.sdf"]
    return args


def _csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _payload(path, batch):
    payload = json.loads(path.read_text())
    return payload["results"]["pose/one"] if batch else payload


@pytest.mark.parametrize("batch", [False, True])
def test_primary_reports_contain_only_finals_and_sidecars_contain_all_candidates(
    result, tmp_path, monkeypatch, capsys, batch,
):
    captured = _stub_analysis(monkeypatch, result, batch)
    original = deepcopy(result.to_dict(include_raw=True))
    json_path = tmp_path / "contacts.v1.json"
    csv_path = tmp_path / "contacts.v1.csv"
    assert cli.main(_args(batch) + [
        "--include-raw", "--no-split-by-ligand",
        "--out", str(json_path), "--csv", str(csv_path),
    ]) == 0
    assert captured["include_raw"] is True
    final = _payload(json_path, batch)
    assert len(final["interactions"]) == 1
    assert "raw_candidates" not in final
    assert {row["assigned"] for row in _csv_rows(csv_path)} == {"True"}
    raw_json = tmp_path / "contacts.v1.raw.json"
    raw_csv = tmp_path / "contacts.v1.raw.csv"
    raw = _payload(raw_json, batch)
    assert raw["interactions"] == []
    assert len(raw["raw_candidates"]) == 3
    # Successful candidates remain part of the raw diagnostic report.
    assert raw["raw_candidates"][0]["rejection_reason"] is None
    assert raw["raw_candidates"][1]["rejection_reason"] == "occluded"
    raw_rows = _csv_rows(raw_csv)
    assert len(raw_rows) == 3
    assert {row["assigned"] for row in raw_rows} == {"False"}
    assert [json.loads(row["metadata"]) for row in raw_rows] == [
        candidate.metadata for candidate in result.raw_candidates
    ]
    assert final["interactions"][0]["metadata"] == result.interactions[0].metadata
    assert [record["metadata"] for record in raw["raw_candidates"]] == [
        candidate.metadata for candidate in result.raw_candidates
    ]
    output = capsys.readouterr().out
    assert "Wrote 2 raw candidate reports" in output
    assert str(raw_json) in output and str(raw_csv) in output
    assert result.to_dict(include_raw=True) == original


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("output_format", ["json", "csv", "both"])
def test_ligand_sidecars_preserve_raw_partner_context_and_requested_formats(
    result, tmp_path, monkeypatch, batch, output_format,
):
    _stub_analysis(monkeypatch, result, batch)
    original = deepcopy(result.to_dict(include_raw=True))
    output_root = tmp_path / "ligands"
    assert cli.main(_args(batch) + [
        "--include-raw", "--split-by-ligand-dir", str(output_root),
        "--split-by-ligand-format", output_format,
    ]) == 0
    directory = output_root / "pose_one" if batch else output_root
    formats = {"json", "csv"} if output_format == "both" else {output_format}
    assert {path.name for path in directory.iterdir()} == {
        f"A_LIG_{aid}{raw}.{suffix}"
        for aid in (0, 3) for raw in ("", ".raw") for suffix in formats
    }
    if "json" in formats:
        final = json.loads((directory / "A_LIG_0.json").read_text())
        raw = json.loads((directory / "A_LIG_0.raw.json").read_text())
        assert "raw_candidates" not in final
        assert len(final["interactions"]) == 1
        assert {atom["atom_id"] for atom in final["atoms"]} == {0, 1}
        assert raw["interactions"] == []
        assert len(raw["raw_candidates"]) == 2
        assert {atom["atom_id"] for atom in raw["atoms"]} == {0, 1, 2}
        assert {feature["feature_id"] for feature in raw["features"]} == {0, 1, 2}
        assert "A:ALA:2" in {component["component_id"] for component in raw["components"]}
        assert json.loads((directory / "A_LIG_3.json").read_text())["interactions"] == []
        assert len(json.loads((directory / "A_LIG_3.raw.json").read_text())["raw_candidates"]) == 1
    if "csv" in formats:
        assert {row["assigned"] for row in _csv_rows(directory / "A_LIG_0.csv")} == {"True"}
        raw_rows = _csv_rows(directory / "A_LIG_0.raw.csv")
        assert len(raw_rows) == 2
        assert {row["assigned"] for row in raw_rows} == {"False"}
        assert json.loads(raw_rows[1]["metadata"])["contact_atoms"] == [0, 2]
        assert _csv_rows(directory / "A_LIG_3.csv") == []
    assert result.to_dict(include_raw=True) == original


@pytest.mark.parametrize("batch", [False, True])
def test_without_include_raw_creates_only_final_files(result, tmp_path, monkeypatch, batch):
    captured = _stub_analysis(monkeypatch, result, batch)
    json_path, csv_path = tmp_path / "contacts.json", tmp_path / "contacts.csv"
    assert cli.main(_args(batch) + [
        "--no-split-by-ligand", "--out", str(json_path), "--csv", str(csv_path),
    ]) == 0
    assert captured["include_raw"] is False
    assert {path.name for path in tmp_path.iterdir()} == {"contacts.json", "contacts.csv"}
    assert "raw_candidates" not in _payload(json_path, batch)
    assert {row["assigned"] for row in _csv_rows(csv_path)} == {"True"}


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("output_format", ["json", "csv"])
def test_empty_raw_reports_are_written_only_in_requested_format(
    result, tmp_path, monkeypatch, batch, output_format,
):
    result.raw_candidates = []
    _stub_analysis(monkeypatch, result, batch)
    path = tmp_path / f"contacts.{output_format}"
    flag = "--out" if output_format == "json" else "--csv"
    assert cli.main(_args(batch) + [
        "--include-raw", "--no-split-by-ligand", flag, str(path),
    ]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {
        f"contacts.{output_format}", f"contacts.raw.{output_format}",
    }
    raw_path = tmp_path / f"contacts.raw.{output_format}"
    if output_format == "json":
        raw = _payload(raw_path, batch)
        assert raw["interactions"] == []
        assert raw["raw_candidates"] == []
    else:
        assert _csv_rows(raw_path) == []
        assert "assigned" in raw_path.read_text().splitlines()[0]


@pytest.mark.parametrize("batch", [False, True])
def test_stdout_only_raw_request_fails_before_analysis(monkeypatch, capsys, batch):
    def unexpected_analysis(*args, **kwargs):
        pytest.fail("Analysis must not run without a raw report destination")

    monkeypatch.setattr(cli, "analyze", unexpected_analysis)
    monkeypatch.setattr(cli, "analyze_batch", unexpected_analysis)
    with pytest.raises(SystemExit) as error:
        cli.main(_args(batch) + ["--include-raw", "--no-split-by-ligand"])
    assert error.value.code == 2
    assert "--include-raw requires --out, --csv, or per-ligand outputs" in capsys.readouterr().err
