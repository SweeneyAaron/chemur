from pathlib import Path

from chemur import cli


class DummyResult:
    def __init__(self):
        self.interactions = [object()]
        self.split_args = None

    def to_json(self, path=None, include_raw=False):
        if path is not None:
            return None
        return '{"ok": true}'

    def to_csv(self, path, include_raw=False):
        return None

    def to_dict(self, include_raw=False):
        return {"interactions": []}

    def write_ligand_outputs(self, output_dir, include_raw=False, output_format="both"):
        self.split_args = (output_dir, include_raw, output_format)
        return [Path(output_dir) / "A_LIG_1.json"]


def test_cli_splits_by_ligand_by_default(monkeypatch, capsys):
    holder = {}

    def fake_analyze(*args, **kwargs):
        holder["result"] = DummyResult()
        return holder["result"]

    monkeypatch.setattr(cli, "analyze", fake_analyze)

    status = cli.main(["analyze", "structure.cif"])
    captured = capsys.readouterr()

    assert status == 0
    assert holder["result"].split_args == ("ligand_outputs", False, "both")
    assert "Wrote 1 per-ligand output files to ligand_outputs" in captured.out


def test_cli_no_split_by_ligand_restores_stdout_json(monkeypatch, capsys):
    holder = {}

    def fake_analyze(*args, **kwargs):
        holder["result"] = DummyResult()
        return holder["result"]

    monkeypatch.setattr(cli, "analyze", fake_analyze)

    status = cli.main(["analyze", "structure.cif", "--no-split-by-ligand"])
    captured = capsys.readouterr()

    assert status == 0
    assert holder["result"].split_args is None
    assert captured.out == '{"ok": true}\n'


def test_cli_batch_routes_to_analyze_batch_and_splits_by_batch(monkeypatch, capsys):
    holder = {}

    def fake_analyze_batch(*args, **kwargs):
        holder["result"] = DummyResult()
        holder["kwargs"] = kwargs
        return {"ligand-one": holder["result"]}

    monkeypatch.setattr(cli, "analyze_batch", fake_analyze_batch)

    status = cli.main(["analyze", "structure.cif", "--ligand-sdf", "ligand.sdf", "--batch"])
    captured = capsys.readouterr()

    assert status == 0
    assert holder["kwargs"]["ligand_sdf"] == ["ligand.sdf"]
    assert holder["result"].split_args == (Path("ligand_outputs") / "ligand-one", False, "both")
    assert "across 1 batch analyses" in captured.out
