import sys
import types

from chemur.protonation import prepare_ligand_smiles


def test_prepare_ligand_smiles_protonates_and_prints_debug(capsys, monkeypatch):
    module = types.ModuleType("dimorphite_dl")

    def protonate_smiles(smiles, ph_min, ph_max, precision, max_variants):
        assert smiles == "CCN"
        assert ph_min == 6.8
        assert ph_max == 7.4
        assert precision == 0.5
        assert max_variants == 4
        return ["CC[NH3+]"]

    module.protonate_smiles = protonate_smiles
    monkeypatch.setitem(sys.modules, "dimorphite_dl", module)

    prepared = prepare_ligand_smiles(
        {"LIG": "CCN"},
        protonate=True,
        ph_min=6.8,
        ph_max=7.4,
        precision=0.5,
        max_variants=4,
    )

    assert prepared == {"LIG": "CC[NH3+]"}
    assert "using SMILES 'CC[NH3+]'" in capsys.readouterr().err


def test_prepare_ligand_smiles_debug_without_protonation(capsys):
    prepared = prepare_ligand_smiles({"LIG": "CCN"}, debug=True)

    assert prepared == {"LIG": "CCN"}
    assert "using SMILES 'CCN'" in capsys.readouterr().err

