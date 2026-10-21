"""Clients share effective geometry and historical data remains interpretable."""
import json

import numpy as np
import pytest

from chemur import analyze
from chemur.cli import RULE_CUTOFF_FLAGS, main
from chemur.geometry_catalogue import GEOMETRY_REVISION, get_geometry_catalogue, geometry_parameters
from chemur.interaction_types import INTERACTION_TYPES, canonical_family
from chemur.profile import apply_rule_overrides, load_profile
from chemur.trajectory import ArrayFrameSource, analyze_trajectory


def test_catalogue_exposes_same_effective_parameters_as_cli_without_sharing_state():
    profile = apply_rule_overrides(load_profile(), {"hbond_pi": {"hydrogen_distance": 3.2}})
    parameters = {(item["rule"], item["key"]): item for item in geometry_parameters(profile)}
    assert set(parameters) == {(rule, key) for rule, key, _, _ in RULE_CUTOFF_FLAGS}
    hydrogen = parameters["hbond_pi", "hydrogen_distance"]
    assert hydrogen["value"] == 3.2
    assert hydrogen["default"] == 3.5
    assert "H-ring centroid" in hydrogen["endpoints"]
    assert hydrogen["references"]
    hydrogen["value"] = 99
    assert profile["rules"]["hbond_pi"]["hydrogen_distance"] == 3.2
    assert load_profile()["rules"]["hbond_pi"]["hydrogen_distance"] == 3.5


def test_active_catalogue_excludes_anion_pi_but_historical_family_survives():
    assert "anion_pi" not in get_geometry_catalogue()["rules"]
    assert "anion_pi" not in {item.name for item in INTERACTION_TYPES}
    assert canonical_family("anion_pi") == "ionic"
    assert get_geometry_catalogue()["rules"]["anion_aromatic_edge"]["enabled"] is False


@pytest.mark.parametrize("flag", ["--anion-pi-distance", "--anion-pi-offset=2"])
def test_removed_detector_cli_request_is_actionable(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "unused.cif", flag])
    assert exc.value.code == 2
    assert "detection has been removed" in capsys.readouterr().err


def test_old_pi_heavy_distance_is_not_silently_reinterpreted(capsys):
    with pytest.raises(SystemExit):
        main(["analyze", "unused.cif", "--ch-pi-distance", "3.3"])
    assert "--ch-pi-hydrogen-distance" in capsys.readouterr().err


def test_static_and_trajectory_native_results_snapshot_effective_settings(tmp_path):
    path = tmp_path / "protein.pdb"
    path.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    overrides = {"hydrophobic": {"distance": 3.75}}
    static = analyze(path, fetch_ccd_smiles=False, rule_overrides=overrides)
    trajectory = analyze_trajectory(ArrayFrameSource(path, np.zeros((1, 1, 3))),
                                    fetch_ccd_smiles=False, rule_overrides=overrides)
    for result in (static, trajectory):
        metadata = json.loads(result.to_json())["metadata"]
        assert metadata["geometry_catalogue_revision"] == GEOMETRY_REVISION
        assert metadata["effective_profile"]["rules"]["hydrophobic"]["distance"] == 3.75
    overrides["hydrophobic"]["distance"] = 8
    assert static.metadata["effective_profile"]["rules"]["hydrophobic"]["distance"] == 3.75
    assert static.metadata["rule_overrides"]["hydrophobic"]["distance"] == 3.75
    assert trajectory.metadata["effective_profile"]["rules"]["hydrophobic"]["distance"] == 3.75


def test_polymer_sequence_cutoff_rejects_fractional_cli_input(capsys):
    with pytest.raises(SystemExit):
        main(["analyze", "unused.cif", "--n-pi-star-min-residue-separation", "2.8"])
    assert "invalid int value" in capsys.readouterr().err


def test_catalogue_defines_optional_rule_endpoints_and_cation_plane_angle():
    rules = get_geometry_catalogue()["rules"]
    assert "Ring-centroid" in rules["aliphatic_stack"]["parameters"]["distance"]["endpoints"]
    assert "Anion feature centre" in rules["anion_aromatic_edge"]["parameters"]["distance"]["endpoints"]
    assert "aromatic-ring plane" in rules["cation_pi"]["parameters"]["tertiary_amine_angle"]["endpoints"]
    assert rules["n_pi_star"]["parameters"]["min_residue_separation"]["units"] == "residues"
