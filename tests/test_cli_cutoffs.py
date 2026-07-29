import argparse

from chemur.cli import _add_rule_cutoff_args, _rule_overrides_from_args


def test_cli_hbond_distance_override():
    parser = argparse.ArgumentParser()
    _add_rule_cutoff_args(parser)

    args = parser.parse_args(["--hb-distance", "3.1", "--cation-pi-offset", "2.5"])
    overrides = _rule_overrides_from_args(args)

    assert overrides["hbond"]["distance"] == 3.1
    assert overrides["cation_pi"]["offset"] == 2.5
