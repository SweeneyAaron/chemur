"""The interaction-rule name set is duplicated in several places; keep them agreed.

Three copies live in this package:

1. ``chemeleonx/profiles/default.yaml``      -- the shipped profile
2. ``chemeleonx.profile.DEFAULT_PROFILE``    -- the hard-coded fallback
3. ``chemeleonx.cli.RULE_CUTOFF_FLAGS``      -- the ``--*-distance/-angle`` flags

A fourth lives in the ChimeraX bundle (``colors.py``'s pseudobond style table)
and is guarded by ``tests/test_interaction_names.py`` in that repo.

1 vs 2 is the dangerous pair: ``profile.load_profile`` wraps the packaged-YAML
lookup in a bare ``except Exception`` that returns ``DEFAULT_PROFILE``, so if
the YAML ever stops shipping, drift between them is completely silent.
"""

from chemeleonx.cli import RULE_CUTOFF_FLAGS
from chemeleonx.profile import DEFAULT_PROFILE, load_profile


def test_packaged_yaml_is_actually_readable():
    """Guard load_profile's silent except-fallback.

    If ``profiles/default.yaml`` is missing from the wheel, ``load_profile``
    still returns a usable profile -- so asserting only on the *content* would
    pass. Assert on a field the hard-coded fallback cannot supply instead.
    """
    from importlib import resources

    text = resources.files("chemeleonx.profiles").joinpath("default.yaml").read_text()
    assert "rules:" in text, "packaged default.yaml is present but not a profile"


def test_yaml_profile_matches_hardcoded_default():
    yaml_rules = set(load_profile("default")["rules"])
    assert yaml_rules, "no rules loaded at all"
    default_rules = set(DEFAULT_PROFILE["rules"])
    assert yaml_rules == default_rules, (
        "profiles/default.yaml and profile.DEFAULT_PROFILE have drifted: "
        f"{sorted(yaml_rules ^ default_rules)}"
    )


def test_cli_cutoff_flags_reference_real_rules():
    yaml_rules = set(load_profile("default")["rules"])
    flagged = {rule for rule, _key, _flags, _help in RULE_CUTOFF_FLAGS}
    assert flagged <= yaml_rules, (
        f"CLI exposes cutoff flags for unknown rules: {sorted(flagged - yaml_rules)}"
    )
