"""The interaction-rule name set is duplicated in several places; keep them agreed.

``chemeleonx.interaction_types.INTERACTION_TYPES`` is the single source of truth.
Four things have to agree with it:

1. ``chemeleonx/profiles/default.yaml``      -- the shipped profile
2. ``chemeleonx.profile.DEFAULT_PROFILE``    -- the hard-coded fallback
3. ``chemeleonx.cli.RULE_CUTOFF_FLAGS``      -- the ``--*-distance/-angle`` flags
4. ``chemeleonx.interactions._DETECTORS``    -- the geometry that actually runs

A fifth lives in the ChimeraX bundle (``colors.py``'s pseudobond style table)
and is guarded by ``tests/test_interaction_names.py`` in that repo.

1 vs 2 is the dangerous pair: ``profile.load_profile`` wraps the packaged-YAML
lookup in a bare ``except Exception`` that returns ``DEFAULT_PROFILE``, so if
the YAML ever stops shipping, drift between them is completely silent. Comparing
only the rule *names* is not enough -- the two once disagreed on the ``hbond``
angle (90 vs 120 degrees) for exactly that reason, so the values are compared too.
"""

from chemeleonx.cli import RULE_CUTOFF_FLAGS
from chemeleonx.ifm_config import DEFAULT_FAMILY_WEIGHTS, IFMConfig
from chemeleonx.interaction_types import (
    CANONICAL_FAMILIES,
    INTERACTION_TYPES,
    canonical_family,
)
from chemeleonx.interactions import _DETECTORS
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


def test_yaml_profile_matches_hardcoded_default_value_by_value():
    """Names agreeing is not enough -- the cutoffs have silently disagreed before."""
    loaded = load_profile("default")
    mismatches = [
        f"{rule}.{key}: yaml={value!r} fallback={DEFAULT_PROFILE['rules'][rule].get(key)!r}"
        for rule, body in loaded["rules"].items()
        for key, value in body.items()
        if DEFAULT_PROFILE["rules"].get(rule, {}).get(key) != value
    ]
    assert not mismatches, "cutoff values have drifted: " + "; ".join(mismatches)
    assert loaded == DEFAULT_PROFILE


def test_cli_cutoff_flags_reference_real_rules():
    yaml_rules = set(load_profile("default")["rules"])
    flagged = {rule for rule, _key, _flags, _help in RULE_CUTOFF_FLAGS}
    assert flagged <= yaml_rules, (
        f"CLI exposes cutoff flags for unknown rules: {sorted(flagged - yaml_rules)}"
    )


def test_cli_cutoff_flags_reference_real_cutoffs():
    """A flag for a key the rule does not define raises at override time, not here."""
    rules = load_profile("default")["rules"]
    missing = [
        f"--{rule}-{key}"
        for rule, key, _flags, _help in RULE_CUTOFF_FLAGS
        if key not in rules[rule]
    ]
    assert not missing, f"CLI exposes flags for cutoffs no rule defines: {missing}"


def test_registry_matches_the_shipped_profile():
    registry = {interaction.name for interaction in INTERACTION_TYPES}
    yaml_rules = set(load_profile("default")["rules"])
    assert registry == yaml_rules, (
        "interaction_types.INTERACTION_TYPES and the profile have drifted: "
        f"{sorted(registry ^ yaml_rules)}"
    )


def test_every_registered_type_has_a_detector():
    """amide_bridge and amide_pi were declared everywhere but never detected."""
    registry = {interaction.name for interaction in INTERACTION_TYPES}
    assert registry == set(_DETECTORS), (
        "registry and detector table have drifted: "
        f"{sorted(registry ^ set(_DETECTORS))}"
    )


def test_every_registered_type_maps_to_a_canonical_family():
    """An unmapped type is dropped from every score with only a warning."""
    unmapped = [
        interaction.name
        for interaction in INTERACTION_TYPES
        if canonical_family(interaction.name) is None
    ]
    assert not unmapped, f"interaction types with no scoring family: {unmapped}"


def test_every_canonical_family_has_a_default_weight():
    missing = [family for family in CANONICAL_FAMILIES if family not in DEFAULT_FAMILY_WEIGHTS]
    assert not missing, f"canonical families with no IFM weight: {missing}"
    IFMConfig().validate()
