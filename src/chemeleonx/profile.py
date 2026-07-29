from __future__ import annotations

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

from .errors import DependencyMissingError

# Fallback used only when the packaged profiles/default.yaml cannot be read.
# It must stay byte-for-byte equivalent to that YAML; tests/test_profile_consistency.py
# compares the two rule-by-rule and value-by-value, because load_profile swallows
# every read error and silently returns this instead.
DEFAULT_PROFILE: dict[str, Any] = {
    'name': 'default',
    'neighbor_cutoff': 6.0,
    'occlusion_radius': 1.0,
    'occlusion_include_multi_atom': True,
    'assignment': {'intra_before_inter': True},
    'rules': {
        'metal_coordination': {'priority': 10, 'distance': 3.0, 'consume': {'acceptor': 1}},
        'salt_bridge': {'priority': 20, 'distance': 4.5},
        'amide_bridge': {
            'priority': 25,
            'distance': 3.3,
            'angle': 120.0,
            'min_residue_separation': 2,
            'consume': {'donor': 1, 'acceptor': 1},
        },
        'hbond': {
            'priority': 30,
            'distance': 3.5,
            'angle': 120.0,
            'sulfur_distance': 4.0,
            'short_distance': 2.7,
            'consume': {'donor': 1, 'acceptor': 1},
        },
        'solvent_bridge': {
            'priority': 40,
            'distance': 3.5,
            'consume': {'donor': 1, 'acceptor': 1},
        },
        'weak_hbond': {
            'priority': 50,
            'distance': 4.0,
            'angle': 90.0,
            'consume': {'donor': 1, 'acceptor': 1},
        },
        'pipi_stack': {
            'priority': 60,
            'distance': 6.0,
            'angle': 45.0,
            'offset': 3.0,
            'require_aromatic': True,
        },
        'aliphatic_pi_stack': {'priority': 65, 'distance': 5.5, 'angle': 30.0, 'offset': 2.0},
        'cation_pi': {
            'priority': 70,
            'distance': 5.0,
            'angle': 45.0,
            'offset': 3.0,
            'require_aromatic': True,
        },
        'anion_pi': {
            'priority': 80,
            'distance': 5.0,
            'angle': 45.0,
            'offset': 3.0,
            'require_aromatic': True,
        },
        'anion_aromatic_edge': {
            'priority': 85,
            'distance': 4.5,
            'min_angle': 60.0,
            'min_offset': 1.5,
            'offset': 4.5,
            'require_aromatic': True,
        },
        'hbond_pi': {
            'priority': 90,
            'distance': 4.5,
            'angle': 40.0,
            'donor_angle': 90.0,
            'offset': 3.0,
            'require_aromatic': True,
            'consume': {'donor': 1},
        },
        'n_pi_star': {
            'priority': 95,
            'distance': 3.22,
            'min_angle': 99.0,
            'angle': 119.0,
            'approach_angle': 60.0,
            'min_residue_separation': 1,
        },
        'amide_pi': {
            'priority': 100,
            'distance': 4.5,
            'angle': 40.0,
            'offset': 3.0,
            'require_aromatic': True,
        },
        'halogen_bond': {
            'priority': 110,
            'distance': 4.1,
            'angle': 100.0,
            'consume': {'halogen_donor': 1, 'acceptor': 1},
        },
        'chalcogen_bond': {
            'priority': 115,
            'distance': 3.6,
            'angle': 140.0,
            'require_activated_sigma_hole': True,
            'consume': {'chalcogen_donor': 1, 'acceptor': 1},
        },
        'halogen_pi': {
            'priority': 120,
            'distance': 6.0,
            'offset': 3.0,
            'require_aromatic': True,
            'consume': {'halogen_donor': 1},
        },
        'chalcogen_pi': {
            'priority': 122,
            'distance': 5.5,
            'angle': 45.0,
            'offset': 2.5,
            'require_aromatic': True,
            'consume': {'chalcogen_donor': 1},
        },
        'tetrel_bond': {
            'priority': 125,
            'min_distance': 2.5,
            'distance': 3.6,
            'angle': 160.0,
            'hydrogen_angle': 60.0,
            'consume': {'tetrel_donor': 1, 'acceptor': 1},
        },
        'ch_pi': {
            'priority': 130,
            'distance': 5.0,
            'angle': 40.0,
            'donor_angle': 90.0,
            'offset': 3.0,
            'require_aromatic': True,
            'consume': {'carbon_donor': 1},
        },
        'hydrophobic': {'priority': 140, 'distance': 4.0},
        'aliphatic_stack': {
            'priority': 145,
            'distance': 5.5,
            'angle': 30.0,
            'offset': 2.0,
            'enabled': False,
        },
    },
}


def load_profile(profile: str | Path | dict[str, Any] = "default") -> dict[str, Any]:
    if isinstance(profile, dict):
        return _merge_profile(profile)

    profile_path = Path(profile)
    if profile_path.exists():
        return _merge_profile(_load_yaml(profile_path, require_yaml=True))

    if str(profile) != "default":
        raise FileNotFoundError(f"Profile {profile!r} was not found")

    try:
        with resources.files("chemeleonx.profiles").joinpath("default.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            return _merge_profile(_load_yaml_text(handle.read(), require_yaml=False))
    except Exception:
        return deepcopy(DEFAULT_PROFILE)


def apply_rule_overrides(
    profile: dict[str, Any],
    rule_overrides: dict[str, dict[str, float | int]] | None,
) -> dict[str, Any]:
    if not rule_overrides:
        return deepcopy(profile)

    updated = deepcopy(profile)
    rules = updated.setdefault("rules", {})
    for rule_name, override_values in rule_overrides.items():
        if rule_name not in rules:
            raise ValueError(f"Unknown interaction rule override: {rule_name}")
        for key, value in override_values.items():
            if key not in rules[rule_name]:
                raise ValueError(f"Unknown cutoff override for {rule_name}: {key}")
            rules[rule_name][key] = value
    return updated


def _load_yaml(path: Path, require_yaml: bool) -> dict[str, Any]:
    return _load_yaml_text(path.read_text(encoding="utf-8"), require_yaml=require_yaml)


def _load_yaml_text(text: str, require_yaml: bool) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        if require_yaml:
            raise DependencyMissingError(
                "PyYAML is required to load custom profile files. Install pyyaml "
                "or use the built-in default profile."
            )
        return deepcopy(DEFAULT_PROFILE)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("Profile YAML must contain a mapping at the top level")
    return loaded


def _merge_profile(profile: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_PROFILE)
    for key, value in profile.items():
        if key == "rules" and isinstance(value, dict):
            for rule_name, rule_value in value.items():
                merged["rules"].setdefault(rule_name, {})
                merged["rules"][rule_name].update(rule_value or {})
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged
