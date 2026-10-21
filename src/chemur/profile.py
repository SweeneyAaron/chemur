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
DEFAULT_PROFILE: dict[str, Any] = {'name': 'default',
 'neighbor_cutoff': 6.0,
 'occlusion_radius': 1.0,
 'occlusion_include_multi_atom': True,
 'assignment': {'intra_before_inter': True},
 'rules': {'metal_coordination': {'priority': 10,
                                  'distance': 2.6,
                                  'consume': {'acceptor': 1},
                                  'min_distance': 1.7,
                                  'alkali_min_distance': 2.0,
                                  'alkali_distance': 2.9},
           'salt_bridge': {'priority': 20, 'distance': 4.0},
           'amide_bridge': {'priority': 25,
                            'min_residue_separation': 2,
                            'consume': {'donor': 1, 'acceptor': 1}},
           'hbond': {'priority': 30,
                     'distance': 3.5,
                     'angle': 120.0,
                     'consume': {'donor': 1, 'acceptor': 1},
                     'min_distance': 2.4,
                     'hydrogen_min_distance': 1.5,
                     'hydrogen_distance': 2.5,
                     'acceptor_angle': 90.0,
                     'water_distance': 3.5,
                     'water_angle': 120.0},
           'solvent_bridge': {'priority': 40, 'distance': 3.5, 'consume': {}},
           'weak_hbond': {'priority': 50,
                          'distance': 3.6,
                          'angle': 130.0,
                          'consume': {'donor': 1, 'acceptor': 1}},
           'pipi_stack': {'priority': 60,
                          'distance': 5.5,
                          'angle': 30.0,
                          'offset': 2.0,
                          'require_aromatic': True,
                          'min_perpendicular_angle': 60.0},
           'aliphatic_pi_stack': {'priority': 65,
                                  'distance': 7.0,
                                  'angle': 10.0,
                                  'plane_separation': 5.0},
           'cation_pi': {'priority': 70,
                         'distance': 6.0,
                         'offset': 2.0,
                         'require_aromatic': True,
                         'tertiary_amine_angle': 30.0},
           'anion_aromatic_edge': {'priority': 85,
                                   'distance': 4.5,
                                   'min_angle': 60.0,
                                   'min_offset': 1.5,
                                   'offset': 4.5,
                                   'require_aromatic': True,
                                   'enabled': False},
           'hbond_pi': {'priority': 90,
                        'angle': 40.0,
                        'donor_angle': 100.0,
                        'require_aromatic': True,
                        'consume': {'donor': 1},
                        'hydrogen_distance': 3.5},
           'n_pi_star': {'priority': 95,
                         'distance': 3.2,
                         'min_angle': 99.0,
                         'angle': 119.0,
                         'min_residue_separation': 1},
           'amide_pi': {'priority': 100,
                        'distance': 6.0,
                        'angle': 30.0,
                        'require_aromatic': True,
                        'approach_angle': 30.0},
           'halogen_bond': {'priority': 110,
                            'distance': 4.0,
                            'angle': 135.0,
                            'consume': {'halogen_donor': 1, 'acceptor': 1},
                            'acceptor_min_angle': 90.0,
                            'acceptor_angle': 150.0},
           'chalcogen_bond': {'priority': 115,
                              'angle': 130.0,
                              'consume': {'chalcogen_donor': 1, 'acceptor': 1},
                              's_o_min_distance': 2.0,
                              's_o_distance': 3.32,
                              'se_o_min_distance': 2.0,
                              'se_o_distance': 3.42,
                              'se_s_min_distance': 2.5,
                              'se_s_distance': 4.07},
           'halogen_pi': {'priority': 120,
                          'distance': 6.0,
                          'require_aromatic': True,
                          'consume': {'halogen_donor': 1},
                          'donor_angle': 90.0},
           'chalcogen_pi': {'priority': 122,
                            'distance': 7.0,
                            'angle': 60.0,
                            'require_aromatic': True,
                            'consume': {},
                            'directional_distance': 3.8,
                            'directional_angle': 155.0},
           'tetrel_bond': {'priority': 125,
                           'min_distance': 2.5,
                           'distance': 3.6,
                           'angle': 160.0,
                           'consume': {'tetrel_donor': 1, 'acceptor': 1},
                           'acceptor_angle': 160.0},
           'ch_pi': {'priority': 130,
                     'angle': 40.0,
                     'donor_angle': 100.0,
                     'require_aromatic': True,
                     'consume': {'carbon_donor': 1},
                     'hydrogen_distance': 3.5},
           'hydrophobic_enclosure': {'priority': 135, 'enabled': True, 'consume': {}},
           'hydrophobic': {'priority': 140, 'distance': 4.0},
           'aliphatic_stack': {'priority': 145,
                               'distance': 5.5,
                               'angle': 30.0,
                               'offset': 2.0,
                               'enabled': False}}}


def load_profile(profile: str | Path | dict[str, Any] = "default") -> dict[str, Any]:
    if isinstance(profile, dict):
        return _merge_profile(profile)

    profile_path = Path(profile)
    if profile_path.exists():
        return _merge_profile(_load_yaml(profile_path, require_yaml=True))

    if str(profile) != "default":
        raise FileNotFoundError(f"Profile {profile!r} was not found")

    try:
        with resources.files("chemur.profiles").joinpath("default.yaml").open(
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

    _validate_profile_rules(rule_overrides)

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
    _validate_profile_rules(profile.get("rules", {}))
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


def _validate_profile_rules(rules: dict[str, Any]) -> None:
    if "anion_pi" in rules:
        raise ValueError(
            "anion_pi detection has been removed because its geometry was not validated; "
            "remove this rule from the profile. Historical results remain readable."
        )
    for name in ("hbond_pi", "ch_pi"):
        values = rules.get(name) or {}
        if "distance" in values or "offset" in values:
            raise ValueError(
                f"{name} now measures hydrogen–ring-centroid distance: use hydrogen_distance "
                "instead of distance and remove offset; the previous heavy-atom and offset gates were removed."
            )
    removed = {
        "hbond": {"sulfur_distance", "short_distance"},
        "amide_bridge": {"distance", "angle"},
        "aliphatic_pi_stack": {"offset"},
        "cation_pi": {"angle"},
        "n_pi_star": {"approach_angle"},
        "amide_pi": {"offset"},
        "chalcogen_bond": {"distance", "require_activated_sigma_hole"},
        "halogen_pi": {"offset"},
        "chalcogen_pi": {"offset"},
        "tetrel_bond": {"hydrogen_angle"},
    }
    for name, keys in removed.items():
        obsolete = keys.intersection(rules.get(name) or {})
        if obsolete:
            raise ValueError(
                f"{name} no longer supports {', '.join(sorted(obsolete))}; remove these "
                "cutoffs and consult chemur.get_geometry_catalogue() for the current geometry."
            )
