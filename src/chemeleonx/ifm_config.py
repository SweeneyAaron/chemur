from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

CANONICAL_FAMILIES = (
    "hydrogen_bond",
    "ionic",
    "hydrophobic",
    "aromatic_stacking",
    "cation_pi",
    "halogen_bond",
    "metal_coordination",
    "water_bridge",
)

DEFAULT_FAMILY_WEIGHTS = {
    "metal_coordination": 4.0,
    "ionic": 3.0,
    "hydrogen_bond": 3.0,
    "aromatic_stacking": 2.5,
    "cation_pi": 2.5,
    "halogen_bond": 2.0,
    "water_bridge": 1.5,
    "hydrophobic": 1.0,
}


@dataclass(slots=True)
class ModeWeights:
    protein: float = 0.65
    ligand: float = 0.35
    normalize: bool = True

    def normalized(self) -> tuple[float, float]:
        if not self.normalize:
            return self.protein, self.ligand
        total = self.protein + self.ligand
        if total == 0.0:
            return 0.0, 0.0
        return self.protein / total, self.ligand / total


@dataclass(slots=True)
class PenaltyConfig:
    missing_reference_scale: float = 1.0
    extra_query_scale: float = 0.25


@dataclass(slots=True)
class GeometryConfig:
    distance_alpha: float = 1.0
    angle_alpha: float = 1.0
    offset_alpha: float = 1.0
    planarity_alpha: float = 1.0
    distance_tolerance: float = 0.5
    angle_tolerance: float = 20.0
    distance_bucket_width: float = 0.5
    angle_bucket_width: float = 20.0
    offset_bucket_width: float = 0.5


@dataclass(slots=True)
class MatchingConfig:
    use_bipartite_assignment: bool = True
    strict_target_identity: bool = True
    allow_moiety_partial_credit: bool = True
    allow_entity_class_partial_credit: bool = True
    allow_entity_kind_partial_credit: bool = True
    subtype_partial_credit: float = 0.75
    moiety_partial_credit: float = 0.75
    entity_class_partial_credit: float = 0.5
    entity_kind_partial_credit: float = 0.25
    minimum_mapping_confidence: float = 0.4


@dataclass(slots=True)
class OutputConfig:
    emit_pairwise_matrix: bool = True
    emit_alignment_tables: bool = True
    emit_consensus_summary: bool = True


@dataclass(slots=True)
class SpecificityConfig:
    enabled: bool = False
    key: str = "target_family_subtype_role"
    scale: float = 1.0
    max_multiplier: float = 3.0
    min_multiplier: float = 0.25


@dataclass(slots=True)
class IFMConfig:
    mode_weights: ModeWeights = field(default_factory=ModeWeights)
    family_weights: dict[str, float] = field(
        default_factory=lambda: deepcopy(DEFAULT_FAMILY_WEIGHTS)
    )
    penalties: PenaltyConfig = field(default_factory=PenaltyConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    priority_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "required": 2.0,
            "preferred": 1.25,
            "optional": 1.0,
            "ignored": 0.0,
        }
    )
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    specificity: SpecificityConfig = field(default_factory=SpecificityConfig)
    normalization_strategy: str = "penalized_jaccard"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> "IFMConfig":
        cfg = cls()
        if not data:
            return cfg.validate()

        merged = cfg.to_dict()
        _deep_update(merged, data)
        cfg = cls(
            mode_weights=ModeWeights(**merged["mode_weights"]),
            family_weights=dict(merged["family_weights"]),
            penalties=PenaltyConfig(**merged["penalties"]),
            geometry=GeometryConfig(**merged["geometry"]),
            priority_multipliers=dict(merged["priority_multipliers"]),
            matching=MatchingConfig(**merged["matching"]),
            output=OutputConfig(**merged["output"]),
            specificity=SpecificityConfig(**merged.get("specificity", {})),
            normalization_strategy=merged.get(
                "normalization_strategy", "penalized_jaccard"
            ),
        )
        return cfg.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, overrides: dict[str, Any] | None) -> "IFMConfig":
        if not overrides:
            return self.validate()
        merged = self.to_dict()
        _deep_update(merged, overrides)
        return IFMConfig.from_dict(merged)

    def validate(self) -> "IFMConfig":
        _require_nonnegative("mode_weights.protein", self.mode_weights.protein)
        _require_nonnegative("mode_weights.ligand", self.mode_weights.ligand)
        for family in CANONICAL_FAMILIES:
            if family not in self.family_weights:
                raise ValueError(f"Missing IFM family weight for {family!r}")
        for family, weight in self.family_weights.items():
            _require_nonnegative(f"family_weights.{family}", weight)
        _require_nonnegative(
            "penalties.missing_reference_scale",
            self.penalties.missing_reference_scale,
        )
        _require_nonnegative("penalties.extra_query_scale", self.penalties.extra_query_scale)
        for name in (
            "distance_alpha",
            "angle_alpha",
            "offset_alpha",
            "planarity_alpha",
            "distance_tolerance",
            "angle_tolerance",
            "distance_bucket_width",
            "angle_bucket_width",
            "offset_bucket_width",
        ):
            _require_nonnegative(f"geometry.{name}", getattr(self.geometry, name))
        for name, value in self.priority_multipliers.items():
            _require_nonnegative(f"priority_multipliers.{name}", value)
        for name in (
            "subtype_partial_credit",
            "moiety_partial_credit",
            "entity_class_partial_credit",
            "entity_kind_partial_credit",
            "minimum_mapping_confidence",
        ):
            _require_fraction(f"matching.{name}", getattr(self.matching, name))
        if self.specificity.key not in {
            "target_family",
            "target_family_subtype",
            "target_family_subtype_role",
        }:
            raise ValueError(f"Unsupported specificity.key: {self.specificity.key!r}")
        _require_nonnegative("specificity.scale", self.specificity.scale)
        _require_nonnegative(
            "specificity.min_multiplier",
            self.specificity.min_multiplier,
        )
        _require_nonnegative(
            "specificity.max_multiplier",
            self.specificity.max_multiplier,
        )
        if self.specificity.max_multiplier < self.specificity.min_multiplier:
            raise ValueError(
                "specificity.max_multiplier must be >= specificity.min_multiplier"
            )
        if self.normalization_strategy != "penalized_jaccard":
            raise ValueError(
                "Unsupported IFM normalization_strategy: "
                f"{self.normalization_strategy!r}"
            )
        return self


def load_ifm_config(config: IFMConfig | dict[str, Any] | str | Path | None) -> IFMConfig:
    if config is None:
        return IFMConfig()
    if isinstance(config, IFMConfig):
        return config.validate()
    if isinstance(config, (str, Path)):
        data = json.loads(Path(config).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("IFM config file must contain a JSON object")
        return IFMConfig.from_dict(data)
    if isinstance(config, dict):
        return IFMConfig.from_dict(config)
    raise TypeError("config must be an IFMConfig, dict, path, or None")


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _require_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_fraction(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
