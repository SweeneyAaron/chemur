from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Vec3 = tuple[float, float, float]


@dataclass(slots=True)
class LigandFeature:
    feature_id: str
    feature_type: str
    atom_ids: tuple[int, ...] = ()
    centroid: Vec3 | None = None
    normal: Vec3 | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InteractionRecord:
    interaction_id: str
    ligand_id: str
    target_id: str
    interaction_family: str
    interaction_subtype: str | None
    ligand_feature_id: str | None
    ligand_atom_ids: tuple[int, ...]
    target_entity_kind: str
    target_entity_id: str
    target_moiety: str
    target_atom_ids: tuple[int, ...]
    geometry: dict[str, float | None] = field(default_factory=dict)
    directionality: dict[str, bool] = field(default_factory=dict)
    strength_weight: float = 1.0
    confidence: float = 1.0
    priority: str = "optional"
    target_entity_class: str | None = None
    source_interaction_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def alignment_key(self, *, include_subtype: bool = True) -> str:
        parts = [self.target_entity_id, self.interaction_family]
        if include_subtype and self.interaction_subtype:
            parts.append(self.interaction_subtype)
        role = self.role_key()
        if role:
            parts.append(role)
        return " | ".join(parts)

    def role_key(self) -> str | None:
        ligand_is_donor = bool(self.directionality.get("ligand_is_donor"))
        target_is_donor = bool(self.directionality.get("target_is_donor"))
        if ligand_is_donor and not target_is_donor:
            return "ligand_donor"
        if target_is_donor and not ligand_is_donor:
            return "target_donor"
        return None

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.target_entity_id,
            self.interaction_family,
            self.interaction_subtype or "",
            self.ligand_feature_id or "",
            self.interaction_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FingerprintComplex:
    ligand_id: str
    target_id: str
    interaction_records: list[InteractionRecord]
    ligand_features: list[LigandFeature] = field(default_factory=list)
    pose_id: str | None = None
    upstream_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MatchResult:
    ligand_id: str
    reference_ligand_id: str
    mode_used: str
    protein_score: float = 0.0
    ligand_score: float = 0.0
    hybrid_score: float = 0.0
    strict_similarity: float = 0.0
    relaxed_similarity: float = 0.0
    raw_alignment_score: float = 0.0
    mapping_confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
    matched_interactions: list[dict[str, Any]] = field(default_factory=list)
    missing_interactions: list[dict[str, Any]] = field(default_factory=list)
    gained_interactions: list[dict[str, Any]] = field(default_factory=list)
    per_target_summary: list[dict[str, Any]] = field(default_factory=list)
    per_ligand_feature_summary: list[dict[str, Any]] = field(default_factory=list)
    medicinal_chemistry_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BatchMatchResult:
    reference_ligand_id: str
    results: list[MatchResult]
    pairwise_similarity_matrix: dict[str, Any] | list[Any] = field(default_factory=dict)
    ranked_ligands: list[dict[str, Any]] = field(default_factory=list)
    consensus_interactions: list[dict[str, Any]] = field(default_factory=list)
    cluster_labels: list[Any] = field(default_factory=list)
    config_used: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["results"] = [result.to_dict() for result in self.results]
        return data
