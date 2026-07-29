from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import exp, log, sqrt
import os
from pathlib import Path
from typing import Any, Iterable
import json
import warnings as py_warnings

from .ifm_config import CANONICAL_FAMILIES, IFMConfig, load_ifm_config
from .ifm_models import (
    BatchMatchResult,
    FingerprintComplex,
    InteractionRecord,
    LigandFeature,
    MatchResult,
)
from .ifm_scoring import (
    extra_penalty,
    missing_penalty,
    normalized_similarity,
    score_interaction_pair,
)
from .models import AnalysisResult

VALID_MODES = {"protein", "ligand", "hybrid"}
VALID_STRICTNESS = {"strict", "relaxed", "both"}
VALID_IFM_ENGINES = {"auto", "cpp", "python"}

_FAMILY_ORDER = {family: index for index, family in enumerate(CANONICAL_FAMILIES)}
_SUPPORTED_SUBTYPES = {
    "backbone_hbond",
    "sidechain_hbond",
    "base_edge_hbond",
    "base_stacking",
    "t_stack",
    "parallel_stack",
    "phosphate_contact",
    "sugar_contact",
    "donor_role",
    "acceptor_role",
}
_BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT", "H", "HN", "HA"}
_PHOSPHATE_PREFIXES = ("P", "OP", "O1P", "O2P", "O3P")
_SUGAR_MARKERS = ("'", "*")


@dataclass(slots=True)
class _AlignmentOutcome:
    raw_score: float
    similarity: float
    matched_weight: float
    missing_weight: float
    extra_weight: float
    matched_interactions: list[dict[str, Any]]
    missing_interactions: list[dict[str, Any]]
    gained_interactions: list[dict[str, Any]]
    per_target_summary: list[dict[str, Any]]
    per_ligand_feature_summary: list[dict[str, Any]]
    warnings: list[str]
    mapping_confidence: float = 1.0


@dataclass(slots=True)
class _ProteinScoreParams:
    use_bipartite_assignment: bool
    strict_target_identity: bool
    allow_moiety_partial_credit: bool
    allow_entity_class_partial_credit: bool
    allow_entity_kind_partial_credit: bool
    subtype_partial_credit: float
    moiety_partial_credit: float
    entity_class_partial_credit: float
    entity_kind_partial_credit: float
    distance_alpha: float
    angle_alpha: float
    offset_alpha: float
    planarity_alpha: float


@dataclass(slots=True)
class _ProteinScoringRecord:
    sort_rank: int
    family_code: int
    subtype_code: int
    target_id_code: int
    target_kind_code: int
    target_moiety_code: int
    target_class_code: int
    role_mask: int
    geometry_mask: int
    distance: float
    angle: float
    offset: float
    planarity: float
    strength_weight: float
    confidence: float
    interaction_weight: float
    missing_penalty: float
    extra_penalty: float


@dataclass(slots=True)
class _ProteinScoringComplex:
    ligand_id: str
    target_id: str
    records: list[_ProteinScoringRecord]
    strict_groups: dict[tuple[int, int], list[_ProteinScoringRecord]]
    relaxed_groups: dict[tuple[int, int], list[_ProteinScoringRecord]]
    record_count: int
    warnings: tuple[str, ...]
    cpp_payload: tuple[str, list[tuple[Any, ...]]]


@dataclass(slots=True)
class ProteinIFMScoringBatch:
    """Compact, pre-indexed protein-mode IFM scoring batch."""

    config: IFMConfig
    complexes: list[_ProteinScoringComplex]
    by_ligand_id: dict[str, int]
    cpp_payloads: list[tuple[str, list[tuple[Any, ...]]]]


@dataclass(slots=True)
class _ProteinScore:
    raw_score: float
    similarity: float


_GEOM_DISTANCE = 1
_GEOM_ANGLE = 2
_GEOM_OFFSET = 4
_GEOM_PLANARITY = 8


def match_fingerprints(
    complexes,
    reference_ligand_id: str | None = None,
    mode: str = "hybrid",
    strictness: str = "both",
    ligand_mapping="auto",
    key_interactions=None,
    target_alignment=None,
    config: IFMConfig | dict[str, Any] | str | Path | None = None,
) -> BatchMatchResult:
    """Rank ligand complexes by interaction-fingerprint similarity."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(VALID_MODES))}")
    if strictness not in VALID_STRICTNESS:
        raise ValueError(
            f"strictness must be one of: {', '.join(sorted(VALID_STRICTNESS))}"
        )

    cfg = load_ifm_config(config)
    batch = _coerce_complexes(complexes, key_interactions=key_interactions)
    if not batch:
        raise ValueError("match_fingerprints requires at least one complex")

    reference = _select_reference(batch, reference_ligand_id)
    protein_available, protein_warning = _protein_mode_available(batch, target_alignment)
    reference_zero_warning = ""
    if not reference.interaction_records:
        reference_zero_warning = (
            f"Reference ligand {reference.ligand_id} has zero canonical interactions"
        )
        py_warnings.warn(reference_zero_warning, RuntimeWarning, stacklevel=2)

    results: list[MatchResult] = []
    for query in batch:
        result = _match_complex_pair(
            reference,
            query,
            cfg,
            mode=mode,
            strictness=strictness,
            ligand_mapping=ligand_mapping,
            protein_available=protein_available,
            protein_warning=protein_warning,
        )
        if reference_zero_warning:
            result.warnings.append(reference_zero_warning)
        if not query.interaction_records:
            result.warnings.append(
                f"Query ligand {query.ligand_id} has zero canonical interactions"
            )
        if cfg.output.emit_alignment_tables:
            result.medicinal_chemistry_summary = _medchem_summary(result)
        results.append(result)

    ranked_results = sorted(
        results,
        key=lambda result: (
            -_ranking_score(result, mode),
            result.ligand_id,
        ),
    )
    ranked_ligands = [
        {
            "rank": index + 1,
            "ligand_id": result.ligand_id,
            "score": _ranking_score(result, mode),
            "strict_similarity": result.strict_similarity,
            "relaxed_similarity": result.relaxed_similarity,
            "warnings": list(result.warnings),
        }
        for index, result in enumerate(ranked_results)
    ]

    pairwise_matrix: dict[str, Any] | list[Any] = []
    if cfg.output.emit_pairwise_matrix:
        pairwise_matrix = _pairwise_similarity_matrix(
            batch,
            cfg,
            protein_available=protein_available,
        )

    consensus = []
    if cfg.output.emit_consensus_summary:
        consensus = _consensus_interactions(batch)

    return BatchMatchResult(
        reference_ligand_id=reference.ligand_id,
        results=ranked_results,
        pairwise_similarity_matrix=pairwise_matrix,
        ranked_ligands=ranked_ligands,
        consensus_interactions=consensus,
        cluster_labels=[],
        config_used=cfg.to_dict(),
    )


def prepare_protein_ifm_scoring_batch(
    complexes,
    *,
    config: IFMConfig | dict[str, Any] | str | Path | None = None,
    key_interactions=None,
) -> ProteinIFMScoringBatch:
    """Coerce and pre-index complexes for repeated protein-mode IFM scoring."""
    cfg = load_ifm_config(config)
    batch = _coerce_complexes(complexes, key_interactions=key_interactions)
    codebook: dict[str, int] = {}
    specificity_multipliers = _specificity_multipliers(batch, cfg)
    indexed = [
        _build_protein_scoring_complex(
            complex_item,
            cfg,
            codebook,
            specificity_multipliers,
        )
        for complex_item in batch
    ]
    by_ligand_id: dict[str, int] = {}
    for index, complex_item in enumerate(indexed):
        by_ligand_id.setdefault(complex_item.ligand_id, index)
    return ProteinIFMScoringBatch(
        config=cfg,
        complexes=indexed,
        by_ligand_id=by_ligand_id,
        cpp_payloads=[complex_item.cpp_payload for complex_item in indexed],
    )


def score_protein_ifm_batch(
    scoring_batch,
    *,
    reference_ligand_id: str,
    config: IFMConfig | dict[str, Any] | str | Path | None = None,
    engine: str = "auto",
    workers: int | str = 1,
) -> list[dict[str, Any]]:
    """Score a pre-indexed batch against one reference in protein mode."""
    if engine not in VALID_IFM_ENGINES:
        raise ValueError(f"engine must be one of: {', '.join(sorted(VALID_IFM_ENGINES))}")
    if isinstance(scoring_batch, ProteinIFMScoringBatch):
        if config is not None:
            raise ValueError(
                "config must be supplied when preparing ProteinIFMScoringBatch, "
                "not when scoring it"
            )
        prepared = scoring_batch
    else:
        prepared = prepare_protein_ifm_scoring_batch(scoring_batch, config=config)

    try:
        reference = prepared.complexes[prepared.by_ligand_id[reference_ligand_id]]
    except KeyError as exc:
        raise ValueError(f"Reference ligand not found: {reference_ligand_id}") from exc

    target_ids = {complex_item.target_id for complex_item in prepared.complexes}
    protein_warning = ""
    protein_available = len(target_ids) <= 1
    if not protein_available:
        protein_warning = (
            "Protein-anchored mode unavailable: complexes do not share one target_id "
            "and no target correspondence map was supplied"
        )

    if engine in {"auto", "cpp"} and protein_available:
        try:
            return _score_protein_ifm_batch_cpp(prepared, reference, workers=workers)
        except RuntimeError:
            if engine == "cpp":
                raise

    return _score_protein_ifm_batch_python(
        prepared,
        reference,
        protein_available=protein_available,
        protein_warning=protein_warning,
    )


def analysis_result_to_ifm_payloads(
    result: AnalysisResult,
    *,
    key_interactions=None,
) -> list[dict[str, Any]]:
    """Convert an AnalysisResult into compact fingerprint-ready payloads."""
    return [
        _fingerprint_complex_to_payload(complex_item)
        for complex_item in _coerce_one_complex(
            result,
            default_ligand_id=None,
            key_interactions=key_interactions,
        )
    ]


def _build_protein_scoring_complex(
    complex_item: FingerprintComplex,
    config: IFMConfig,
    codebook: dict[str, int],
    specificity_multipliers: dict[str, float],
) -> _ProteinScoringComplex:
    active_source_records = _active_records(complex_item.interaction_records, config)
    sort_ranks = {
        id(record): index
        for index, record in enumerate(sorted(active_source_records, key=lambda item: item.sort_key()))
    }
    records = [
        _protein_scoring_record(
            record,
            config,
            codebook,
            sort_ranks[id(record)],
            specificity_multipliers,
        )
        for record in active_source_records
    ]
    strict_groups = _protein_record_groups(records, strict=True)
    relaxed_groups = _protein_record_groups(records, strict=False)
    cpp_records = [_cpp_record_payload(record) for record in records]
    return _ProteinScoringComplex(
        ligand_id=complex_item.ligand_id,
        target_id=complex_item.target_id,
        records=records,
        strict_groups=strict_groups,
        relaxed_groups=relaxed_groups,
        record_count=len(complex_item.interaction_records),
        warnings=tuple(complex_item.warnings),
        cpp_payload=(complex_item.ligand_id, cpp_records),
    )


def _protein_scoring_record(
    record: InteractionRecord,
    config: IFMConfig,
    codebook: dict[str, int],
    sort_rank: int,
    specificity_multipliers: dict[str, float],
) -> _ProteinScoringRecord:
    geometry_mask = 0
    distance = angle = offset = planarity = 0.0
    if record.geometry.get("distance") is not None:
        geometry_mask |= _GEOM_DISTANCE
        distance = float(record.geometry["distance"])
    if record.geometry.get("angle") is not None:
        geometry_mask |= _GEOM_ANGLE
        angle = float(record.geometry["angle"])
    if record.geometry.get("offset") is not None:
        geometry_mask |= _GEOM_OFFSET
        offset = float(record.geometry["offset"])
    if record.geometry.get("planarity") is not None:
        geometry_mask |= _GEOM_PLANARITY
        planarity = float(record.geometry["planarity"])

    role_mask = 0
    if bool(record.directionality.get("ligand_is_donor")):
        role_mask |= 1
    if bool(record.directionality.get("target_is_donor")):
        role_mask |= 2

    family_weight = config.family_weights.get(record.interaction_family, 1.0)
    priority_multiplier = config.priority_multipliers.get(record.priority, 1.0)
    specificity_multiplier = specificity_multipliers.get(
        _specificity_key(record, config),
        1.0,
    )
    interaction = family_weight * priority_multiplier * specificity_multiplier
    return _ProteinScoringRecord(
        sort_rank=sort_rank,
        family_code=_code_value(codebook, record.interaction_family),
        subtype_code=_optional_code(codebook, record.interaction_subtype),
        target_id_code=_code_value(codebook, record.target_entity_id),
        target_kind_code=_code_value(codebook, record.target_entity_kind),
        target_moiety_code=_code_value(codebook, record.target_moiety),
        target_class_code=_optional_code(codebook, record.target_entity_class),
        role_mask=role_mask,
        geometry_mask=geometry_mask,
        distance=distance,
        angle=angle,
        offset=offset,
        planarity=planarity,
        strength_weight=record.strength_weight,
        confidence=record.confidence,
        interaction_weight=interaction,
        missing_penalty=interaction * config.penalties.missing_reference_scale,
        extra_penalty=family_weight * config.penalties.extra_query_scale,
    )


def _code_value(codebook: dict[str, int], value: Any) -> int:
    key = str(value)
    code = codebook.get(key)
    if code is None:
        code = len(codebook)
        codebook[key] = code
    return code


def _optional_code(codebook: dict[str, int], value: Any) -> int:
    if value is None or value == "":
        return -1
    return _code_value(codebook, value)


def _specificity_multipliers(
    complexes: list[FingerprintComplex],
    config: IFMConfig,
) -> dict[str, float]:
    if not config.specificity.enabled:
        return {}

    total = max(len(complexes), 1)
    counts: dict[str, int] = defaultdict(int)
    for complex_item in complexes:
        keys = {
            _specificity_key(record, config)
            for record in _active_records(complex_item.interaction_records, config)
        }
        for key in keys:
            counts[key] += 1
    if not counts:
        return {}

    raw_idf = {
        key: log((total + 1.0) / (count + 1.0))
        for key, count in counts.items()
    }
    mean_idf = sum(raw_idf.values()) / len(raw_idf)
    return {
        key: min(
            config.specificity.max_multiplier,
            max(
                config.specificity.min_multiplier,
                exp(config.specificity.scale * (value - mean_idf)),
            ),
        )
        for key, value in raw_idf.items()
    }


def _specificity_key(record: InteractionRecord, config: IFMConfig) -> str:
    parts = [record.target_entity_id, record.interaction_family]
    if config.specificity.key in {
        "target_family_subtype",
        "target_family_subtype_role",
    }:
        parts.append(record.interaction_subtype or "")
    if config.specificity.key == "target_family_subtype_role":
        parts.append(record.role_key() or "")
    return " | ".join(parts)


def _protein_record_groups(
    records: list[_ProteinScoringRecord],
    *,
    strict: bool,
) -> dict[tuple[int, int], list[_ProteinScoringRecord]]:
    groups: dict[tuple[int, int], list[_ProteinScoringRecord]] = defaultdict(list)
    for record in records:
        target_code = record.target_id_code if strict else record.target_kind_code
        groups[(target_code, record.family_code)].append(record)
    return groups


def _cpp_record_payload(record: _ProteinScoringRecord) -> tuple[Any, ...]:
    return (
        record.sort_rank,
        record.family_code,
        record.subtype_code,
        record.target_id_code,
        record.target_kind_code,
        record.target_moiety_code,
        record.target_class_code,
        record.role_mask,
        record.geometry_mask,
        record.distance,
        record.angle,
        record.offset,
        record.planarity,
        record.strength_weight,
        record.confidence,
        record.interaction_weight,
        record.missing_penalty,
        record.extra_penalty,
    )


def _score_protein_ifm_batch_python(
    prepared: ProteinIFMScoringBatch,
    reference: _ProteinScoringComplex,
    *,
    protein_available: bool,
    protein_warning: str,
) -> list[dict[str, Any]]:
    params = _protein_score_params(prepared.config)
    reference_zero_warning = (
        f"Reference ligand {reference.ligand_id} has zero canonical interactions"
        if reference.record_count == 0
        else ""
    )
    rows: list[dict[str, Any]] = []
    for query in prepared.complexes:
        if protein_available:
            strict_score = _score_protein_indexed_pair(reference, query, params, strict=True)
            relaxed_score = _score_protein_indexed_pair(reference, query, params, strict=False)
            protein_score = strict_score.raw_score
            strict_similarity = strict_score.similarity
            relaxed_similarity = relaxed_score.similarity
            raw_alignment_score = protein_score
        else:
            protein_score = raw_alignment_score = 0.0
            strict_similarity = relaxed_similarity = 0.0

        warnings = [*reference.warnings, *query.warnings]
        if not protein_available:
            warnings.append(protein_warning)
        if reference_zero_warning:
            warnings.append(reference_zero_warning)
        if query.record_count == 0:
            warnings.append(f"Query ligand {query.ligand_id} has zero canonical interactions")
        rows.append(
            {
                "ligand_id": query.ligand_id,
                "reference_ligand_id": reference.ligand_id,
                "mode_used": "protein",
                "protein_score": protein_score,
                "ligand_score": 0.0,
                "hybrid_score": 0.0,
                "strict_similarity": strict_similarity,
                "relaxed_similarity": relaxed_similarity,
                "raw_alignment_score": raw_alignment_score,
                "mapping_confidence": 1.0,
                "warnings": _dedupe(warnings),
            }
        )
    rows.sort(key=lambda result: (-float(result["protein_score"]), str(result["ligand_id"])))
    return rows


def _score_protein_indexed_pair(
    reference: _ProteinScoringComplex,
    query: _ProteinScoringComplex,
    params: _ProteinScoreParams,
    *,
    strict: bool,
) -> _ProteinScore:
    ref_groups = reference.strict_groups if strict else reference.relaxed_groups
    qry_groups = query.strict_groups if strict else query.relaxed_groups
    matched_weight = 0.0
    missing_weight = 0.0
    extra_weight = 0.0
    for group_key in sorted(set(ref_groups) | set(qry_groups)):
        refs = ref_groups.get(group_key, [])
        qrys = qry_groups.get(group_key, [])
        if not refs:
            extra_weight += sum(record.extra_penalty for record in qrys)
            continue
        if not qrys:
            missing_weight += sum(record.missing_penalty for record in refs)
            continue
        matched, missing, extra = _assign_indexed_records(
            refs,
            qrys,
            params,
            strict=strict,
        )
        matched_weight += matched
        missing_weight += missing
        extra_weight += extra
    denominator = matched_weight + missing_weight + extra_weight
    similarity = matched_weight / denominator if denominator > 0.0 else 0.0
    return _ProteinScore(
        raw_score=matched_weight - missing_weight - extra_weight,
        similarity=similarity,
    )


def _assign_indexed_records(
    reference_records: list[_ProteinScoringRecord],
    query_records: list[_ProteinScoringRecord],
    params: _ProteinScoreParams,
    *,
    strict: bool,
) -> tuple[float, float, float]:
    if len(reference_records) == 1 and len(query_records) == 1:
        score = _score_indexed_interaction_pair(
            reference_records[0],
            query_records[0],
            params,
            strict=strict,
        )
        if score > 0.0:
            return score, 0.0, 0.0
        return (
            0.0,
            reference_records[0].missing_penalty,
            query_records[0].extra_penalty,
        )

    if params.use_bipartite_assignment:
        return _hungarian_assign_indexed(
            reference_records,
            query_records,
            params,
            strict=strict,
        )
    return _greedy_assign_indexed(
        reference_records,
        query_records,
        params,
        strict=strict,
    )


def _hungarian_assign_indexed(
    reference_records: list[_ProteinScoringRecord],
    query_records: list[_ProteinScoringRecord],
    params: _ProteinScoreParams,
    *,
    strict: bool,
) -> tuple[float, float, float]:
    refs = sorted(reference_records, key=lambda record: record.sort_rank)
    qrys = sorted(query_records, key=lambda record: record.sort_rank)
    scores = [
        [_score_indexed_interaction_pair(ref, qry, params, strict=strict) for qry in qrys]
        for ref in refs
    ]
    max_score = max((score for row in scores for score in row), default=0.0)
    dummy_columns = len(refs)
    cost_matrix = [
        [max_score - score for score in row] + [max_score] * dummy_columns
        for row in scores
    ]
    assignment = _hungarian_min_cost(cost_matrix)
    matched_weight = 0.0
    missing_weight = 0.0
    used_qry: set[int] = set()
    for ref_index, column in enumerate(assignment):
        if 0 <= column < len(qrys):
            score = scores[ref_index][column]
            if score > 0.0:
                matched_weight += score
                used_qry.add(column)
                continue
        missing_weight += refs[ref_index].missing_penalty
    extra_weight = sum(
        record.extra_penalty
        for index, record in enumerate(qrys)
        if index not in used_qry
    )
    return matched_weight, missing_weight, extra_weight


def _greedy_assign_indexed(
    reference_records: list[_ProteinScoringRecord],
    query_records: list[_ProteinScoringRecord],
    params: _ProteinScoreParams,
    *,
    strict: bool,
) -> tuple[float, float, float]:
    candidates = []
    for ref_index, ref in enumerate(reference_records):
        for qry_index, qry in enumerate(query_records):
            score = _score_indexed_interaction_pair(ref, qry, params, strict=strict)
            if score > 0.0:
                candidates.append((score, ref.sort_rank, qry.sort_rank, ref_index, qry_index))
    candidates.sort(reverse=True)

    matched_weight = 0.0
    used_ref: set[int] = set()
    used_qry: set[int] = set()
    for score, _, _, ref_index, qry_index in candidates:
        if ref_index in used_ref or qry_index in used_qry:
            continue
        used_ref.add(ref_index)
        used_qry.add(qry_index)
        matched_weight += score
    missing_weight = sum(
        record.missing_penalty
        for index, record in enumerate(reference_records)
        if index not in used_ref
    )
    extra_weight = sum(
        record.extra_penalty
        for index, record in enumerate(query_records)
        if index not in used_qry
    )
    return matched_weight, missing_weight, extra_weight


def _score_indexed_interaction_pair(
    reference: _ProteinScoringRecord,
    query: _ProteinScoringRecord,
    params: _ProteinScoreParams,
    *,
    strict: bool,
) -> float:
    if reference.family_code != query.family_code or reference.interaction_weight == 0.0:
        return 0.0
    return (
        reference.interaction_weight
        * reference.strength_weight
        * query.strength_weight
        * _indexed_geometry_similarity(reference, query, params)
        * _indexed_role_similarity(reference, query)
        * _indexed_target_similarity(reference, query, params, strict=strict)
        * _indexed_subtype_similarity(reference, query, params)
        * sqrt(max(reference.confidence, 0.0) * max(query.confidence, 0.0))
    )


def _indexed_geometry_similarity(
    reference: _ProteinScoringRecord,
    query: _ProteinScoringRecord,
    params: _ProteinScoreParams,
) -> float:
    shared_mask = reference.geometry_mask & query.geometry_mask
    score = 1.0
    if shared_mask & _GEOM_DISTANCE:
        score *= exp(-params.distance_alpha * abs(reference.distance - query.distance))
    if shared_mask & _GEOM_ANGLE:
        angle_error = abs(reference.angle - query.angle)
        angle_error = min(angle_error, 360.0 - angle_error) / 180.0
        score *= exp(-params.angle_alpha * angle_error)
    if shared_mask & _GEOM_OFFSET:
        score *= exp(-params.offset_alpha * abs(reference.offset - query.offset))
    if shared_mask & _GEOM_PLANARITY:
        score *= exp(-params.planarity_alpha * abs(reference.planarity - query.planarity))
    return score


def _indexed_role_similarity(
    reference: _ProteinScoringRecord,
    query: _ProteinScoringRecord,
) -> float:
    return 1.0 if reference.role_mask == query.role_mask else 0.5


def _indexed_target_similarity(
    reference: _ProteinScoringRecord,
    query: _ProteinScoringRecord,
    params: _ProteinScoreParams,
    *,
    strict: bool,
) -> float:
    if reference.target_id_code == query.target_id_code:
        return 1.0
    if strict and params.strict_target_identity:
        return 0.0
    if reference.target_kind_code != query.target_kind_code:
        return 0.0
    if params.allow_moiety_partial_credit and reference.target_moiety_code == query.target_moiety_code:
        return params.moiety_partial_credit
    if (
        params.allow_entity_class_partial_credit
        and reference.target_class_code >= 0
        and reference.target_class_code == query.target_class_code
    ):
        return params.entity_class_partial_credit
    if params.allow_entity_kind_partial_credit:
        return params.entity_kind_partial_credit
    return 0.0


def _indexed_subtype_similarity(
    reference: _ProteinScoringRecord,
    query: _ProteinScoringRecord,
    params: _ProteinScoreParams,
) -> float:
    if reference.subtype_code == query.subtype_code:
        return 1.0
    if reference.subtype_code < 0 or query.subtype_code < 0:
        return 1.0
    return params.subtype_partial_credit


def _protein_score_params(config: IFMConfig) -> _ProteinScoreParams:
    return _ProteinScoreParams(
        use_bipartite_assignment=config.matching.use_bipartite_assignment,
        strict_target_identity=config.matching.strict_target_identity,
        allow_moiety_partial_credit=config.matching.allow_moiety_partial_credit,
        allow_entity_class_partial_credit=config.matching.allow_entity_class_partial_credit,
        allow_entity_kind_partial_credit=config.matching.allow_entity_kind_partial_credit,
        subtype_partial_credit=config.matching.subtype_partial_credit,
        moiety_partial_credit=config.matching.moiety_partial_credit,
        entity_class_partial_credit=config.matching.entity_class_partial_credit,
        entity_kind_partial_credit=config.matching.entity_kind_partial_credit,
        distance_alpha=config.geometry.distance_alpha,
        angle_alpha=config.geometry.angle_alpha,
        offset_alpha=config.geometry.offset_alpha,
        planarity_alpha=config.geometry.planarity_alpha,
    )


def _score_protein_ifm_batch_cpp(
    prepared: ProteinIFMScoringBatch,
    reference: _ProteinScoringComplex,
    *,
    workers: int | str,
) -> list[dict[str, Any]]:
    try:
        from . import core
    except ImportError as exc:
        raise RuntimeError("C++ ChemeleonX core is unavailable") from exc

    scorer = getattr(core, "score_ifm_protein_batch", None)
    if scorer is None:
        raise RuntimeError("C++ ChemeleonX core does not expose score_ifm_protein_batch")

    worker_count = _resolve_ifm_worker_count(workers)
    cpp_rows = scorer(
        reference.cpp_payload,
        prepared.cpp_payloads,
        _cpp_score_config(prepared.config),
        worker_count,
    )
    if len(cpp_rows) != len(prepared.complexes):
        raise RuntimeError("C++ IFM scorer returned an unexpected number of rows")

    reference_zero_warning = (
        f"Reference ligand {reference.ligand_id} has zero canonical interactions"
        if reference.record_count == 0
        else ""
    )
    rows: list[dict[str, Any]] = []
    for query, cpp_row in zip(prepared.complexes, cpp_rows):
        strict_raw_score = float(cpp_row["strict_raw_score"])
        warnings = [*reference.warnings, *query.warnings]
        if reference_zero_warning:
            warnings.append(reference_zero_warning)
        if query.record_count == 0:
            warnings.append(f"Query ligand {query.ligand_id} has zero canonical interactions")
        rows.append(
            {
                "ligand_id": query.ligand_id,
                "reference_ligand_id": reference.ligand_id,
                "mode_used": "protein",
                "protein_score": strict_raw_score,
                "ligand_score": 0.0,
                "hybrid_score": 0.0,
                "strict_similarity": float(cpp_row["strict_similarity"]),
                "relaxed_similarity": float(cpp_row["relaxed_similarity"]),
                "raw_alignment_score": strict_raw_score,
                "mapping_confidence": 1.0,
                "warnings": _dedupe(warnings),
            }
        )
    rows.sort(key=lambda result: (-float(result["protein_score"]), str(result["ligand_id"])))
    return rows


def _cpp_score_config(config: IFMConfig) -> dict[str, Any]:
    params = _protein_score_params(config)
    return asdict(params)


def _resolve_ifm_worker_count(workers: int | str) -> int:
    if isinstance(workers, str):
        value = workers.strip().lower()
        if value == "auto":
            return max(1, (os.cpu_count() or 1) - 1)
        return max(1, int(value))
    return max(1, int(workers))


def _match_complex_pair(
    reference: FingerprintComplex,
    query: FingerprintComplex,
    config: IFMConfig,
    *,
    mode: str,
    strictness: str,
    ligand_mapping,
    protein_available: bool,
    protein_warning: str,
) -> MatchResult:
    zero = _empty_outcome()
    protein_strict = protein_relaxed = zero
    ligand_strict = ligand_relaxed = zero
    warnings: list[str] = [*reference.warnings, *query.warnings]

    run_protein = mode in {"protein", "hybrid"}
    run_ligand = mode in {"ligand", "hybrid"}

    if run_protein:
        if protein_available:
            protein_strict = _protein_anchored_match(
                reference.interaction_records,
                query.interaction_records,
                config,
                strict=True,
            )
            protein_relaxed = _protein_anchored_match(
                reference.interaction_records,
                query.interaction_records,
                config,
                strict=False,
            )
        else:
            warnings.append(protein_warning)

    if run_ligand:
        ligand_strict = _ligand_anchored_match(
            reference,
            query,
            config,
            strict=True,
            ligand_mapping=ligand_mapping,
        )
        ligand_relaxed = _ligand_anchored_match(
            reference,
            query,
            config,
            strict=False,
            ligand_mapping=ligand_mapping,
        )

    protein_primary = protein_relaxed if strictness == "relaxed" else protein_strict
    ligand_primary = ligand_relaxed if strictness == "relaxed" else ligand_strict
    warnings.extend(protein_primary.warnings)
    warnings.extend(ligand_primary.warnings)

    if mode == "protein":
        primary = protein_primary
        mode_used = "protein"
        raw_alignment_score = primary.raw_score
        strict_similarity = protein_strict.similarity
        relaxed_similarity = protein_relaxed.similarity
        protein_score = primary.raw_score
        ligand_score = 0.0
        hybrid_score = 0.0
    elif mode == "ligand":
        primary = ligand_primary
        mode_used = "ligand"
        raw_alignment_score = primary.raw_score
        strict_similarity = ligand_strict.similarity
        relaxed_similarity = ligand_relaxed.similarity
        protein_score = 0.0
        ligand_score = primary.raw_score
        hybrid_score = 0.0
    else:
        protein_score = protein_primary.raw_score
        ligand_score = ligand_primary.raw_score
        if not protein_available:
            primary = ligand_primary
            mode_used = "ligand"
            raw_alignment_score = ligand_score
            hybrid_score = ligand_score
            strict_similarity = ligand_strict.similarity
            relaxed_similarity = ligand_relaxed.similarity
        else:
            protein_weight, ligand_weight = config.mode_weights.normalized()
            primary = protein_primary
            mode_used = "hybrid"
            hybrid_score = protein_weight * protein_score + ligand_weight * ligand_score
            raw_alignment_score = hybrid_score
            strict_similarity = (
                protein_weight * protein_strict.similarity
                + ligand_weight * ligand_strict.similarity
            )
            relaxed_similarity = (
                protein_weight * protein_relaxed.similarity
                + ligand_weight * ligand_relaxed.similarity
            )

    return MatchResult(
        ligand_id=query.ligand_id,
        reference_ligand_id=reference.ligand_id,
        mode_used=mode_used,
        protein_score=protein_score,
        ligand_score=ligand_score,
        hybrid_score=hybrid_score,
        strict_similarity=strict_similarity,
        relaxed_similarity=relaxed_similarity,
        raw_alignment_score=raw_alignment_score,
        mapping_confidence=ligand_primary.mapping_confidence,
        warnings=_dedupe(warnings),
        matched_interactions=primary.matched_interactions,
        missing_interactions=primary.missing_interactions,
        gained_interactions=primary.gained_interactions,
        per_target_summary=primary.per_target_summary,
        per_ligand_feature_summary=primary.per_ligand_feature_summary,
    )


def _protein_anchored_match(
    reference_records: list[InteractionRecord],
    query_records: list[InteractionRecord],
    config: IFMConfig,
    *,
    strict: bool,
) -> _AlignmentOutcome:
    ref_groups = _group_records(reference_records, strict=strict)
    qry_groups = _group_records(query_records, strict=strict)

    matched: list[tuple[InteractionRecord, InteractionRecord, float]] = []
    missing: list[tuple[InteractionRecord, float]] = []
    gained: list[tuple[InteractionRecord, float]] = []

    for group_key in sorted(set(ref_groups) | set(qry_groups)):
        pairs, unpaired_ref, unpaired_qry = _assign_records(
            ref_groups.get(group_key, []),
            qry_groups.get(group_key, []),
            config,
            strict=strict,
        )
        matched.extend(pairs)
        missing.extend((record, missing_penalty(record, config)) for record in unpaired_ref)
        gained.extend((record, extra_penalty(record, config)) for record in unpaired_qry)

    return _finalize_alignment(
        matched,
        missing,
        gained,
        config,
        mode_component="protein",
    )


def _ligand_anchored_match(
    reference: FingerprintComplex,
    query: FingerprintComplex,
    config: IFMConfig,
    *,
    strict: bool,
    ligand_mapping,
) -> _AlignmentOutcome:
    feature_map, mapping_confidence, mapping_warnings = _map_ligand_features(
        reference,
        query,
        config,
        ligand_mapping,
    )
    ref_by_feature = _records_by_ligand_feature(reference.interaction_records)
    qry_by_feature = _records_by_ligand_feature(query.interaction_records)

    matched: list[tuple[InteractionRecord, InteractionRecord, float]] = []
    missing: list[tuple[InteractionRecord, float]] = []
    gained: list[tuple[InteractionRecord, float]] = []
    used_qry_features: set[str] = set()

    for ref_feature_id, qry_feature_id in sorted(feature_map.items()):
        used_qry_features.add(qry_feature_id)
        pairs, unpaired_ref, unpaired_qry = _assign_records(
            ref_by_feature.get(ref_feature_id, []),
            qry_by_feature.get(qry_feature_id, []),
            config,
            strict=strict,
        )
        matched.extend(
            (ref_record, qry_record, score * mapping_confidence)
            for ref_record, qry_record, score in pairs
        )
        missing.extend((record, missing_penalty(record, config)) for record in unpaired_ref)
        gained.extend((record, extra_penalty(record, config)) for record in unpaired_qry)

    for ref_feature_id, records in ref_by_feature.items():
        if ref_feature_id not in feature_map:
            missing.extend((record, missing_penalty(record, config)) for record in records)

    for qry_feature_id, records in qry_by_feature.items():
        if qry_feature_id not in used_qry_features:
            gained.extend((record, extra_penalty(record, config)) for record in records)

    outcome = _finalize_alignment(
        matched,
        missing,
        gained,
        config,
        mode_component="ligand",
        mapping_confidence=mapping_confidence,
    )
    outcome.warnings.extend(mapping_warnings)
    if mapping_confidence < config.matching.minimum_mapping_confidence:
        outcome.warnings.append(
            "Ligand mapping confidence below configured minimum: "
            f"{mapping_confidence:.3f} < {config.matching.minimum_mapping_confidence:.3f}"
        )
    return outcome


def _assign_records(
    reference_records: list[InteractionRecord],
    query_records: list[InteractionRecord],
    config: IFMConfig,
    *,
    strict: bool,
) -> tuple[
    list[tuple[InteractionRecord, InteractionRecord, float]],
    list[InteractionRecord],
    list[InteractionRecord],
]:
    reference_records = _active_records(reference_records, config)
    query_records = _active_records(query_records, config)
    pairs: list[tuple[InteractionRecord, InteractionRecord, float]] = []
    unpaired_ref: list[InteractionRecord] = []
    unpaired_qry: list[InteractionRecord] = []
    families = sorted(
        {record.interaction_family for record in reference_records}
        | {record.interaction_family for record in query_records},
        key=lambda family: (_FAMILY_ORDER.get(family, 999), family),
    )
    for family in families:
        ref_family = [
            record for record in reference_records if record.interaction_family == family
        ]
        qry_family = [
            record for record in query_records if record.interaction_family == family
        ]
        if not ref_family:
            unpaired_qry.extend(qry_family)
            continue
        if not qry_family:
            unpaired_ref.extend(ref_family)
            continue
        if config.matching.use_bipartite_assignment:
            family_pairs, family_unpaired_ref, family_unpaired_qry = _hungarian_assign(
                ref_family,
                qry_family,
                config,
                strict=strict,
            )
        else:
            family_pairs, family_unpaired_ref, family_unpaired_qry = _greedy_assign(
                ref_family,
                qry_family,
                config,
                strict=strict,
            )
        pairs.extend(family_pairs)
        unpaired_ref.extend(family_unpaired_ref)
        unpaired_qry.extend(family_unpaired_qry)
    return pairs, unpaired_ref, unpaired_qry


def _hungarian_assign(
    reference_records: list[InteractionRecord],
    query_records: list[InteractionRecord],
    config: IFMConfig,
    *,
    strict: bool,
) -> tuple[
    list[tuple[InteractionRecord, InteractionRecord, float]],
    list[InteractionRecord],
    list[InteractionRecord],
]:
    refs = sorted(reference_records, key=lambda record: record.sort_key())
    qrys = sorted(query_records, key=lambda record: record.sort_key())
    scores = [
        [score_interaction_pair(ref, qry, config, strict=strict) for qry in qrys]
        for ref in refs
    ]
    max_score = max((score for row in scores for score in row), default=0.0)
    dummy_columns = len(refs)
    cost_matrix = [
        [max_score - score for score in row] + [max_score] * dummy_columns
        for row in scores
    ]
    assignment = _hungarian_min_cost(cost_matrix)

    pairs: list[tuple[InteractionRecord, InteractionRecord, float]] = []
    used_qry: set[int] = set()
    unpaired_ref: list[InteractionRecord] = []
    for ref_index, column in enumerate(assignment):
        if 0 <= column < len(qrys):
            score = scores[ref_index][column]
            if score > 0.0:
                pairs.append((refs[ref_index], qrys[column], score))
                used_qry.add(column)
                continue
        unpaired_ref.append(refs[ref_index])
    unpaired_qry = [qry for index, qry in enumerate(qrys) if index not in used_qry]
    return pairs, unpaired_ref, unpaired_qry


def _hungarian_min_cost(cost_matrix: list[list[float]]) -> list[int]:
    if not cost_matrix:
        return []
    row_count = len(cost_matrix)
    column_count = len(cost_matrix[0])
    if row_count > column_count:
        raise ValueError("Hungarian assignment requires row_count <= column_count")

    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)

    for i in range(1, row_count + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, column_count + 1):
                if used[j]:
                    continue
                cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(column_count + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * row_count
    for j in range(1, column_count + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def _greedy_assign(
    reference_records: list[InteractionRecord],
    query_records: list[InteractionRecord],
    config: IFMConfig,
    *,
    strict: bool,
) -> tuple[
    list[tuple[InteractionRecord, InteractionRecord, float]],
    list[InteractionRecord],
    list[InteractionRecord],
]:
    candidates = []
    for ref_index, ref in enumerate(reference_records):
        for qry_index, qry in enumerate(query_records):
            score = score_interaction_pair(ref, qry, config, strict=strict)
            if score > 0.0:
                candidates.append((score, ref.sort_key(), qry.sort_key(), ref_index, qry_index))
    candidates.sort(reverse=True)

    used_ref: set[int] = set()
    used_qry: set[int] = set()
    pairs: list[tuple[InteractionRecord, InteractionRecord, float]] = []
    for score, _, _, ref_index, qry_index in candidates:
        if ref_index in used_ref or qry_index in used_qry:
            continue
        used_ref.add(ref_index)
        used_qry.add(qry_index)
        pairs.append((reference_records[ref_index], query_records[qry_index], score))
    unpaired_ref = [
        record for index, record in enumerate(reference_records) if index not in used_ref
    ]
    unpaired_qry = [
        record for index, record in enumerate(query_records) if index not in used_qry
    ]
    return pairs, unpaired_ref, unpaired_qry


def _finalize_alignment(
    matched: list[tuple[InteractionRecord, InteractionRecord, float]],
    missing: list[tuple[InteractionRecord, float]],
    gained: list[tuple[InteractionRecord, float]],
    config: IFMConfig,
    *,
    mode_component: str,
    mapping_confidence: float = 1.0,
) -> _AlignmentOutcome:
    matched_weight = sum(score for _, _, score in matched)
    missing_weight = sum(penalty for _, penalty in missing)
    extra_weight = sum(penalty for _, penalty in gained)
    raw_score = matched_weight - missing_weight - extra_weight
    similarity = normalized_similarity(
        matched_weight,
        missing_weight,
        extra_weight,
        config,
    )

    if not config.output.emit_alignment_tables:
        return _AlignmentOutcome(
            raw_score=raw_score,
            similarity=similarity,
            matched_weight=matched_weight,
            missing_weight=missing_weight,
            extra_weight=extra_weight,
            matched_interactions=[],
            missing_interactions=[],
            gained_interactions=[],
            per_target_summary=[],
            per_ligand_feature_summary=[],
            warnings=[],
            mapping_confidence=mapping_confidence,
        )

    matched_entries = [
        {
            "mode_component": mode_component,
            "reference_interaction": ref.to_dict(),
            "query_interaction": qry.to_dict(),
            "score": score,
        }
        for ref, qry, score in sorted(matched, key=lambda item: item[0].sort_key())
    ]
    missing_entries = [
        {
            "mode_component": mode_component,
            "reference_interaction": record.to_dict(),
            "penalty": penalty,
        }
        for record, penalty in sorted(missing, key=lambda item: item[0].sort_key())
        if penalty > 0.0
    ]
    gained_entries = [
        {
            "mode_component": mode_component,
            "query_interaction": record.to_dict(),
            "penalty": penalty,
        }
        for record, penalty in sorted(gained, key=lambda item: item[0].sort_key())
        if penalty > 0.0
    ]

    return _AlignmentOutcome(
        raw_score=raw_score,
        similarity=similarity,
        matched_weight=matched_weight,
        missing_weight=missing_weight,
        extra_weight=extra_weight,
        matched_interactions=matched_entries,
        missing_interactions=missing_entries,
        gained_interactions=gained_entries,
        per_target_summary=_per_target_summary(
            matched_entries,
            missing_entries,
            gained_entries,
        ),
        per_ligand_feature_summary=_per_ligand_feature_summary(
            matched_entries,
            missing_entries,
            gained_entries,
        ),
        warnings=[],
        mapping_confidence=mapping_confidence,
    )


def _group_records(
    records: list[InteractionRecord],
    *,
    strict: bool,
) -> dict[tuple[str, ...], list[InteractionRecord]]:
    grouped: dict[tuple[str, ...], list[InteractionRecord]] = defaultdict(list)
    for record in records:
        if strict:
            key = (record.target_entity_id,)
        else:
            key = (record.target_entity_kind,)
        grouped[key].append(record)
    return grouped


def _records_by_ligand_feature(
    records: list[InteractionRecord],
) -> dict[str, list[InteractionRecord]]:
    grouped: dict[str, list[InteractionRecord]] = defaultdict(list)
    for record in records:
        if record.ligand_feature_id is None:
            continue
        grouped[record.ligand_feature_id].append(record)
    return grouped


def _active_records(
    records: list[InteractionRecord],
    config: IFMConfig,
) -> list[InteractionRecord]:
    return [
        record
        for record in records
        if config.priority_multipliers.get(record.priority, 1.0) > 0.0
        and config.family_weights.get(record.interaction_family, 0.0) > 0.0
    ]


def _map_ligand_features(
    reference: FingerprintComplex,
    query: FingerprintComplex,
    config: IFMConfig,
    ligand_mapping,
) -> tuple[dict[str, str], float, list[str]]:
    warnings: list[str] = []
    ref_features = sorted(reference.ligand_features, key=lambda feature: feature.feature_id)
    qry_features = sorted(query.ligand_features, key=lambda feature: feature.feature_id)
    if not ref_features:
        return {}, 0.0, ["Reference ligand exposes no mappable ligand features"]
    if not qry_features:
        return {}, 0.0, ["Query ligand exposes no mappable ligand features"]

    if isinstance(ligand_mapping, dict):
        feature_map = {
            str(ref_feature_id): str(qry_feature_id)
            for ref_feature_id, qry_feature_id in ligand_mapping.items()
        }
        mapped_fraction = len(feature_map) / max(len(ref_features), 1)
        return feature_map, min(1.0, 0.8 * mapped_fraction), []

    if ligand_mapping == "none":
        return {}, 0.0, ["Ligand feature mapping disabled"]

    qry_by_id = {feature.feature_id: feature for feature in qry_features}
    exact_map = {
        ref.feature_id: ref.feature_id
        for ref in ref_features
        if ref.feature_id in qry_by_id
        and qry_by_id[ref.feature_id].feature_type == ref.feature_type
    }
    if len(exact_map) == len(ref_features):
        return exact_map, 1.0, []

    atom_map = _map_features_by_atom_sets(ref_features, qry_features)
    if len(atom_map) == len(ref_features):
        return atom_map, 1.0, []

    centroid_map, avg_distance, ambiguous = _map_features_by_type_and_centroid(
        ref_features,
        qry_features,
    )
    if ambiguous:
        warnings.append("Ligand feature mapping ambiguous for one or more features")
    if centroid_map:
        mapped_fraction = len(centroid_map) / max(len(ref_features), 1)
        geo_confidence = exp(-config.geometry.distance_alpha * avg_distance)
        confidence = min(0.75, 0.65 * mapped_fraction + 0.10 * geo_confidence)
        if len(centroid_map) < len(ref_features):
            warnings.append(
                "Ligand feature mapping incomplete: "
                f"{len(centroid_map)} of {len(ref_features)} reference features mapped"
            )
        else:
            warnings.append("Ligand feature mapping used feature-type/centroid fallback")
        return centroid_map, confidence, warnings

    return {}, 0.0, ["Ligand feature mapping failed"]


def _map_features_by_atom_sets(
    ref_features: list[LigandFeature],
    qry_features: list[LigandFeature],
) -> dict[str, str]:
    qry_by_atoms: dict[tuple[int, ...], LigandFeature] = {
        tuple(sorted(feature.atom_ids)): feature for feature in qry_features
    }
    feature_map: dict[str, str] = {}
    for ref in ref_features:
        qry = qry_by_atoms.get(tuple(sorted(ref.atom_ids)))
        if qry is not None and qry.feature_type == ref.feature_type:
            feature_map[ref.feature_id] = qry.feature_id
    return feature_map


def _map_features_by_type_and_centroid(
    ref_features: list[LigandFeature],
    qry_features: list[LigandFeature],
) -> tuple[dict[str, str], float, bool]:
    used_query: set[str] = set()
    feature_map: dict[str, str] = {}
    distances: list[float] = []
    ambiguous = False

    for ref in ref_features:
        candidates = [
            (
                _feature_distance(ref, qry),
                abs(len(ref.atom_ids) - len(qry.atom_ids)),
                qry.feature_id,
                qry,
            )
            for qry in qry_features
            if qry.feature_id not in used_query and qry.feature_type == ref.feature_type
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        best = candidates[0]
        if len(candidates) > 1 and abs(candidates[1][0] - best[0]) < 0.2:
            ambiguous = True
        used_query.add(best[2])
        feature_map[ref.feature_id] = best[2]
        if best[0] != float("inf"):
            distances.append(best[0])

    avg_distance = sum(distances) / len(distances) if distances else 5.0
    return feature_map, avg_distance, ambiguous


def _feature_distance(reference: LigandFeature, query: LigandFeature) -> float:
    if reference.centroid is None or query.centroid is None:
        return float("inf")
    return sqrt(
        sum((reference.centroid[index] - query.centroid[index]) ** 2 for index in range(3))
    )


def _coerce_complexes(complexes, *, key_interactions=None) -> list[FingerprintComplex]:
    if isinstance(complexes, (str, Path)):
        payload = json.loads(Path(complexes).read_text(encoding="utf-8"))
        return _coerce_complexes(payload, key_interactions=key_interactions)

    if isinstance(complexes, dict) and "results" in complexes:
        coerced: list[FingerprintComplex] = []
        for key, payload in sorted(complexes["results"].items()):
            coerced.extend(
                _coerce_one_complex(
                    payload,
                    default_ligand_id=key,
                    key_interactions=key_interactions,
                )
            )
        return coerced

    if isinstance(complexes, dict) and "complexes" in complexes:
        source = complexes["complexes"]
    else:
        source = complexes

    if isinstance(source, (list, tuple)):
        coerced = []
        for index, item in enumerate(source):
            coerced.extend(
                _coerce_one_complex(
                    item,
                    default_ligand_id=f"ligand_{index + 1}",
                    key_interactions=key_interactions,
                )
            )
        return coerced

    return _coerce_one_complex(
        source,
        default_ligand_id=None,
        key_interactions=key_interactions,
    )


def _coerce_one_complex(
    item,
    *,
    default_ligand_id: str | None,
    key_interactions=None,
) -> list[FingerprintComplex]:
    if isinstance(item, FingerprintComplex):
        _apply_key_interaction_priorities(item.interaction_records, key_interactions)
        return [item]

    if isinstance(item, AnalysisResult):
        split = item.split_by_ligand()
        if split:
            return [
                _complex_from_analysis_payload(
                    result.to_dict(),
                    default_ligand_id=ligand_id,
                    key_interactions=key_interactions,
                )
                for ligand_id, result in sorted(split.items())
            ]
        return [
            _complex_from_analysis_payload(
                item.to_dict(),
                default_ligand_id=default_ligand_id,
                key_interactions=key_interactions,
            )
        ]

    if not isinstance(item, dict):
        raise TypeError("Complex entries must be dicts, AnalysisResult, or FingerprintComplex")

    if "interaction_records" in item:
        return [
            _complex_from_explicit_payload(
                item,
                default_ligand_id=default_ligand_id,
                key_interactions=key_interactions,
            )
        ]

    return [
        _complex_from_analysis_payload(
            item,
            default_ligand_id=default_ligand_id,
            key_interactions=key_interactions,
        )
    ]


def _complex_from_explicit_payload(
    payload: dict[str, Any],
    *,
    default_ligand_id: str | None,
    key_interactions=None,
) -> FingerprintComplex:
    ligand_id = str(payload.get("ligand_id") or default_ligand_id or "ligand")
    target_id = str(payload.get("target_id") or "target")
    records = [
        _record_from_canonical_dict(record, ligand_id=ligand_id, target_id=target_id)
        for record in payload.get("interaction_records", [])
    ]
    _apply_key_interaction_priorities(records, key_interactions)
    features = [
        _feature_from_dict(feature)
        for feature in payload.get("ligand_features", payload.get("features", []))
    ]
    return FingerprintComplex(
        ligand_id=ligand_id,
        target_id=target_id,
        interaction_records=records,
        ligand_features=features,
        pose_id=payload.get("pose_id"),
        upstream_score=payload.get("upstream_score"),
        metadata=dict(payload.get("metadata") or {}),
        warnings=list(payload.get("warnings") or []),
    )


def _fingerprint_complex_to_payload(complex_item: FingerprintComplex) -> dict[str, Any]:
    return {
        "ligand_id": complex_item.ligand_id,
        "target_id": complex_item.target_id,
        "interaction_records": [
            record.to_dict() for record in complex_item.interaction_records
        ],
        "ligand_features": [
            feature.to_dict() for feature in complex_item.ligand_features
        ],
        "pose_id": complex_item.pose_id,
        "upstream_score": complex_item.upstream_score,
        "metadata": dict(complex_item.metadata),
        "warnings": list(complex_item.warnings),
    }


def _complex_from_analysis_payload(
    payload: dict[str, Any],
    *,
    default_ligand_id: str | None,
    key_interactions=None,
) -> FingerprintComplex:
    metadata = dict(payload.get("metadata") or {})
    ligand_id = (
        payload.get("ligand_id")
        or metadata.get("ligand_component_id")
        or _single_ligand_component_id(payload)
        or default_ligand_id
    )
    if ligand_id is None:
        raise ValueError("Could not infer ligand_id for IFM complex")
    ligand_id = str(ligand_id)
    target_id = str(payload.get("target_id") or metadata.get("structure") or "target")
    warnings: list[str] = []
    records = _canonical_records_from_analysis_payload(
        payload,
        ligand_id=ligand_id,
        target_id=target_id,
        warnings=warnings,
    )
    _apply_key_interaction_priorities(records, key_interactions)
    features = _ligand_features_from_analysis_payload(payload, ligand_id=ligand_id)
    return FingerprintComplex(
        ligand_id=ligand_id,
        target_id=target_id,
        interaction_records=records,
        ligand_features=features,
        pose_id=payload.get("pose_id") or metadata.get("pose_id"),
        upstream_score=payload.get("upstream_score") or metadata.get("upstream_score"),
        metadata=metadata,
        warnings=warnings,
    )


def _record_from_canonical_dict(
    data: dict[str, Any],
    *,
    ligand_id: str,
    target_id: str,
) -> InteractionRecord:
    geometry = {
        key: _float_or_none(value)
        for key, value in dict(data.get("geometry") or {}).items()
    }
    return InteractionRecord(
        interaction_id=str(data.get("interaction_id") or f"{ligand_id}:{len(geometry)}"),
        ligand_id=str(data.get("ligand_id") or ligand_id),
        target_id=str(data.get("target_id") or target_id),
        interaction_family=str(data["interaction_family"]),
        interaction_subtype=data.get("interaction_subtype"),
        ligand_feature_id=(
            None
            if data.get("ligand_feature_id") is None
            else str(data.get("ligand_feature_id"))
        ),
        ligand_atom_ids=_int_tuple(data.get("ligand_atom_ids", ())),
        target_entity_kind=str(data.get("target_entity_kind") or "residue"),
        target_entity_id=str(data["target_entity_id"]),
        target_moiety=str(data.get("target_moiety") or "unknown"),
        target_atom_ids=_int_tuple(data.get("target_atom_ids", ())),
        geometry=geometry,
        directionality=dict(data.get("directionality") or {}),
        strength_weight=float(data.get("strength_weight", 1.0)),
        confidence=float(data.get("confidence", 1.0)),
        priority=str(data.get("priority") or "optional"),
        target_entity_class=data.get("target_entity_class")
        or _entity_class_from_target_id(str(data.get("target_entity_id", ""))),
        source_interaction_type=data.get("source_interaction_type"),
        metadata=dict(data.get("metadata") or {}),
    )


def _canonical_records_from_analysis_payload(
    payload: dict[str, Any],
    *,
    ligand_id: str,
    target_id: str,
    warnings: list[str],
) -> list[InteractionRecord]:
    atoms_by_id = {
        int(atom["atom_id"]): atom
        for atom in payload.get("atoms", [])
        if "atom_id" in atom
    }
    components_by_id = {
        str(component["component_id"]): component
        for component in payload.get("components", [])
        if "component_id" in component
    }
    features_by_id = {
        int(feature["feature_id"]): feature
        for feature in payload.get("features", [])
        if "feature_id" in feature
    }
    records: list[InteractionRecord] = []
    for index, interaction in enumerate(payload.get("interactions", []), start=1):
        component_ids = tuple(str(value) for value in interaction.get("component_ids", ()))
        if ligand_id not in component_ids:
            continue
        target_component_id = _choose_target_component_id(
            component_ids,
            ligand_id=ligand_id,
            components_by_id=components_by_id,
        )
        if target_component_id is None:
            continue

        raw_type = str(interaction.get("interaction_type") or "")
        family = _canonical_family(raw_type)
        if family is None:
            warnings.append(f"Unsupported interaction type skipped: {raw_type}")
            continue

        target_component = components_by_id.get(target_component_id, {})
        atom_ids = _int_tuple(interaction.get("atom_ids", ()))
        ligand_atom_ids = tuple(
            atom_id
            for atom_id in atom_ids
            if atoms_by_id.get(atom_id, {}).get("component_id") == ligand_id
        )
        target_atom_ids = tuple(
            atom_id
            for atom_id in atom_ids
            if atoms_by_id.get(atom_id, {}).get("component_id") == target_component_id
        )
        target_kind = _target_entity_kind(target_component, target_atom_ids, atoms_by_id)
        target_moiety = _target_moiety(
            target_kind,
            target_component,
            target_atom_ids,
            atoms_by_id,
        )
        subtype = _canonical_subtype(raw_type, interaction, target_moiety)
        if subtype and subtype not in _SUPPORTED_SUBTYPES:
            warnings.append(f"Unsupported interaction subtype encountered: {subtype}")

        feature_ids = _int_tuple(interaction.get("feature_ids", ()))
        ligand_feature_ids = [
            str(feature_id)
            for feature_id in feature_ids
            if features_by_id.get(feature_id, {}).get("component_id") == ligand_id
        ]
        ligand_feature_id = "+".join(ligand_feature_ids) if ligand_feature_ids else None
        assignment_order = interaction.get("assignment_order", index)
        records.append(
            InteractionRecord(
                interaction_id=f"{ligand_id}:{assignment_order}",
                ligand_id=ligand_id,
                target_id=target_id,
                interaction_family=family,
                interaction_subtype=subtype,
                ligand_feature_id=ligand_feature_id,
                ligand_atom_ids=ligand_atom_ids,
                target_entity_kind=target_kind,
                target_entity_id=target_component_id,
                target_moiety=target_moiety,
                target_atom_ids=target_atom_ids,
                geometry={
                    "distance": _float_or_none(interaction.get("distance")),
                    "angle": _float_or_none(interaction.get("angle")),
                    "offset": _float_or_none(interaction.get("offset")),
                    "planarity": _float_or_none(
                        (interaction.get("metadata") or {}).get("planarity")
                    ),
                },
                directionality=_directionality(
                    interaction,
                    ligand_atom_ids=set(ligand_atom_ids),
                    target_atom_ids=set(target_atom_ids),
                ),
                strength_weight=float(
                    (interaction.get("metadata") or {}).get("strength_weight", 1.0)
                ),
                confidence=float((interaction.get("metadata") or {}).get("confidence", 1.0)),
                priority=str((interaction.get("metadata") or {}).get("ifm_priority", "optional")),
                target_entity_class=_target_entity_class(target_component),
                source_interaction_type=raw_type,
                metadata={"source_priority": interaction.get("priority")},
            )
        )
    return sorted(records, key=lambda record: record.sort_key())


def _ligand_features_from_analysis_payload(
    payload: dict[str, Any],
    *,
    ligand_id: str,
) -> list[LigandFeature]:
    features: list[LigandFeature] = []
    for feature in payload.get("features", []):
        if feature.get("component_id") != ligand_id:
            continue
        features.append(
            LigandFeature(
                feature_id=str(feature["feature_id"]),
                feature_type=_canonical_feature_type(str(feature.get("feature_type", ""))),
                atom_ids=_int_tuple(feature.get("atom_ids", ())),
                centroid=_vec3_or_none(feature.get("center") or feature.get("centroid")),
                normal=_vec3_or_none(feature.get("normal")),
                properties=dict(feature.get("metadata") or {}),
            )
        )
    return sorted(features, key=lambda feature: feature.feature_id)


def _feature_from_dict(data: dict[str, Any]) -> LigandFeature:
    return LigandFeature(
        feature_id=str(data["feature_id"]),
        feature_type=_canonical_feature_type(str(data.get("feature_type", ""))),
        atom_ids=_int_tuple(data.get("atom_ids", ())),
        centroid=_vec3_or_none(data.get("centroid") or data.get("center")),
        normal=_vec3_or_none(data.get("normal")),
        properties=dict(data.get("properties") or data.get("metadata") or {}),
    )


def _canonical_feature_type(feature_type: str) -> str:
    return {
        "donor": "donor",
        "acceptor": "acceptor",
        "cation": "cationic_center",
        "anion": "anionic_center",
        "hydrophobe": "hydrophobe",
        "ring": "aromatic_ring",
        "halogen_donor": "halogen_donor",
        "metal": "metal_binder",
    }.get(feature_type, feature_type or "grouped_pharmacophore")


def _canonical_family(interaction_type: str) -> str | None:
    return {
        "hbond": "hydrogen_bond",
        "weak_hbond": "hydrogen_bond",
        "hbond_pi": "hydrogen_bond",
        "salt_bridge": "ionic",
        "anion_pi": "ionic",
        "hydrophobic": "hydrophobic",
        "ch_pi": "hydrophobic",
        "pipi_stack": "aromatic_stacking",
        "amide_pi": "aromatic_stacking",
        "cation_pi": "cation_pi",
        "halogen_bond": "halogen_bond",
        "halogen_pi": "halogen_bond",
        "metal_coordination": "metal_coordination",
        "solvent_bridge": "water_bridge",
        "amide_bridge": "hydrogen_bond",
    }.get(interaction_type)


def _canonical_subtype(
    interaction_type: str,
    interaction: dict[str, Any],
    target_moiety: str,
) -> str | None:
    metadata = dict(interaction.get("metadata") or {})
    explicit = metadata.get("interaction_subtype") or interaction.get("interaction_subtype")
    if explicit:
        return str(explicit)
    if interaction_type in {"hbond", "weak_hbond", "amide_bridge"}:
        if target_moiety == "backbone":
            return "backbone_hbond"
        if target_moiety == "base":
            return "base_edge_hbond"
        return "sidechain_hbond"
    if interaction_type == "pipi_stack":
        if metadata.get("geometry") == "parallel":
            return "parallel_stack"
        return "t_stack"
    if interaction_type == "salt_bridge" and target_moiety == "phosphate":
        return "phosphate_contact"
    if interaction_type == "hbond_pi":
        return "donor_role"
    return None


def _choose_target_component_id(
    component_ids: tuple[str, ...],
    *,
    ligand_id: str,
    components_by_id: dict[str, dict[str, Any]],
) -> str | None:
    candidates = [component_id for component_id in component_ids if component_id != ligand_id]
    if not candidates:
        return None
    priority = {"protein": 0, "nucleotide": 1, "ion": 2, "solvent": 3}
    return sorted(
        candidates,
        key=lambda component_id: (
            priority.get(
                str(components_by_id.get(component_id, {}).get("molecule_type", "")),
                4,
            ),
            component_id,
        ),
    )[0]


def _target_entity_kind(
    component: dict[str, Any],
    target_atom_ids: tuple[int, ...],
    atoms_by_id: dict[int, dict[str, Any]],
) -> str:
    molecule_type = str(component.get("molecule_type") or "")
    if molecule_type == "nucleotide":
        return "nucleotide"
    if molecule_type == "solvent":
        return "water"
    if molecule_type == "ion":
        return "metal"
    if molecule_type == "protein":
        return "residue"
    if any(str(atoms_by_id.get(atom_id, {}).get("molecule_type")) == "solvent" for atom_id in target_atom_ids):
        return "water"
    return "residue"


def _target_moiety(
    target_kind: str,
    component: dict[str, Any],
    target_atom_ids: tuple[int, ...],
    atoms_by_id: dict[int, dict[str, Any]],
) -> str:
    if target_kind == "metal":
        return "metal"
    if target_kind == "water":
        return "water"
    atom_names = [
        str(atoms_by_id.get(atom_id, {}).get("name", "")).strip().upper()
        for atom_id in target_atom_ids
    ]
    if target_kind == "nucleotide":
        if any(name.startswith(_PHOSPHATE_PREFIXES) for name in atom_names):
            return "phosphate"
        if atom_names and all(any(marker in name for marker in _SUGAR_MARKERS) for name in atom_names):
            return "sugar"
        return "base"
    if target_kind == "residue":
        if atom_names and all(name in _BACKBONE_ATOMS for name in atom_names):
            return "backbone"
        return "sidechain"
    return str(component.get("molecule_type") or "unknown")


def _target_entity_class(component: dict[str, Any]) -> str | None:
    name = str(component.get("name") or "").upper()
    molecule_type = str(component.get("molecule_type") or "")
    if molecule_type == "protein":
        return _protein_residue_class(name)
    if molecule_type == "nucleotide":
        return _nucleotide_class(name)
    if molecule_type == "ion":
        return name or "metal"
    if molecule_type == "solvent":
        return "water"
    return name or None


def _entity_class_from_target_id(target_id: str) -> str | None:
    parts = target_id.split(":")
    if len(parts) >= 2:
        residue_name = parts[1].upper()
        return _protein_residue_class(residue_name) or _nucleotide_class(residue_name)
    return None


def _protein_residue_class(name: str) -> str | None:
    if name in {"ASP", "GLU"}:
        return "acidic"
    if name in {"ARG", "HIS", "LYS"}:
        return "basic"
    if name in {"ASN", "GLN", "SER", "THR", "TYR", "CYS"}:
        return "polar"
    if name in {"PHE", "TRP", "TYR", "HIS"}:
        return "aromatic"
    if name in {"ALA", "VAL", "ILE", "LEU", "MET", "PRO"}:
        return "hydrophobic"
    if name in {"GLY"}:
        return "special"
    return None


def _nucleotide_class(name: str) -> str | None:
    if name in {"A", "G", "DA", "DG", "ADE", "GUA"}:
        return "purine"
    if name in {"C", "U", "T", "DC", "DU", "DT", "CYT", "URA", "THY"}:
        return "pyrimidine"
    return None


def _directionality(
    interaction: dict[str, Any],
    *,
    ligand_atom_ids: set[int],
    target_atom_ids: set[int],
) -> dict[str, bool]:
    ligand_is_donor = False
    target_is_donor = False
    for demand in interaction.get("resource_demands", ()):
        resource_type = str(demand.get("resource_type") if isinstance(demand, dict) else demand.resource_type)
        atom_id = int(demand.get("atom_id") if isinstance(demand, dict) else demand.atom_id)
        if resource_type not in {"donor", "halogen_donor"}:
            continue
        if atom_id in ligand_atom_ids:
            ligand_is_donor = True
        if atom_id in target_atom_ids:
            target_is_donor = True
    return {
        "ligand_is_donor": ligand_is_donor,
        "target_is_donor": target_is_donor,
    }


def _single_ligand_component_id(payload: dict[str, Any]) -> str | None:
    ligand_components = [
        str(component["component_id"])
        for component in payload.get("components", [])
        if component.get("molecule_type") == "ligand"
        and not (component.get("metadata") or {}).get("ignored_by_template")
    ]
    if len(ligand_components) == 1:
        return ligand_components[0]
    return None


def _apply_key_interaction_priorities(
    records: list[InteractionRecord],
    key_interactions,
) -> None:
    if not key_interactions:
        return
    if isinstance(key_interactions, dict):
        priority_by_key = {str(key): str(value) for key, value in key_interactions.items()}
        selectors: list[dict[str, Any]] = []
    else:
        priority_by_key = {}
        selectors = list(key_interactions)

    for record in records:
        if record.interaction_id in priority_by_key:
            record.priority = priority_by_key[record.interaction_id]
        alignment_key = record.alignment_key(include_subtype=True)
        loose_key = record.alignment_key(include_subtype=False)
        if alignment_key in priority_by_key:
            record.priority = priority_by_key[alignment_key]
        if loose_key in priority_by_key:
            record.priority = priority_by_key[loose_key]
        for selector in selectors:
            priority = selector.get("priority", "optional")
            if _record_matches_selector(record, selector):
                record.priority = str(priority)


def _record_matches_selector(record: InteractionRecord, selector: dict[str, Any]) -> bool:
    for key in (
        "interaction_id",
        "target_entity_id",
        "interaction_family",
        "interaction_subtype",
        "ligand_feature_id",
    ):
        if key in selector and selector[key] != getattr(record, key):
            return False
    if "alignment_key" in selector:
        return selector["alignment_key"] in {
            record.alignment_key(include_subtype=True),
            record.alignment_key(include_subtype=False),
        }
    return True


def _select_reference(
    complexes: list[FingerprintComplex],
    reference_ligand_id: str | None,
) -> FingerprintComplex:
    if reference_ligand_id is not None:
        for complex_item in complexes:
            if complex_item.ligand_id == reference_ligand_id:
                return complex_item
        raise ValueError(f"Reference ligand not found: {reference_ligand_id}")
    for complex_item in complexes:
        if complex_item.interaction_records:
            return complex_item
    return complexes[0]


def _protein_mode_available(
    complexes: list[FingerprintComplex],
    target_alignment,
) -> tuple[bool, str]:
    if target_alignment:
        return True, ""
    target_ids = {complex_item.target_id for complex_item in complexes}
    if len(target_ids) <= 1:
        return True, ""
    return (
        False,
        "Protein-anchored mode unavailable: complexes do not share one target_id "
        "and no target correspondence map was supplied",
    )


def _pairwise_similarity_matrix(
    complexes: list[FingerprintComplex],
    config: IFMConfig,
    *,
    protein_available: bool,
) -> dict[str, Any]:
    ligand_ids = [complex_item.ligand_id for complex_item in complexes]
    values: list[list[float]] = []
    for row_complex in complexes:
        row: list[float] = []
        for column_complex in complexes:
            if row_complex.ligand_id == column_complex.ligand_id:
                row.append(1.0 if row_complex.interaction_records else 0.0)
            elif protein_available:
                row.append(
                    _protein_anchored_match(
                        row_complex.interaction_records,
                        column_complex.interaction_records,
                        config,
                        strict=True,
                    ).similarity
                )
            else:
                row.append(
                    _ligand_anchored_match(
                        row_complex,
                        column_complex,
                        config,
                        strict=True,
                        ligand_mapping="auto",
                    ).similarity
                )
        values.append(row)
    return {"ligand_ids": ligand_ids, "values": values}


def _consensus_interactions(
    complexes: list[FingerprintComplex],
) -> list[dict[str, Any]]:
    counts: dict[str, set[str]] = defaultdict(set)
    exemplars: dict[str, InteractionRecord] = {}
    for complex_item in complexes:
        for record in complex_item.interaction_records:
            key = record.alignment_key(include_subtype=True)
            counts[key].add(complex_item.ligand_id)
            exemplars.setdefault(key, record)
    total = max(len(complexes), 1)
    rows = []
    for key, ligand_ids in counts.items():
        exemplar = exemplars[key]
        rows.append(
            {
                "alignment_key": key,
                "frequency": len(ligand_ids) / total,
                "ligand_count": len(ligand_ids),
                "interaction_family": exemplar.interaction_family,
                "interaction_subtype": exemplar.interaction_subtype,
                "target_entity_id": exemplar.target_entity_id,
            }
        )
    return sorted(rows, key=lambda row: (-row["frequency"], row["alignment_key"]))


def _per_target_summary(
    matched_entries: list[dict[str, Any]],
    missing_entries: list[dict[str, Any]],
    gained_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    def row(target_entity_id: str) -> dict[str, Any]:
        return summary.setdefault(
            target_entity_id,
            {
                "target_entity_id": target_entity_id,
                "matched": 0,
                "missing": 0,
                "gained": 0,
                "score": 0.0,
                "penalty": 0.0,
            },
        )

    for entry in matched_entries:
        record = entry["reference_interaction"]
        item = row(record["target_entity_id"])
        item["matched"] += 1
        item["score"] += entry["score"]
    for entry in missing_entries:
        record = entry["reference_interaction"]
        item = row(record["target_entity_id"])
        item["missing"] += 1
        item["penalty"] += entry["penalty"]
    for entry in gained_entries:
        record = entry["query_interaction"]
        item = row(record["target_entity_id"])
        item["gained"] += 1
        item["penalty"] += entry["penalty"]
    return sorted(summary.values(), key=lambda item: item["target_entity_id"])


def _per_ligand_feature_summary(
    matched_entries: list[dict[str, Any]],
    missing_entries: list[dict[str, Any]],
    gained_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    def row(feature_id: str | None) -> dict[str, Any]:
        key = feature_id or "unmapped"
        return summary.setdefault(
            key,
            {
                "ligand_feature_id": feature_id,
                "matched": 0,
                "missing": 0,
                "gained": 0,
                "score": 0.0,
                "penalty": 0.0,
            },
        )

    for entry in matched_entries:
        record = entry["reference_interaction"]
        item = row(record.get("ligand_feature_id"))
        item["matched"] += 1
        item["score"] += entry["score"]
    for entry in missing_entries:
        record = entry["reference_interaction"]
        item = row(record.get("ligand_feature_id"))
        item["missing"] += 1
        item["penalty"] += entry["penalty"]
    for entry in gained_entries:
        record = entry["query_interaction"]
        item = row(record.get("ligand_feature_id"))
        item["gained"] += 1
        item["penalty"] += entry["penalty"]
    return sorted(summary.values(), key=lambda item: str(item["ligand_feature_id"]))


def _medchem_summary(result: MatchResult) -> str:
    preserved = [
        _contact_label(entry["reference_interaction"])
        for entry in result.matched_interactions[:3]
    ]
    lost = [
        _contact_label(entry["reference_interaction"])
        for entry in result.missing_interactions[:3]
    ]
    gained = [
        _contact_label(entry["query_interaction"])
        for entry in result.gained_interactions[:3]
    ]
    clauses = []
    if preserved:
        clauses.append("preserves " + ", ".join(preserved))
    else:
        clauses.append("preserves no reference contacts")
    if lost:
        clauses.append("loses " + ", ".join(lost))
    if gained:
        clauses.append("gains " + ", ".join(gained))
    sentence = "; ".join(clauses)
    return sentence[:1].upper() + sentence[1:] + "."


def _contact_label(record: dict[str, Any]) -> str:
    family = str(record["interaction_family"]).replace("_", " ")
    target = record["target_entity_id"]
    return f"{family} at {target}"


def _ranking_score(result: MatchResult, mode: str) -> float:
    if mode == "protein":
        return result.protein_score
    if mode == "ligand":
        return result.ligand_score
    return result.hybrid_score if result.mode_used == "hybrid" else result.raw_alignment_score


def _empty_outcome() -> _AlignmentOutcome:
    return _AlignmentOutcome(
        raw_score=0.0,
        similarity=0.0,
        matched_weight=0.0,
        missing_weight=0.0,
        extra_weight=0.0,
        matched_interactions=[],
        missing_interactions=[],
        gained_interactions=[],
        per_target_summary=[],
        per_ligand_feature_summary=[],
        warnings=[],
    )


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _int_tuple(values: Iterable[Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _vec3_or_none(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        return None
    return values


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    return float(value)
