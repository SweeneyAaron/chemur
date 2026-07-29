from __future__ import annotations

from math import exp, sqrt

from .ifm_config import IFMConfig
from .ifm_models import InteractionRecord


def score_interaction_pair(
    reference: InteractionRecord,
    query: InteractionRecord,
    config: IFMConfig,
    *,
    strict: bool,
) -> float:
    if reference.interaction_family != query.interaction_family:
        return 0.0

    base_weight = interaction_weight(reference, config)
    if base_weight == 0.0:
        return 0.0

    return (
        base_weight
        * reference.strength_weight
        * query.strength_weight
        * geometry_similarity(reference, query, config)
        * role_similarity(reference, query)
        * target_similarity(reference, query, config, strict=strict)
        * subtype_similarity(reference, query, config)
        * confidence_factor(reference, query)
    )


def interaction_weight(record: InteractionRecord, config: IFMConfig) -> float:
    family_weight = config.family_weights.get(record.interaction_family, 1.0)
    priority_multiplier = config.priority_multipliers.get(record.priority, 1.0)
    return family_weight * priority_multiplier


def missing_penalty(record: InteractionRecord, config: IFMConfig) -> float:
    return interaction_weight(record, config) * config.penalties.missing_reference_scale


def extra_penalty(record: InteractionRecord, config: IFMConfig) -> float:
    if config.priority_multipliers.get(record.priority, 1.0) == 0.0:
        return 0.0
    return config.family_weights.get(record.interaction_family, 1.0) * (
        config.penalties.extra_query_scale
    )


def normalized_similarity(
    matched_weight: float,
    missing_weight: float,
    extra_weight: float,
    config: IFMConfig,
) -> float:
    if config.normalization_strategy != "penalized_jaccard":
        raise ValueError(
            "Unsupported normalization strategy: "
            f"{config.normalization_strategy!r}"
        )
    denominator = matched_weight + missing_weight + extra_weight
    if denominator <= 0.0:
        return 0.0
    return matched_weight / denominator


def geometry_similarity(
    reference: InteractionRecord,
    query: InteractionRecord,
    config: IFMConfig,
) -> float:
    score = 1.0
    ref_distance = reference.geometry.get("distance")
    qry_distance = query.geometry.get("distance")
    if ref_distance is not None and qry_distance is not None:
        score *= exp(-config.geometry.distance_alpha * abs(ref_distance - qry_distance))

    ref_angle = reference.geometry.get("angle")
    qry_angle = query.geometry.get("angle")
    if ref_angle is not None and qry_angle is not None:
        angle_error = abs(ref_angle - qry_angle)
        angle_error = min(angle_error, 360.0 - angle_error) / 180.0
        score *= exp(-config.geometry.angle_alpha * angle_error)

    ref_offset = reference.geometry.get("offset")
    qry_offset = query.geometry.get("offset")
    if ref_offset is not None and qry_offset is not None:
        score *= exp(-config.geometry.offset_alpha * abs(ref_offset - qry_offset))

    ref_planarity = reference.geometry.get("planarity")
    qry_planarity = query.geometry.get("planarity")
    if ref_planarity is not None and qry_planarity is not None:
        score *= exp(-config.geometry.planarity_alpha * abs(ref_planarity - qry_planarity))

    return score


def role_similarity(reference: InteractionRecord, query: InteractionRecord) -> float:
    ref_role = (
        bool(reference.directionality.get("ligand_is_donor")),
        bool(reference.directionality.get("target_is_donor")),
    )
    qry_role = (
        bool(query.directionality.get("ligand_is_donor")),
        bool(query.directionality.get("target_is_donor")),
    )
    if ref_role == qry_role:
        return 1.0
    if not any(ref_role) and not any(qry_role):
        return 1.0
    return 0.5


def target_similarity(
    reference: InteractionRecord,
    query: InteractionRecord,
    config: IFMConfig,
    *,
    strict: bool,
) -> float:
    if reference.target_entity_id == query.target_entity_id:
        return 1.0
    if strict and config.matching.strict_target_identity:
        return 0.0
    if reference.target_entity_kind != query.target_entity_kind:
        return 0.0
    if (
        config.matching.allow_moiety_partial_credit
        and reference.target_moiety == query.target_moiety
    ):
        return config.matching.moiety_partial_credit
    if (
        config.matching.allow_entity_class_partial_credit
        and reference.target_entity_class
        and reference.target_entity_class == query.target_entity_class
    ):
        return config.matching.entity_class_partial_credit
    if config.matching.allow_entity_kind_partial_credit:
        return config.matching.entity_kind_partial_credit
    return 0.0


def subtype_similarity(
    reference: InteractionRecord,
    query: InteractionRecord,
    config: IFMConfig,
) -> float:
    if reference.interaction_subtype == query.interaction_subtype:
        return 1.0
    if not reference.interaction_subtype or not query.interaction_subtype:
        return 1.0
    return config.matching.subtype_partial_credit


def confidence_factor(reference: InteractionRecord, query: InteractionRecord) -> float:
    return sqrt(max(reference.confidence, 0.0) * max(query.confidence, 0.0))
