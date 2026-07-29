"""Score docked ligand poses against a known 3D binder.

This is the user-facing entry point for the CASF-tuned CoSPLIF + IFM hybrid
scorer. It compares each docked pose's protein-ligand interaction pattern (and
3D pharmacophore blocks) to a reference binder and returns the poses ranked by
similarity.

High-level usage::

    import chemeleonx
    ranked = chemeleonx.score_poses(
        "receptor.pdb",
        ligand_sdf_dir="docked_poses",
        reference_sdf="crystal_ligand.sdf",
    )
    for row in ranked[:5]:
        print(row["rank"], row["ligand_id"], row["cosplif_score"])

The default profile is ``cosplif_pose_v1`` (hybrid_ifm, ligand_block_weight=0.40),
the best profile from the CASF-2016 block-weight sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .api import analyze, analyze_batch
from .cosplif import (
    CoSPLIFProfile,
    build_cosplif_payload,
    cosplif_profile,
    cosplif_profiles_from_config,
    score_cosplif_batch,
)
from .ifm_config import IFMConfig
from .ifm_matcher import (
    analysis_result_to_ifm_payloads,
    prepare_protein_ifm_scoring_batch,
    score_protein_ifm_batch,
)
from .models import AnalysisResult

DEFAULT_POSE_PROFILE_NAME = "cosplif_pose_v1"

__all__ = [
    "DEFAULT_POSE_PROFILE_NAME",
    "build_scoring_batch",
    "score_pose_batch",
    "score_poses",
]


def _resolve_profile(
    profile: CoSPLIFProfile | str | None,
    cosplif_config: dict[str, Any] | str | Path | None = None,
) -> CoSPLIFProfile:
    """Resolve a profile object from a CoSPLIFProfile, a name, or a config file."""
    if isinstance(profile, CoSPLIFProfile):
        return profile
    name = profile or DEFAULT_POSE_PROFILE_NAME
    if cosplif_config is not None:
        custom = cosplif_profiles_from_config(cosplif_config)
        if name in custom:
            return custom[name]
    return cosplif_profile(name)


def build_scoring_batch(results: dict[str, AnalysisResult]) -> dict[str, Any]:
    """Assemble a CoSPLIF + IFM scoring batch from analyzed poses.

    For each ``AnalysisResult`` this builds the compact IFM interaction-record
    payload and attaches the CoSPLIF context payload (tokens + ligand blocks).
    Each pose is given a unique ligand id (its key in ``results``) so it can be
    addressed individually by the scorers, while the CoSPLIF context is built
    from the pose's real ligand component id.
    """
    out: dict[str, Any] = {}
    for key, result in results.items():
        payloads = analysis_result_to_ifm_payloads(result)
        if not payloads:
            continue
        for index, payload in enumerate(payloads):
            component_id = payload.get("ligand_id")
            if not component_id:
                continue
            entry_key = key if len(payloads) == 1 else f"{key}:{index}"
            payload["cosplif"] = build_cosplif_payload(result, component_id)
            payload["ligand_id"] = entry_key
            payload["source_component_id"] = component_id
            out[entry_key] = payload
    return {"results": out}


def _apply_hybrid_ifm(
    cosplif_rows: list[dict[str, Any]],
    ifm_rows: list[dict[str, Any]],
    *,
    ifm_weight: float,
) -> None:
    """Blend the CoSPLIF score with the strict protein-IFM similarity in place."""
    ifm_by_ligand = {row["ligand_id"]: row for row in ifm_rows}
    for row in cosplif_rows:
        ifm = ifm_by_ligand.get(row["ligand_id"], {})
        strict = float(ifm.get("strict_similarity") or 0.0)
        base = float(row.get("cosplif_score") or 0.0)
        row["strict_ifm_similarity"] = strict
        row["cosplif_score"] = (1.0 - ifm_weight) * base + ifm_weight * strict


def score_pose_batch(
    batch_payload: dict[str, Any],
    reference_ligand_id: str,
    *,
    profile: CoSPLIFProfile | str | None = None,
    cosplif_config: dict[str, Any] | str | Path | None = None,
    ifm_config: IFMConfig | dict[str, Any] | str | Path | None = None,
    ifm_engine: str = "python",
    ifm_workers: int | str = "auto",
) -> list[dict[str, Any]]:
    """Score a pre-built scoring batch against one reference, ranked best first.

    ``batch_payload`` must be a ``{"results": {id: payload}}`` mapping as produced
    by :func:`build_scoring_batch` (each payload carries ``interaction_records``
    and a ``cosplif`` context). For ``hybrid_ifm`` profiles this also runs the
    protein-anchored IFM scorer and blends it in.
    """
    resolved = _resolve_profile(profile, cosplif_config)
    rows = score_cosplif_batch(
        batch_payload,
        reference_ligand_id=reference_ligand_id,
        profile=resolved,
    )
    if resolved.score_mode == "hybrid_ifm":
        ifm_batch = prepare_protein_ifm_scoring_batch(batch_payload, config=ifm_config)
        ifm_rows = score_protein_ifm_batch(
            ifm_batch,
            reference_ligand_id=reference_ligand_id,
            engine=ifm_engine,
            workers=ifm_workers,
        )
        _apply_hybrid_ifm(rows, ifm_rows, ifm_weight=resolved.hybrid_ifm_weight)
    rows.sort(key=lambda row: float(row.get("cosplif_score") or 0.0), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def score_poses(
    structure: str | Path,
    *,
    ligand_sdf: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    ligand_sdf_dir: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    reference_sdf: str | Path | None = None,
    reference_id: str | None = None,
    profile: CoSPLIFProfile | str = DEFAULT_POSE_PROFILE_NAME,
    cosplif_config: dict[str, Any] | str | Path | None = None,
    analysis_profile: str | Path | dict[str, Any] = "default",
    protonate: bool = False,
    protonate_ph_min: float = 7.4,
    protonate_ph_max: float = 7.4,
    scope: str = "all",
    threads: int | str = "auto",
    ifm_config: IFMConfig | dict[str, Any] | str | Path | None = None,
    ifm_engine: str = "python",
    ifm_workers: int | str = "auto",
    include_reference: bool = False,
) -> list[dict[str, Any]]:
    """Analyze docked poses against a receptor and rank them against a reference.

    Provide the docked poses via ``ligand_sdf``/``ligand_sdf_dir`` and the known
    binder via either ``reference_sdf`` (a separate SDF, takes precedence) or
    ``reference_id`` (the key of one of the supplied poses). Returns the poses
    ranked best-first; each row has ``rank``, ``ligand_id``, ``cosplif_score``,
    the CoSPLIF component scores, and ``strict_ifm_similarity`` for hybrid
    profiles. The reference itself is excluded unless ``include_reference=True``.

    ``profile`` is the CoSPLIF scoring profile (default ``cosplif_pose_v1``);
    ``analysis_profile`` is the ChemeleonX interaction-detection rule profile.
    """
    if ligand_sdf is None and ligand_sdf_dir is None:
        raise ValueError("score_poses requires ligand_sdf or ligand_sdf_dir")
    if reference_sdf is None and reference_id is None:
        raise ValueError("score_poses requires either reference_sdf or reference_id")

    analyze_opts: dict[str, Any] = dict(
        profile=analysis_profile,
        protonate=protonate,
        protonate_ph_min=protonate_ph_min,
        protonate_ph_max=protonate_ph_max,
        scope=scope,
        threads=threads,
    )

    results = analyze_batch(
        structure,
        ligand_sdf=ligand_sdf,
        ligand_sdf_dir=ligand_sdf_dir,
        **analyze_opts,
    )
    if not results:
        raise ValueError("score_poses found no docked poses to score")

    if reference_sdf is not None:
        reference_key = "__reference__"
        while reference_key in results:
            reference_key += "_"
        results = {
            **results,
            reference_key: analyze(structure, ligand_sdf=reference_sdf, **analyze_opts),
        }
    else:
        if reference_id not in results:
            raise ValueError(
                f"reference_id {reference_id!r} is not among the supplied poses: "
                f"{sorted(results)}"
            )
        reference_key = reference_id

    batch = build_scoring_batch(results)
    if reference_key not in batch["results"]:
        raise ValueError(
            f"reference {reference_key!r} produced no scorable interactions"
        )

    rows = score_pose_batch(
        batch,
        reference_ligand_id=reference_key,
        profile=profile,
        cosplif_config=cosplif_config,
        ifm_config=ifm_config,
        ifm_engine=ifm_engine,
        ifm_workers=ifm_workers,
    )
    for row in rows:
        row["is_reference"] = row.get("ligand_id") == reference_key
    if not include_reference:
        rows = [row for row in rows if not row["is_reference"]]
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
    return rows
