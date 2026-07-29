from __future__ import annotations

import json
from pathlib import Path

from .ifm_models import BatchMatchResult, MatchResult


def batch_to_json(result: BatchMatchResult, path: str | Path | None = None) -> str:
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if path is not None:
        Path(path).write_text(payload + "\n", encoding="utf-8")
    return payload


def render_batch_report(result: BatchMatchResult) -> str:
    lines = [f"Reference ligand: {result.reference_ligand_id}", ""]
    lines.append("Rank  Ligand  Score    Strict   Relaxed  Summary")
    for ranked in result.ranked_ligands:
        match = _result_by_ligand(result, ranked["ligand_id"])
        lines.append(
            f"{ranked['rank']:>4}  "
            f"{ranked['ligand_id']:<7} "
            f"{ranked['score']:>7.3f}  "
            f"{ranked['strict_similarity']:>7.3f}  "
            f"{ranked['relaxed_similarity']:>7.3f}  "
            f"{match.medicinal_chemistry_summary if match else ''}"
        )
    warning_lines = []
    for match in result.results:
        for warning in match.warnings:
            warning_lines.append(f"{match.ligand_id}: {warning}")
    if warning_lines:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in sorted(set(warning_lines)))
    return "\n".join(lines) + "\n"


def _result_by_ligand(
    batch_result: BatchMatchResult,
    ligand_id: str,
) -> MatchResult | None:
    for result in batch_result.results:
        if result.ligand_id == ligand_id:
            return result
    return None
