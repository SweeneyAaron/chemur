"""Expand grouped interactions into unique contact legs for pair-based scoring."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Iterable, TypeVar

from .models import AssignedInteraction

_Interaction = TypeVar("_Interaction", AssignedInteraction, dict[str, Any])


def _value(interaction, key: str, default=None):
    return interaction.get(key, default) if isinstance(interaction, dict) else getattr(interaction, key, default)


def project_interactions_for_scoring(interactions: Iterable[_Interaction]) -> list[_Interaction]:
    """Preserve ordinary records and score each enclosure center-shell pair once.

    Projection precedes ligand selection: shell atoms are not mutually interacting
    simply because they belong to one enclosure. An edge shared by two enclosed
    centers contributes once, with both original groups retained as provenance.
    Input records and their metadata are never mutated.
    """
    ordinary = []
    legs_by_pair: dict[tuple[int, int], list[tuple[Any, dict, dict]]] = {}
    for interaction in interactions:
        if _value(interaction, "interaction_type") != "hydrophobic_enclosure":
            ordinary.append(interaction)
            continue
        metadata = _value(interaction, "metadata", {}) or {}
        center_id = metadata.get("center_atom_id")
        legs = metadata.get("legs")
        if center_id is None or not legs:
            raise ValueError("Hydrophobic enclosure scoring requires center_atom_id and contact legs")
        for leg in legs:
            pair = tuple(sorted(int(atom_id) for atom_id in leg["atom_ids"]))
            if len(pair) != 2 or pair[0] == pair[1] or center_id not in pair:
                raise ValueError("Hydrophobic enclosure legs must join the center to one neighbor")
            legs_by_pair.setdefault(pair, []).append((interaction, metadata, leg))

    # Native assignment already replaces these pairs. Also handle imported
    # payloads that contain both the grouped contact and its ordinary legs.
    projected = [
        interaction for interaction in ordinary
        if not (_value(interaction, "interaction_type") == "hydrophobic"
                and tuple(sorted(_value(interaction, "atom_ids", ()))) in legs_by_pair)
    ]
    for pair, sources in sorted(legs_by_pair.items()):
        sources.sort(key=lambda source: (
            int(source[1]["center_atom_id"]),
            _value(source[0], "assignment_order", 0),
        ))
        exemplar, original_metadata, leg = sources[0]
        groups = []
        seen_centers = set()
        for source, source_metadata, _ in sources:
            center_id = int(source_metadata["center_atom_id"])
            if center_id in seen_centers:
                continue
            seen_centers.add(center_id)
            groups.append({
                **deepcopy(source_metadata),
                "assignment_order": _value(source, "assignment_order"),
            })
        metadata = {
            key: deepcopy(original_metadata[key])
            for key in ("strength_weight", "confidence", "ifm_priority")
            if key in original_metadata
        }
        metadata.update({
            "interaction_subtype": "hydrophobic_enclosure",
            "enclosure_leg_atom_ids": list(pair),
            "enclosure_center_atom_ids": sorted(seen_centers),
            "enclosure_groups": groups,
        })
        changes = {
            "atom_ids": pair,
            "feature_ids": tuple(leg["feature_ids"]),
            "component_ids": tuple(leg["component_ids"]),
            "distance": float(leg["distance"]),
            "angle": None,
            "offset": None,
            "resource_demands": (),
            "metadata": metadata,
        }
        if isinstance(exemplar, dict):
            changes = {key: list(value) if isinstance(value, tuple) else value
                       for key, value in changes.items()}
            projected.append({**exemplar, **changes})
        else:
            projected.append(replace(exemplar, **changes))
    return projected
