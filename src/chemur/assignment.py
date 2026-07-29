from __future__ import annotations

from collections import defaultdict

from .models import AssignedInteraction, AtomRecord, CandidateInteraction, ResourceDemand


def assign_interactions(
    atoms: list[AtomRecord],
    candidates: list[CandidateInteraction],
    profile: dict,
) -> list[AssignedInteraction]:
    capacities = _initial_capacities(atoms)
    assigned: list[AssignedInteraction] = []

    for candidate in sorted(candidates, key=_assignment_key):
        if candidate.rejection_reason:
            continue
        failed = _first_failed_demand(capacities, candidate.resource_demands)
        if failed is not None:
            candidate.rejection_reason = (
                f"resource_exhausted:{failed.resource_type}:{failed.atom_id}"
            )
            continue
        _consume(capacities, candidate.resource_demands)
        assigned.append(AssignedInteraction.from_candidate(candidate, len(assigned) + 1))

    return assigned


def _initial_capacities(atoms: list[AtomRecord]) -> dict[tuple[str, int], int]:
    capacities: dict[tuple[str, int], int] = defaultdict(int)
    for atom in atoms:
        capacities[("donor", atom.atom_id)] += atom.donor_capacity
        capacities[("acceptor", atom.atom_id)] += atom.acceptor_capacity
        capacities[("halogen_donor", atom.atom_id)] += atom.halogen_donor_capacity
        capacities[("chalcogen_donor", atom.atom_id)] += atom.chalcogen_donor_capacity
        capacities[("tetrel_donor", atom.atom_id)] += atom.tetrel_donor_capacity
        capacities[("carbon_donor", atom.atom_id)] += atom.carbon_donor_capacity
    return capacities


def _first_failed_demand(
    capacities: dict[tuple[str, int], int],
    demands: tuple[ResourceDemand, ...],
) -> ResourceDemand | None:
    requested: dict[tuple[str, int], int] = defaultdict(int)
    for demand in demands:
        key = (demand.resource_type, demand.atom_id)
        requested[key] += demand.amount
        if capacities.get(key, 0) < requested[key]:
            return demand
    return None


def _consume(
    capacities: dict[tuple[str, int], int],
    demands: tuple[ResourceDemand, ...],
) -> None:
    for demand in demands:
        key = (demand.resource_type, demand.atom_id)
        capacities[key] -= demand.amount


def _assignment_key(candidate: CandidateInteraction):
    stage_order = 0 if candidate.stage == "intra" else 1
    return (
        stage_order,
        candidate.priority,
        round(candidate.distance, 6),
        candidate.interaction_type,
        candidate.component_ids,
        candidate.atom_ids,
        candidate.feature_ids,
    )

