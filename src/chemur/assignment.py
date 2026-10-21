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

    ordinary = [candidate for candidate in candidates if candidate.interaction_type != "solvent_bridge"]
    bridges = [candidate for candidate in candidates if candidate.interaction_type == "solvent_bridge"]
    for candidate in sorted(ordinary, key=_assignment_key):
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

    assigned = _replace_enclosing_contacts(assigned, candidates)

    hydrogen_bonds = {
        interaction.feature_ids: interaction
        for interaction in assigned if interaction.interaction_type == "hbond"
    }
    for candidate in sorted(bridges, key=_assignment_key):
        if candidate.rejection_reason:
            continue
        leg_keys = _bridge_leg_keys(candidate)
        if leg_keys is None:
            candidate.rejection_reason = "missing_hbond_leg_dependencies"
            continue
        if any(key not in hydrogen_bonds for key in leg_keys):
            candidate.rejection_reason = "unassigned_hbond_leg"
            continue
        # These resources have already been charged through the assigned legs.
        # A bridge never competes with its own legs, regardless of its priority.
        candidate.resource_demands = ()
        candidate.metadata["leg_assignment_orders"] = [
            hydrogen_bonds[key].assignment_order for key in leg_keys
        ]
        assigned.append(AssignedInteraction.from_candidate(candidate, len(assigned) + 1))

    return assigned


def _replace_enclosing_contacts(
    assigned: list[AssignedInteraction], candidates: list[CandidateInteraction],
) -> list[AssignedInteraction]:
    """Only accepted enclosures replace their own centre-to-neighbour contacts."""
    enclosing_centers: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for interaction in assigned:
        if interaction.interaction_type == "hydrophobic_enclosure":
            for leg in interaction.metadata["legs"]:
                enclosing_centers[tuple(sorted(leg["atom_ids"]))].add(
                    interaction.metadata["center_atom_id"]
                )
    if not enclosing_centers:
        return assigned
    for candidate in candidates:
        centers = enclosing_centers.get(tuple(sorted(candidate.atom_ids)))
        if candidate.interaction_type == "hydrophobic" and not candidate.rejection_reason and centers:
            candidate.rejection_reason = "subsumed_by_hydrophobic_enclosure"
            candidate.metadata["enclosure_center_atom_ids"] = sorted(centers)
    retained = [
        interaction for interaction in assigned
        if interaction.interaction_type != "hydrophobic"
        or tuple(sorted(interaction.atom_ids)) not in enclosing_centers
    ]
    for order, interaction in enumerate(retained, start=1):
        interaction.assignment_order = order
    return retained


def _bridge_leg_keys(candidate: CandidateInteraction) -> tuple[tuple[int, int], ...] | None:
    keys = candidate.metadata.get("leg_feature_ids")
    if not isinstance(keys, (list, tuple)) or len(keys) != 2:
        return None
    if any(
        not isinstance(key, (list, tuple))
        or len(key) != 2
        or not all(isinstance(value, int) for value in key)
        for key in keys
    ):
        return None
    directed_keys = tuple(tuple(key) for key in keys)
    return directed_keys if directed_keys[0] != directed_keys[1] else None


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
        candidate.metadata.get("center_atom_id", -1),
    )
