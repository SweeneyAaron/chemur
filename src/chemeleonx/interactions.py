from __future__ import annotations

from math import acos, degrees, sqrt
from typing import Iterable

from .core import angle as geom_angle
from .core import distance, is_occluded, neighbor_pairs, point_plane_offset
from .models import AtomRecord, CandidateInteraction, FeatureRecord, ResourceDemand


def generate_candidates(
    atoms: list[AtomRecord],
    features: list[FeatureRecord],
    profile: dict,
) -> list[CandidateInteraction]:
    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    features_by_type: dict[str, list[FeatureRecord]] = {}
    for feature in features:
        features_by_type.setdefault(feature.feature_type, []).append(feature)

    candidates: list[CandidateInteraction] = []
    candidates.extend(_hbond_candidates(atoms_by_id, features_by_type, profile))
    candidates.extend(_salt_bridge_candidates(features_by_type, profile))
    candidates.extend(_metal_candidates(features_by_type, profile))
    candidates.extend(_ring_ring_candidates(features_by_type, profile))
    candidates.extend(_charge_pi_candidates(features_by_type, profile, "cation", "cation_pi"))
    candidates.extend(_charge_pi_candidates(features_by_type, profile, "anion", "anion_pi"))
    candidates.extend(_point_pi_candidates(atoms_by_id, features_by_type, profile, "donor", "hbond_pi"))
    candidates.extend(_point_pi_candidates(atoms_by_id, features_by_type, profile, "donor", "ch_pi"))
    candidates.extend(_point_pi_candidates(atoms_by_id, features_by_type, profile, "halogen_donor", "halogen_pi"))
    candidates.extend(_halogen_bond_candidates(atoms_by_id, features_by_type, profile))
    candidates.extend(_hydrophobic_candidates(features_by_type, profile))
    candidates.extend(_solvent_bridge_candidates(candidates, atoms_by_id, profile))

    return _with_occlusion_rejections(candidates, atoms, profile)


def _hbond_candidates(atoms_by_id, features_by_type, profile) -> list[CandidateInteraction]:
    rule = profile["rules"]["hbond"]
    weak_rule = profile["rules"]["weak_hbond"]
    donors = features_by_type.get("donor", [])
    acceptors = features_by_type.get("acceptor", [])
    candidates: list[CandidateInteraction] = []
    for donor in donors:
        donor_atom_id = donor.atom_ids[0]
        donor_atom = atoms_by_id[donor_atom_id]
        for acceptor in acceptors:
            acceptor_atom_id = acceptor.atom_ids[0]
            if donor_atom_id == acceptor_atom_id:
                continue
            acceptor_atom = atoms_by_id[acceptor_atom_id]
            dist = distance(donor.center, acceptor.center)
            if donor_atom.element in {"N", "O", "S"} and acceptor_atom.element in {"N", "O", "S"}:
                if dist <= rule["distance"]:
                    candidate = _candidate(
                        "hbond",
                        donor,
                        acceptor,
                        dist,
                        rule,
                        (
                            ResourceDemand("donor", donor_atom_id),
                            ResourceDemand("acceptor", acceptor_atom_id),
                        ),
                    )
                    _set_donor_angle(candidate, donor_atom, acceptor.center, atoms_by_id, rule["angle"])
                    candidates.append(candidate)
            elif donor_atom.element == "C" and acceptor_atom.element in {"N", "O", "S"}:
                if dist <= weak_rule["distance"]:
                    candidate = _candidate(
                        "weak_hbond",
                        donor,
                        acceptor,
                        dist,
                        weak_rule,
                        (
                            ResourceDemand("donor", donor_atom_id),
                            ResourceDemand("acceptor", acceptor_atom_id),
                        ),
                    )
                    _set_donor_angle(candidate, donor_atom, acceptor.center, atoms_by_id, weak_rule["angle"])
                    candidates.append(candidate)
    return candidates


def _salt_bridge_candidates(features_by_type, profile) -> list[CandidateInteraction]:
    rule = profile["rules"]["salt_bridge"]
    candidates: list[CandidateInteraction] = []
    for cation in features_by_type.get("cation", []):
        for anion in features_by_type.get("anion", []):
            if cation.component_id == anion.component_id:
                continue
            dist = distance(cation.center, anion.center)
            if dist <= rule["distance"]:
                candidates.append(_candidate("salt_bridge", cation, anion, dist, rule))
    return candidates


def _metal_candidates(features_by_type, profile) -> list[CandidateInteraction]:
    rule = profile["rules"]["metal_coordination"]
    candidates: list[CandidateInteraction] = []
    for metal in features_by_type.get("metal", []):
        for acceptor in features_by_type.get("acceptor", []):
            if metal.component_id == acceptor.component_id:
                continue
            dist = distance(metal.center, acceptor.center)
            if dist <= rule["distance"]:
                candidates.append(
                    _candidate(
                        "metal_coordination",
                        metal,
                        acceptor,
                        dist,
                        rule,
                        (ResourceDemand("acceptor", acceptor.atom_ids[0]),),
                    )
                )
    return candidates


def _ring_ring_candidates(features_by_type, profile) -> list[CandidateInteraction]:
    rule = profile["rules"]["pipi_stack"]
    rings = features_by_type.get("ring", [])
    candidates: list[CandidateInteraction] = []
    for ring_a, ring_b, dist in _nearby_feature_pairs(rings, rule["distance"]):
        if ring_a.component_id == ring_b.component_id:
            continue
        angle = _normal_angle(ring_a.normal, ring_b.normal)
        offset = min(
            point_plane_offset(ring_b.center, ring_a.center, ring_a.normal or (0.0, 0.0, 1.0)),
            point_plane_offset(ring_a.center, ring_b.center, ring_b.normal or (0.0, 0.0, 1.0)),
        )
        if offset <= rule["offset"]:
            geometry = "parallel" if angle <= rule.get("angle", 45.0) else "t_shaped"
            candidates.append(
                _candidate(
                    "pipi_stack",
                    ring_a,
                    ring_b,
                    dist,
                    rule,
                    angle=angle,
                    offset=offset,
                    metadata={"geometry": geometry},
                )
            )
    return candidates


def _charge_pi_candidates(features_by_type, profile, charge_type: str, interaction_type: str):
    rule = profile["rules"][interaction_type]
    candidates: list[CandidateInteraction] = []
    for point in features_by_type.get(charge_type, []):
        for ring in features_by_type.get("ring", []):
            if point.component_id == ring.component_id:
                continue
            dist = distance(point.center, ring.center)
            if dist > rule["distance"]:
                continue
            angle = _point_ring_angle(point, ring)
            offset = point_plane_offset(point.center, ring.center, ring.normal or (0.0, 0.0, 1.0))
            if angle <= rule.get("angle", 90.0) and offset <= rule["offset"]:
                candidates.append(
                    _candidate(
                        interaction_type,
                        point,
                        ring,
                        dist,
                        rule,
                        angle=angle,
                        offset=offset,
                    )
                )
    return candidates


def _point_pi_candidates(atoms_by_id, features_by_type, profile, point_type: str, interaction_type: str):
    rule = profile["rules"][interaction_type]
    candidates: list[CandidateInteraction] = []
    for point in features_by_type.get(point_type, []):
        point_atom = atoms_by_id[point.atom_ids[0]]
        if interaction_type == "hbond_pi" and point_atom.element not in {"N", "O", "S"}:
            continue
        if interaction_type == "ch_pi" and point_atom.element != "C":
            continue
        for ring in features_by_type.get("ring", []):
            if point.component_id == ring.component_id:
                continue
            dist = distance(point.center, ring.center)
            if dist > rule["distance"]:
                continue
            point_plane_angle = _point_ring_angle(point, ring)
            offset = point_plane_offset(point.center, ring.center, ring.normal or (0.0, 0.0, 1.0))
            if point_plane_angle <= rule.get("angle", 90.0) and offset <= rule["offset"]:
                demands = ()
                if point_type == "donor":
                    demands = (ResourceDemand("donor", point.atom_ids[0]),)
                elif point_type == "halogen_donor":
                    demands = (ResourceDemand("halogen_donor", point.atom_ids[0]),)

                candidate = _candidate(
                    interaction_type,
                    point,
                    ring,
                    dist,
                    rule,
                    demands,
                    offset=offset,
                    metadata={"point_plane_angle": point_plane_angle},
                )
                if point_type == "donor":
                    min_angle = float(rule.get("donor_angle", 90.0))
                    _set_donor_angle(candidate, point_atom, ring.center, atoms_by_id, min_angle)
                elif point_type == "halogen_donor":
                    candidate.angle = point_plane_angle
                candidates.append(candidate)
    return candidates


def _halogen_bond_candidates(atoms_by_id, features_by_type, profile):
    rule = profile["rules"]["halogen_bond"]
    candidates: list[CandidateInteraction] = []
    for halogen in features_by_type.get("halogen_donor", []):
        for acceptor in features_by_type.get("acceptor", []):
            if halogen.component_id == acceptor.component_id:
                continue
            dist = distance(halogen.center, acceptor.center)
            if dist <= rule["distance"]:
                candidate = _candidate(
                    "halogen_bond",
                    halogen,
                    acceptor,
                    dist,
                    rule,
                    (
                        ResourceDemand("halogen_donor", halogen.atom_ids[0]),
                        ResourceDemand("acceptor", acceptor.atom_ids[0]),
                    ),
                )
                _set_halogen_angle(candidate, atoms_by_id[halogen.atom_ids[0]], acceptor.center, atoms_by_id, rule["angle"])
                candidates.append(candidate)
    return candidates


def _hydrophobic_candidates(features_by_type, profile):
    rule = profile["rules"]["hydrophobic"]
    hydrophobes = features_by_type.get("hydrophobe", [])
    candidates: list[CandidateInteraction] = []
    for feature_a, feature_b, dist in _nearby_feature_pairs(hydrophobes, rule["distance"]):
        if feature_a.component_id == feature_b.component_id:
            continue
        candidates.append(_candidate("hydrophobic", feature_a, feature_b, dist, rule))
    return candidates


def _solvent_bridge_candidates(candidates, atoms_by_id, profile):
    rule = profile["rules"]["solvent_bridge"]
    solvent_legs = [
        candidate
        for candidate in candidates
        if candidate.interaction_type == "hbond"
        and any(atoms_by_id[atom_id].molecule_type == "solvent" for atom_id in candidate.atom_ids)
    ]
    bridges: list[CandidateInteraction] = []
    for i, leg_a in enumerate(solvent_legs):
        solvent_components_a = {
            atoms_by_id[atom_id].component_id
            for atom_id in leg_a.atom_ids
            if atoms_by_id[atom_id].molecule_type == "solvent"
        }
        for leg_b in solvent_legs[i + 1 :]:
            solvent_components_b = {
                atoms_by_id[atom_id].component_id
                for atom_id in leg_b.atom_ids
                if atoms_by_id[atom_id].molecule_type == "solvent"
            }
            shared_solvent = solvent_components_a & solvent_components_b
            if not shared_solvent:
                continue
            non_solvent_components = tuple(
                sorted(
                    component_id
                    for component_id in set(leg_a.component_ids + leg_b.component_ids)
                    if component_id not in shared_solvent
                )
            )
            if len(non_solvent_components) < 2:
                continue
            bridges.append(
                CandidateInteraction(
                    interaction_type="solvent_bridge",
                    feature_ids=tuple(sorted(set(leg_a.feature_ids + leg_b.feature_ids))),
                    atom_ids=tuple(sorted(set(leg_a.atom_ids + leg_b.atom_ids))),
                    component_ids=tuple(sorted(set(leg_a.component_ids + leg_b.component_ids))),
                    distance=leg_a.distance + leg_b.distance,
                    priority=rule["priority"],
                    stage="inter",
                    resource_demands=tuple(leg_a.resource_demands + leg_b.resource_demands),
                    metadata={"legs": [leg_a.metadata, leg_b.metadata]},
                )
            )
    return bridges


def _candidate(
    interaction_type: str,
    feature_a: FeatureRecord,
    feature_b: FeatureRecord,
    dist: float,
    rule: dict,
    resource_demands: tuple[ResourceDemand, ...] = (),
    angle: float | None = None,
    offset: float | None = None,
    metadata: dict | None = None,
) -> CandidateInteraction:
    component_ids = (feature_a.component_id, feature_b.component_id)
    stage = "intra" if feature_a.component_id == feature_b.component_id else "inter"
    return CandidateInteraction(
        interaction_type=interaction_type,
        feature_ids=(feature_a.feature_id, feature_b.feature_id),
        atom_ids=tuple(sorted(set(feature_a.atom_ids + feature_b.atom_ids))),
        component_ids=component_ids,
        distance=dist,
        priority=rule["priority"],
        stage=stage,
        angle=angle,
        offset=offset,
        resource_demands=resource_demands,
        metadata=metadata or {},
    )


def _set_donor_angle(
    candidate: CandidateInteraction,
    donor_atom: AtomRecord,
    target_center: tuple[float, float, float],
    atoms_by_id: dict[int, AtomRecord],
    min_angle: float,
) -> None:
    hydrogen_atoms = [
        atoms_by_id[neighbor_id]
        for neighbor_id in donor_atom.bonds
        if neighbor_id in atoms_by_id and atoms_by_id[neighbor_id].element == "H"
    ]
    if not hydrogen_atoms:
        candidate.rejection_reason = "missing_donor_hydrogen_geometry"
        return

    best_hydrogen = max(
        hydrogen_atoms,
        key=lambda hydrogen: geom_angle(donor_atom.coord, hydrogen.coord, target_center),
    )
    best_angle = geom_angle(donor_atom.coord, best_hydrogen.coord, target_center)
    candidate.angle = best_angle
    candidate.metadata["hydrogen_atom_id"] = best_hydrogen.atom_id
    if best_angle < min_angle:
        candidate.rejection_reason = f"angle_below_cutoff:{round(best_angle, 3)}<{min_angle}"


def _set_halogen_angle(
    candidate: CandidateInteraction,
    halogen_atom: AtomRecord,
    target_center: tuple[float, float, float],
    atoms_by_id: dict[int, AtomRecord],
    min_angle: float,
) -> None:
    root_atoms = [
        atoms_by_id[neighbor_id]
        for neighbor_id in halogen_atom.bonds
        if neighbor_id in atoms_by_id and atoms_by_id[neighbor_id].element != "H"
    ]
    if not root_atoms:
        candidate.rejection_reason = "missing_halogen_root_geometry"
        return
    root_atom = root_atoms[0]
    bond_angle = geom_angle(root_atom.coord, halogen_atom.coord, target_center)
    candidate.angle = bond_angle
    candidate.metadata["halogen_root_atom_id"] = root_atom.atom_id
    if bond_angle < min_angle:
        candidate.rejection_reason = f"angle_below_cutoff:{round(bond_angle, 3)}<{min_angle}"


def _with_occlusion_rejections(
    candidates: list[CandidateInteraction],
    atoms: list[AtomRecord],
    profile: dict,
) -> list[CandidateInteraction]:
    atom_ids = [atom.atom_id for atom in atoms]
    coords = [atom.coord for atom in atoms]
    coord_index = {atom_id: idx for idx, atom_id in enumerate(atom_ids)}
    ignored_indices = {
        idx for idx, atom in enumerate(atoms) if atom.metadata.get("ignored_by_template")
    }
    radius = float(profile.get("occlusion_radius", 1.0))
    for candidate in candidates:
        if candidate.rejection_reason:
            continue
        if len(candidate.atom_ids) != 2:
            continue
        start_id, end_id = candidate.atom_ids[0], candidate.atom_ids[-1]
        if start_id not in coord_index or end_id not in coord_index:
            continue
        ignored_atom_ids = set(candidate.atom_ids)
        for metadata_key in ("hydrogen_atom_id", "halogen_root_atom_id"):
            if metadata_key in candidate.metadata:
                ignored_atom_ids.add(candidate.metadata[metadata_key])
        if is_occluded(
            coords[coord_index[start_id]],
            coords[coord_index[end_id]],
            coords,
            {
                coord_index[atom_id]
                for atom_id in ignored_atom_ids
                if atom_id in coord_index
            }
            | ignored_indices,
            radius,
        ):
            candidate.rejection_reason = "occluded"
    return candidates


def _nearby_feature_pairs(features: list[FeatureRecord], cutoff: float):
    coords = [feature.center for feature in features]
    for i, j, dist in neighbor_pairs(coords, cutoff):
        yield features[i], features[j], dist


def _normal_angle(normal_a, normal_b) -> float:
    if normal_a is None or normal_b is None:
        return 0.0
    dot = abs(sum(normal_a[i] * normal_b[i] for i in range(3)))
    norm_a = sqrt(sum(v * v for v in normal_a))
    norm_b = sqrt(sum(v * v for v in normal_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    value = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return degrees(acos(value))


def _point_ring_angle(point: FeatureRecord, ring: FeatureRecord) -> float:
    if ring.normal is None:
        return 0.0
    vector = tuple(point.center[i] - ring.center[i] for i in range(3))
    norm_v = sqrt(sum(v * v for v in vector))
    norm_n = sqrt(sum(v * v for v in ring.normal))
    if norm_v == 0.0 or norm_n == 0.0:
        return 0.0
    value = abs(sum(vector[i] * ring.normal[i] for i in range(3))) / (norm_v * norm_n)
    value = max(-1.0, min(1.0, value))
    return degrees(acos(value))
