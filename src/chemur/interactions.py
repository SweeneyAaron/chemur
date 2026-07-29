from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, degrees, sqrt
from typing import Callable, Iterable

from .core import angle as geom_angle
from .core import distance, is_occluded, neighbor_pairs, point_plane_offset
from .interaction_types import INTERACTION_TYPES
from .models import AtomRecord, CandidateInteraction, FeatureRecord, ResourceDemand

Vec3 = tuple[float, float, float]


@dataclass(slots=True)
class DetectorContext:
    """Everything a detector needs, assembled once per structure."""

    atoms_by_id: dict[int, AtomRecord]
    features_by_type: dict[str, list[FeatureRecord]]
    profile: dict
    candidates: list[CandidateInteraction] = field(default_factory=list)

    def features(self, feature_type: str) -> list[FeatureRecord]:
        return self.features_by_type.get(feature_type, [])

    def rings(self, rule: dict) -> list[FeatureRecord]:
        """Ring features, filtered to aromatic ones when the rule demands it.

        Ligand rings come from RDKit's full SSSR, so a saturated ring is a ring
        feature too. Rules describing genuine pi interactions set
        ``require_aromatic`` so a cyclohexyl cannot stand in for a phenyl.
        """
        rings = self.features("ring")
        if not rule.get("require_aromatic", False):
            return rings
        return [ring for ring in rings if ring.metadata.get("aromatic", True)]

    def aliphatic_rings(self) -> list[FeatureRecord]:
        return [
            ring
            for ring in self.features("ring")
            if not ring.metadata.get("aromatic", True)
        ]


def generate_candidates(
    atoms: list[AtomRecord],
    features: list[FeatureRecord],
    profile: dict,
) -> list[CandidateInteraction]:
    features_by_type: dict[str, list[FeatureRecord]] = {}
    for feature in features:
        features_by_type.setdefault(feature.feature_type, []).append(feature)

    context = DetectorContext(
        atoms_by_id={atom.atom_id: atom for atom in atoms},
        features_by_type=features_by_type,
        profile=profile,
    )

    rules = profile.get("rules", {})
    # Two passes: second-order rules (solvent bridges) are built from the
    # first-order candidates, so they cannot run until those exist.
    for second_order in (False, True):
        for interaction in INTERACTION_TYPES:
            if _is_second_order(interaction.name) != second_order:
                continue
            rule = rules.get(interaction.name)
            if rule is None or not rule.get("enabled", True):
                continue
            detector = _DETECTORS.get(interaction.name)
            if detector is None:
                continue
            context.candidates.extend(detector(context, rule))

    return _with_occlusion_rejections(context.candidates, atoms, features, profile)


def _is_second_order(name: str) -> bool:
    return name in _SECOND_ORDER_RULES


_SECOND_ORDER_RULES = {"solvent_bridge"}


# --------------------------------------------------------------------------
# Hydrogen bonds
# --------------------------------------------------------------------------


def _hbond_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    candidates: list[CandidateInteraction] = []
    sulfur_distance = float(rule.get("sulfur_distance", rule["distance"]))
    short_distance = float(rule.get("short_distance", 0.0))

    for donor in context.features("donor"):
        donor_atom = context.atoms_by_id[donor.atom_ids[0]]
        if donor_atom.element not in {"N", "O", "S", "Se"}:
            continue
        for acceptor in context.features("acceptor"):
            acceptor_atom_id = acceptor.atom_ids[0]
            if donor.atom_ids[0] == acceptor_atom_id:
                continue
            acceptor_atom = context.atoms_by_id[acceptor_atom_id]
            if acceptor_atom.element not in {"N", "O", "S", "Se"}:
                continue

            # Sulfur H-bonds are longer than the N/O case, so the cutoff follows
            # the heavier partner rather than truncating them at the N/O distance.
            cutoff = (
                sulfur_distance
                if "S" in {donor_atom.element, acceptor_atom.element}
                or "Se" in {donor_atom.element, acceptor_atom.element}
                else float(rule["distance"])
            )
            dist = distance(donor.center, acceptor.center)
            if dist > cutoff:
                continue

            candidate = _candidate(
                "hbond",
                donor,
                acceptor,
                dist,
                rule,
                (
                    ResourceDemand("donor", donor.atom_ids[0]),
                    ResourceDemand("acceptor", acceptor_atom_id),
                ),
            )
            if short_distance and dist < short_distance:
                # A short, near-symmetric H-bond is the low-barrier regime. Recorded
                # as a property of the bond rather than a separate interaction type,
                # because the geometry is identical and only the length differs.
                candidate.metadata["short_strong"] = True
                candidate.metadata["interaction_subtype"] = "low_barrier"
            _set_donor_angle(
                candidate, donor_atom, acceptor.center, context.atoms_by_id, rule["angle"]
            )
            candidates.append(candidate)
    return candidates


def _weak_hbond_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    candidates: list[CandidateInteraction] = []
    for donor in context.features("carbon_donor"):
        donor_atom = context.atoms_by_id[donor.atom_ids[0]]
        for acceptor in context.features("acceptor"):
            acceptor_atom_id = acceptor.atom_ids[0]
            if donor.atom_ids[0] == acceptor_atom_id:
                continue
            if context.atoms_by_id[acceptor_atom_id].element not in {"N", "O", "S", "Se"}:
                continue
            dist = distance(donor.center, acceptor.center)
            if dist > rule["distance"]:
                continue
            candidate = _candidate(
                "weak_hbond",
                donor,
                acceptor,
                dist,
                rule,
                (
                    ResourceDemand("carbon_donor", donor.atom_ids[0]),
                    ResourceDemand("acceptor", acceptor_atom_id),
                ),
            )
            _set_donor_angle(
                candidate, donor_atom, acceptor.center, context.atoms_by_id, rule["angle"]
            )
            candidates.append(candidate)
    return candidates


def _amide_bridge_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    """Reciprocal amide-amide pair: N-H...O=C running in both directions.

    This is the beta-sheet-like double hydrogen bond, and it is a single event
    rather than two independent bonds -- hence its own type, and a priority above
    ``hbond`` so it claims both amides before they are consumed one leg at a time.
    """
    amides = context.features("amide")
    cutoff = float(rule["distance"])
    candidates: list[CandidateInteraction] = []

    separation = int(rule.get("min_residue_separation", 2))
    for index, first in enumerate(amides):
        for second in amides[index + 1 :]:
            if first.component_id == second.component_id:
                continue
            if not _amides_are_far_enough_apart(first, second, context, separation):
                continue
            forward = distance(
                context.atoms_by_id[first.metadata["nitrogen_atom_id"]].coord,
                context.atoms_by_id[second.metadata["oxygen_atom_id"]].coord,
            )
            reverse = distance(
                context.atoms_by_id[second.metadata["nitrogen_atom_id"]].coord,
                context.atoms_by_id[first.metadata["oxygen_atom_id"]].coord,
            )
            if forward > cutoff or reverse > cutoff:
                continue

            candidate = _candidate(
                "amide_bridge",
                first,
                second,
                (forward + reverse) / 2.0,
                rule,
                (
                    ResourceDemand("donor", first.metadata["nitrogen_atom_id"]),
                    ResourceDemand("acceptor", second.metadata["oxygen_atom_id"]),
                    ResourceDemand("donor", second.metadata["nitrogen_atom_id"]),
                    ResourceDemand("acceptor", first.metadata["oxygen_atom_id"]),
                ),
                metadata={"forward_distance": forward, "reverse_distance": reverse},
            )
            _set_reciprocal_amide_angles(candidate, first, second, context, rule["angle"])
            candidates.append(candidate)
    return candidates


def _amides_are_far_enough_apart(
    first: FeatureRecord,
    second: FeatureRecord,
    context: DetectorContext,
    separation: int,
) -> bool:
    """Reject amide pairs that are merely consecutive along the same chain.

    A backbone amide feature spans two residues -- C(=O) of residue i with N of
    residue i+1 -- so consecutive peptide units *overlap*, and every one of them sits
    at a short N...O distance simply because that is how a polypeptide is built.
    Comparing ``component_id`` alone does not exclude them (adjacent residues have
    different component ids), which made every candidate a chain-local artefact:
    in 1NTP the closest was Asp71/Asn72 against Asn72/Ile73, at a perfect 2.78 A but
    with the N-H and C=O nearly perpendicular. A real bridge is between amides that
    are not sequence neighbours.
    """
    if set(first.atom_ids) & set(second.atom_ids):
        return False
    return _residue_separation_ok(
        context.atoms_by_id[first.metadata["carbon_atom_id"]],
        context.atoms_by_id[second.metadata["carbon_atom_id"]],
        separation,
    )


def _residue_separation_ok(first: AtomRecord, second: AtomRecord, separation: int) -> bool:
    """Whether two atoms are at least ``separation`` residues apart in sequence.

    Atoms on different chains always pass -- they cannot be sequence neighbours. Used
    to keep covalently-constrained arrangements out of interactions that are supposed
    to be non-covalent contacts between distinct groups.
    """
    if first.chain_id != second.chain_id:
        return True
    numbers = (_residue_number(first.residue_id), _residue_number(second.residue_id))
    if any(number is None for number in numbers):
        return True
    return abs(numbers[0] - numbers[1]) >= separation


def _residue_number(residue_id: str) -> int | None:
    digits = ""
    for character in residue_id:
        if character.isdigit() or (character == "-" and not digits):
            digits += character
        else:
            break
    try:
        return int(digits)
    except ValueError:
        return None


def _set_reciprocal_amide_angles(
    candidate: CandidateInteraction,
    first: FeatureRecord,
    second: FeatureRecord,
    context: DetectorContext,
    min_angle: float,
) -> None:
    angles = []
    for donor_feature, acceptor_feature in ((first, second), (second, first)):
        donor_atom = context.atoms_by_id[donor_feature.metadata["nitrogen_atom_id"]]
        acceptor_atom = context.atoms_by_id[acceptor_feature.metadata["oxygen_atom_id"]]
        leg = _best_donor_angle(donor_atom, acceptor_atom.coord, context.atoms_by_id)
        if leg is None:
            candidate.rejection_reason = "missing_donor_hydrogen_geometry"
            return
        angles.append(leg)

    candidate.angle = min(value for value, _ in angles)
    candidate.metadata["hydrogen_atom_ids"] = [atom_id for _, atom_id in angles]
    if candidate.angle < min_angle:
        candidate.rejection_reason = f"angle_below_cutoff:{round(candidate.angle, 3)}<{min_angle}"


def _solvent_bridge_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    atoms_by_id = context.atoms_by_id
    solvent_legs = [
        candidate
        for candidate in context.candidates
        # Rejected legs were previously still eligible to form a bridge, so a
        # bridge could be built entirely from H-bonds that were themselves thrown out.
        if candidate.interaction_type == "hbond"
        and not candidate.rejection_reason
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


# --------------------------------------------------------------------------
# Ionic and metal
# --------------------------------------------------------------------------


def _salt_bridge_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    candidates: list[CandidateInteraction] = []
    for cation in context.features("cation"):
        for anion in context.features("anion"):
            if cation.component_id == anion.component_id:
                continue
            dist = distance(cation.center, anion.center)
            if dist <= rule["distance"]:
                candidates.append(_candidate("salt_bridge", cation, anion, dist, rule))
    return candidates


def _metal_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    candidates: list[CandidateInteraction] = []
    for metal in context.features("metal"):
        for acceptor in context.features("acceptor"):
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


def _hydrophobic_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    hydrophobes = context.features("hydrophobe")
    candidates: list[CandidateInteraction] = []
    for feature_a, feature_b, dist in _nearby_feature_pairs(hydrophobes, rule["distance"]):
        if feature_a.component_id == feature_b.component_id:
            continue
        candidates.append(_candidate("hydrophobic", feature_a, feature_b, dist, rule))
    return candidates


# --------------------------------------------------------------------------
# Ring stacking
# --------------------------------------------------------------------------


def _ring_stack_candidates(
    interaction_type: str,
    rule: dict,
    pairs: Iterable[tuple[FeatureRecord, FeatureRecord]],
    *,
    gate_angle: bool,
) -> list[CandidateInteraction]:
    """Shared face-to-face stacking geometry.

    ``gate_angle`` distinguishes the two conventions in play: ``pipi_stack`` has
    always used its angle as a parallel/T-shaped *label* and accepted both, while
    the aliphatic stacks genuinely require near-parallel faces -- a saturated ring
    has no quadrupole to stabilise an edge-on approach.
    """
    candidates: list[CandidateInteraction] = []
    for ring_a, ring_b in pairs:
        if ring_a.component_id == ring_b.component_id:
            continue
        dist = distance(ring_a.center, ring_b.center)
        if dist > rule["distance"]:
            continue
        angle = _normal_angle(ring_a.normal, ring_b.normal)
        offset = min(
            point_plane_offset(ring_b.center, ring_a.center, ring_a.normal or (0.0, 0.0, 1.0)),
            point_plane_offset(ring_a.center, ring_b.center, ring_b.normal or (0.0, 0.0, 1.0)),
        )
        if offset > rule["offset"]:
            continue
        if gate_angle and angle > rule["angle"]:
            continue
        geometry = "parallel" if angle <= rule.get("angle", 45.0) else "t_shaped"
        candidates.append(
            _candidate(
                interaction_type,
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


def _pipi_stack_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    rings = context.rings(rule)
    return _ring_stack_candidates(
        "pipi_stack",
        rule,
        ((rings[i], rings[j]) for i, j, _ in neighbor_pairs([r.center for r in rings], rule["distance"])),
        gate_angle=False,
    )


def _aliphatic_pi_stack_candidates(
    context: DetectorContext, rule: dict
) -> list[CandidateInteraction]:
    aromatic = [
        ring for ring in context.features("ring") if ring.metadata.get("aromatic", True)
    ]
    aliphatic = context.aliphatic_rings()
    return _ring_stack_candidates(
        "aliphatic_pi_stack",
        rule,
        ((saturated, ring) for saturated in aliphatic for ring in aromatic),
        gate_angle=True,
    )


def _aliphatic_stack_candidates(
    context: DetectorContext, rule: dict
) -> list[CandidateInteraction]:
    aliphatic = context.aliphatic_rings()
    return _ring_stack_candidates(
        "aliphatic_stack",
        rule,
        (
            (aliphatic[i], aliphatic[j])
            for i in range(len(aliphatic))
            for j in range(i + 1, len(aliphatic))
        ),
        gate_angle=True,
    )


def _amide_pi_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    """Amide plane stacked on an aromatic face."""
    return _ring_stack_candidates(
        "amide_pi",
        rule,
        (
            (amide, ring)
            for amide in context.features("amide")
            for ring in context.rings(rule)
        ),
        gate_angle=True,
    )


# --------------------------------------------------------------------------
# Point-to-ring interactions
# --------------------------------------------------------------------------


def _charge_pi_candidates(
    context: DetectorContext,
    rule: dict,
    charge_type: str,
    interaction_type: str,
) -> list[CandidateInteraction]:
    candidates: list[CandidateInteraction] = []
    for point in context.features(charge_type):
        for ring in context.rings(rule):
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


def _cation_pi_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    return _charge_pi_candidates(context, rule, "cation", "cation_pi")


def _anion_pi_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    return _charge_pi_candidates(context, rule, "anion", "anion_pi")


def _anion_aromatic_edge_candidates(
    context: DetectorContext, rule: dict
) -> list[CandidateInteraction]:
    """Anion approaching the ring edge rather than its face.

    The edgewise geometry is a different interaction from anion...pi -- it is an
    ionic contact with the ring C-H rim and is far stronger (paper: up to
    -8 kcal/mol, against -1.3 for the face-on case). ``anion_pi`` only ever finds
    the face-on approach because it caps both the angle and the offset.
    """
    candidates: list[CandidateInteraction] = []
    min_angle = float(rule["min_angle"])
    min_offset = float(rule["min_offset"])
    for anion in context.features("anion"):
        for ring in context.rings(rule):
            if anion.component_id == ring.component_id:
                continue
            dist = distance(anion.center, ring.center)
            if dist > rule["distance"]:
                continue
            angle = _point_ring_angle(anion, ring)
            offset = point_plane_offset(anion.center, ring.center, ring.normal or (0.0, 0.0, 1.0))
            if angle < min_angle:
                continue
            if not min_offset <= offset <= rule["offset"]:
                continue
            candidates.append(
                _candidate(
                    "anion_aromatic_edge",
                    anion,
                    ring,
                    dist,
                    rule,
                    angle=angle,
                    offset=offset,
                    metadata={"geometry": "edgewise"},
                )
            )
    return candidates


def _point_pi_candidates(
    context: DetectorContext,
    rule: dict,
    point_type: str,
    interaction_type: str,
    point_elements: set[str] | None = None,
) -> list[CandidateInteraction]:
    candidates: list[CandidateInteraction] = []
    for point in context.features(point_type):
        point_atom = context.atoms_by_id[point.atom_ids[0]]
        if point_elements is not None and point_atom.element not in point_elements:
            continue
        for ring in context.rings(rule):
            if point.component_id == ring.component_id:
                continue
            dist = distance(point.center, ring.center)
            if dist > rule["distance"]:
                continue
            point_plane_angle = _point_ring_angle(point, ring)
            offset = point_plane_offset(point.center, ring.center, ring.normal or (0.0, 0.0, 1.0))
            if point_plane_angle > rule.get("angle", 90.0) or offset > rule["offset"]:
                continue

            demands: tuple[ResourceDemand, ...] = ()
            if point_type in _PI_POINT_RESOURCES:
                demands = (ResourceDemand(_PI_POINT_RESOURCES[point_type], point.atom_ids[0]),)

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
            if "donor_angle" in rule:
                _set_donor_angle(
                    candidate,
                    point_atom,
                    ring.center,
                    context.atoms_by_id,
                    float(rule["donor_angle"]),
                )
            else:
                candidate.angle = point_plane_angle
            candidates.append(candidate)
    return candidates


_PI_POINT_RESOURCES = {
    "donor": "donor",
    "carbon_donor": "carbon_donor",
    "halogen_donor": "halogen_donor",
    "chalcogen_donor": "chalcogen_donor",
}


def _hbond_pi_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    # Carbons still carry a `donor` feature, so restrict this to the heteroatom
    # donors; a C-H pointing at a ring is a CH-pi, which has its own rule.
    return _point_pi_candidates(
        context, rule, "donor", "hbond_pi", point_elements={"N", "O", "S", "Se"}
    )


def _ch_pi_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    # carbon_donor is a dedicated feature: reusing `donor` meant a templated
    # ligand's carbons -- which _is_donor restricts to N/O/S -- could never form a
    # CH-pi at all, and the ones that did consumed hydrogen-bond capacity.
    return _point_pi_candidates(context, rule, "carbon_donor", "ch_pi")


def _halogen_pi_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    return _point_pi_candidates(context, rule, "halogen_donor", "halogen_pi")


def _chalcogen_pi_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    return _point_pi_candidates(context, rule, "chalcogen_donor", "chalcogen_pi")


# --------------------------------------------------------------------------
# Sigma-hole interactions
# --------------------------------------------------------------------------


def _sigma_hole_candidates(
    context: DetectorContext,
    rule: dict,
    *,
    interaction_type: str,
    donor_type: str,
    resource: str,
    root_elements: set[str] | None = None,
    record_elevation: bool = False,
    record_activation: bool = False,
    exclude_hydrogen_contacts: bool = False,
) -> list[CandidateInteraction]:
    """Shared sigma-hole geometry for halogen, chalcogen and tetrel bonds.

    A sigma-hole sits on the extension of an R-D bond, so acceptance turns on the
    R-D...A angle. Where the donor carries several substituents the best of them is
    used -- the previous halogen-only implementation took ``root_atoms[0]``, which
    silently mismeasured any multi-substituted donor.
    """
    candidates: list[CandidateInteraction] = []
    min_angle = float(rule["angle"])
    max_distance = float(rule["distance"])
    min_distance = float(rule.get("min_distance", 0.0))

    for donor in context.features(donor_type):
        donor_atom = context.atoms_by_id[donor.atom_ids[0]]
        roots = _heavy_neighbors(donor_atom, context.atoms_by_id, root_elements)
        for acceptor in context.features("acceptor"):
            acceptor_atom_id = acceptor.atom_ids[0]
            if donor.component_id == acceptor.component_id:
                continue
            dist = distance(donor.center, acceptor.center)
            if not min_distance <= dist <= max_distance:
                continue

            candidate = _candidate(
                interaction_type,
                donor,
                acceptor,
                dist,
                rule,
                (
                    ResourceDemand(resource, donor.atom_ids[0]),
                    ResourceDemand("acceptor", acceptor_atom_id),
                ),
            )
            if not roots:
                candidate.rejection_reason = "missing_sigma_hole_root_geometry"
                candidates.append(candidate)
                continue

            best_angle, best_root = max(
                (
                    (geom_angle(root.coord, donor_atom.coord, acceptor.center), root)
                    for root in roots
                ),
                key=lambda pair: pair[0],
            )
            candidate.angle = best_angle
            candidate.metadata["sigma_hole_root_atom_id"] = best_root.atom_id
            if best_angle < min_angle:
                candidate.rejection_reason = (
                    f"angle_below_cutoff:{round(best_angle, 3)}<{min_angle}"
                )
                candidates.append(candidate)
                continue

            if record_activation:
                candidate.metadata["sigma_hole_activated"] = donor_atom.sigma_hole_activated
                if rule.get("require_activated_sigma_hole") and not donor_atom.sigma_hole_activated:
                    # Checked after the geometry so the rejected set stays meaningful:
                    # "this is a chalcogen bond geometrically, but the donor cannot
                    # donate". Checking first would emit one of these for every Met or
                    # Cys that happens to sit near an acceptor.
                    candidate.rejection_reason = "sigma_hole_not_activated"
                    candidates.append(candidate)
                    continue

            if record_elevation:
                # How far the acceptor sits out of the R1-D-R2 plane. Reported, not
                # gated: the sigma-hole angle above already confines this to 40 deg,
                # so a separate cutoff could never reject anything.
                elevation = _out_of_plane_elevation(
                    donor_atom, acceptor.center, context.atoms_by_id
                )
                if elevation is not None:
                    candidate.metadata["elevation"] = elevation

            if exclude_hydrogen_contacts:
                hydrogen_angle = _closest_hydrogen_angle(
                    donor_atom, acceptor.center, context.atoms_by_id
                )
                if hydrogen_angle is not None:
                    candidate.metadata["hydrogen_angle"] = hydrogen_angle
                    limit = float(rule["hydrogen_angle"])
                    if hydrogen_angle < limit:
                        # A C-H pointing at the acceptor is a C-H...A contact. Without
                        # this the far cheaper weak_hbond rule would claim every one
                        # of these first anyway, and the label would be wrong.
                        candidate.rejection_reason = (
                            f"hydrogen_contact:{round(hydrogen_angle, 3)}<{limit}"
                        )
                        candidates.append(candidate)
                        continue

            candidates.append(candidate)
    return candidates


def _halogen_bond_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    return _sigma_hole_candidates(
        context,
        rule,
        interaction_type="halogen_bond",
        donor_type="halogen_donor",
        resource="halogen_donor",
    )


def _chalcogen_bond_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    return _sigma_hole_candidates(
        context,
        rule,
        interaction_type="chalcogen_bond",
        donor_type="chalcogen_donor",
        resource="chalcogen_donor",
        record_elevation=True,
        record_activation=True,
    )


def _tetrel_bond_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    from .parser import TETREL_WITHDRAWING_ELEMENTS

    return _sigma_hole_candidates(
        context,
        rule,
        interaction_type="tetrel_bond",
        donor_type="tetrel_donor",
        resource="tetrel_donor",
        # The sigma-hole is trans to the withdrawing substituent, so only those
        # count as the axis -- an ordinary C-C bond does not open one.
        root_elements=TETREL_WITHDRAWING_ELEMENTS,
        exclude_hydrogen_contacts=True,
    )


# --------------------------------------------------------------------------
# n -> pi*
# --------------------------------------------------------------------------


def _n_pi_star_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    """Carbonyl lone pair donating into an adjacent carbonyl's pi* orbital.

    Burgi-Dunitz geometry: the donor oxygen approaches the acceptor carbon at
    roughly 109 degrees to its C=O axis and close to perpendicular to the sp2
    plane. The paper's quoted optimum of 102 degrees sits inside the accepted window.
    """
    carbonyls = context.features("carbonyl")
    max_distance = float(rule["distance"])
    min_angle = float(rule["min_angle"])
    max_angle = float(rule["angle"])
    approach_limit = float(rule["approach_angle"])
    separation = int(rule.get("min_residue_separation", 1))
    candidates: list[CandidateInteraction] = []

    for donor in carbonyls:
        donor_oxygen = context.atoms_by_id[donor.metadata["oxygen_atom_id"]]
        donor_carbon = context.atoms_by_id[donor.metadata["carbon_atom_id"]]
        for acceptor in carbonyls:
            if donor.feature_id == acceptor.feature_id:
                continue
            acceptor_carbon = context.atoms_by_id[acceptor.metadata["carbon_atom_id"]]
            acceptor_oxygen = context.atoms_by_id[acceptor.metadata["oxygen_atom_id"]]
            if donor_oxygen.atom_id in {acceptor_carbon.atom_id, acceptor_oxygen.atom_id}:
                continue
            # An ASP/ASN side-chain carboxylate sits within the Burgi-Dunitz window of
            # its own backbone carbonyl purely because the two are three bonds apart.
            # Those made up ~4% of every hit and are covalent geometry, not a contact.
            if not _residue_separation_ok(donor_carbon, acceptor_carbon, separation):
                continue

            dist = distance(donor_oxygen.coord, acceptor_carbon.coord)
            if dist > max_distance:
                continue
            burgi_dunitz = geom_angle(
                donor_oxygen.coord, acceptor_carbon.coord, acceptor_oxygen.coord
            )
            if not min_angle <= burgi_dunitz <= max_angle:
                continue

            approach = _plane_normal_angle(
                acceptor.normal,
                tuple(
                    donor_oxygen.coord[i] - acceptor_carbon.coord[i] for i in range(3)
                ),
            )
            if approach is not None and approach > approach_limit:
                continue

            candidate = _candidate(
                "n_pi_star",
                donor,
                acceptor,
                dist,
                rule,
                angle=burgi_dunitz,
                metadata={
                    "donor_oxygen_atom_id": donor_oxygen.atom_id,
                    "acceptor_carbon_atom_id": acceptor_carbon.atom_id,
                    "approach_angle": approach,
                    "interaction_subtype": "burgi_dunitz",
                },
            )
            candidates.append(candidate)
    return candidates


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _heavy_neighbors(
    atom: AtomRecord,
    atoms_by_id: dict[int, AtomRecord],
    elements: set[str] | None = None,
) -> list[AtomRecord]:
    neighbors = [
        atoms_by_id[neighbor_id]
        for neighbor_id in atom.bonds
        if neighbor_id in atoms_by_id and atoms_by_id[neighbor_id].element != "H"
    ]
    if elements is None:
        return neighbors
    return [neighbor for neighbor in neighbors if neighbor.element in elements]


def _out_of_plane_elevation(
    donor_atom: AtomRecord,
    target: Vec3,
    atoms_by_id: dict[int, AtomRecord],
) -> float | None:
    """Angle of donor->target out of the R1-D-R2 plane, in degrees.

    Both sigma-holes of a divalent chalcogen lie in the plane of its substituents,
    so 0 means the acceptor is aimed straight at one of them and 90 means it sits
    over the lone-pair belt instead. Returns None when the donor has fewer than two
    substituents and no plane exists.
    """
    roots = _heavy_neighbors(donor_atom, atoms_by_id)
    if len(roots) != 2:
        # One substituent gives no plane. Three (a pyramidal sulfonium) gives no single
        # plane either -- picking two of them would report an arbitrary number.
        return None
    first = tuple(roots[0].coord[i] - donor_atom.coord[i] for i in range(3))
    second = tuple(roots[1].coord[i] - donor_atom.coord[i] for i in range(3))
    normal = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    vector = tuple(target[i] - donor_atom.coord[i] for i in range(3))
    angle_to_normal = _vector_angle(normal, vector)
    if angle_to_normal is None:
        return None
    return abs(90.0 - angle_to_normal)


def _closest_hydrogen_angle(
    donor_atom: AtomRecord,
    target: Vec3,
    atoms_by_id: dict[int, AtomRecord],
) -> float | None:
    """Smallest H-D...A angle over the donor's hydrogens, or None if it has none."""
    angles = [
        geom_angle(atoms_by_id[neighbor_id].coord, donor_atom.coord, target)
        for neighbor_id in donor_atom.bonds
        if neighbor_id in atoms_by_id and atoms_by_id[neighbor_id].element == "H"
    ]
    return min(angles) if angles else None


def _plane_normal_angle(normal: Vec3 | None, vector: Vec3) -> float | None:
    """Angle between ``vector`` and a plane normal, folded into [0, 90]."""
    if normal is None:
        return None
    angle = _vector_angle(normal, vector)
    if angle is None:
        return None
    return min(angle, 180.0 - angle)


def _vector_angle(first: Vec3, second: Vec3) -> float | None:
    norm_a = sqrt(sum(v * v for v in first))
    norm_b = sqrt(sum(v * v for v in second))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    value = sum(first[i] * second[i] for i in range(3)) / (norm_a * norm_b)
    return degrees(acos(max(-1.0, min(1.0, value))))


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


def _best_donor_angle(
    donor_atom: AtomRecord,
    target_center: Vec3,
    atoms_by_id: dict[int, AtomRecord],
) -> tuple[float, int] | None:
    hydrogen_atoms = [
        atoms_by_id[neighbor_id]
        for neighbor_id in donor_atom.bonds
        if neighbor_id in atoms_by_id and atoms_by_id[neighbor_id].element == "H"
    ]
    if not hydrogen_atoms:
        return None
    best_hydrogen = max(
        hydrogen_atoms,
        key=lambda hydrogen: geom_angle(donor_atom.coord, hydrogen.coord, target_center),
    )
    return (
        geom_angle(donor_atom.coord, best_hydrogen.coord, target_center),
        best_hydrogen.atom_id,
    )


def _set_donor_angle(
    candidate: CandidateInteraction,
    donor_atom: AtomRecord,
    target_center: Vec3,
    atoms_by_id: dict[int, AtomRecord],
    min_angle: float,
) -> None:
    best = _best_donor_angle(donor_atom, target_center, atoms_by_id)
    if best is None:
        candidate.rejection_reason = "missing_donor_hydrogen_geometry"
        return

    best_angle, hydrogen_atom_id = best
    candidate.angle = best_angle
    candidate.metadata["hydrogen_atom_id"] = hydrogen_atom_id
    if best_angle < min_angle:
        candidate.rejection_reason = f"angle_below_cutoff:{round(best_angle, 3)}<{min_angle}"


def _with_occlusion_rejections(
    candidates: list[CandidateInteraction],
    atoms: list[AtomRecord],
    features: list[FeatureRecord],
    profile: dict,
) -> list[CandidateInteraction]:
    coords = [atom.coord for atom in atoms]
    coord_index = {atom.atom_id: idx for idx, atom in enumerate(atoms)}
    bonds_by_id = {atom.atom_id: atom.bonds for atom in atoms}
    features_by_id = {feature.feature_id: feature for feature in features}
    ignored_indices = {
        idx for idx, atom in enumerate(atoms) if atom.metadata.get("ignored_by_template")
    }
    radius = float(profile.get("occlusion_radius", 1.0))
    include_multi_atom = bool(profile.get("occlusion_include_multi_atom", True))

    for candidate in candidates:
        if candidate.rejection_reason:
            continue

        endpoints = _occlusion_endpoints(
            candidate, coord_index, coords, features_by_id, bonds_by_id, include_multi_atom
        )
        if endpoints is None:
            continue
        start, end, ignored_atom_ids = endpoints

        if is_occluded(
            start,
            end,
            coords,
            {coord_index[atom_id] for atom_id in ignored_atom_ids if atom_id in coord_index}
            | ignored_indices,
            radius,
        ):
            candidate.rejection_reason = "occluded"
    return candidates


def _occlusion_endpoints(
    candidate: CandidateInteraction,
    coord_index: dict[int, int],
    coords: list[Vec3],
    features_by_id: dict[int, FeatureRecord],
    bonds_by_id: dict[int, tuple[int, ...]],
    include_multi_atom: bool,
) -> tuple[Vec3, Vec3, set[int]] | None:
    """Segment to test for blocking atoms, plus the atoms exempt from blocking it.

    Two-atom interactions run atom to atom, unchanged. Anything built from groups or
    rings -- every pi interaction, every bridge, and group-feature salt bridges --
    used to skip the check entirely; those run centroid to centroid instead.

    On the centroid path the participants' own bonded neighbours are exempt as well.
    A centroid-to-centroid segment is long enough to pass close to the substituents
    of the very atoms taking part: biotin's S...pi to Trp108 was rejected because
    the segment clipped C6, the ring carbon bonded to that same sulfur. A covalent
    neighbour is a substituent, not an obstruction.
    """
    ignored_atom_ids = set(candidate.atom_ids)
    for metadata_key in ("hydrogen_atom_id", "sigma_hole_root_atom_id"):
        if metadata_key in candidate.metadata:
            ignored_atom_ids.add(candidate.metadata[metadata_key])
    ignored_atom_ids.update(candidate.metadata.get("hydrogen_atom_ids", ()))

    if len(candidate.atom_ids) == 2:
        start_id, end_id = candidate.atom_ids
        if start_id not in coord_index or end_id not in coord_index:
            return None
        return coords[coord_index[start_id]], coords[coord_index[end_id]], ignored_atom_ids

    if not include_multi_atom or len(candidate.feature_ids) != 2:
        return None
    first = features_by_id.get(candidate.feature_ids[0])
    second = features_by_id.get(candidate.feature_ids[1])
    if first is None or second is None:
        return None
    ignored_atom_ids.update(first.atom_ids)
    ignored_atom_ids.update(second.atom_ids)
    for atom_id in tuple(ignored_atom_ids):
        ignored_atom_ids.update(bonds_by_id.get(atom_id, ()))
    return first.center, second.center, ignored_atom_ids


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
    angle = _plane_normal_angle(ring.normal, vector)
    return 0.0 if angle is None else angle


_DETECTORS: dict[str, Callable[[DetectorContext, dict], list[CandidateInteraction]]] = {
    "metal_coordination": _metal_candidates,
    "salt_bridge": _salt_bridge_candidates,
    "amide_bridge": _amide_bridge_candidates,
    "hbond": _hbond_candidates,
    "solvent_bridge": _solvent_bridge_candidates,
    "weak_hbond": _weak_hbond_candidates,
    "pipi_stack": _pipi_stack_candidates,
    "aliphatic_pi_stack": _aliphatic_pi_stack_candidates,
    "aliphatic_stack": _aliphatic_stack_candidates,
    "cation_pi": _cation_pi_candidates,
    "anion_pi": _anion_pi_candidates,
    "anion_aromatic_edge": _anion_aromatic_edge_candidates,
    "hbond_pi": _hbond_pi_candidates,
    "n_pi_star": _n_pi_star_candidates,
    "amide_pi": _amide_pi_candidates,
    "halogen_bond": _halogen_bond_candidates,
    "chalcogen_bond": _chalcogen_bond_candidates,
    "halogen_pi": _halogen_pi_candidates,
    "chalcogen_pi": _chalcogen_pi_candidates,
    "tetrel_bond": _tetrel_bond_candidates,
    "ch_pi": _ch_pi_candidates,
    "hydrophobic": _hydrophobic_candidates,
}
