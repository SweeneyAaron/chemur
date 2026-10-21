from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, degrees, isfinite, sqrt
from typing import Callable, Iterable

from .core import distance, is_enclosed, is_occluded, neighbor_pairs, point_plane_offset
from .interaction_types import INTERACTION_TYPES
from .models import AtomRecord, CandidateInteraction, FeatureRecord, ResourceDemand

Vec3 = tuple[float, float, float]

# Capability marker for clients that must reject older same-version engine builds.
HYDROPHOBIC_ENDPOINT_HYDROGEN_EXEMPTION = True


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
    # Constituent contacts must pass every geometry check, including occlusion, before
    # they can form a second-order candidate.
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
        if not second_order:
            _with_covalent_neighborhood_rejections(context.candidates, atoms, features)
            _with_occlusion_rejections(context.candidates, atoms, features, profile)

    return context.candidates


def _is_second_order(name: str) -> bool:
    return name in _SECOND_ORDER_RULES


_SECOND_ORDER_RULES = {"amide_bridge", "solvent_bridge", "hydrophobic_enclosure"}


# --------------------------------------------------------------------------
# Hydrogen bonds
# --------------------------------------------------------------------------


def _hbond_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    candidates = []
    for donor in context.features("donor"):
        donor_atom = context.atoms_by_id[donor.atom_ids[0]]
        if donor_atom.element not in {"N", "O", "S", "Se"}:
            continue
        for acceptor in context.features("acceptor"):
            acceptor_atom = context.atoms_by_id[acceptor.atom_ids[0]]
            if donor_atom.atom_id == acceptor_atom.atom_id or acceptor_atom.element not in {"N", "O", "S", "Se"}:
                continue
            water = "solvent" in {donor_atom.molecule_type, acceptor_atom.molecule_type}
            dist = distance(donor.center, acceptor.center)
            lower = 0.0 if water else float(rule["min_distance"])
            upper = float(rule["water_distance"] if water else rule["distance"])
            if not lower <= dist <= upper:
                continue
            candidate = _candidate("hbond", donor, acceptor, dist, rule, (
                ResourceDemand("donor", donor_atom.atom_id),
                ResourceDemand("acceptor", acceptor_atom.atom_id),
            ))
            if water:
                # Preserve the existing explicit-H water-leg definition.
                _set_donor_angle(candidate, donor_atom, acceptor.center, context.atoms_by_id, rule["water_angle"])
            else:
                _set_hydrogen_geometry(candidate, donor_atom, acceptor.center, context.atoms_by_id,
                    min_angle=float(rule["angle"]), hydrogen_min_distance=float(rule["hydrogen_min_distance"]),
                    hydrogen_distance=float(rule["hydrogen_distance"]), acceptor_atom=acceptor_atom,
                    acceptor_min_angle=float(rule["acceptor_angle"]))
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
            if context.atoms_by_id[acceptor_atom_id].element != "O":
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
                metadata={"weak_hbond_activated": donor_atom.weak_hbond_activated},
            )
            _set_donor_angle(
                candidate, donor_atom, acceptor.center, context.atoms_by_id, rule["angle"]
            )
            if candidate.rejection_reason is None and not donor_atom.weak_hbond_activated:
                # Keep geometrically valid contacts in the raw diagnostics, but
                # leave their donor/acceptor capacity available for other rules.
                candidate.rejection_reason = "carbon_not_activated"
            candidates.append(candidate)
    return candidates


def _amide_bridge_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    """Two independently valid reciprocal hydrogen bonds between amide groups."""
    amides = context.features("amide")
    if len(amides) < 2:
        return []
    leg_candidates = [c for c in context.candidates if c.interaction_type == "hbond"]
    if not leg_candidates:
        # A separately enabled amide rule still uses the same hydrogen-bond definition.
        leg_candidates = _hbond_candidates(context, context.profile["rules"]["hbond"])
        atoms = list(context.atoms_by_id.values())
        features = [f for group in context.features_by_type.values() for f in group]
        _with_covalent_neighborhood_rejections(leg_candidates, atoms, features)
        _with_occlusion_rejections(leg_candidates, atoms, features, context.profile)
    valid = {(c.resource_demands[0].atom_id, c.resource_demands[1].atom_id): c
             for c in leg_candidates if not c.rejection_reason}
    candidates = []
    for index, first in enumerate(amides):
        for second in amides[index + 1:]:
            if _same_feature_entity(first, second) or not _amides_are_far_enough_apart(
                    first, second, context, int(rule.get("min_residue_separation", 2))):
                continue
            forward = valid.get((first.metadata["nitrogen_atom_id"], second.metadata["oxygen_atom_id"]))
            reverse = valid.get((second.metadata["nitrogen_atom_id"], first.metadata["oxygen_atom_id"]))
            if forward is None or reverse is None:
                continue
            candidate = _candidate("amide_bridge", first, second,
                (forward.distance + reverse.distance) / 2.0, rule,
                forward.resource_demands + reverse.resource_demands,
                angle=min(forward.angle, reverse.angle),
                metadata={"forward_distance": forward.distance, "reverse_distance": reverse.distance,
                    "hydrogen_atom_ids": [forward.metadata["hydrogen_atom_id"], reverse.metadata["hydrogen_atom_id"]],
                    "legs": [dict(c.metadata, atom_ids=list(c.atom_ids), distance=c.distance, angle=c.angle)
                             for c in (forward, reverse)]})
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
    """Apply sequence exclusions only within the same actual polymer chain."""
    if separation <= 0:
        return True
    if first.molecule_type == second.molecule_type == "ligand":
        return not (first.metadata.get("ligand_assembly_id") and _atom_entity_id(first) == _atom_entity_id(second))
    if first.molecule_type not in {"protein", "nucleotide"} or second.molecule_type != first.molecule_type:
        return True
    def chain_identity(atom):
        return (atom.metadata.get("source_model_id", atom.metadata.get("source_model")),
                atom.metadata.get("source_entity_id"),
                atom.metadata.get("source_label_asym_id") or atom.chain_id)
    if chain_identity(first) != chain_identity(second):
        return True
    indices = [a.metadata.get("source_polymer_sequence_index", a.metadata.get("source_label_seq_id"))
               for a in (first, second)]
    try:
        return abs(int(indices[0]) - int(indices[1])) >= separation
    except (TypeError, ValueError):
        # Deposited numbering is not sequence: insertion codes and gaps matter.
        # Covalent-neighbourhood exclusions still apply without sequence metadata.
        return first.component_id != second.component_id






def _solvent_bridge_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    atoms_by_id = context.atoms_by_id
    cutoff = float(rule["distance"])
    # Keying by the directed donor/acceptor feature pair also removes duplicate
    # input legs without collapsing chemically different donor directions.
    legs_by_solvent: dict[int, dict[tuple[int, ...], CandidateInteraction]] = {}
    for candidate in context.candidates:
        if (
            candidate.interaction_type != "hbond"
            or candidate.rejection_reason
            or candidate.distance > cutoff
        ):
            continue
        solvent_atom_ids = [
            atom_id for atom_id in candidate.atom_ids
            if atoms_by_id[atom_id].molecule_type == "solvent"
        ]
        # A bridge is one water between two non-solvent partners, not a chain
        # involving a water-water hydrogen bond.
        if len(candidate.atom_ids) != 2 or len(solvent_atom_ids) != 1:
            continue
        legs_by_solvent.setdefault(solvent_atom_ids[0], {})[candidate.feature_ids] = candidate

    bridges: list[CandidateInteraction] = []
    for solvent_atom_id in sorted(legs_by_solvent):
        solvent_legs = [
            leg for _, leg in sorted(legs_by_solvent[solvent_atom_id].items())
        ]
        for i, leg_a in enumerate(solvent_legs):
            for leg_b in solvent_legs[i + 1 :]:
                endpoint_components = {
                    _atom_entity_id(atoms_by_id[atom_id])
                    for atom_id in leg_a.atom_ids + leg_b.atom_ids
                    if atom_id != solvent_atom_id
                }
                if len(endpoint_components) != 2:
                    continue
                legs = (leg_a, leg_b)
                bridges.append(
                    CandidateInteraction(
                        interaction_type="solvent_bridge",
                        feature_ids=tuple(sorted(set(leg_a.feature_ids + leg_b.feature_ids))),
                        atom_ids=tuple(sorted(set(leg_a.atom_ids + leg_b.atom_ids))),
                        component_ids=tuple(sorted(set(leg_a.component_ids + leg_b.component_ids))),
                        distance=leg_a.distance + leg_b.distance,
                        priority=rule["priority"],
                        stage="inter",
                        # The two hydrogen bonds consume these resources. The
                        # bridge only describes their joint connectivity.
                        resource_demands=(),
                        metadata={
                            "legs": [dict(leg.metadata) for leg in legs],
                            "leg_feature_ids": [list(leg.feature_ids) for leg in legs],
                            "leg_donor_atom_ids": [
                                demand.atom_id for leg in legs for demand in leg.resource_demands
                                if demand.resource_type == "donor"
                            ],
                            "leg_acceptor_atom_ids": [
                                demand.atom_id for leg in legs for demand in leg.resource_demands
                                if demand.resource_type == "acceptor"
                            ],
                        },
                    )
                )
    return bridges


# --------------------------------------------------------------------------
# Ionic and metal
# --------------------------------------------------------------------------


def _salt_bridge_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    candidates = []
    def charged_atoms(feature):
        atoms = [context.atoms_by_id[i] for i in feature.atom_ids]
        if feature.metadata.get("source") == "protein_arg":
            atoms = [a for a in atoms if a.element == "N"]
        elif feature.metadata.get("source") == "nucleotide_phosphate":
            atoms = [a for a in atoms if a.element == "O"]
        return atoms
    for cation in context.features("cation"):
        for anion in context.features("anion"):
            if _same_feature_entity(cation, anion):
                continue
            pairs = [(distance(a.coord, b.coord), a.atom_id, b.atom_id)
                     for a in charged_atoms(cation) for b in charged_atoms(anion)]
            if not pairs:
                continue
            dist, first, second = min(pairs)
            if dist <= rule["distance"]:
                candidates.append(_candidate("salt_bridge", cation, anion, dist, rule,
                    metadata={"distance_atom_ids": [first, second]}))
    return candidates


def _metal_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    candidates = []
    for metal in context.features("metal"):
        symbol = context.atoms_by_id[metal.atom_ids[0]].element
        if symbol in {"Na", "K"}:
            lower, upper = rule["alkali_min_distance"], rule["alkali_distance"]
        elif symbol in {"Ca", "Co", "Cu", "Fe", "Mg", "Mn", "Ni", "Zn"}:
            lower, upper = rule["min_distance"], rule["distance"]
        else:
            continue
        for acceptor in context.features("acceptor"):
            if _same_feature_entity(metal, acceptor):
                continue
            dist = distance(metal.center, acceptor.center)
            if lower <= dist <= upper:
                candidates.append(_candidate("metal_coordination", metal, acceptor, dist, rule,
                    (ResourceDemand("acceptor", acceptor.atom_ids[0]),)))
    return candidates


def _hydrophobic_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    hydrophobes = context.features("hydrophobe")
    candidates: list[CandidateInteraction] = []
    for feature_a, feature_b, dist in _nearby_feature_pairs(hydrophobes, rule["distance"]):
        if _same_feature_entity(feature_a, feature_b):
            continue
        candidates.append(_candidate("hydrophobic", feature_a, feature_b, dist, rule))
    return candidates


def _hydrophobic_enclosure_candidates(
    context: DetectorContext, rule: dict,
) -> list[CandidateInteraction]:
    """Group valid hydrophobic contacts that strictly surround a central atom.

    Build every group from the same unmodified contact graph. Assignment replaces
    constituent pairs later, so neighbouring enclosed atoms can share a contact.
    """
    contacts: dict[int, dict[int, CandidateInteraction]] = {}
    for contact in context.candidates:
        if (
            contact.interaction_type != "hydrophobic"
            or contact.rejection_reason
            or len(contact.atom_ids) != 2
            or not isfinite(contact.distance)
            or contact.distance <= 0.0
        ):
            continue
        first, second = contact.atom_ids
        contacts.setdefault(first, {})[second] = contact
        contacts.setdefault(second, {})[first] = contact

    enclosures: list[CandidateInteraction] = []
    for center_id, neighbors in sorted(contacts.items()):
        if context.atoms_by_id[center_id].molecule_type != "ligand" or len(neighbors) < 4:
            continue
        neighbor_ids = sorted(neighbors)
        if not is_enclosed(
            context.atoms_by_id[center_id].coord,
            [context.atoms_by_id[atom_id].coord for atom_id in neighbor_ids],
        ):
            continue
        legs = [neighbors[atom_id] for atom_id in neighbor_ids]
        enclosures.append(CandidateInteraction(
            interaction_type="hydrophobic_enclosure",
            feature_ids=tuple(sorted({feature_id for leg in legs for feature_id in leg.feature_ids})),
            atom_ids=tuple(sorted([center_id, *neighbor_ids])),
            component_ids=tuple(sorted({component_id for leg in legs for component_id in leg.component_ids})),
            distance=max(leg.distance for leg in legs),
            priority=rule["priority"],
            stage="inter",
            resource_demands=(),
            metadata={
                "center_atom_id": center_id,
                "neighbor_atom_ids": neighbor_ids,
                "interaction_subtype": "hydrophobic_enclosure",
                "legs": [
                    {
                        "atom_ids": list(leg.atom_ids),
                        "feature_ids": list(leg.feature_ids),
                        "component_ids": list(leg.component_ids),
                        "distance": leg.distance,
                    }
                    for leg in legs
                ],
            },
        ))
    return enclosures


# --------------------------------------------------------------------------
# Ring stacking
# --------------------------------------------------------------------------


def _ring_stack_candidates(interaction_type: str, rule: dict,
        pairs: Iterable[tuple[FeatureRecord, FeatureRecord]], *, gate_angle: bool) -> list[CandidateInteraction]:
    candidates = []
    for ring_a, ring_b in pairs:
        if _same_feature_entity(ring_a, ring_b):
            continue
        dist = distance(ring_a.center, ring_b.center)
        angle = _normal_angle(ring_a.normal, ring_b.normal)
        if not all(isfinite(v) for v in (dist, angle)):
            continue
        if interaction_type == "aliphatic_pi_stack":
            # ring_b is aromatic; the published height is measured to its plane.
            separation = abs(sum((ring_a.center[i] - ring_b.center[i]) * ring_b.normal[i] for i in range(3))) / sqrt(sum(v*v for v in ring_b.normal))
            if not (dist < rule["distance"] and angle < rule["angle"] and separation < rule["plane_separation"]):
                continue
        else:
            separation = None
            if dist > rule["distance"]:
                continue
        offset = min(point_plane_offset(ring_b.center, ring_a.center, ring_a.normal),
                     point_plane_offset(ring_a.center, ring_b.center, ring_b.normal))
        if "offset" in rule and offset > rule["offset"]:
            continue
        if interaction_type == "pipi_stack":
            if rule["angle"] < angle < rule["min_perpendicular_angle"]:
                continue
        elif gate_angle and angle > rule["angle"]:
            continue
        geometry = "parallel" if angle <= rule.get("angle", 30.0) else "t_shaped"
        metadata = {"geometry": geometry}
        if separation is not None:
            metadata["plane_separation"] = separation
        candidates.append(_candidate(interaction_type, ring_a, ring_b, dist, rule,
            angle=angle, offset=offset, metadata=metadata))
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
    candidates = []
    for amide in context.features("amide"):
        carbon = context.atoms_by_id[amide.metadata["carbon_atom_id"]]
        nitrogen = context.atoms_by_id[amide.metadata["nitrogen_atom_id"]]
        center = tuple((carbon.coord[i] + nitrogen.coord[i]) / 2.0 for i in range(3))
        for ring in context.rings(rule):
            if _same_feature_entity(amide, ring):
                continue
            dist = distance(center, ring.center)
            angle = _normal_angle(amide.normal, ring.normal)
            approach = _plane_normal_angle(amide.normal, tuple(ring.center[i] - center[i] for i in range(3)))
            if not isfinite(dist) or not isfinite(angle) or approach is None:
                continue
            if dist <= rule["distance"] and angle <= rule["angle"] and approach <= rule["approach_angle"]:
                candidates.append(_candidate("amide_pi", amide, ring, dist, rule, angle=angle,
                    metadata={"approach_angle": approach, "amide_center": list(center)}))
    return candidates


# --------------------------------------------------------------------------
# Point-to-ring interactions
# --------------------------------------------------------------------------




def _cation_pi_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    candidates = []
    for cation in context.features("cation"):
        for ring in context.rings(rule):
            if _same_feature_entity(cation, ring):
                continue
            dist = distance(cation.center, ring.center)
            angle = _point_ring_angle(cation, ring)
            if not isfinite(angle) or dist > rule["distance"]:
                continue
            offset = point_plane_offset(cation.center, ring.center, ring.normal)
            if offset > rule["offset"]:
                continue
            metadata = {"point_plane_angle": angle}
            if len(cation.atom_ids) == 1:
                atom = context.atoms_by_id[cation.atom_ids[0]]
                roots = _heavy_neighbors(atom, context.atoms_by_id)
                if atom.metadata.get("tertiary_amine_cation") and len(roots) == 3:
                    # PLIP compares the amine substituent plane with the aromatic plane.
                    a = tuple(roots[1].coord[i] - roots[0].coord[i] for i in range(3))
                    b = tuple(roots[2].coord[i] - roots[0].coord[i] for i in range(3))
                    normal = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
                    approach = _normal_angle(normal, ring.normal)
                    if not isfinite(approach) or approach > rule["tertiary_amine_angle"]:
                        continue
                    metadata["tertiary_amine_angle"] = approach
            candidates.append(_candidate("cation_pi", cation, ring, dist, rule, angle=angle, offset=offset, metadata=metadata))
    return candidates




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
            if _same_feature_entity(anion, ring):
                continue
            dist = distance(anion.center, ring.center)
            if dist > rule["distance"]:
                continue
            angle = _point_ring_angle(anion, ring)
            offset = point_plane_offset(anion.center, ring.center, ring.normal or (0.0, 0.0, 1.0))
            if not isfinite(angle) or angle < min_angle:
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


def _point_pi_candidates(context: DetectorContext, rule: dict, point_type: str,
        interaction_type: str, point_elements: set[str] | None = None) -> list[CandidateInteraction]:
    candidates = []
    for point in context.features(point_type):
        point_atom = context.atoms_by_id[point.atom_ids[0]]
        if point_elements is not None and point_atom.element not in point_elements:
            continue
        for ring in context.rings(rule):
            if _same_feature_entity(point, ring):
                continue
            dist = distance(point.center, ring.center)
            face_angle = _point_ring_angle(point, ring)
            if not isfinite(dist) or not isfinite(face_angle):
                continue
            if face_angle > rule.get("angle", 90.0):
                continue
            if "distance" in rule and dist > rule["distance"]:
                continue
            if "hydrogen_distance" in rule:
                # Safe prefilter only: the measured cutoff remains H–centroid.
                lengths = [distance(point_atom.coord, context.atoms_by_id[i].coord) for i in point_atom.bonds
                           if i in context.atoms_by_id and context.atoms_by_id[i].element in {"H", "D", "T"}
                           and not context.atoms_by_id[i].metadata.get("ignored_by_template")]
                if lengths and dist > rule["hydrogen_distance"] + max(lengths):
                    continue
            offset = point_plane_offset(point.center, ring.center, ring.normal)
            demands = ((ResourceDemand(_PI_POINT_RESOURCES[point_type], point.atom_ids[0]),)
                       if point_type in _PI_POINT_RESOURCES else ())
            candidate = _candidate(interaction_type, point, ring, dist, rule, demands,
                offset=offset, metadata={"point_plane_angle": face_angle})
            if "hydrogen_distance" in rule:
                _set_hydrogen_geometry(candidate, point_atom, ring.center, context.atoms_by_id,
                    min_angle=float(rule["donor_angle"]), hydrogen_distance=float(rule["hydrogen_distance"]))
            elif interaction_type == "halogen_pi":
                roots = _heavy_neighbors(point_atom, context.atoms_by_id, {"C"})
                if not roots:
                    candidate.rejection_reason = "missing_sigma_hole_root_geometry"
                else:
                    candidate.angle, root_id = max((_bond_angle(root.coord, point_atom.coord, ring.center), root.atom_id) for root in roots)
                    candidate.metadata["sigma_hole_root_atom_id"] = root_id
                    if not isfinite(candidate.angle) or candidate.angle <= rule["donor_angle"]:
                        candidate.rejection_reason = "donor_angle_not_above_cutoff"
            else:
                candidate.angle = face_angle
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
        context, rule, "donor", "hbond_pi", point_elements={"N", "O"}
    )


def _ch_pi_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    # carbon_donor is a dedicated feature: reusing `donor` meant a templated
    # ligand's carbons -- which _is_donor restricts to N/O/S -- could never form a
    # CH-pi at all, and the ones that did consumed hydrogen-bond capacity.
    return _point_pi_candidates(context, rule, "carbon_donor", "ch_pi")


def _halogen_pi_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    return _point_pi_candidates(context, rule, "halogen_donor", "halogen_pi")


def _chalcogen_pi_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    """Union of directional sulfur–pi and broad thioether/sulfoxide–aromatic contacts."""
    candidates = []
    seen = set()
    for sulfur in context.features("sulfur_pi"):
        atom = context.atoms_by_id[sulfur.atom_ids[0]]
        types = set(sulfur.metadata.get("sulfur_pi_types", ()))
        roots = _heavy_neighbors(atom, context.atoms_by_id, {"C", "S"})
        for ring in context.rings(rule):
            identity = (atom.atom_id, tuple(sorted(ring.atom_ids)))
            if identity in seen or _same_feature_entity(sulfur, ring):
                continue
            dist = distance(sulfur.center, ring.center)
            face = _point_ring_angle(sulfur, ring)
            if not isfinite(dist) or not isfinite(face):
                continue
            root_angles = [(_bond_angle(r.coord, atom.coord, ring.center), r.atom_id) for r in roots]
            directional_angle, root_id = max((value for value in root_angles if isfinite(value[0])), default=(0.0, None))
            classifications = []
            if types & {"thioether", "disulfide"} and dist <= rule["directional_distance"] and directional_angle >= rule["directional_angle"]:
                classifications.append("directional")
            if types & {"thioether", "sulfoxide"} and dist < rule["distance"] and face < rule["angle"]:
                classifications.append("sulfur_aromatic")
            if not classifications:
                continue
            seen.add(identity)
            metadata = {"geometry_classifications": classifications, "point_plane_angle": face,
                        "sulfur_pi_types": sorted(types), "directional_angle": directional_angle}
            if root_id is not None:
                metadata["sigma_hole_root_atom_id"] = root_id
            candidates.append(_candidate("chalcogen_pi", sulfur, ring, dist, rule, angle=face,
                offset=point_plane_offset(sulfur.center, ring.center, ring.normal), metadata=metadata))
    return candidates


# --------------------------------------------------------------------------
# Sigma-hole interactions
# --------------------------------------------------------------------------


def _sigma_hole_candidates(context: DetectorContext, rule: dict, *, interaction_type: str,
        donor_type: str, resource: str, root_elements: set[str] | None = None,
        record_elevation: bool = False, record_activation: bool = False) -> list[CandidateInteraction]:
    candidates = []
    carbonyl_oxygen = {f.metadata["oxygen_atom_id"]: f.metadata["carbon_atom_id"]
                      for f in context.features("carbonyl")}
    for donor in context.features(donor_type):
        donor_atom = context.atoms_by_id[donor.atom_ids[0]]
        roots = _heavy_neighbors(donor_atom, context.atoms_by_id, root_elements)
        if interaction_type == "tetrel_bond" and "tetrel_root_atom_ids" in donor_atom.metadata:
            valid_roots = set(donor_atom.metadata["tetrel_root_atom_ids"])
            roots = [root for root in roots if root.atom_id in valid_roots]
        for acceptor in context.features("acceptor"):
            acceptor_atom = context.atoms_by_id[acceptor.atom_ids[0]]
            if _same_feature_entity(donor, acceptor):
                continue
            if interaction_type in {"halogen_bond", "tetrel_bond"} and acceptor_atom.element != "O":
                continue
            if interaction_type == "tetrel_bond" and acceptor_atom.atom_id not in carbonyl_oxygen:
                continue
            if interaction_type == "chalcogen_bond":
                prefix = {("S", "O"): "s_o", ("Se", "O"): "se_o", ("Se", "S"): "se_s"}.get((donor_atom.element, acceptor_atom.element))
                if prefix is None:
                    continue
                lower, upper = rule[prefix + "_min_distance"], rule[prefix + "_distance"]
            else:
                lower, upper = rule.get("min_distance", 0.0), rule["distance"]
            dist = distance(donor.center, acceptor.center)
            if not lower <= dist <= upper or (interaction_type == "chalcogen_bond" and dist <= lower):
                continue
            candidate = _candidate(interaction_type, donor, acceptor, dist, rule,
                (ResourceDemand(resource, donor_atom.atom_id), ResourceDemand("acceptor", acceptor_atom.atom_id)))
            if record_activation:
                candidate.metadata["sigma_hole_activated"] = donor_atom.sigma_hole_activated
            if not roots:
                candidate.rejection_reason = "missing_sigma_hole_root_geometry"
            else:
                candidate.angle, root_id = max((_bond_angle(root.coord, donor_atom.coord, acceptor.center), root.atom_id) for root in roots)
                candidate.metadata["sigma_hole_root_atom_id"] = root_id
                if not isfinite(candidate.angle) or candidate.angle < rule["angle"]:
                    candidate.rejection_reason = f"angle_below_cutoff:{round(candidate.angle, 3)}<{rule['angle']}"
            if candidate.rejection_reason is None and interaction_type in {"halogen_bond", "tetrel_bond"}:
                if interaction_type == "tetrel_bond":
                    antecedents = [context.atoms_by_id[carbonyl_oxygen[acceptor_atom.atom_id]]]
                    minimum, maximum = rule["acceptor_angle"], 180.0
                else:
                    antecedents = _heavy_neighbors(acceptor_atom, context.atoms_by_id, {"C", "N", "P", "S"})
                    minimum, maximum = rule["acceptor_min_angle"], rule["acceptor_angle"]
                values = [(_bond_angle(donor_atom.coord, acceptor_atom.coord, a.coord), a.atom_id) for a in antecedents]
                valid = [item for item in values if minimum <= item[0] <= maximum]
                if not values:
                    candidate.rejection_reason = "missing_acceptor_antecedent_geometry"
                else:
                    angle, atom_id = max(valid or values)
                    candidate.metadata.update(acceptor_angle=angle, acceptor_antecedent_atom_id=atom_id)
                    if not valid:
                        candidate.rejection_reason = "acceptor_angle_outside_cutoff"
            if record_elevation:
                elevation = _out_of_plane_elevation(donor_atom, acceptor.center, context.atoms_by_id)
                if elevation is not None:
                    candidate.metadata["elevation"] = elevation
            candidates.append(candidate)
    return candidates


def _halogen_bond_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    return _sigma_hole_candidates(context, rule, interaction_type="halogen_bond",
        donor_type="halogen_donor", resource="halogen_donor", root_elements={"C"})


def _chalcogen_bond_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    return _sigma_hole_candidates(
        context,
        rule,
        interaction_type="chalcogen_bond",
        donor_type="chalcogen_donor",
        resource="chalcogen_donor",
        root_elements={"C"},
        record_elevation=True,
        record_activation=True,
    )


def _tetrel_bond_candidates(context: DetectorContext, rule: dict) -> list[CandidateInteraction]:
    return _sigma_hole_candidates(context, rule, interaction_type="tetrel_bond",
        donor_type="tetrel_donor", resource="tetrel_donor", root_elements={"C", "N", "O", "S"})


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
            burgi_dunitz = _bond_angle(
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




def _plane_normal_angle(normal: Vec3 | None, vector: Vec3) -> float | None:
    """Angle between ``vector`` and a plane normal, folded into [0, 90]."""
    if normal is None:
        return None
    angle = _vector_angle(normal, vector)
    if angle is None:
        return None
    return min(angle, 180.0 - angle)


def _bond_angle(first: Vec3, center: Vec3, last: Vec3) -> float:
    value = _vector_angle(tuple(first[i] - center[i] for i in range(3)),
                          tuple(last[i] - center[i] for i in range(3)))
    return float("nan") if value is None else value


def _vector_angle(first: Vec3, second: Vec3) -> float | None:
    norm_a = sqrt(sum(v * v for v in first))
    norm_b = sqrt(sum(v * v for v in second))
    if not isfinite(norm_a + norm_b) or norm_a <= 1e-12 or norm_b <= 1e-12:
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
    first_components = _feature_component_ids(feature_a)
    second_components = _feature_component_ids(feature_b)
    component_ids = first_components + second_components
    if len(first_components) > 1 or len(second_components) > 1:
        component_ids = tuple(dict.fromkeys(component_ids))
    stage = "intra" if _same_feature_entity(feature_a, feature_b) else "inter"
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


def _feature_component_ids(feature: FeatureRecord) -> tuple[str, ...]:
    return tuple(feature.metadata.get("component_ids") or (feature.component_id,))


def _feature_entity_id(feature: FeatureRecord) -> tuple[str, str]:
    assembly_id = feature.metadata.get("ligand_assembly_id")
    if assembly_id and not feature.metadata.get("crosses_attachment_boundary"):
        return "assembly", str(assembly_id)
    return "component", feature.component_id


def _same_feature_entity(first: FeatureRecord, second: FeatureRecord) -> bool:
    return _feature_entity_id(first) == _feature_entity_id(second)


def _atom_entity_id(atom: AtomRecord) -> tuple[str, str]:
    assembly_id = atom.metadata.get("ligand_assembly_id")
    if atom.molecule_type == "ligand" and assembly_id:
        return "assembly", str(assembly_id)
    return "component", atom.component_id


def _with_covalent_neighborhood_rejections(
    candidates: list[CandidateInteraction],
    atoms: list[AtomRecord],
    features: list[FeatureRecord],
) -> None:
    """Exclude 1-2 and 1-3 heavy-atom contacts across an imported covalent link.

    Existing intracomponent chemistry keeps its established treatment. Only a
    path containing an explicitly imported link triggers this diagnostic, and
    the path must join sites from the two participating features.
    """
    heavy_atoms = {atom.atom_id: atom for atom in atoms if atom.element != "H"}
    links = {
        tuple(sorted((atom.atom_id, neighbor_id)))
        for atom in heavy_atoms.values()
        for neighbor_id in atom.metadata.get("assembly_link_neighbors", ())
        if neighbor_id in heavy_atoms and neighbor_id != atom.atom_id
    }
    if not links:
        return
    neighbors: dict[int, set[int]] = {atom_id: set() for atom_id in heavy_atoms}
    for atom in heavy_atoms.values():
        for neighbor_id in atom.bonds:
            if neighbor_id in heavy_atoms and neighbor_id != atom.atom_id:
                neighbors[atom.atom_id].add(neighbor_id)
                neighbors[neighbor_id].add(atom.atom_id)
    for first, second in links:
        neighbors[first].add(second)
        neighbors[second].add(first)
    excluded_pairs = set(links)
    for first, second in links:
        excluded_pairs.update(
            tuple(sorted((neighbor, second)))
            for neighbor in neighbors[first] if neighbor != second
        )
        excluded_pairs.update(
            tuple(sorted((first, neighbor)))
            for neighbor in neighbors[second] if neighbor != first
        )
    features_by_id = {feature.feature_id: feature for feature in features}
    for candidate in candidates:
        if len(candidate.feature_ids) != 2:
            continue
        first = features_by_id.get(candidate.feature_ids[0])
        second = features_by_id.get(candidate.feature_ids[1])
        if first is None or second is None:
            continue
        if any(
            tuple(sorted((first_id, second_id))) in excluded_pairs
            for first_id in first.atom_ids if first_id in heavy_atoms
            for second_id in second.atom_ids if second_id in heavy_atoms
        ):
            candidate.rejection_reason = "covalent_neighborhood"


def _set_hydrogen_geometry(
    candidate: CandidateInteraction,
    donor_atom: AtomRecord,
    target_center: Vec3,
    atoms_by_id: dict[int, AtomRecord],
    *,
    min_angle: float,
    hydrogen_distance: float,
    hydrogen_min_distance: float = 0.0,
    acceptor_atom: AtomRecord | None = None,
    acceptor_min_angle: float | None = None,
) -> None:
    """All hydrogen-dependent criteria must hold for the same explicit hydrogen."""
    hydrogens = [atoms_by_id[i] for i in donor_atom.bonds
                 if i in atoms_by_id and atoms_by_id[i].element in {"H", "D", "T"}
                 and not atoms_by_id[i].metadata.get("ignored_by_template")]
    if not hydrogens:
        candidate.rejection_reason = "missing_donor_hydrogen_geometry"
        return
    antecedents = [a for a in _heavy_neighbors(acceptor_atom, atoms_by_id)
                   if all(isfinite(v) for v in a.coord + target_center)
                   and distance(a.coord, target_center) > 1e-12] if acceptor_atom else []
    if acceptor_min_angle is not None and not antecedents:
        candidate.rejection_reason = "missing_acceptor_antecedent_geometry"
        return
    evaluated = []
    for hydrogen in hydrogens:
        dha = _bond_angle(donor_atom.coord, hydrogen.coord, target_center)
        ha = distance(hydrogen.coord, target_center)
        metadata = {"hydrogen_atom_id": hydrogen.atom_id, "hydrogen_distance": ha}
        reason = None
        if not all(isfinite(v) for v in (dha, ha)) or distance(donor_atom.coord, hydrogen.coord) <= 1e-12:
            reason = "invalid_donor_hydrogen_geometry"
        elif dha < min_angle:
            reason = f"angle_below_cutoff:{round(dha, 3)}<{min_angle}"
        elif not hydrogen_min_distance <= ha <= hydrogen_distance:
            reason = "hydrogen_distance_outside_cutoff"
        if antecedents:
            values = [(_bond_angle(a.coord, target_center, hydrogen.coord), a.atom_id) for a in antecedents]
            values = [item for item in values if isfinite(item[0])]
            if not values:
                reason = reason or "invalid_acceptor_antecedent_geometry"
            else:
                antecedent_angle, antecedent_id = max(values)
                metadata.update(acceptor_angle=antecedent_angle, acceptor_antecedent_atom_id=antecedent_id)
                if reason is None and acceptor_min_angle is not None and antecedent_angle < acceptor_min_angle:
                    reason = "acceptor_angle_below_cutoff"
        evaluated.append((reason is not None, -dha, ha, hydrogen.atom_id, reason, metadata))
    failed, negative_angle, _, _, reason, metadata = min(evaluated, key=lambda item: item[:4])
    candidate.angle = -negative_angle
    candidate.metadata.update(metadata)
    candidate.rejection_reason = reason if failed else None


def _best_donor_angle(
    donor_atom: AtomRecord,
    target_center: Vec3,
    atoms_by_id: dict[int, AtomRecord],
) -> tuple[float, int] | None:
    hydrogen_atoms = [
        atoms_by_id[neighbor_id]
        for neighbor_id in donor_atom.bonds
        if neighbor_id in atoms_by_id
        and atoms_by_id[neighbor_id].element in {"H", "D", "T"}
        and not atoms_by_id[neighbor_id].metadata.get("ignored_by_template")
        and all(isfinite(v) for v in atoms_by_id[neighbor_id].coord + donor_atom.coord + target_center)
        and distance(donor_atom.coord, atoms_by_id[neighbor_id].coord) > 1e-12
        and distance(target_center, atoms_by_id[neighbor_id].coord) > 1e-12
    ]
    if not hydrogen_atoms:
        return None
    best_hydrogen = max(
        hydrogen_atoms,
        key=lambda hydrogen: _bond_angle(donor_atom.coord, hydrogen.coord, target_center),
    )
    return (
        _bond_angle(donor_atom.coord, best_hydrogen.coord, target_center),
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
    hydrogen_atom_ids = {atom.atom_id for atom in atoms if atom.element in {"H", "D", "T"}}
    features_by_id = {feature.feature_id: feature for feature in features}
    ignored_indices = {
        idx for idx, atom in enumerate(atoms) if atom.metadata.get("ignored_by_template")
    }
    radius = float(profile.get("occlusion_radius", 1.0))
    include_multi_atom = bool(profile.get("occlusion_include_multi_atom", True))

    for candidate in candidates:
        if candidate.rejection_reason:
            continue
        if candidate.interaction_type == "amide_bridge":
            # Both actual hydrogen-bond legs have already passed occlusion.
            continue

        endpoints = _occlusion_endpoints(
            candidate, coord_index, coords, features_by_id, bonds_by_id, include_multi_atom
        )
        if endpoints is None:
            continue
        start, end, ignored_atom_ids = endpoints
        if candidate.interaction_type == "hydrophobic" and len(candidate.atom_ids) == 2:
            # An endpoint's C-H group supplies the hydrophobic contact; its own
            # hydrogen is not an intervening obstruction. Keep unrelated H and
            # every non-participating heavy atom subject to the existing check.
            for atom_id in candidate.atom_ids:
                ignored_atom_ids.update(hydrogen_atom_ids.intersection(bonds_by_id.get(atom_id, ())))

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

    measured_pair = candidate.metadata.get("distance_atom_ids")
    if measured_pair and all(atom_id in coord_index for atom_id in measured_pair):
        return (coords[coord_index[measured_pair[0]]], coords[coord_index[measured_pair[1]]],
                ignored_atom_ids)

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
    start = tuple(candidate.metadata["amide_center"]) if candidate.interaction_type == "amide_pi" else first.center
    return start, second.center, ignored_atom_ids


def _nearby_feature_pairs(features: list[FeatureRecord], cutoff: float):
    coords = [feature.center for feature in features]
    for i, j, dist in neighbor_pairs(coords, cutoff):
        yield features[i], features[j], dist


def _normal_angle(normal_a, normal_b) -> float:
    if normal_a is None or normal_b is None:
        return float("nan")
    dot = abs(sum(normal_a[i] * normal_b[i] for i in range(3)))
    norm_a = sqrt(sum(v * v for v in normal_a))
    norm_b = sqrt(sum(v * v for v in normal_b))
    if not isfinite(norm_a + norm_b) or norm_a <= 1e-12 or norm_b <= 1e-12:
        return float("nan")
    value = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return degrees(acos(value))


def _point_ring_angle(point: FeatureRecord, ring: FeatureRecord) -> float:
    if ring.normal is None:
        return float("nan")
    vector = tuple(point.center[i] - ring.center[i] for i in range(3))
    angle = _plane_normal_angle(ring.normal, vector)
    return float("nan") if angle is None else angle


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
    "hydrophobic_enclosure": _hydrophobic_enclosure_candidates,
}
