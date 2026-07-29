from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from hashlib import sha1
from math import exp, floor, log
from pathlib import Path
from typing import Any

from .models import AnalysisResult, AssignedInteraction, AtomRecord, ComponentRecord

COSPLIF_VERSION = "cosplif_context_v1"
LIGAND_BLOCKS_VERSION = "ligand_blocks_v1"
DEFAULT_LIGAND_RADIUS = 2
DEFAULT_TARGET_RADIUS = 1
COSPLIF_PROFILE_NAMES = (
    "cosplif_plain_v1",
    "cosplif_plain_v2",
    "cosplif_contrast_v1",
    "cosplif_reference_containment_v1",
    "cosplif_hybrid_ifm_v1",
    "cosplif_hybrid_ifm_v2",
    "cosplif_pose_v1",
)


@dataclass(frozen=True, slots=True)
class CoSPLIFProfile:
    name: str
    use_background_idf: bool = True
    score_mode: str = "blend"
    containment_weight: float = 0.7
    tanimoto_weight: float = 0.3
    idf_scale: float = 1.0
    min_multiplier: float = 0.25
    max_multiplier: float = 4.0
    hybrid_ifm_weight: float = 0.3
    ligand_block_weight: float = 0.0
    ligand_block_spatial_cutoff: float = 4.0
    sigma_hydrophobic: float = 2.5
    sigma_aromatic: float = 1.8
    sigma_polar: float = 1.0
    sigma_charged: float = 1.5
    sigma_metal: float = 0.8
    sigma_fallback: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "use_background_idf": self.use_background_idf,
            "score_mode": self.score_mode,
            "containment_weight": self.containment_weight,
            "tanimoto_weight": self.tanimoto_weight,
            "idf_scale": self.idf_scale,
            "min_multiplier": self.min_multiplier,
            "max_multiplier": self.max_multiplier,
            "hybrid_ifm_weight": self.hybrid_ifm_weight,
            "ligand_block_weight": self.ligand_block_weight,
            "ligand_block_spatial_cutoff": self.ligand_block_spatial_cutoff,
            "sigma_hydrophobic": self.sigma_hydrophobic,
            "sigma_aromatic": self.sigma_aromatic,
            "sigma_polar": self.sigma_polar,
            "sigma_charged": self.sigma_charged,
            "sigma_metal": self.sigma_metal,
            "sigma_fallback": self.sigma_fallback,
        }

    def with_overrides(self, overrides: dict[str, Any] | None) -> "CoSPLIFProfile":
        """Return a copy of this profile with the given fields replaced."""
        fields = {key: value for key, value in (overrides or {}).items() if key != "base"}
        if not fields:
            return self
        unknown = set(fields) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"Unknown CoSPLIF profile field(s): {', '.join(sorted(unknown))}"
            )
        updated = replace(self, **fields)
        updated._validate()
        return updated

    def _validate(self) -> None:
        for name in (
            "containment_weight",
            "tanimoto_weight",
            "idf_scale",
            "min_multiplier",
            "max_multiplier",
            "hybrid_ifm_weight",
            "ligand_block_weight",
            "ligand_block_spatial_cutoff",
            "sigma_hydrophobic",
            "sigma_aromatic",
            "sigma_polar",
            "sigma_charged",
            "sigma_metal",
            "sigma_fallback",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"CoSPLIF profile {name} must be non-negative")
        if self.max_multiplier < self.min_multiplier:
            raise ValueError("CoSPLIF profile max_multiplier must be >= min_multiplier")


def cosplif_profile(profile_name: str) -> CoSPLIFProfile:
    if profile_name == "cosplif_plain_v1":
        return CoSPLIFProfile(
            name=profile_name,
            use_background_idf=False,
            score_mode="blend",
        )
    if profile_name == "cosplif_plain_v2":
        return CoSPLIFProfile(
            name=profile_name,
            use_background_idf=False,
            score_mode="blend",
            ligand_block_weight=0.3,
        )
    if profile_name == "cosplif_contrast_v1":
        return CoSPLIFProfile(name=profile_name)
    if profile_name == "cosplif_reference_containment_v1":
        return CoSPLIFProfile(
            name=profile_name,
            use_background_idf=True,
            score_mode="containment",
            containment_weight=1.0,
            tanimoto_weight=0.0,
        )
    if profile_name == "cosplif_hybrid_ifm_v1":
        return CoSPLIFProfile(
            name=profile_name,
            use_background_idf=True,
            score_mode="hybrid_ifm",
        )
    if profile_name == "cosplif_hybrid_ifm_v2":
        return CoSPLIFProfile(
            name=profile_name,
            use_background_idf=True,
            score_mode="hybrid_ifm",
            ligand_block_weight=0.3,
        )
    if profile_name == "cosplif_pose_v1":
        # Production default: CASF-tuned hybrid CoSPLIF + IFM with the best
        # ligand-block weight (0.40) from the block-weight sweep.
        return CoSPLIFProfile(
            name=profile_name,
            use_background_idf=True,
            score_mode="hybrid_ifm",
            ligand_block_weight=0.40,
        )
    raise ValueError(f"Unknown CoSPLIF profile: {profile_name}")


def cosplif_profiles_from_config(
    config: dict[str, Any] | str | Path | None,
) -> dict[str, CoSPLIFProfile]:
    """Build custom named CoSPLIF profiles from a JSON config.

    Schema: ``{"profiles": {<name>: {"base": <builtin>, <field overrides...>}}}``.
    Each entry starts from the named base profile (default
    ``cosplif_hybrid_ifm_v1``) and applies the remaining keys as field overrides.
    Returns an empty mapping when ``config`` is ``None``.
    """
    if config is None:
        return {}
    if isinstance(config, (str, Path)):
        data = json.loads(Path(config).read_text(encoding="utf-8"))
    elif isinstance(config, dict):
        data = config
    else:
        raise TypeError("cosplif config must be a dict, path, or None")
    profiles_spec = data.get("profiles", data)
    if not isinstance(profiles_spec, dict):
        raise ValueError("cosplif config must contain a 'profiles' object")
    profiles: dict[str, CoSPLIFProfile] = {}
    for name, spec in profiles_spec.items():
        if not isinstance(spec, dict):
            raise ValueError(f"cosplif profile {name!r} must be a JSON object")
        base = spec.get("base", "cosplif_hybrid_ifm_v1")
        overrides = {key: value for key, value in spec.items() if key != "base"}
        profile = cosplif_profile(base).with_overrides(overrides)
        profiles[str(name)] = replace(profile, name=str(name))
    return profiles


def build_cosplif_tokens(
    analysis_result: AnalysisResult,
    ligand_id: str,
    *,
    ligand_radius: int = DEFAULT_LIGAND_RADIUS,
    target_radius: int = DEFAULT_TARGET_RADIUS,
) -> dict[str, int]:
    """Build hashed contextual SPLIF tokens for one ligand in an analysis result."""
    atoms_by_id = {atom.atom_id: atom for atom in analysis_result.atoms}
    components_by_id = {
        component.component_id: component for component in analysis_result.components
    }
    tokens: Counter[str] = Counter()
    for interaction in analysis_result.interactions:
        if ligand_id not in interaction.component_ids:
            continue
        target_component_id = _choose_target_component_id(
            interaction,
            ligand_id=ligand_id,
            components_by_id=components_by_id,
        )
        if target_component_id is None:
            continue
        target_component = components_by_id.get(target_component_id)
        if target_component is None:
            continue

        ligand_atom_ids = tuple(
            atom_id
            for atom_id in interaction.atom_ids
            if atoms_by_id.get(atom_id) is not None
            and atoms_by_id[atom_id].component_id == ligand_id
        )
        target_atom_ids = tuple(
            atom_id
            for atom_id in interaction.atom_ids
            if atoms_by_id.get(atom_id) is not None
            and atoms_by_id[atom_id].component_id == target_component_id
        )
        if not ligand_atom_ids or not target_atom_ids:
            continue

        target_kind = _target_kind(target_component)
        target_moiety = _target_moiety(target_kind, target_component, target_atom_ids, atoms_by_id)
        target_class = _target_class(target_component)
        family = _canonical_family(interaction.interaction_type)
        if family is None:
            continue
        subtype = _canonical_subtype(interaction, target_moiety)
        role = _role_key(interaction, set(ligand_atom_ids), set(target_atom_ids))
        geometry = _geometry_context(interaction)

        for ligand_atom_id in sorted(set(ligand_atom_ids)):
            ligand_env = _atom_environment(
                ligand_atom_id,
                atoms_by_id,
                components_by_id,
                radius=ligand_radius,
                resolution="ligand",
                target_moiety="ligand",
                target_class=None,
            )
            for target_atom_id in sorted(set(target_atom_ids)):
                for resolution in ("specific", "class"):
                    target_env = _atom_environment(
                        target_atom_id,
                        atoms_by_id,
                        components_by_id,
                        radius=target_radius,
                        resolution=resolution,
                        target_moiety=target_moiety,
                        target_class=target_class,
                    )
                    token_text = "|".join(
                        (
                            COSPLIF_VERSION,
                            f"res={resolution}",
                            f"lig={ligand_env}",
                            f"tar={target_env}",
                            f"fam={family}",
                            f"sub={subtype or 'none'}",
                            f"role={role}",
                            geometry,
                        )
                    )
                    tokens[_hash_token(token_text)] += 1
    return dict(tokens)


def build_cosplif_payload(
    analysis_result: AnalysisResult,
    ligand_id: str,
    *,
    ligand_radius: int = DEFAULT_LIGAND_RADIUS,
    target_radius: int = DEFAULT_TARGET_RADIUS,
) -> dict[str, Any]:
    return {
        "version": COSPLIF_VERSION,
        "ligand_radius": ligand_radius,
        "target_radius": target_radius,
        "tokens": build_cosplif_tokens(
            analysis_result,
            ligand_id,
            ligand_radius=ligand_radius,
            target_radius=target_radius,
        ),
        "ligand_blocks_v1": build_ligand_blocks(analysis_result, ligand_id),
    }


def build_ligand_blocks(
    analysis_result: AnalysisResult,
    ligand_id: str,
) -> dict[str, Any]:
    """Build deterministic, non-overlapping chemical blocks for one ligand."""
    atoms_by_id = {atom.atom_id: atom for atom in analysis_result.atoms}
    components_by_id = {
        component.component_id: component for component in analysis_result.components
    }
    component = components_by_id.get(ligand_id)
    warnings: list[str] = []
    if component is None:
        return {
            "version": LIGAND_BLOCKS_VERSION,
            "ligand_id": ligand_id,
            "heavy_atom_count": 0,
            "blocks": [],
            "warnings": [f"Ligand component not found: {ligand_id}"],
        }

    heavy_ids = {
        atom_id
        for atom_id in component.atom_ids
        if (atom := atoms_by_id.get(atom_id)) is not None
        and atom.element != "H"
        and not atom.metadata.get("ignored_by_template")
    }
    bond_orders = _bond_order_lookup(component)
    rdkit_context = _rdkit_fingerprint_context(component)
    seeds: list[dict[str, Any]] = []
    unassigned = set(heavy_ids)

    for ring_atoms in _ring_systems(component, heavy_ids, atoms_by_id):
        _append_block_seed(
            seeds,
            "ring_system",
            ring_atoms,
            unassigned,
            source="template_rings",
        )

    functional_candidates = _functional_group_candidates(
        heavy_ids,
        atoms_by_id,
        bond_orders,
    )
    for candidate in functional_candidates:
        _append_block_seed(
            seeds,
            candidate["block_type"],
            set(candidate["atom_ids"]),
            unassigned,
            source="functional_group",
            labels=set(candidate.get("labels", ())),
        )

    for group in _hydrophobic_substituent_groups(unassigned, atoms_by_id, bond_orders):
        _append_block_seed(
            seeds,
            "hydrophobic_substituent",
            group,
            unassigned,
            source="hydrophobic_graph",
            labels={"hydrophobic"},
        )

    for group in _connected_components(unassigned, atoms_by_id):
        block_type, labels = _linker_block_type(group, atoms_by_id)
        _append_block_seed(
            seeds,
            block_type,
            group,
            unassigned,
            source="linker_graph",
            labels=labels,
        )

    _merge_trivial_singletons(seeds, atoms_by_id)

    blocks = _finalize_ligand_blocks(
        seeds,
        atoms_by_id,
        bond_orders,
        rdkit_context,
        analysis_result.interactions,
        ligand_id=ligand_id,
    )
    assigned = {atom_id for block in blocks for atom_id in block["atom_ids"]}
    if assigned != heavy_ids:
        missing = sorted(heavy_ids - assigned)
        duplicated = sorted(atom_id for atom_id in assigned if atom_id not in heavy_ids)
        if missing:
            warnings.append(f"Ligand block coverage missing atom ids: {missing}")
        if duplicated:
            warnings.append(f"Ligand block coverage has non-ligand atom ids: {duplicated}")

    return {
        "version": LIGAND_BLOCKS_VERSION,
        "ligand_id": ligand_id,
        "heavy_atom_count": len(heavy_ids),
        "blocks": blocks,
        "warnings": warnings,
    }


_HALOGEN_ELEMENTS = {"F", "Cl", "Br", "I"}
_HYDROPHOBIC_ELEMENTS = {"C", "S", "F", "Cl", "Br", "I"}


def _append_block_seed(
    seeds: list[dict[str, Any]],
    block_type: str,
    atom_ids: set[int],
    unassigned: set[int],
    *,
    source: str,
    labels: set[str] | None = None,
) -> None:
    selected = set(atom_ids) & unassigned
    if not selected:
        return
    seeds.append(
        {
            "block_type": block_type,
            "atom_ids": selected,
            "source": source,
            "labels": set(labels or ()),
        }
    )
    unassigned.difference_update(selected)


def _ring_systems(
    component: ComponentRecord,
    heavy_ids: set[int],
    atoms_by_id: dict[int, AtomRecord],
) -> list[set[int]]:
    rings = [
        set(int(atom_id) for atom_id in ring if int(atom_id) in heavy_ids)
        for ring in component.metadata.get("template_rings", [])
    ]
    rings = [ring for ring in rings if len(ring) >= 3]
    if not rings:
        aromatic_ids = {
            atom_id
            for atom_id in heavy_ids
            if atoms_by_id[atom_id].is_aromatic
        }
        rings = [
            group
            for group in _connected_components(aromatic_ids, atoms_by_id)
            if len(group) >= 3
        ]

    systems: list[set[int]] = []
    for ring in rings:
        merged = set(ring)
        changed = True
        while changed:
            changed = False
            remaining: list[set[int]] = []
            for system in systems:
                if merged & system:
                    merged.update(system)
                    changed = True
                else:
                    remaining.append(system)
            systems = remaining
        systems.append(merged)
    return sorted(systems, key=lambda item: (min(item), len(item)))


def _functional_group_candidates(
    heavy_ids: set[int],
    atoms_by_id: dict[int, AtomRecord],
    bond_orders: dict[frozenset[int], float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add(
        priority: int,
        block_type: str,
        atom_ids,
        labels=(),
    ) -> None:
        selected = {int(atom_id) for atom_id in atom_ids if int(atom_id) in heavy_ids}
        if selected:
            candidates.append(
                {
                    "priority": priority,
                    "block_type": block_type,
                    "atom_ids": selected,
                    "labels": set(labels),
                }
            )

    for atom_id in sorted(heavy_ids):
        atom = atoms_by_id[atom_id]
        neighbors = _heavy_neighbors(atom_id, atoms_by_id, heavy_ids)
        if atom.element == "P":
            oxygens = [
                neighbor_id
                for neighbor_id in neighbors
                if atoms_by_id[neighbor_id].element == "O"
            ]
            if len(oxygens) >= 2:
                add(10, "phosphate", {atom_id, *oxygens}, {"charged", "polar", "acceptor"})

        if atom.element == "S":
            oxygens = [
                neighbor_id
                for neighbor_id in neighbors
                if atoms_by_id[neighbor_id].element == "O"
            ]
            nitrogens = [
                neighbor_id
                for neighbor_id in neighbors
                if atoms_by_id[neighbor_id].element == "N"
            ]
            if len(oxygens) >= 2:
                block_type = "sulfonamide" if nitrogens else "sulfonate"
                add(
                    15,
                    block_type,
                    {atom_id, *oxygens, *nitrogens},
                    {"charged", "polar", "acceptor"},
                )
            elif oxygens:
                add(65, "sulfoxide", {atom_id, *oxygens}, {"polar", "acceptor"})
            elif any(atoms_by_id[neighbor_id].element == "C" for neighbor_id in neighbors):
                add(80, "thioether", {atom_id}, {"hydrophobic"})

        if atom.element == "C":
            all_oxygen_neighbors = [
                neighbor_id
                for neighbor_id in neighbors
                if atoms_by_id[neighbor_id].element == "O"
            ]
            oxygen_neighbors = [
                neighbor_id
                for neighbor_id in all_oxygen_neighbors
                if _is_carbonyl_like(atom_id, neighbor_id, atoms_by_id, bond_orders)
            ]
            nitrogen_neighbors = [
                neighbor_id
                for neighbor_id in neighbors
                if atoms_by_id[neighbor_id].element == "N"
            ]
            if oxygen_neighbors and len(all_oxygen_neighbors) >= 2:
                add(
                    20,
                    "carboxylate",
                    {atom_id, *all_oxygen_neighbors},
                    {"charged", "polar", "acceptor"},
                )
            elif oxygen_neighbors and nitrogen_neighbors:
                add(
                    30,
                    "amide",
                    {atom_id, *oxygen_neighbors, *nitrogen_neighbors},
                    {"polar", "donor", "acceptor"},
                )
            elif len(nitrogen_neighbors) >= 2:
                add(
                    35,
                    "amidine_guanidine",
                    {atom_id, *nitrogen_neighbors},
                    {"charged", "polar", "donor"},
                )
            elif oxygen_neighbors:
                add(
                    60,
                    "carbonyl",
                    {atom_id, *oxygen_neighbors},
                    {"polar", "acceptor"},
                )
            for nitrogen_id in nitrogen_neighbors:
                if _bond_order(atom_id, nitrogen_id, bond_orders) >= 2.5:
                    add(40, "nitrile", {atom_id, nitrogen_id}, {"polar", "acceptor"})

        if atom.element == "N":
            if atom.formal_charge > 0 or atom.donor_capacity > 0:
                add(50, "amine", {atom_id}, {"charged" if atom.formal_charge > 0 else "polar", "donor"})

        if atom.element == "O":
            carbon_neighbors = [
                neighbor_id
                for neighbor_id in neighbors
                if atoms_by_id[neighbor_id].element == "C"
            ]
            if any(_is_carbonyl_like(neighbor_id, atom_id, atoms_by_id, bond_orders) for neighbor_id in carbon_neighbors):
                continue
            if atom.donor_capacity > 0 or atom.hydrogen_count > 0:
                add(55, "hydroxyl", {atom_id}, {"polar", "donor", "acceptor"})
            elif len(neighbors) >= 2:
                add(75, "ether", {atom_id}, {"polar", "acceptor"})

        if atom.element in _HALOGEN_ELEMENTS:
            add(90, "halogen_substituent", {atom_id}, {"hydrophobic", "halogen"})

    candidates.sort(
        key=lambda item: (
            int(item["priority"]),
            -len(item["atom_ids"]),
            str(item["block_type"]),
            min(item["atom_ids"]),
        )
    )
    return candidates


def _hydrophobic_substituent_groups(
    unassigned: set[int],
    atoms_by_id: dict[int, AtomRecord],
    bond_orders: dict[frozenset[int], float],
) -> list[set[int]]:
    hydrophobic_ids = {
        atom_id
        for atom_id in unassigned
        if _is_hydrophobic_atom(atoms_by_id[atom_id])
    }
    groups: list[set[int]] = []
    visited: set[int] = set()
    for start_id in sorted(hydrophobic_ids):
        if start_id in visited:
            continue
        group = set()
        queue = deque([start_id])
        visited.add(start_id)
        while queue:
            atom_id = queue.popleft()
            group.add(atom_id)
            for neighbor_id in sorted(atoms_by_id[atom_id].bonds):
                if neighbor_id not in hydrophobic_ids or neighbor_id in visited:
                    continue
                if _is_split_hydrophobic_bond(
                    atom_id,
                    neighbor_id,
                    atoms_by_id,
                    bond_orders,
                ):
                    continue
                visited.add(neighbor_id)
                queue.append(neighbor_id)
        if group:
            groups.append(group)
    return groups


def _connected_components(
    atom_ids: set[int],
    atoms_by_id: dict[int, AtomRecord],
) -> list[set[int]]:
    groups: list[set[int]] = []
    visited: set[int] = set()
    for start_id in sorted(atom_ids):
        if start_id in visited:
            continue
        group = set()
        queue = deque([start_id])
        visited.add(start_id)
        while queue:
            atom_id = queue.popleft()
            group.add(atom_id)
            for neighbor_id in sorted(atoms_by_id[atom_id].bonds):
                if neighbor_id in atom_ids and neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)
        groups.append(group)
    return groups


def _merge_trivial_singletons(
    seeds: list[dict[str, Any]],
    atoms_by_id: dict[int, AtomRecord],
) -> None:
    atom_to_seed: dict[int, int] = {}
    for index, seed in enumerate(seeds):
        for atom_id in seed["atom_ids"]:
            atom_to_seed[atom_id] = index

    for index, seed in enumerate(list(seeds)):
        atom_ids = set(seed["atom_ids"])
        if len(atom_ids) != 1:
            continue
        atom_id = next(iter(atom_ids))
        atom = atoms_by_id[atom_id]
        if _is_special_singleton(atom, atoms_by_id):
            continue
        neighbor_seed_ids = {
            atom_to_seed[neighbor_id]
            for neighbor_id in atom.bonds
            if neighbor_id in atom_to_seed and atom_to_seed[neighbor_id] != index
        }
        if not neighbor_seed_ids:
            continue
        target_index = max(
            neighbor_seed_ids,
            key=lambda seed_index: len(seeds[seed_index]["atom_ids"]),
        )
        seeds[target_index]["atom_ids"].add(atom_id)
        seeds[target_index]["labels"].update(seed.get("labels", set()))
        seed["atom_ids"].clear()
        atom_to_seed[atom_id] = target_index

    seeds[:] = [seed for seed in seeds if seed["atom_ids"]]


def _finalize_ligand_blocks(
    seeds: list[dict[str, Any]],
    atoms_by_id: dict[int, AtomRecord],
    bond_orders: dict[frozenset[int], float],
    rdkit_context: tuple[Any, dict[int, int]] | None,
    interactions: list[AssignedInteraction],
    *,
    ligand_id: str,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for index, seed in enumerate(
        sorted(seeds, key=lambda item: (min(item["atom_ids"]), len(item["atom_ids"])))
    ):
        atom_ids = tuple(sorted(seed["atom_ids"]))
        labels = sorted(_chemistry_labels(seed["block_type"], atom_ids, atoms_by_id) | set(seed.get("labels", ())))
        centroid = _centroid([atoms_by_id[atom_id].coord for atom_id in atom_ids])
        radius = max((_distance(centroid, atoms_by_id[atom_id].coord) for atom_id in atom_ids), default=0.0)
        interaction_counts = _block_interaction_counts(
            set(atom_ids),
            interactions,
            atoms_by_id,
            ligand_id=ligand_id,
        )
        blocks.append(
            {
                "block_id": f"{ligand_id}:block:{index}",
                "block_type": str(seed["block_type"]),
                "source": str(seed.get("source") or "graph"),
                "atom_ids": list(atom_ids),
                "centroid": [float(value) for value in centroid],
                "radius": float(radius),
                "chemistry_labels": labels,
                "charge_class": _charge_class(atom_ids, atoms_by_id),
                "donor_count": sum(atoms_by_id[atom_id].donor_capacity for atom_id in atom_ids),
                "acceptor_count": sum(atoms_by_id[atom_id].acceptor_capacity for atom_id in atom_ids),
                "hydrophobe_count": sum(1 for atom_id in atom_ids if atoms_by_id[atom_id].is_hydrophobe),
                "aromatic_count": sum(1 for atom_id in atom_ids if atoms_by_id[atom_id].is_aromatic),
                "halogen_count": sum(1 for atom_id in atom_ids if atoms_by_id[atom_id].element in _HALOGEN_ELEMENTS),
                "heavy_atom_count": len(atom_ids),
                "fingerprint": (
                    _rdkit_block_fingerprint(rdkit_context, atom_ids)
                    or _block_fingerprint(atom_ids, labels, atoms_by_id, bond_orders)
                ),
                "interaction_counts": interaction_counts,
                "interaction_count": sum(interaction_counts.values()),
                "attachments": _block_attachments(set(atom_ids), atoms_by_id),
            }
        )
    return blocks


def _block_interaction_counts(
    block_atom_ids: set[int],
    interactions: list[AssignedInteraction],
    atoms_by_id: dict[int, AtomRecord],
    *,
    ligand_id: str,
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for interaction in interactions:
        if ligand_id not in interaction.component_ids:
            continue
        family = _canonical_family(interaction.interaction_type)
        if family is None:
            continue
        ligand_atom_ids = {
            atom_id
            for atom_id in interaction.atom_ids
            if atoms_by_id.get(atom_id) is not None
            and atoms_by_id[atom_id].component_id == ligand_id
        }
        if block_atom_ids & ligand_atom_ids:
            counts[family] += 1
    return {key: float(counts[key]) for key in sorted(counts)}


def _bond_order_lookup(component: ComponentRecord) -> dict[frozenset[int], float]:
    lookup: dict[frozenset[int], float] = {}
    for record in component.metadata.get("bond_orders", []):
        try:
            atom_id_1 = int(record["atom_id_1"])
            atom_id_2 = int(record["atom_id_2"])
            order = float(record.get("order", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        lookup[frozenset((atom_id_1, atom_id_2))] = order
    return lookup


def _bond_order(
    atom_id_1: int,
    atom_id_2: int,
    bond_orders: dict[frozenset[int], float],
) -> float:
    return bond_orders.get(frozenset((atom_id_1, atom_id_2)), 1.0)


def _heavy_neighbors(
    atom_id: int,
    atoms_by_id: dict[int, AtomRecord],
    allowed_ids: set[int] | None = None,
) -> list[int]:
    allowed = allowed_ids or set(atoms_by_id)
    return [
        neighbor_id
        for neighbor_id in atoms_by_id[atom_id].bonds
        if neighbor_id in allowed
        and atoms_by_id.get(neighbor_id) is not None
        and atoms_by_id[neighbor_id].element != "H"
    ]


def _is_carbonyl_like(
    carbon_id: int,
    oxygen_id: int,
    atoms_by_id: dict[int, AtomRecord],
    bond_orders: dict[frozenset[int], float],
) -> bool:
    carbon = atoms_by_id[carbon_id]
    oxygen = atoms_by_id[oxygen_id]
    if carbon.element != "C" or oxygen.element != "O":
        return False
    if _bond_order(carbon_id, oxygen_id, bond_orders) >= 1.5:
        return True
    return oxygen.formal_charge < 0 and oxygen.acceptor_capacity > 0


def _is_hydrophobic_atom(atom: AtomRecord) -> bool:
    return (
        atom.element in _HYDROPHOBIC_ELEMENTS
        or atom.is_hydrophobe
    ) and atom.formal_charge == 0 and atom.donor_capacity == 0 and atom.acceptor_capacity == 0


def _is_split_hydrophobic_bond(
    atom_id_1: int,
    atom_id_2: int,
    atoms_by_id: dict[int, AtomRecord],
    bond_orders: dict[frozenset[int], float],
) -> bool:
    atom_1 = atoms_by_id[atom_id_1]
    atom_2 = atoms_by_id[atom_id_2]
    if atom_1.element in _HALOGEN_ELEMENTS or atom_2.element in _HALOGEN_ELEMENTS:
        return False
    if atom_1.is_aromatic or atom_2.is_aromatic:
        return False
    if _bond_order(atom_id_1, atom_id_2, bond_orders) > 1.1:
        return False
    return (
        len(_heavy_neighbors(atom_id_1, atoms_by_id)) > 1
        and len(_heavy_neighbors(atom_id_2, atoms_by_id)) > 1
    )


def _linker_block_type(
    atom_ids: set[int],
    atoms_by_id: dict[int, AtomRecord],
) -> tuple[str, set[str]]:
    labels: set[str] = set()
    atoms = [atoms_by_id[atom_id] for atom_id in atom_ids]
    if any(atom.formal_charge for atom in atoms):
        labels.add("charged")
    if any(atom.donor_capacity or atom.acceptor_capacity for atom in atoms):
        labels.add("polar")
    if any(atom.is_hydrophobe or atom.element in _HYDROPHOBIC_ELEMENTS for atom in atoms):
        labels.add("hydrophobic")
    if any(atom.is_aromatic for atom in atoms):
        labels.add("aromatic")
    if labels == {"hydrophobic"}:
        return "hydrophobic_linker", labels
    if "polar" in labels or "charged" in labels:
        return "polar_linker", labels
    return "mixed_linker", labels or {"mixed"}


def _is_special_singleton(
    atom: AtomRecord,
    atoms_by_id: dict[int, AtomRecord],
) -> bool:
    if atom.formal_charge != 0:
        return True
    if atom.donor_capacity > 0 or atom.acceptor_capacity > 0:
        return True
    if atom.element in _HALOGEN_ELEMENTS:
        return True
    heavy_degree = len(_heavy_neighbors(atom.atom_id, atoms_by_id))
    return atom.is_hydrophobe and heavy_degree <= 1


def _chemistry_labels(
    block_type: str,
    atom_ids: tuple[int, ...],
    atoms_by_id: dict[int, AtomRecord],
) -> set[str]:
    labels: set[str] = set()
    atoms = [atoms_by_id[atom_id] for atom_id in atom_ids]
    if block_type in {"ring_system"}:
        labels.add("ring")
    if any(atom.is_aromatic for atom in atoms):
        labels.add("aromatic")
    if any(atom.is_hydrophobe or atom.element in _HYDROPHOBIC_ELEMENTS for atom in atoms):
        labels.add("hydrophobic")
    if any(atom.donor_capacity > 0 for atom in atoms):
        labels.add("donor")
    if any(atom.acceptor_capacity > 0 for atom in atoms):
        labels.add("acceptor")
    if any(atom.formal_charge > 0 for atom in atoms):
        labels.add("positive")
        labels.add("charged")
    if any(atom.formal_charge < 0 for atom in atoms):
        labels.add("negative")
        labels.add("charged")
    if any(atom.element in _HALOGEN_ELEMENTS for atom in atoms):
        labels.add("halogen")
    if any(atom.element not in _HYDROPHOBIC_ELEMENTS for atom in atoms):
        labels.add("polar")
    if block_type in {"amide", "carbonyl", "hydroxyl", "ether", "nitrile", "sulfoxide"}:
        labels.add("polar")
    if block_type in {"carboxylate", "phosphate", "sulfonate", "amidine_guanidine"}:
        labels.add("charged")
    return labels


def _charge_class(
    atom_ids: tuple[int, ...],
    atoms_by_id: dict[int, AtomRecord],
) -> str:
    total = sum(atoms_by_id[atom_id].formal_charge for atom_id in atom_ids)
    has_positive = any(atoms_by_id[atom_id].formal_charge > 0 for atom_id in atom_ids)
    has_negative = any(atoms_by_id[atom_id].formal_charge < 0 for atom_id in atom_ids)
    if has_positive and has_negative:
        return "zwitterionic"
    if total > 0:
        return "positive"
    if total < 0:
        return "negative"
    return "neutral"


def _rdkit_fingerprint_context(
    component: ComponentRecord,
) -> tuple[Any, dict[int, int]] | None:
    smiles = component.metadata.get("smiles")
    mapping = component.metadata.get("template_mapping")
    if not smiles or not isinstance(mapping, dict):
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    template_index_by_atom_id: dict[int, int] = {}
    for template_index, atom_id in mapping.items():
        try:
            template_index_by_atom_id[int(atom_id)] = int(template_index)
        except (TypeError, ValueError):
            continue
    if not template_index_by_atom_id:
        return None
    return molecule, template_index_by_atom_id


def _rdkit_block_fingerprint(
    rdkit_context: tuple[Any, dict[int, int]] | None,
    atom_ids: tuple[int, ...],
) -> list[str]:
    if rdkit_context is None:
        return []
    molecule, template_index_by_atom_id = rdkit_context
    from_atoms = [
        template_index_by_atom_id[atom_id]
        for atom_id in atom_ids
        if atom_id in template_index_by_atom_id
    ]
    if not from_atoms:
        return []
    try:
        from rdkit import rdBase
        from rdkit.Chem import rdMolDescriptors

        blocker = rdBase.BlockLogs()
        try:
            fingerprint = rdMolDescriptors.GetMorganFingerprint(
                molecule,
                2,
                fromAtoms=from_atoms,
            )
        finally:
            del blocker
        return sorted(
            f"rdkit-morgan:{bit}:{count}"
            for bit, count in fingerprint.GetNonzeroElements().items()
        )
    except Exception:
        return []


def _block_fingerprint(
    atom_ids: tuple[int, ...],
    labels: list[str],
    atoms_by_id: dict[int, AtomRecord],
    bond_orders: dict[frozenset[int], float],
) -> list[str]:
    atom_id_set = set(atom_ids)
    descriptors: list[str] = [f"label:{label}" for label in labels]
    for atom_id in atom_ids:
        atom = atoms_by_id[atom_id]
        internal_degree = sum(1 for neighbor_id in atom.bonds if neighbor_id in atom_id_set)
        descriptors.append(
            "|".join(
                (
                    "atom",
                    f"el={atom.element}",
                    f"q={_charge_class((atom_id,), atoms_by_id)}",
                    f"aro={int(atom.is_aromatic)}",
                    f"don={int(atom.donor_capacity > 0)}",
                    f"acc={int(atom.acceptor_capacity > 0)}",
                    f"hyd={int(atom.is_hydrophobe)}",
                    f"deg={internal_degree}",
                )
            )
        )
    for atom_id in atom_ids:
        atom = atoms_by_id[atom_id]
        for neighbor_id in atom.bonds:
            if neighbor_id in atom_id_set and atom_id < neighbor_id:
                pair = sorted((atom.element, atoms_by_id[neighbor_id].element))
                descriptors.append(
                    f"bond:{pair[0]}-{pair[1]}:{_bond_order(atom_id, neighbor_id, bond_orders):.1f}"
                )
    return sorted(_hash_token(f"ligand-block|{descriptor}")[:16] for descriptor in descriptors)


def _block_attachments(
    atom_ids: set[int],
    atoms_by_id: dict[int, AtomRecord],
) -> list[int]:
    attachments = {
        neighbor_id
        for atom_id in atom_ids
        for neighbor_id in atoms_by_id[atom_id].bonds
        if neighbor_id not in atom_ids
        and atoms_by_id.get(neighbor_id) is not None
        and atoms_by_id[neighbor_id].element != "H"
    }
    return sorted(attachments)


def _centroid(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if not points:
        return (0.0, 0.0, 0.0)
    count = float(len(points))
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def _distance(left, right) -> float:
    return (
        (float(left[0]) - float(right[0])) ** 2
        + (float(left[1]) - float(right[1])) ** 2
        + (float(left[2]) - float(right[2])) ** 2
    ) ** 0.5


def payload_has_cosplif(payload: dict[str, Any]) -> bool:
    cosplif = payload.get("cosplif")
    return (
        isinstance(cosplif, dict)
        and cosplif.get("version") == COSPLIF_VERSION
        and isinstance(cosplif.get("tokens"), dict)
    )


def batch_has_cosplif(batch_payload: dict[str, Any]) -> bool:
    results = batch_payload.get("results", {})
    return bool(results) and all(payload_has_cosplif(payload) for payload in results.values())


def score_cosplif_batch(
    batch_payload: dict[str, Any],
    reference_ligand_id: str,
    profile_name: str = "cosplif_contrast_v1",
    profile: CoSPLIFProfile | None = None,
) -> list[dict[str, Any]]:
    if profile is None:
        profile = cosplif_profile(profile_name)
    entries = _cosplif_entries(batch_payload)
    complexes = {
        ligand_id: entry["tokens"]
        for ligand_id, entry in entries.items()
    }
    if reference_ligand_id not in complexes:
        raise ValueError(f"Reference ligand not found for CoSPLIF: {reference_ligand_id}")
    reference = complexes[reference_ligand_id]
    reference_payload = entries[reference_ligand_id]["cosplif"]
    background = [
        tokens for ligand_id, tokens in complexes.items()
        if ligand_id != reference_ligand_id
    ]
    weights = (
        _background_token_weights(background, reference.keys(), profile)
        if profile.use_background_idf
        else defaultdict(lambda: 1.0)
    )

    rows: list[dict[str, Any]] = []
    for ligand_id in sorted(complexes):
        query = complexes[ligand_id]
        score = _score_token_maps(reference, query, weights, profile)
        warnings = [] if query else ["CoSPLIF query has zero tokens"]
        row = {
            "ligand_id": ligand_id,
            "cosplif_score": score["score"],
            "cosplif_containment": score["containment"],
            "cosplif_tanimoto": score["tanimoto"],
            "cosplif_token_overlap": score["token_overlap"],
            "cosplif_reference_token_count": sum(reference.values()),
            "cosplif_query_token_count": sum(query.values()),
            "warnings": warnings,
        }
        if profile.ligand_block_weight > 0.0:
            block_score = _score_ligand_blocks(
                reference_payload,
                entries[ligand_id]["cosplif"],
                profile,
            )
            base_score = float(score["score"])
            final_score = base_score
            if block_score["available"]:
                block_weight = min(max(profile.ligand_block_weight, 0.0), 1.0)
                final_score = (
                    (1.0 - block_weight) * base_score
                    + block_weight * float(block_score["score"])
                )
            row.update(
                {
                    "cosplif_score": final_score,
                    "cosplif_base_score": base_score,
                    "cosplif_ligand_block_score": block_score["score"],
                    "cosplif_block_matches": block_score["match_count"],
                    "cosplif_block_match_details": block_score["matches"],
                    "cosplif_reference_block_count": block_score["reference_block_count"],
                    "cosplif_query_block_count": block_score["query_block_count"],
                }
            )
            row["warnings"].extend(block_score["warnings"])
        rows.append(row)
    return rows


def _cosplif_complexes(batch_payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        ligand_id: entry["tokens"]
        for ligand_id, entry in _cosplif_entries(batch_payload).items()
    }


def _cosplif_entries(batch_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for payload in batch_payload.get("results", {}).values():
        ligand_id = payload.get("ligand_id")
        if not ligand_id:
            continue
        if not payload_has_cosplif(payload):
            raise ValueError(
                "CoSPLIF scoring requires cosplif_context cache entries. "
                "Regenerate analysis with --fingerprint-method cosplif and --bootstrap-cache."
            )
        tokens = {
            str(token): int(count)
            for token, count in payload["cosplif"]["tokens"].items()
            if int(count) > 0
        }
        entries[str(ligand_id)] = {
            "tokens": tokens,
            "cosplif": payload["cosplif"],
        }
    return entries


def _score_token_maps(
    reference: dict[str, int],
    query: dict[str, int],
    weights,
    profile: CoSPLIFProfile,
) -> dict[str, float | int]:
    intersection = 0.0
    token_overlap = 0
    for token, reference_count in reference.items():
        shared = min(reference_count, query.get(token, 0))
        if shared <= 0:
            continue
        token_overlap += shared
        intersection += shared * float(weights[token])
    reference_total = sum(count * float(weights[token]) for token, count in reference.items())
    query_total = sum(count * float(weights[token]) for token, count in query.items())
    containment = intersection / reference_total if reference_total > 0 else 0.0
    denominator = reference_total + query_total - intersection
    tanimoto = intersection / denominator if denominator > 0 else 0.0
    if profile.score_mode == "containment":
        score = containment
    else:
        score = (
            profile.containment_weight * containment
            + profile.tanimoto_weight * tanimoto
        )
    return {
        "score": score,
        "containment": containment,
        "tanimoto": tanimoto,
        "token_overlap": token_overlap,
    }


def _score_ligand_blocks(
    reference_payload: dict[str, Any],
    query_payload: dict[str, Any],
    profile: CoSPLIFProfile,
) -> dict[str, Any]:
    reference_context = _ligand_block_context(reference_payload)
    query_context = _ligand_block_context(query_payload)
    warnings: list[str] = []
    if reference_context is None:
        return _empty_ligand_block_score(
            "CoSPLIF v2 requires ligand_blocks_v1 reference context; falling back to base score"
        )
    if query_context is None:
        return _empty_ligand_block_score(
            "CoSPLIF v2 requires ligand_blocks_v1 query context; falling back to base score"
        )

    warnings.extend(reference_context.get("warnings", []))
    warnings.extend(query_context.get("warnings", []))
    reference_blocks = [
        block
        for block in reference_context.get("blocks", [])
        if _block_interaction_total(block) > 0.0
    ]
    query_blocks = list(query_context.get("blocks", []))
    if not reference_blocks:
        return _empty_ligand_block_score(
            "CoSPLIF v2 reference has no interaction-supported ligand blocks; falling back to base score",
            reference_block_count=0,
            query_block_count=len(query_blocks),
        )

    candidate_pairs: list[tuple[float, int, int, dict[str, Any]]] = []
    for ref_index, reference_block in enumerate(reference_blocks):
        ref_centroid = reference_block.get("centroid")
        if not _is_vec3(ref_centroid):
            continue
        for query_index, query_block in enumerate(query_blocks):
            query_centroid = query_block.get("centroid")
            if not _is_vec3(query_centroid):
                continue
            centroid_distance = _distance(ref_centroid, query_centroid)
            if centroid_distance > profile.ligand_block_spatial_cutoff:
                continue
            pair = _ligand_block_pair_score(
                reference_block,
                query_block,
                centroid_distance,
                profile,
            )
            if pair["score"] <= 0.0:
                continue
            candidate_pairs.append(
                (float(pair["score"]), ref_index, query_index, pair)
            )

    candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_reference: set[int] = set()
    used_query: set[int] = set()
    matched_score = 0.0
    matches: list[dict[str, Any]] = []
    for score, ref_index, query_index, pair in candidate_pairs:
        if ref_index in used_reference or query_index in used_query:
            continue
        used_reference.add(ref_index)
        used_query.add(query_index)
        matched_score += score
        reference_block = reference_blocks[ref_index]
        query_block = query_blocks[query_index]
        matches.append(
            {
                "reference_block_id": reference_block.get("block_id"),
                "query_block_id": query_block.get("block_id"),
                "reference_block_type": reference_block.get("block_type"),
                "query_block_type": query_block.get("block_type"),
                "distance": pair["distance"],
                "chemical_similarity": pair["chemical_similarity"],
                "interaction_support_score": pair["interaction_support_score"],
                "size_similarity": pair["size_similarity"],
                "pair_score": pair["score"],
            }
        )

    max_score = sum(_reference_block_weight(block) for block in reference_blocks)
    ligand_block_score = matched_score / max_score if max_score > 0.0 else 0.0
    ligand_block_score = min(1.0, max(0.0, ligand_block_score))
    return {
        "available": True,
        "score": ligand_block_score,
        "match_count": len(matches),
        "matches": matches,
        "reference_block_count": len(reference_blocks),
        "query_block_count": len(query_blocks),
        "warnings": warnings,
    }


def _empty_ligand_block_score(
    warning: str,
    *,
    reference_block_count: int = 0,
    query_block_count: int = 0,
) -> dict[str, Any]:
    return {
        "available": False,
        "score": 0.0,
        "match_count": 0,
        "matches": [],
        "reference_block_count": reference_block_count,
        "query_block_count": query_block_count,
        "warnings": [warning],
    }


def _ligand_block_context(cosplif_payload: dict[str, Any]) -> dict[str, Any] | None:
    context = cosplif_payload.get("ligand_blocks_v1")
    if not isinstance(context, dict):
        return None
    if context.get("version") != LIGAND_BLOCKS_VERSION:
        return None
    blocks = context.get("blocks")
    if not isinstance(blocks, list):
        return None
    usable_blocks = [
        block for block in blocks
        if isinstance(block, dict)
        and block.get("atom_ids")
        and _is_vec3(block.get("centroid"))
    ]
    return {
        "blocks": usable_blocks,
        "warnings": list(context.get("warnings") or []),
    }


def _ligand_block_pair_score(
    reference_block: dict[str, Any],
    query_block: dict[str, Any],
    centroid_distance: float,
    profile: CoSPLIFProfile,
) -> dict[str, float]:
    chemical_similarity = _block_chemical_similarity(reference_block, query_block)
    if chemical_similarity <= 0.0:
        return {"score": 0.0}
    interaction_support = _weighted_tanimoto_counts(
        _float_counts(reference_block.get("interaction_counts")),
        _float_counts(query_block.get("interaction_counts")),
    )
    if interaction_support <= 0.0:
        return {"score": 0.0}
    size_similarity = _block_size_similarity(reference_block, query_block)
    if size_similarity <= 0.0:
        return {"score": 0.0}
    spatial_score = _distance_kernel(
        centroid_distance,
        _block_sigma(reference_block, profile),
    )
    reference_weight = _reference_block_weight(reference_block)
    score = (
        spatial_score
        * chemical_similarity
        * interaction_support
        * size_similarity
        * reference_weight
    )
    return {
        "score": score,
        "distance": centroid_distance,
        "chemical_similarity": chemical_similarity,
        "interaction_support_score": interaction_support,
        "size_similarity": size_similarity,
    }


def _block_chemical_similarity(
    reference_block: dict[str, Any],
    query_block: dict[str, Any],
) -> float:
    type_score = _block_type_similarity(reference_block, query_block)
    if type_score <= 0.0:
        return 0.0
    fingerprint_score = _tanimoto_sets(
        set(reference_block.get("fingerprint") or []),
        set(query_block.get("fingerprint") or []),
    )
    if fingerprint_score <= 0.0:
        return type_score
    return 0.7 * type_score + 0.3 * fingerprint_score


def _block_type_similarity(
    reference_block: dict[str, Any],
    query_block: dict[str, Any],
) -> float:
    ref_labels = set(reference_block.get("chemistry_labels") or [])
    qry_labels = set(query_block.get("chemistry_labels") or [])
    ref_charge = str(reference_block.get("charge_class") or "neutral")
    qry_charge = str(query_block.get("charge_class") or "neutral")
    if {ref_charge, qry_charge} == {"positive", "negative"}:
        return 0.0
    ref_donor_only = int(reference_block.get("donor_count") or 0) > 0 and int(reference_block.get("acceptor_count") or 0) == 0
    qry_acceptor_only = int(query_block.get("acceptor_count") or 0) > 0 and int(query_block.get("donor_count") or 0) == 0
    ref_acceptor_only = int(reference_block.get("acceptor_count") or 0) > 0 and int(reference_block.get("donor_count") or 0) == 0
    qry_donor_only = int(query_block.get("donor_count") or 0) > 0 and int(query_block.get("acceptor_count") or 0) == 0
    if (ref_donor_only and qry_acceptor_only) or (ref_acceptor_only and qry_donor_only):
        return 0.0

    if str(reference_block.get("block_type")) == str(query_block.get("block_type")):
        return 1.0
    if "charged" in ref_labels or "charged" in qry_labels:
        return 1.0 if ref_charge == qry_charge and ref_charge != "neutral" else 0.0
    ref_polarish = bool(ref_labels & {"polar", "donor", "acceptor"})
    qry_polarish = bool(qry_labels & {"polar", "donor", "acceptor"})
    if ref_polarish != qry_polarish:
        return 0.0
    if "aromatic" in ref_labels and "aromatic" in qry_labels:
        return 1.0
    if "hydrophobic" in ref_labels and "hydrophobic" in qry_labels:
        if "aromatic" in ref_labels or "aromatic" in qry_labels:
            return 0.6
        return 1.0
    if (
        ("aromatic" in ref_labels and "hydrophobic" in qry_labels)
        or ("hydrophobic" in ref_labels and "aromatic" in qry_labels)
    ):
        return 0.6
    if "polar" in ref_labels and "polar" in qry_labels:
        return 0.7
    return 0.0


def _block_size_similarity(
    reference_block: dict[str, Any],
    query_block: dict[str, Any],
) -> float:
    reference_size = max(int(reference_block.get("heavy_atom_count") or 0), 0)
    query_size = max(int(query_block.get("heavy_atom_count") or 0), 0)
    if reference_size <= 0 or query_size <= 0:
        return 0.0
    return min(reference_size, query_size) / max(reference_size, query_size)


def _reference_block_weight(block: dict[str, Any]) -> float:
    return max(1.0, _block_interaction_total(block))


def _block_interaction_total(block: dict[str, Any]) -> float:
    return sum(_float_counts(block.get("interaction_counts")).values())


def _float_counts(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, float] = {}
    for key, count in value.items():
        try:
            numeric = float(count)
        except (TypeError, ValueError):
            continue
        if numeric > 0.0:
            counts[str(key)] = numeric
    return counts


def _weighted_tanimoto_counts(
    reference_counts: dict[str, float],
    query_counts: dict[str, float],
) -> float:
    keys = set(reference_counts) | set(query_counts)
    if not keys:
        return 0.0
    numerator = sum(
        min(reference_counts.get(key, 0.0), query_counts.get(key, 0.0))
        for key in keys
    )
    denominator = sum(
        max(reference_counts.get(key, 0.0), query_counts.get(key, 0.0))
        for key in keys
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def _tanimoto_sets(reference: set[str], query: set[str]) -> float:
    if not reference and not query:
        return 0.0
    denominator = len(reference | query)
    return len(reference & query) / denominator if denominator else 0.0


def _block_sigma(block: dict[str, Any], profile: CoSPLIFProfile) -> float:
    labels = set(block.get("chemistry_labels") or [])
    if "metal" in labels or str(block.get("block_type")) == "metal":
        return profile.sigma_metal
    if "charged" in labels or str(block.get("charge_class")) != "neutral":
        return profile.sigma_charged
    if "aromatic" in labels:
        return profile.sigma_aromatic
    if "hydrophobic" in labels and "polar" not in labels:
        return profile.sigma_hydrophobic
    if "polar" in labels or "donor" in labels or "acceptor" in labels:
        return profile.sigma_polar
    return profile.sigma_fallback


def _distance_kernel(distance: float, sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0
    return exp(-(distance * distance) / (2.0 * sigma * sigma))


def _is_vec3(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    try:
        float(value[0])
        float(value[1])
        float(value[2])
    except (TypeError, ValueError):
        return False
    return True


def _background_token_weights(
    background: list[dict[str, int]],
    reference_tokens,
    profile: CoSPLIFProfile,
) -> defaultdict[str, float]:
    total = max(len(background), 1)
    document_frequency: Counter[str] = Counter()
    for tokens in background:
        document_frequency.update(tokens.keys())
    raw_idf = {
        token: log((total + 1.0) / (count + 1.0))
        for token, count in document_frequency.items()
    }
    mean_idf = sum(raw_idf.values()) / len(raw_idf) if raw_idf else 0.0

    def weight(token: str) -> float:
        value = raw_idf.get(token, log(total + 1.0))
        return min(
            profile.max_multiplier,
            max(
                profile.min_multiplier,
                exp(profile.idf_scale * (value - mean_idf)),
            ),
        )

    weighted_tokens = set(raw_idf) | set(reference_tokens)
    return defaultdict(lambda: 1.0, {token: weight(token) for token in weighted_tokens})


def _atom_environment(
    atom_id: int,
    atoms_by_id: dict[int, AtomRecord],
    components_by_id: dict[str, ComponentRecord],
    *,
    radius: int,
    resolution: str,
    target_moiety: str,
    target_class: str | None,
) -> str:
    visited = {atom_id}
    queue: deque[tuple[int, int]] = deque([(atom_id, 0)])
    by_shell: dict[int, list[str]] = defaultdict(list)
    while queue:
        current_id, depth = queue.popleft()
        atom = atoms_by_id.get(current_id)
        if atom is None:
            continue
        component = components_by_id.get(atom.component_id)
        by_shell[depth].append(
            _atom_descriptor(
                atom,
                component,
                resolution=resolution,
                target_moiety=target_moiety,
                target_class=target_class,
            )
        )
        if depth >= radius:
            continue
        for neighbor_id in sorted(atom.bonds):
            if neighbor_id in visited or neighbor_id not in atoms_by_id:
                continue
            visited.add(neighbor_id)
            queue.append((neighbor_id, depth + 1))
    return ";".join(
        f"{shell}:{','.join(sorted(descriptors))}"
        for shell, descriptors in sorted(by_shell.items())
    )


def _atom_descriptor(
    atom: AtomRecord,
    component: ComponentRecord | None,
    *,
    resolution: str,
    target_moiety: str,
    target_class: str | None,
) -> str:
    chemistry = ",".join(
        (
            f"el={atom.element}",
            f"q={atom.formal_charge}",
            f"aro={int(atom.is_aromatic)}",
            f"don={atom.donor_capacity}",
            f"acc={atom.acceptor_capacity}",
            f"hal={atom.halogen_donor_capacity}",
            f"hyd={int(atom.is_hydrophobe)}",
            f"deg={len(atom.bonds)}",
            f"h={atom.hydrogen_count}",
        )
    )
    if resolution == "ligand":
        return chemistry
    component = component or ComponentRecord("", "", "", "", "", ())
    if resolution == "specific":
        return ",".join(
            (
                f"res={component.name}",
                f"rid={component.chain_id}:{component.residue_id}",
                f"atom={atom.name}",
                chemistry,
            )
        )
    return ",".join(
        (
            f"kind={component.molecule_type}",
            f"class={target_class or 'unknown'}",
            f"moiety={target_moiety}",
            chemistry,
        )
    )


def _choose_target_component_id(
    interaction: AssignedInteraction,
    *,
    ligand_id: str,
    components_by_id: dict[str, ComponentRecord],
) -> str | None:
    candidates = [component_id for component_id in interaction.component_ids if component_id != ligand_id]
    if not candidates:
        return None
    priority = {"protein": 0, "nucleotide": 1, "ion": 2, "solvent": 3}
    return sorted(
        candidates,
        key=lambda component_id: (
            priority.get(components_by_id.get(component_id, ComponentRecord("", "", "", "", "", ())).molecule_type, 4),
            component_id,
        ),
    )[0]


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
    }.get(str(interaction_type))


def _canonical_subtype(interaction: AssignedInteraction, target_moiety: str) -> str | None:
    metadata = dict(interaction.metadata or {})
    explicit = metadata.get("interaction_subtype")
    if explicit:
        return str(explicit)
    interaction_type = str(interaction.interaction_type)
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


def _target_kind(component: ComponentRecord) -> str:
    if component.molecule_type == "nucleotide":
        return "nucleotide"
    if component.molecule_type == "solvent":
        return "water"
    if component.molecule_type == "ion":
        return "metal"
    return "residue"


def _target_moiety(
    target_kind: str,
    component: ComponentRecord,
    target_atom_ids: tuple[int, ...],
    atoms_by_id: dict[int, AtomRecord],
) -> str:
    if target_kind in {"metal", "water"}:
        return target_kind
    atom_names = [
        atoms_by_id[atom_id].name.strip().upper()
        for atom_id in target_atom_ids
        if atom_id in atoms_by_id
    ]
    if target_kind == "nucleotide":
        if any(name.startswith(("P", "OP", "O1P", "O2P")) for name in atom_names):
            return "phosphate"
        if atom_names and all("'" in name or "*" in name for name in atom_names):
            return "sugar"
        return "base"
    if atom_names and all(name in {"N", "CA", "C", "O", "OXT", "H", "HA"} for name in atom_names):
        return "backbone"
    return "sidechain"


def _target_class(component: ComponentRecord) -> str | None:
    if component.molecule_type == "protein":
        return _protein_residue_class(component.name)
    if component.molecule_type == "nucleotide":
        return _nucleotide_class(component.name)
    if component.molecule_type == "ion":
        return component.name or "metal"
    if component.molecule_type == "solvent":
        return "water"
    return component.name or None


def _protein_residue_class(name: str) -> str | None:
    name = name.upper()
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
    if name == "GLY":
        return "special"
    return None


def _nucleotide_class(name: str) -> str | None:
    name = name.upper()
    if name in {"A", "G", "DA", "DG", "ADE", "GUA"}:
        return "purine"
    if name in {"C", "U", "T", "DC", "DU", "DT", "CYT", "URA", "THY"}:
        return "pyrimidine"
    return None


def _role_key(
    interaction: AssignedInteraction,
    ligand_atom_ids: set[int],
    target_atom_ids: set[int],
) -> str:
    ligand_is_donor = False
    target_is_donor = False
    for demand in interaction.resource_demands:
        if demand.resource_type not in {"donor", "halogen_donor"}:
            continue
        if demand.atom_id in ligand_atom_ids:
            ligand_is_donor = True
        if demand.atom_id in target_atom_ids:
            target_is_donor = True
    if ligand_is_donor and target_is_donor:
        return "both_donor"
    if ligand_is_donor:
        return "ligand_donor"
    if target_is_donor:
        return "target_donor"
    return "none"


def _geometry_context(interaction: AssignedInteraction) -> str:
    metadata = dict(interaction.metadata or {})
    planarity = metadata.get("planarity")
    return ",".join(
        (
            f"dist={_bucket_label(interaction.distance, 0.5)}",
            f"angle={_bucket_label(interaction.angle, 20.0)}",
            f"offset={_bucket_label(interaction.offset, 0.5)}",
            f"planarity={_bucket_label(planarity, 0.25)}",
        )
    )


def _bucket_label(value: Any, width: float) -> str:
    if value is None:
        return "na"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "na"
    lower = floor(numeric / width) * width
    upper = lower + width
    return f"{lower:.2f}-{upper:.2f}"


def _hash_token(token_text: str) -> str:
    return sha1(token_text.encode("utf-8")).hexdigest()
