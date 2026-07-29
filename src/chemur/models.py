from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable

Vec3 = tuple[float, float, float]


@dataclass(slots=True)
class AtomRecord:
    atom_id: int
    name: str
    element: str
    coord: Vec3
    residue_name: str
    residue_id: str
    chain_id: str
    component_id: str
    molecule_type: str
    serial: int | None = None
    formal_charge: int = 0
    donor_capacity: int = 0
    acceptor_capacity: int = 0
    halogen_donor_capacity: int = 0
    chalcogen_donor_capacity: int = 0
    tetrel_donor_capacity: int = 0
    carbon_donor_capacity: int = 0
    hydrogen_count: int = 0
    is_aromatic: bool = False
    is_hydrophobe: bool = False
    # Whether a chalcogen's sigma-hole is actually positive. Geometry cannot tell:
    # at the C-S extension of an ordinary Met or Cys the potential is negative, so
    # nothing is there to donate into. Set during atom typing, gated on in detection.
    sigma_hole_activated: bool = False
    bonds: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return (
            f"{self.chain_id}:{self.residue_name}:{self.residue_id}:"
            f"{self.name}:{self.atom_id}"
        )


@dataclass(slots=True)
class ComponentRecord:
    component_id: str
    name: str
    chain_id: str
    residue_id: str
    molecule_type: str
    atom_ids: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureRecord:
    feature_id: int
    feature_type: str
    component_id: str
    atom_ids: tuple[int, ...]
    center: Vec3
    capacity: int = 0
    normal: Vec3 | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceDemand:
    resource_type: str
    atom_id: int
    amount: int = 1


@dataclass(slots=True)
class CandidateInteraction:
    interaction_type: str
    feature_ids: tuple[int, ...]
    atom_ids: tuple[int, ...]
    component_ids: tuple[str, ...]
    distance: float
    priority: int
    stage: str
    angle: float | None = None
    offset: float | None = None
    resource_demands: tuple[ResourceDemand, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str | None = None

    @property
    def is_intra_component(self) -> bool:
        return len(set(self.component_ids)) == 1


@dataclass(slots=True)
class AssignedInteraction:
    interaction_type: str
    feature_ids: tuple[int, ...]
    atom_ids: tuple[int, ...]
    component_ids: tuple[str, ...]
    distance: float
    priority: int
    stage: str
    assignment_order: int
    angle: float | None = None
    offset: float | None = None
    resource_demands: tuple[ResourceDemand, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_candidate(
        cls, candidate: CandidateInteraction, assignment_order: int
    ) -> "AssignedInteraction":
        return cls(
            interaction_type=candidate.interaction_type,
            feature_ids=candidate.feature_ids,
            atom_ids=candidate.atom_ids,
            component_ids=candidate.component_ids,
            distance=candidate.distance,
            priority=candidate.priority,
            stage=candidate.stage,
            assignment_order=assignment_order,
            angle=candidate.angle,
            offset=candidate.offset,
            resource_demands=candidate.resource_demands,
            metadata=dict(candidate.metadata),
        )


@dataclass(slots=True)
class AnalysisResult:
    atoms: list[AtomRecord]
    components: list[ComponentRecord]
    features: list[FeatureRecord]
    interactions: list[AssignedInteraction]
    raw_candidates: list[CandidateInteraction] = field(default_factory=list)
    profile_name: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        data = {
            "profile": self.profile_name,
            "metadata": self.metadata,
            "atoms": [asdict(atom) for atom in self.atoms],
            "components": [asdict(component) for component in self.components],
            "features": [asdict(feature) for feature in self.features],
            "interactions": [asdict(interaction) for interaction in self.interactions],
        }
        if include_raw:
            data["raw_candidates"] = [asdict(candidate) for candidate in self.raw_candidates]
        return data

    def to_json(self, path: str | Path | None = None, include_raw: bool = False) -> str:
        payload = json.dumps(self.to_dict(include_raw=include_raw), indent=2, sort_keys=True)
        if path is not None:
            Path(path).write_text(payload + "\n", encoding="utf-8")
        return payload

    def rows(self, include_raw: bool = False) -> list[dict[str, Any]]:
        atoms_by_id = atom_index(self.atoms)
        features_by_id = feature_index(self.features)
        assigned_rows = [
            _interaction_row(interaction, assigned=True, atoms_by_id=atoms_by_id, features_by_id=features_by_id)
            for interaction in self.interactions
        ]
        if not include_raw:
            return assigned_rows
        raw_rows = [
            _interaction_row(candidate, assigned=False, atoms_by_id=atoms_by_id, features_by_id=features_by_id)
            for candidate in self.raw_candidates
        ]
        return assigned_rows + raw_rows

    def to_csv(self, path: str | Path, include_raw: bool = False) -> None:
        rows = self.rows(include_raw=include_raw)
        fieldnames = [
            "interaction_type",
            "assigned",
            "stage",
            "priority",
            "component_ids",
            "atom_ids",
            "atom_labels",
            "feature_ids",
            "feature_types",
            "distance",
            "angle",
            "offset",
            "resource_demands",
            "resource_labels",
            "rejection_reason",
        ]
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def split_by_ligand(self, include_raw: bool = False) -> dict[str, "AnalysisResult"]:
        ligand_components = [
            component
            for component in self.components
            if component.molecule_type == "ligand"
            and not component.metadata.get("ignored_by_template")
        ]
        return {
            component.component_id: self._filtered_for_ligand(
                component,
                include_raw=include_raw,
            )
            for component in ligand_components
        }

    def write_ligand_outputs(
        self,
        output_dir: str | Path,
        *,
        include_raw: bool = False,
        output_format: str = "both",
    ) -> list[Path]:
        if output_format not in {"json", "csv", "both"}:
            raise ValueError("output_format must be one of: json, csv, both")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for ligand_id, ligand_result in self.split_by_ligand(include_raw=include_raw).items():
            slug = _slugify_component_id(ligand_id)
            if output_format in {"json", "both"}:
                json_path = output_path / f"{slug}.json"
                ligand_result.to_json(json_path, include_raw=include_raw)
                written.append(json_path)
            if output_format in {"csv", "both"}:
                csv_path = output_path / f"{slug}.csv"
                ligand_result.to_csv(csv_path, include_raw=include_raw)
                written.append(csv_path)
        return written

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required for AnalysisResult.to_dataframe()") from exc
        return pd.DataFrame(self.rows(include_raw=False))

    def _filtered_for_ligand(
        self,
        ligand: ComponentRecord,
        *,
        include_raw: bool,
    ) -> "AnalysisResult":
        ligand_id = ligand.component_id
        interactions = [
            interaction
            for interaction in self.interactions
            if ligand_id in interaction.component_ids
        ]
        raw_candidates = [
            candidate
            for candidate in self.raw_candidates
            if ligand_id in candidate.component_ids
        ] if include_raw else []

        component_ids = {ligand_id}
        atom_ids = set(ligand.atom_ids)
        feature_ids: set[int] = set()
        for interaction in [*interactions, *raw_candidates]:
            component_ids.update(interaction.component_ids)
            atom_ids.update(interaction.atom_ids)
            feature_ids.update(interaction.feature_ids)

        features = [feature for feature in self.features if feature.feature_id in feature_ids]
        for feature in features:
            component_ids.add(feature.component_id)
            atom_ids.update(feature.atom_ids)

        components = [
            component for component in self.components if component.component_id in component_ids
        ]
        atoms = [atom for atom in self.atoms if atom.atom_id in atom_ids]
        metadata = dict(self.metadata)
        metadata["split_by_ligand"] = True
        metadata["ligand_component_id"] = ligand.component_id
        metadata["ligand_name"] = ligand.name
        metadata["ligand_chain_id"] = ligand.chain_id
        metadata["ligand_residue_id"] = ligand.residue_id

        return AnalysisResult(
            atoms=atoms,
            components=components,
            features=features,
            interactions=interactions,
            raw_candidates=raw_candidates,
            profile_name=self.profile_name,
            metadata=metadata,
        )


def _interaction_row(
    interaction: AssignedInteraction | CandidateInteraction,
    assigned: bool,
    atoms_by_id: dict[int, AtomRecord] | None = None,
    features_by_id: dict[int, FeatureRecord] | None = None,
) -> dict[str, Any]:
    atoms_by_id = atoms_by_id or {}
    features_by_id = features_by_id or {}
    return {
        "interaction_type": interaction.interaction_type,
        "assigned": assigned,
        "stage": interaction.stage,
        "priority": interaction.priority,
        "component_ids": ";".join(interaction.component_ids),
        "atom_ids": ";".join(str(atom_id) for atom_id in interaction.atom_ids),
        "atom_labels": ";".join(
            atoms_by_id[atom_id].label if atom_id in atoms_by_id else str(atom_id)
            for atom_id in interaction.atom_ids
        ),
        "feature_ids": ";".join(str(feature_id) for feature_id in interaction.feature_ids),
        "feature_types": ";".join(
            features_by_id[feature_id].feature_type if feature_id in features_by_id else str(feature_id)
            for feature_id in interaction.feature_ids
        ),
        "distance": round(interaction.distance, 4),
        "angle": None if interaction.angle is None else round(interaction.angle, 4),
        "offset": None if interaction.offset is None else round(interaction.offset, 4),
        "resource_demands": ";".join(
            f"{demand.resource_type}:{demand.atom_id}:{demand.amount}"
            for demand in interaction.resource_demands
        ),
        "resource_labels": ";".join(
            _resource_label(demand, atoms_by_id)
            for demand in interaction.resource_demands
        ),
        "rejection_reason": getattr(interaction, "rejection_reason", None),
    }


def _resource_label(
    demand: ResourceDemand,
    atoms_by_id: dict[int, AtomRecord],
) -> str:
    atom_label = (
        atoms_by_id[demand.atom_id].label
        if demand.atom_id in atoms_by_id
        else str(demand.atom_id)
    )
    return f"{demand.resource_type}:{atom_label}:{demand.amount}"


def atom_index(atoms: Iterable[AtomRecord]) -> dict[int, AtomRecord]:
    return {atom.atom_id: atom for atom in atoms}


def feature_index(features: Iterable[FeatureRecord]) -> dict[int, FeatureRecord]:
    return {feature.feature_id: feature for feature in features}


def _slugify_component_id(component_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", component_id.strip())
    return slug.strip("_") or "ligand"
