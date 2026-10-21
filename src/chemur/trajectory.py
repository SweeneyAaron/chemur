"""MD-trajectory interaction analysis.

Computes Chemur interactions for every frame of a trajectory and exposes
time-resolved views (occupancy, per-frame counts, per-interaction time series).

Topology is parsed and ligands are templated **once** (both are coordinate
independent); only the coordinate-dependent stage
(:func:`chemur.api.assign_for_atoms`) reruns per frame. Frames can be processed
in parallel across CPUs with :mod:`multiprocessing`.

Two frame sources are provided:

* :class:`ArrayFrameSource` — topology file + an ``(n_frames, n_atoms, 3)`` array
  (or ``.npy`` path). Used by the ChimeraX plugin, which serialises the loaded
  trajectory's coordsets to disk.
* :class:`MDAnalysisFrameSource` — reads ``.gro``/``.xtc`` (and other formats)
  directly via MDAnalysis. Used by the CLI and tests. Requires the optional
  ``trajectory`` extra (``pip install chemur[trajectory]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
import tempfile
import os
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .api import assign_for_atoms
from .models import AssignedInteraction, AtomRecord, ComponentRecord
from .parser import parse_structure
from .profile import apply_rule_overrides, load_profile
from .geometry_catalogue import geometry_result_metadata
from .preparation import prepare_ligand_chemistry

# Molecule types the parser assigns; used for solvent/ion exclusion.
SOLVENT_TYPE = "solvent"
ION_TYPE = "ion"

InteractionKey = tuple[str, frozenset]


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------
class ArrayFrameSource:
    """Topology file plus an in-memory or on-disk coordinate stack.

    ``coords`` is either a NumPy array shaped ``(n_frames, n_atoms, 3)`` or a path
    to a ``.npy`` file with that shape. The atom order of every frame must match
    the order :func:`parse_structure` produces for ``topology`` (atom ``i`` is
    column ``i``). Coordinates are Angstrom.
    """

    def __init__(self, topology: str | Path, coords):
        self.topology = str(topology)
        if isinstance(coords, (str, Path)):
            self._coords_path = str(coords)
            self._coords = None
        else:
            self._coords_path = None
            self._coords = np.asarray(coords, dtype=np.float32)
        self._cached = None

    def __getstate__(self):
        # Don't pickle the (possibly large) memmap handle; workers reopen lazily.
        state = self.__dict__.copy()
        state["_cached"] = None
        if self._coords_path is not None:
            state["_coords"] = None
        return state

    def _array(self) -> np.ndarray:
        if self._cached is None:
            if self._coords_path is not None:
                self._cached = np.load(self._coords_path, mmap_mode="r")
            else:
                self._cached = self._coords
            if self._cached.ndim != 3 or self._cached.shape[2] != 3:
                raise ValueError(
                    "ArrayFrameSource coords must have shape (n_frames, n_atoms, 3); "
                    f"got {self._cached.shape}"
                )
        return self._cached

    def parse_topology(self) -> tuple[list[AtomRecord], list[ComponentRecord]]:
        return parse_structure(self.topology)

    @property
    def n_frames(self) -> int:
        return int(self._array().shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self._array().shape[1])

    def coords_for(self, frame_index: int) -> np.ndarray:
        return np.asarray(self._array()[frame_index], dtype=float)


class MDAnalysisFrameSource:
    """Read a trajectory (``.gro``/``.xtc`` etc.) directly via MDAnalysis.

    Original PDB/mmCIF chemistry is retained when deposited identities align to
    the trajectory topology. Other formats use a temporary PDB for coordinates
    plus the explicit MDAnalysis bond table; unknown bond orders remain unknown.
    Per-frame coordinates follow MDAnalysis topology order and are in Angstrom.

    PBC caveat: coordinates are used as stored in the trajectory. If the
    trajectory is wrapped into the primary unit cell, molecules can be split
    across periodic boundaries, which breaks distance-inferred bonds and inflates
    apparent contacts. Make molecules whole (e.g. ``gmx trjconv -pbc mol`` or an
    MDAnalysis ``unwrap`` transformation, which needs connectivity) before
    analysing. The ChimeraX plugin path is unaffected: it analyses exactly the
    coordinates ChimeraX displays.
    """

    def __init__(self, topology: str | Path, trajectory: str | Path):
        self.topology = str(topology)
        self.trajectory = str(trajectory)
        self._universe = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_universe"] = None  # Universe is not picklable; workers rebuild it.
        return state

    def _u(self):
        if self._universe is None:
            try:
                import MDAnalysis as mda
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "MDAnalysis is required to read trajectory files. Install it with "
                    "`pip install chemur[trajectory]` or `pip install MDAnalysis`."
                ) from exc
            self._universe = mda.Universe(self.topology, self.trajectory)
        return self._universe

    def parse_topology(self) -> tuple[list[AtomRecord], list[ComponentRecord]]:
        u = self._u()
        u.trajectory[0]

        def optional(record, name, default=None):
            try:
                return getattr(record, name)
            except (AttributeError, ValueError, KeyError):
                # Optional topology attributes raise MDAnalysis.NoDataError
                # (an AttributeError/ValueError subclass) when unavailable.
                return default

        source_path = Path(self.topology)
        if source_path.suffix.lower() in {".pdb", ".ent", ".cif", ".mmcif"} and source_path.is_file():
            # A PDB round-trip discards named dictionaries, branch leaving atoms,
            # and source identifiers. Use the original whenever it is alignable.
            source_atoms, components = parse_structure(source_path)
            atoms = []
            unused = {atom.atom_id: atom for atom in source_atoms}
            for index, topology_atom in enumerate(u.atoms):
                name = str(optional(topology_atom, "name", "")).strip()
                residue_name = str(optional(topology_atom, "resname", "")).strip().upper()
                residue = str(optional(topology_atom, "resid", "")).strip()
                insertion = str(optional(topology_atom, "icode", "") or "").strip(" .?")
                if insertion and not residue.endswith(insertion):
                    residue += insertion
                chain = str(optional(topology_atom, "chainID", "") or "").strip(" .?")
                altloc = str(optional(topology_atom, "altLoc", "") or "").strip("\x00 .?")
                candidates = [atom for atom in unused.values()
                              if atom.name == name and atom.residue_name == residue_name
                              and atom.residue_id == residue
                              and (not chain or atom.chain_id == chain)
                              and (not altloc or atom.metadata.get("source_altloc") == altloc)]
                serial = optional(topology_atom, "id")
                serial_matches = [atom for atom in candidates if atom.serial == serial]
                if len(candidates) != 1 and len(serial_matches) == 1:
                    candidates = serial_matches
                if len(candidates) != 1:
                    raise ValueError(
                        f"Cannot uniquely align MDAnalysis topology atom {index} "
                        f"({residue_name}:{residue}:{name}) to its source structure. "
                        "Source assembly chemistry cannot be transferred safely."
                    )
                atom = candidates[0]
                atoms.append(atom)
                unused.pop(atom.atom_id)
            if unused:
                raise ValueError("Topology atom count mismatch between source structure and MDAnalysis universe")
        else:
            handle = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
            handle.close()
            try:
                # CONECT output can turn unspecified MD bond order into a single
                # bond. Transfer the original bond table below instead.
                u.atoms.write(handle.name, bonds=None)
                atoms, components = parse_structure(handle.name)
            finally:
                try:
                    os.unlink(handle.name)
                except OSError:
                    pass
        if len(atoms) != len(u.atoms):
            raise ValueError(
                "Topology atom count mismatch between the parsed PDB "
                f"({len(atoms)}) and the MDAnalysis universe ({len(u.atoms)}). "
                "The trajectory cannot be aligned to the topology."
            )
        for index, atom in enumerate(atoms):
            atom.coord = tuple(float(value) for value in u.atoms.positions[index])

        from copy import deepcopy
        from .connectivity import _COORDINATION_METALS, _merge_links, _order

        by_component = {component.component_id: component for component in components}
        # Source links occur in both endpoint components. Merge unique records
        # once so repeated topology parsing does not multiply their provenance.
        existing_links = {json.dumps(link, sort_keys=True): deepcopy(link)
                          for component in components
                          for link in component.metadata.get("structure_links", ())}
        links = list(existing_links.values())
        topology_bonds = optional(u, "bonds")
        if topology_bonds is not None:
            for bond in topology_bonds:
                if optional(bond, "is_guessed", False):
                    continue
                indices = tuple(int(index) for index in bond.indices)
                if len(indices) != 2 or any(index < 0 or index >= len(atoms) for index in indices):
                    raise ValueError("MDAnalysis bond references an unavailable topology atom")
                first, second = (atoms[index] for index in indices)
                if (first.element in _COORDINATION_METALS or second.element in _COORDINATION_METALS
                        or first.molecule_type == "ion" or second.molecule_type == "ion"):
                    continue
                raw_order = optional(bond, "order")
                order = _order(raw_order)
                if order is None and raw_order is not None:
                    try:
                        numeric_order = float(raw_order)
                        order = numeric_order if numeric_order in {1., 1.5, 2., 3.} else None
                    except (TypeError, ValueError):
                        pass
                component_ids = list(dict.fromkeys((first.component_id, second.component_id)))
                link = {
                    "atom_ids": [first.atom_id, second.atom_id], "component_ids": component_ids,
                    "order": order, "leaving_atoms": [], "kind": "covalent",
                    "source": "mdanalysis_topology",
                    "provenance": [{"source": "mdanalysis_topology", "record_id": f"{indices[0]}-{indices[1]}"}],
                    "endpoints": [{"component_ids": [atom.component_id], "atom_names": [atom.name]}
                                  for atom in (first, second)],
                }
                if raw_order is not None and order is None:
                    link["error"] = f"Unsupported MDAnalysis covalent bond order: {raw_order}"
                if first.component_id != second.component_id:
                    links.append(link)
                elif order is not None:
                    component = by_component[first.component_id]
                    record = {"atom_id_1": first.atom_id, "atom_id_2": second.atom_id, "order": order}
                    orders = component.metadata.setdefault("bond_orders", [])
                    old = next((entry for entry in orders
                                if {entry["atom_id_1"], entry["atom_id_2"]} == {first.atom_id, second.atom_id}), None)
                    if old is None:
                        orders.append(record)
                    elif old["order"] != order:
                        link["error"] = "Conflicting MDAnalysis intracomponent bond orders"
                        links.append(link)
                if not link.get("error"):
                    first.bonds = tuple(sorted(set(first.bonds) | {second.atom_id}))
                    second.bonds = tuple(sorted(set(second.bonds) | {first.atom_id}))
        merged_links = _merge_links(links)
        for component in components:
            component.metadata.pop("structure_links", None)
        by_id = {atom.atom_id: atom for atom in atoms}
        for link in merged_links:
            for cid in link["component_ids"]:
                by_component[cid].metadata.setdefault("structure_links", []).append(deepcopy(link))
            if link.get("error") or len(link.get("atom_ids", ())) != 2:
                continue
            first, second = (by_id[aid] for aid in link["atom_ids"])
            if first.component_id != second.component_id:
                for atom, neighbor in ((first, second), (second, first)):
                    atom.metadata["assembly_link_neighbors"] = sorted(
                        set(atom.metadata.get("assembly_link_neighbors", ())) | {neighbor.atom_id}
                    )
        return atoms, components

    @property
    def n_frames(self) -> int:
        return int(len(self._u().trajectory))

    @property
    def n_atoms(self) -> int:
        return int(len(self._u().atoms))

    def coords_for(self, frame_index: int) -> np.ndarray:
        u = self._u()
        u.trajectory[frame_index]
        return np.asarray(u.atoms.positions, dtype=float)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FrameInteractions:
    """Interactions assigned for a single analysed frame."""

    frame_index: int
    interactions: list[AssignedInteraction]


@dataclass(slots=True)
class TrajectoryResult:
    """Time-resolved interaction analysis over a trajectory.

    ``atoms``/``components`` are the (filtered) topology; their coordinates hold
    the last analysed frame. ``frame_indices`` are the original trajectory frame
    numbers that were analysed (after start/stop/stride); ``frames`` are aligned
    to ``frame_indices``. Atom IDs retain their topology identity after filtering
    and may be noncontiguous; they are identifiers, not offsets into ``atoms``.
    """

    atoms: list[AtomRecord]
    components: list[ComponentRecord]
    frame_indices: list[int]
    frames: list[FrameInteractions]
    profile_name: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    _agg: dict[str, Any] | None = field(default=None, compare=False, repr=False)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @staticmethod
    def interaction_key(inter: AssignedInteraction) -> InteractionKey:
        """Use the enclosed centre for enclosures, and all atoms for other contacts."""
        if inter.interaction_type == "hydrophobic_enclosure":
            return (inter.interaction_type, frozenset({inter.metadata["center_atom_id"]}))
        return (inter.interaction_type, frozenset(inter.atom_ids))

    def aggregates(self) -> dict[str, Any]:
        """Per-key occupancy, label and mean distance, computed in one pass.

        Cached: building the occupancy table / plots queries these for thousands
        of distinct interactions, so a single O(frames x interactions) sweep here
        avoids re-scanning the trajectory per interaction (which was O(keys x
        frames x interactions) and dominated post-analysis time).
        """
        if self._agg is not None:
            return self._agg
        atoms_by_id = {a.atom_id: a for a in self.atoms}
        present: dict[InteractionKey, int] = {}
        dist_sum: dict[InteractionKey, float] = {}
        dist_n: dict[InteractionKey, int] = {}
        labels: dict[InteractionKey, str] = {}
        for frame in self.frames:
            seen: set[InteractionKey] = set()
            for inter in frame.interactions:
                key = self.interaction_key(inter)
                if key not in seen:
                    seen.add(key)
                    present[key] = present.get(key, 0) + 1
                dist_sum[key] = dist_sum.get(key, 0.0) + inter.distance
                dist_n[key] = dist_n.get(key, 0) + 1
                if key not in labels:
                    if inter.interaction_type == "hydrophobic_enclosure":
                        center_id = inter.metadata["center_atom_id"]
                        center = atoms_by_id.get(center_id)
                        labels[key] = center.label if center is not None else str(center_id)
                        continue
                    residues: list[str] = []
                    rseen: set[str] = set()
                    for atom_id in inter.atom_ids:
                        atom = atoms_by_id.get(atom_id)
                        if atom is None:
                            continue
                        tag = f"{atom.chain_id}:{atom.residue_name}:{atom.residue_id}"
                        if tag not in rseen:
                            rseen.add(tag)
                            residues.append(tag)
                    labels[key] = " <-> ".join(residues) if residues else "?"
        n = float(len(self.frames)) or 1.0
        occupancy = {k: present[k] / n for k in present}
        mean_distance = {k: dist_sum[k] / dist_n[k] for k in dist_sum}
        ordered = sorted(occupancy, key=lambda k: (-occupancy[k], labels.get(k, "")))
        self._agg = {
            "occupancy": occupancy,
            "labels": labels,
            "mean_distance": mean_distance,
            "ordered_keys": ordered,
        }
        return self._agg

    def _ordered_keys(self) -> list[InteractionKey]:
        """All keys seen, sorted by descending occupancy then label."""
        return self.aggregates()["ordered_keys"]

    def occupancy(self) -> dict[InteractionKey, float]:
        """Fraction of analysed frames in which each interaction is present."""
        return self.aggregates()["occupancy"]

    def timeseries(self, key: InteractionKey) -> list[dict[str, Any]]:
        """Per-frame geometry for one interaction key (None where absent)."""
        series: list[dict[str, Any]] = []
        for frame in self.frames:
            best: AssignedInteraction | None = None
            for inter in frame.interactions:
                if self.interaction_key(inter) == key:
                    if best is None or inter.distance < best.distance:
                        best = inter
            series.append(
                {
                    "frame": frame.frame_index,
                    "present": best is not None,
                    "distance": None if best is None else round(best.distance, 4),
                    "angle": None if best is None or best.angle is None else round(best.angle, 4),
                    "offset": None if best is None or best.offset is None else round(best.offset, 4),
                }
            )
            if key[0] == "hydrophobic_enclosure":
                series[-1]["metadata"] = {} if best is None else dict(best.metadata)
        return series

    def counts_per_frame(self) -> dict[str, Any]:
        """Total and per-type interaction counts for each analysed frame."""
        types = sorted({i.interaction_type for f in self.frames for i in f.interactions})
        total: list[int] = []
        by_type: dict[str, list[int]] = {t: [] for t in types}
        for frame in self.frames:
            total.append(len(frame.interactions))
            per: dict[str, int] = {}
            for inter in frame.interactions:
                per[inter.interaction_type] = per.get(inter.interaction_type, 0) + 1
            for t in types:
                by_type[t].append(per.get(t, 0))
        return {"frames": list(self.frame_indices), "total": total, "by_type": by_type}

    def interaction_labels(self) -> dict[InteractionKey, str]:
        """Residue-pair labels, or full centre-atom labels for enclosures."""
        return self.aggregates()["labels"]

    def rows(self) -> list[dict[str, Any]]:
        """Long-format rows (one per interaction per frame) with a ``frame`` column."""
        atoms_by_id = {a.atom_id: a for a in self.atoms}
        rows: list[dict[str, Any]] = []
        for frame in self.frames:
            for inter in frame.interactions:
                rows.append(
                    {
                        "frame": frame.frame_index,
                        "interaction_type": inter.interaction_type,
                        "component_ids": ";".join(inter.component_ids),
                        "atom_ids": ";".join(str(a) for a in inter.atom_ids),
                        "atom_labels": ";".join(
                            atoms_by_id[a].label if a in atoms_by_id else str(a)
                            for a in inter.atom_ids
                        ),
                        "distance": round(inter.distance, 4),
                        "angle": None if inter.angle is None else round(inter.angle, 4),
                        "offset": None if inter.offset is None else round(inter.offset, 4),
                        "metadata": dict(inter.metadata),
                    }
                )
        return rows

    def to_csv(self, path: str | Path) -> None:
        fieldnames = [
            "frame",
            "interaction_type",
            "component_ids",
            "atom_ids",
            "atom_labels",
            "distance",
            "angle",
            "offset",
            "metadata",
        ]
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {**row, "metadata": json.dumps(row["metadata"], sort_keys=True)}
                for row in self.rows()
            )

    def to_dict(self) -> dict[str, Any]:
        labels = self.interaction_labels()
        occ = self.occupancy()
        return {
            "profile": self.profile_name,
            "metadata": self.metadata,
            "frame_indices": list(self.frame_indices),
            "occupancy": [
                {"label": labels.get(k, "?"), "type": k[0], "occupancy": round(occ[k], 4)}
                for k in self._ordered_keys()
            ],
            "rows": self.rows(),
        }

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=False)
        if path is not None:
            Path(path).write_text(payload + "\n", encoding="utf-8")
        return payload


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
# Per-worker state, populated by _worker_init in each pool process.
_WORKER: dict[str, Any] = {}


def _set_frame_coords(
    atoms: list[AtomRecord], keep_index: np.ndarray, coords: np.ndarray
) -> None:
    """Write a frame's coordinates onto the kept atoms (in topology order)."""
    selected = coords[keep_index]
    for atom, xyz in zip(atoms, selected):
        atom.coord = (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _worker_init(atoms, components, profile_data, keep_index, source) -> None:
    _WORKER.clear()
    _WORKER.update(
        atoms=atoms,
        components=components,
        profile_data=profile_data,
        keep_index=keep_index,
        source=source,
    )


def _worker_frame(frame_index: int) -> FrameInteractions:
    atoms = _WORKER["atoms"]
    coords = _WORKER["source"].coords_for(frame_index)
    _set_frame_coords(atoms, _WORKER["keep_index"], coords)
    _, _, interactions = assign_for_atoms(atoms, _WORKER["components"], _WORKER["profile_data"])
    return FrameInteractions(frame_index, interactions)


def _resolve_frame_indices(
    n_frames: int, start: int, stop: int | None, stride: int
) -> list[int]:
    if stride < 1:
        raise ValueError("frame_stride must be >= 1")
    stop = n_frames if stop is None else min(stop, n_frames)
    return list(range(max(0, start), stop, stride))


def _build_keep_mask(
    atoms: list[AtomRecord],
    components: list[ComponentRecord],
    exclude_solvent: bool,
    exclude_ions: bool,
) -> tuple[list[AtomRecord], list[ComponentRecord], np.ndarray]:
    """Filter solvent/ion atoms and return (atoms, components, original-index array).

    ``keep_index`` maps the kept atoms back to their column in the full coordinate
    stack. Atom IDs remain unchanged so bonds and chemistry metadata keep their
    original meaning. Coordinate columns follow topology order, independently of
    atom IDs.
    """
    drop_types: set[str] = set()
    if exclude_solvent:
        drop_types.add(SOLVENT_TYPE)
    if exclude_ions:
        drop_types.add(ION_TYPE)

    if not drop_types:
        keep_index = np.arange(len(atoms), dtype=np.int64)
        return atoms, components, keep_index

    keep_index_list = [
        index for index, atom in enumerate(atoms) if atom.molecule_type not in drop_types
    ]
    kept_atoms = [atoms[index] for index in keep_index_list]
    kept_ids = {atom.atom_id for atom in kept_atoms}
    for atom in kept_atoms:
        atom.bonds = tuple(neighbor for neighbor in atom.bonds if neighbor in kept_ids)

    kept_components: list[ComponentRecord] = []
    for comp in components:
        retained_ids = tuple(atom_id for atom_id in comp.atom_ids if atom_id in kept_ids)
        if retained_ids:
            comp.atom_ids = retained_ids
            kept_components.append(comp)

    return kept_atoms, kept_components, np.array(keep_index_list, dtype=np.int64)


def analyze_trajectory(
    frame_source: ArrayFrameSource | MDAnalysisFrameSource,
    *,
    profile: str | Path | dict[str, Any] = "default",
    rule_overrides: dict[str, dict[str, float | int]] | None = None,
    ligand_smiles: dict[str, str] | None = None,
    protonate: bool = False,
    protonate_ph_min: float = 7.4,
    protonate_ph_max: float = 7.4,
    fetch_ccd_smiles: bool = False,
    exclude_solvent: bool = False,
    exclude_ions: bool = False,
    frame_start: int = 0,
    frame_stop: int | None = None,
    frame_stride: int = 1,
    processes: int = 1,
    progress: Callable[[int, int], None] | None = None,
    assembly_overrides: dict | str | Path | None = None,
    strict_ligand_templates: bool = False,
) -> TrajectoryResult:
    """Analyse interactions for every selected frame of a trajectory.

    ``frame_source`` supplies the topology and per-frame coordinates. Ligand
    templating is performed once; the coordinate-dependent stage reruns per frame,
    optionally across ``processes`` worker processes. Solvent/ions can be dropped
    from the calculation with ``exclude_solvent``/``exclude_ions``.
    """
    profile_data = apply_rule_overrides(load_profile(profile), rule_overrides)
    atoms, components = frame_source.parse_topology()

    if len(atoms) != frame_source.n_atoms:
        raise ValueError(
            f"Topology has {len(atoms)} atoms but the coordinate stack has "
            f"{frame_source.n_atoms}; cannot align frames to topology."
        )

    # The same topology preparation as static analysis, once before workers.
    chemistry_metadata = prepare_ligand_chemistry(
        atoms, components, ligand_smiles=ligand_smiles,
        assembly_overrides=assembly_overrides, fetch_ccd=fetch_ccd_smiles,
        protonate=protonate, ph_min=protonate_ph_min, ph_max=protonate_ph_max,
        strict=strict_ligand_templates,
    )

    # Filter solvent/ions AFTER templating, preserving atom IDs and returning the
    # topology-column map back into the full coordinate stack.
    atoms, components, keep_index = _build_keep_mask(
        atoms, components, exclude_solvent, exclude_ions
    )

    frame_indices = _resolve_frame_indices(
        frame_source.n_frames, frame_start, frame_stop, frame_stride
    )
    n = len(frame_indices)
    frames: list[FrameInteractions] = []

    if processes <= 1:
        for done, idx in enumerate(frame_indices, start=1):
            coords = frame_source.coords_for(idx)
            _set_frame_coords(atoms, keep_index, coords)
            _, _, interactions = assign_for_atoms(atoms, components, profile_data)
            frames.append(FrameInteractions(idx, interactions))
            if progress is not None:
                progress(done, n)
    else:
        import multiprocessing as mp

        # spawn is the safe cross-platform default (workers re-import a clean
        # chemur); CHEMUR_MP_CONTEXT lets Linux users opt into faster fork.
        ctx = mp.get_context(os.environ.get("CHEMUR_MP_CONTEXT", "spawn"))
        with ctx.Pool(
            processes=processes,
            initializer=_worker_init,
            initargs=(atoms, components, profile_data, keep_index, frame_source),
        ) as pool:
            for done, frame in enumerate(
                pool.imap(_worker_frame, frame_indices, chunksize=1), start=1
            ):
                frames.append(frame)
                if progress is not None:
                    progress(done, n)
        frames.sort(key=lambda f: f.frame_index)
        # Workers update their own topology copies. Keep the returned topology's
        # coordinates consistent with serial execution and the last analysed frame.
        if frame_indices:
            _set_frame_coords(atoms, keep_index, frame_source.coords_for(frame_indices[-1]))

    return TrajectoryResult(
        atoms=atoms,
        components=components,
        frame_indices=frame_indices,
        frames=frames,
        profile_name=str(profile_data.get("name", "default")),
        metadata={
            **geometry_result_metadata(profile_data),
            "n_frames_total": frame_source.n_frames,
            "n_frames_analysed": n,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
            "frame_stride": frame_stride,
            "exclude_solvent": exclude_solvent,
            "exclude_ions": exclude_ions,
            "processes": processes,
            "n_atoms": len(atoms),
            **chemistry_metadata,
            "strict_ligand_templates": strict_ligand_templates,
        },
    )
