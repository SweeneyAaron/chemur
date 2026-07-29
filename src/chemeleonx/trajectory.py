"""MD-trajectory interaction analysis.

Computes ChemeleonX interactions for every frame of a trajectory and exposes
time-resolved views (occupancy, per-frame counts, per-interaction time series).

Topology is parsed and ligands are templated **once** (both are coordinate
independent); only the coordinate-dependent stage
(:func:`chemeleonx.api.assign_for_atoms`) reruns per frame. Frames can be processed
in parallel across CPUs with :mod:`multiprocessing`.

Two frame sources are provided:

* :class:`ArrayFrameSource` — topology file + an ``(n_frames, n_atoms, 3)`` array
  (or ``.npy`` path). Used by the ChimeraX plugin, which serialises the loaded
  trajectory's coordsets to disk.
* :class:`MDAnalysisFrameSource` — reads ``.gro``/``.xtc`` (and other formats)
  directly via MDAnalysis. Used by the CLI and tests. Requires the optional
  ``trajectory`` extra (``pip install chemeleonx[trajectory]``).
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
from .chemistry import apply_ligand_smiles_templates, mark_untemplated_ligands_ignored
from .ccd import fetch_missing_ligand_smiles_from_ccd
from .models import AssignedInteraction, AtomRecord, ComponentRecord
from .parser import parse_structure
from .profile import apply_rule_overrides, load_profile
from .protonation import prepare_ligand_smiles

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

    The topology is parsed by writing frame 0 to a temporary PDB and handing that
    to :func:`parse_structure` (gemmi cannot read ``.gro``/``.xtc``). Per-frame
    coordinates are streamed from the MDAnalysis ``Universe`` in topology atom
    order. MDAnalysis reports positions in Angstrom, matching ChemeleonX.

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
                    "`pip install chemeleonx[trajectory]` or `pip install MDAnalysis`."
                ) from exc
            self._universe = mda.Universe(self.topology, self.trajectory)
        return self._universe

    def parse_topology(self) -> tuple[list[AtomRecord], list[ComponentRecord]]:
        u = self._u()
        handle = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        handle.close()
        try:
            u.trajectory[0]
            u.atoms.write(handle.name)
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
    to ``frame_indices``.
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
        """Stable identity for a contact across frames: type + atom-id set."""
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
                key = (inter.interaction_type, frozenset(inter.atom_ids))
                if key not in seen:
                    seen.add(key)
                    present[key] = present.get(key, 0) + 1
                dist_sum[key] = dist_sum.get(key, 0.0) + inter.distance
                dist_n[key] = dist_n.get(key, 0) + 1
                if key not in labels:
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
        """Human-readable ``residueA <-> residueB`` label per interaction key."""
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
        ]
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows())

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
    stack. Atom ids are renumbered 0..k-1 so downstream code stays consistent.
    """
    drop_types: set[str] = set()
    if exclude_solvent:
        drop_types.add(SOLVENT_TYPE)
    if exclude_ions:
        drop_types.add(ION_TYPE)

    if not drop_types:
        keep_index = np.array([a.atom_id for a in atoms], dtype=np.int64)
        return atoms, components, keep_index

    # Two passes: build the *complete* old->new id map for all kept atoms first,
    # then remap bonds. A single pass would drop each atom's bonds to atoms that
    # appear later in the list (e.g. a donor N's bond to its hydrogens), which
    # silently breaks H-bond detection (it relies on donor_atom.bonds -> H).
    kept_atoms: list[AtomRecord] = [a for a in atoms if a.molecule_type not in drop_types]
    keep_index_list: list[int] = [a.atom_id for a in kept_atoms]
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(keep_index_list)}
    for atom in kept_atoms:
        new_bonds = tuple(old_to_new[b] for b in atom.bonds if b in old_to_new)
        atom.atom_id = old_to_new[atom.atom_id]
        atom.bonds = new_bonds

    kept_components: list[ComponentRecord] = []
    for comp in components:
        new_ids = tuple(old_to_new[a] for a in comp.atom_ids if a in old_to_new)
        if new_ids:
            comp.atom_ids = new_ids
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

    # One-time ligand templating (topology-based, coordinate independent).
    input_smiles = {k.upper(): v for k, v in (ligand_smiles or {}).items()}
    if not input_smiles and fetch_ccd_smiles:
        ccd = fetch_missing_ligand_smiles_from_ccd(components)
        input_smiles = ccd.smiles
    smiles = prepare_ligand_smiles(
        input_smiles,
        protonate=protonate,
        ph_min=protonate_ph_min,
        ph_max=protonate_ph_max,
    )
    has_ligands = any(c.molecule_type == "ligand" for c in components)
    if has_ligands:
        if smiles:
            mark_untemplated_ligands_ignored(
                atoms, components, smiles, reason="No SMILES template was available"
            )
        apply_ligand_smiles_templates(atoms, components, smiles)

    # Filter solvent/ions AFTER templating; renumbers atom ids and yields the
    # column map back into the full coordinate stack.
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
        # chemeleonx); CHEMELEONX_MP_CONTEXT lets Linux users opt into faster fork.
        ctx = mp.get_context(os.environ.get("CHEMELEONX_MP_CONTEXT", "spawn"))
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

    return TrajectoryResult(
        atoms=atoms,
        components=components,
        frame_indices=frame_indices,
        frames=frames,
        profile_name=str(profile_data.get("name", "default")),
        metadata={
            "n_frames_total": frame_source.n_frames,
            "n_frames_analysed": n,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
            "frame_stride": frame_stride,
            "exclude_solvent": exclude_solvent,
            "exclude_ions": exclude_ions,
            "processes": processes,
            "n_atoms": len(atoms),
            "ligand_smiles": smiles,
        },
    )
