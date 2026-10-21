from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
import warnings

from .assignment import assign_interactions
from .features import perceive_features
from .interactions import generate_candidates
from .models import (
    AnalysisResult,
    AssignedInteraction,
    AtomRecord,
    CandidateInteraction,
    ComponentRecord,
    FeatureRecord,
)
from .parser import parse_embedded_ligand_smiles, parse_structure
from .profile import apply_rule_overrides, load_profile
from .geometry_catalogue import geometry_result_metadata
from .preparation import prepare_ligand_chemistry
from .sdf import SdfLigand, add_sdf_ligands, load_sdf_ligands


def analyze(
    structure: str | Path,
    *,
    ligand_smiles: dict[str, str] | None = None,
    ligand_smiles_file: str | Path | None = None,
    ligand_sdf: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    ligand_sdf_dir: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    scope: str = "all",
    profile: str | Path | dict[str, Any] = "default",
    rule_overrides: dict[str, dict[str, float | int]] | None = None,
    protonate: bool = False,
    protonate_ph_min: float = 7.4,
    protonate_ph_max: float = 7.4,
    protonate_precision: float = 0.0,
    protonate_max_variants: int = 1,
    fetch_ccd_smiles: bool = True,
    ccd_timeout: float = 10.0,
    debug: bool = False,
    include_raw: bool = False,
    threads: int | str = "auto",
    strict_ligand_templates: bool = False,
    assembly_overrides: dict | str | Path | None = None,
) -> AnalysisResult:
    sdf_ligands = load_sdf_ligands(ligand_sdf, ligand_sdf_dir)
    return _analyze_loaded_sdf_ligands(
        structure,
        sdf_ligands=sdf_ligands,
        ligand_smiles=ligand_smiles,
        ligand_smiles_file=ligand_smiles_file,
        scope=scope,
        profile=profile,
        rule_overrides=rule_overrides,
        protonate=protonate,
        protonate_ph_min=protonate_ph_min,
        protonate_ph_max=protonate_ph_max,
        protonate_precision=protonate_precision,
        protonate_max_variants=protonate_max_variants,
        fetch_ccd_smiles=fetch_ccd_smiles,
        ccd_timeout=ccd_timeout,
        debug=debug,
        include_raw=include_raw,
        threads=threads,
        strict_ligand_templates=strict_ligand_templates,
        assembly_overrides=assembly_overrides,
    )


def analyze_batch(
    structure: str | Path,
    *,
    ligand_sdf: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    ligand_sdf_dir: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    ligand_smiles: dict[str, str] | None = None,
    ligand_smiles_file: str | Path | None = None,
    scope: str = "all",
    profile: str | Path | dict[str, Any] = "default",
    rule_overrides: dict[str, dict[str, float | int]] | None = None,
    protonate: bool = False,
    protonate_ph_min: float = 7.4,
    protonate_ph_max: float = 7.4,
    protonate_precision: float = 0.0,
    protonate_max_variants: int = 1,
    debug: bool = False,
    include_raw: bool = False,
    threads: int | str = "auto",
    strict_ligand_templates: bool = False,
    assembly_overrides: dict | str | Path | None = None,
) -> dict[str, AnalysisResult]:
    sdf_ligands = load_sdf_ligands(ligand_sdf, ligand_sdf_dir)
    if not sdf_ligands:
        raise ValueError("analyze_batch requires at least one ligand SDF file or directory")

    results: dict[str, AnalysisResult] = {}
    for ligand in sdf_ligands:
        batch_key = _unique_batch_key(ligand.batch_key, results)
        results[batch_key] = _analyze_loaded_sdf_ligands(
            structure,
            sdf_ligands=[ligand],
            ligand_smiles=ligand_smiles,
            ligand_smiles_file=ligand_smiles_file,
            scope=scope,
            profile=profile,
            rule_overrides=rule_overrides,
            protonate=protonate,
            protonate_ph_min=protonate_ph_min,
            protonate_ph_max=protonate_ph_max,
            protonate_precision=protonate_precision,
            protonate_max_variants=protonate_max_variants,
            fetch_ccd_smiles=False,
            ccd_timeout=10.0,
            debug=debug,
            include_raw=include_raw,
            threads=threads,
            batch_ligand_key=batch_key,
            strict_ligand_templates=strict_ligand_templates,
            assembly_overrides=assembly_overrides,
        )
    return results


def assign_for_atoms(
    atoms: list[AtomRecord],
    components: list[ComponentRecord],
    profile_data: dict[str, Any],
) -> tuple[list[FeatureRecord], list[CandidateInteraction], list[AssignedInteraction]]:
    """Run the coordinate-dependent stage of the pipeline on templated atoms.

    Ligand templating and chemistry assignment are topology-based and only need to
    happen once, but feature perception derives ring centroids/normals and feature
    centres from atom coordinates, so this must be rerun whenever coordinates change
    (e.g. per trajectory frame). Returns ``(features, candidates, interactions)``.
    """
    features = perceive_features(atoms, components)
    candidates = generate_candidates(atoms, features, profile_data)
    interactions = assign_interactions(atoms, candidates, profile_data)
    return features, candidates, interactions


def _analyze_loaded_sdf_ligands(
    structure: str | Path,
    *,
    sdf_ligands: list[SdfLigand],
    ligand_smiles: dict[str, str] | None,
    ligand_smiles_file: str | Path | None,
    scope: str,
    profile: str | Path | dict[str, Any],
    rule_overrides: dict[str, dict[str, float | int]] | None,
    protonate: bool,
    protonate_ph_min: float,
    protonate_ph_max: float,
    protonate_precision: float,
    protonate_max_variants: int,
    fetch_ccd_smiles: bool,
    ccd_timeout: float,
    debug: bool,
    include_raw: bool,
    threads: int | str,
    batch_ligand_key: str | None = None,
    strict_ligand_templates: bool = False,
    assembly_overrides: dict | str | Path | None = None,
) -> AnalysisResult:
    if scope != "all":
        raise ValueError("Standalone Chemur v1 currently supports scope='all' only")

    input_smiles = _merge_ligand_smiles(ligand_smiles, ligand_smiles_file)
    profile_data = apply_rule_overrides(load_profile(profile), rule_overrides)
    sdf_mode = bool(sdf_ligands)

    atoms, components = parse_structure(structure, include_ligands=not sdf_mode)
    sdf_input_smiles: dict[str, str] = {}
    sdf_smiles_overrides: list[str] = []
    if sdf_mode:
        sdf_input_smiles = add_sdf_ligands(atoms, components, sdf_ligands)
        input_smiles, sdf_smiles_overrides = _select_sdf_smiles(
            components, sdf_input_smiles, input_smiles,
        )
    chemistry_metadata = prepare_ligand_chemistry(
        atoms, components, ligand_smiles=input_smiles,
        embedded_smiles=parse_embedded_ligand_smiles(structure) if not sdf_mode else {},
        assembly_overrides=assembly_overrides, fetch_ccd=fetch_ccd_smiles,
        ccd_timeout=ccd_timeout, protonate=protonate, ph_min=protonate_ph_min,
        ph_max=protonate_ph_max, precision=protonate_precision,
        max_variants=protonate_max_variants, debug=debug, strict=strict_ligand_templates,
        sdf_mode=sdf_mode, sdf_overrides=sdf_smiles_overrides,
    )

    features, candidates, interactions = assign_for_atoms(atoms, components, profile_data)

    skipped_ligands = {
        component.component_id: component.metadata.get("ignored_reason", "")
        for component in components
        if component.molecule_type == "ligand"
        and component.metadata.get("ignored_by_template")
    }

    return AnalysisResult(
        atoms=atoms,
        components=components,
        features=features,
        interactions=interactions,
        raw_candidates=candidates if include_raw else [],
        profile_name=str(profile_data.get("name", "default")),
        metadata={
            **geometry_result_metadata(profile_data),
            "structure": str(structure),
            "scope": scope,
            "threads": threads,
            "structure_ligands_included": not sdf_mode,
            "ligand_sdf_paths": sorted({str(ligand.source_path) for ligand in sdf_ligands}),
            "sdf_ligands": _sdf_component_metadata(components),
            "sdf_smiles_overrides": sdf_smiles_overrides,
            "batch_ligand_key": batch_ligand_key,
            "sdf_input_ligand_smiles": sdf_input_smiles,
            **chemistry_metadata,
            "protonate": protonate,
            "protonate_ph_min": protonate_ph_min,
            "protonate_ph_max": protonate_ph_max,
            "protonate_precision": protonate_precision,
            "protonate_max_variants": protonate_max_variants,
            "strict_ligand_templates": strict_ligand_templates,
            "skipped_ligands": skipped_ligands,
            "rule_overrides": deepcopy(rule_overrides or {}),
        },
    )


def _select_sdf_smiles(
    components: list[ComponentRecord],
    sdf_smiles: dict[str, str],
    user_smiles: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    selected: dict[str, str] = {}
    overrides: list[str] = []
    user_templates = {key.upper(): value for key, value in user_smiles.items()}

    for component in components:
        if component.component_id not in sdf_smiles:
            continue
        component_key = component.component_id.upper()
        name_key = component.name.upper()
        if component_key in user_templates:
            selected[component.component_id] = user_templates[component_key]
            overrides.append(component.component_id)
        elif name_key in user_templates:
            selected[component.component_id] = user_templates[name_key]
            overrides.append(component.component_id)
        else:
            selected[component.component_id] = sdf_smiles[component.component_id]

    return selected, overrides


def _sdf_component_metadata(components: list[ComponentRecord]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for component in components:
        if component.metadata.get("source") != "sdf":
            continue
        records.append(
            {
                "component_id": component.component_id,
                "name": component.name,
                "chain_id": component.chain_id,
                "residue_id": component.residue_id,
                "source_path": component.metadata.get("source_path"),
                "sdf_molecule_index": component.metadata.get("sdf_molecule_index"),
                "sdf_name": component.metadata.get("sdf_name"),
                "batch_key": component.metadata.get("batch_key"),
            }
        )
    return records


def _unique_batch_key(candidate: str, results: dict[str, AnalysisResult]) -> str:
    if candidate not in results:
        return candidate
    index = 2
    while f"{candidate}:{index}" in results:
        index += 1
    return f"{candidate}:{index}"


def _merge_ligand_smiles(
    ligand_smiles: dict[str, str] | None,
    ligand_smiles_file: str | Path | None,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    if ligand_smiles_file is not None:
        path = Path(ligand_smiles_file)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in payload.items()
        ):
            raise ValueError("Ligand SMILES file must contain a JSON object of string keys and values")
        merged.update({key.upper(): value for key, value in payload.items()})
    if ligand_smiles:
        merged.update({key.upper(): value for key, value in ligand_smiles.items()})
    return merged
