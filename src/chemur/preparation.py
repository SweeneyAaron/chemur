"""Shared topology chemistry preparation for static and trajectory analysis."""
from __future__ import annotations

import warnings

from .assemblies import prepare_assemblies, assembly_registry, chemical_product_registry
from .ccd import fetch_missing_ligand_smiles_from_ccd
from .chemistry import apply_ligand_smiles_templates, mark_untemplated_ligands_ignored
from .protonation import prepare_ligand_smiles


MODIFIED_POLYMER_CHEMISTRY = True


def prepare_ligand_chemistry(
    atoms, components, *, ligand_smiles=None, embedded_smiles=None,
    assembly_overrides=None, fetch_ccd=True, ccd_timeout=10.0,
    protonate=False, ph_min=7.4, ph_max=7.4, precision=0.0, max_variants=1,
    debug=False, strict=False, sdf_mode=False, sdf_overrides=(),
):
    """Prepare each complete ligand once and return serializable provenance."""
    input_smiles = {str(k).upper(): v for k, v in (ligand_smiles or {}).items()}
    handled = prepare_assemblies(
        atoms, components, ligand_smiles=input_smiles, assembly_overrides=assembly_overrides,
        fetch_ccd=fetch_ccd, ccd_timeout=ccd_timeout, protonate=protonate,
        ph_min=ph_min, ph_max=ph_max, precision=precision, max_variants=max_variants,
        debug=debug, strict=strict,
    )
    ordinary = [c for c in components if c.component_id not in handled]
    fetched = used_embedded = False
    missing = {}
    auto_sourced = not sdf_mode and not input_smiles
    if auto_sourced:
        names = {c.name.upper() for c in ordinary if c.molecule_type == "ligand"}
        embedded = embedded_smiles or {}
        input_smiles = {name: embedded[name] for name in names if name in embedded}
        used_embedded = bool(input_smiles)
        remaining = [c for c in ordinary if c.molecule_type == "ligand" and c.name.upper() not in input_smiles]
        if remaining and fetch_ccd:
            result = fetch_missing_ligand_smiles_from_ccd(remaining, timeout=ccd_timeout)
            input_smiles.update(result.smiles)
            missing = result.missing
            fetched = True
            for name, reason in missing.items():
                warnings.warn(f"Could not find CCD SMILES for ligand {name}; that ligand will be skipped. Reason: {reason}", RuntimeWarning, stacklevel=2)
    # Do not protonate monomer overrides after they have already been applied to
    # the assembled product (especially a fixed covalent attachment).
    ordinary_keys = {key for c in ordinary if c.molecule_type == "ligand"
                     for key in (c.component_id.upper(), c.name.upper())}
    selected = {key: value for key, value in input_smiles.items() if key in ordinary_keys}
    prepared = prepare_ligand_smiles(selected, protonate=protonate, ph_min=ph_min, ph_max=ph_max,
                                      precision=precision, max_variants=max_variants, debug=debug)
    if sdf_mode:
        if protonate or sdf_overrides:
            apply_ligand_smiles_templates(atoms, ordinary, prepared, strict=strict)
    else:
        if auto_sourced and (used_embedded or fetched):
            # Match both component IDs and residue names, unlike a name-only
            # blanket exclusion that would discard specific user templates.
            unresolved = [c for c in ordinary if c.component_id.upper() not in prepared]
            mark_untemplated_ligands_ignored(atoms, unresolved, prepared,
                                           reason="No embedded or CCD SMILES template was available")
        apply_ligand_smiles_templates(atoms, ordinary, prepared, strict=strict)
    registry = assembly_registry(components)
    prepared.update({key: record["smiles"] for key, record in registry.items() if record.get("status") == "validated"})
    return {
        "chemical_products": chemical_product_registry(components),
        "modified_polymer_chemistry": {key: record for key, record in chemical_product_registry(components).items()
                                       if record.get("product_kind") == "modified_polymer"},
        "ligand_assemblies": registry, "ligand_smiles": prepared,
        "input_ligand_smiles": input_smiles, "ligand_smiles_keys": sorted(prepared),
        "ccd_smiles_lookup": fetched, "embedded_smiles_lookup": used_embedded,
        "ccd_missing": missing,
    }
