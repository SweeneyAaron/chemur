from __future__ import annotations

import sys

from .errors import DependencyMissingError, LigandTemplateError


def prepare_ligand_smiles(
    ligand_smiles: dict[str, str],
    *,
    protonate: bool = False,
    ph_min: float = 7.4,
    ph_max: float = 7.4,
    precision: float = 0.0,
    max_variants: int = 1,
    debug: bool = False,
) -> dict[str, str]:
    prepared: dict[str, str] = {}
    for ligand_name, smiles in ligand_smiles.items():
        selected_smiles = smiles
        protonated_variants: list[str] = []
        if protonate:
            protonated_variants = _protonate_smiles(
                smiles,
                ph_min=ph_min,
                ph_max=ph_max,
                precision=precision,
                max_variants=max_variants,
            )
            if not protonated_variants:
                raise LigandTemplateError(
                    f"Dimorphite-DL did not return a protonated SMILES for ligand {ligand_name}"
                )
            selected_smiles = protonated_variants[0]

        prepared[ligand_name] = selected_smiles
        if debug or protonate:
            message = (
                f"[chemur] ligand {ligand_name}: using SMILES {selected_smiles!r}"
            )
            if protonate:
                message += f" (input {smiles!r}, variants={len(protonated_variants)})"
            print(message, file=sys.stderr)
    return prepared


def _protonate_smiles(
    smiles: str,
    *,
    ph_min: float,
    ph_max: float,
    precision: float,
    max_variants: int,
) -> list[str]:
    try:
        from dimorphite_dl import protonate_smiles
    except ImportError as exc:
        raise DependencyMissingError(
            "Dimorphite-DL is required when --protonate is used. Install it with "
            "`pip install dimorphite_dl` or `pip install -e '.[protonation]'`."
        ) from exc

    try:
        variants = protonate_smiles(
            smiles,
            ph_min=ph_min,
            ph_max=ph_max,
            precision=precision,
            max_variants=max_variants,
        )
    except TypeError:
        variants = protonate_smiles(
            smiles,
            ph_min=ph_min,
            ph_max=ph_max,
            precision=precision,
        )

    return [variant for variant in variants if isinstance(variant, str) and variant]

