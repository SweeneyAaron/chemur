from __future__ import annotations

from dataclasses import dataclass
import shlex
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .models import ComponentRecord

CCD_DOWNLOAD_URL = "https://files.rcsb.org/ligands/download/{component_id}.cif"


@dataclass(frozen=True)
class CcdSmilesResult:
    smiles: dict[str, str]
    missing: dict[str, str]


def fetch_missing_ligand_smiles_from_ccd(
    components: list[ComponentRecord],
    *,
    timeout: float = 10.0,
) -> CcdSmilesResult:
    ligand_names = sorted(
        {
            component.name.upper()
            for component in components
            if component.molecule_type == "ligand"
        }
    )
    smiles: dict[str, str] = {}
    missing: dict[str, str] = {}
    for ligand_name in ligand_names:
        try:
            smiles[ligand_name] = fetch_ccd_smiles(ligand_name, timeout=timeout)
        except Exception as exc:
            missing[ligand_name] = str(exc)
    return CcdSmilesResult(smiles=smiles, missing=missing)


def fetch_ccd_smiles(component_id: str, *, timeout: float = 10.0) -> str:
    component_id = component_id.strip().upper()
    url = CCD_DOWNLOAD_URL.format(component_id=component_id)
    try:
        with urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except HTTPError as exc:
        raise LookupError(f"CCD entry {component_id} was not found at {url}") from exc
    except URLError as exc:
        raise LookupError(f"Could not fetch CCD entry {component_id}: {exc.reason}") from exc

    descriptors = _extract_descriptors(text)
    smiles = _choose_smiles(descriptors)
    if smiles is None:
        raise LookupError(f"CCD entry {component_id} does not contain a SMILES descriptor")
    return smiles


def _choose_smiles(descriptors: list[dict[str, str]]) -> str | None:
    preferences = [
        ("Canonical SMILES", "OpenEye OEToolkits"),
        ("Canonical SMILES", "CACTVS"),
        ("SMILES", "OpenEye OEToolkits"),
        ("SMILES", "CACTVS"),
        ("SMILES", "ACDLabs"),
    ]
    for descriptor_type, program in preferences:
        for descriptor in descriptors:
            if (
                descriptor.get("type") == descriptor_type
                and descriptor.get("program") == program
            ):
                return descriptor.get("descriptor")
    for descriptor in descriptors:
        if "SMILES" in descriptor.get("type", "") and descriptor.get("descriptor"):
            return descriptor["descriptor"]
    return None


def _extract_descriptors(cif_text: str) -> list[dict[str, str]]:
    descriptors: list[dict[str, str]] = []
    lines = cif_text.splitlines()
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() != "loop_":
            idx += 1
            continue

        idx += 1
        tags: list[str] = []
        while idx < len(lines) and lines[idx].strip().startswith("_"):
            tags.append(lines[idx].strip())
            idx += 1

        if "_pdbx_chem_comp_descriptor.descriptor" not in tags:
            while idx < len(lines) and not _starts_new_block(lines[idx]):
                idx += 1
            continue

        idx = _parse_descriptor_rows(lines, idx, tags, descriptors)
    return descriptors


def _parse_descriptor_rows(
    lines: list[str],
    idx: int,
    tags: list[str],
    descriptors: list[dict[str, str]],
) -> int:
    type_idx = tags.index("_pdbx_chem_comp_descriptor.type")
    program_idx = tags.index("_pdbx_chem_comp_descriptor.program")
    descriptor_idx = tags.index("_pdbx_chem_comp_descriptor.descriptor")
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line == "#":
            idx += 1
            break
        if _starts_new_block(line):
            break

        row_text = line
        row = _split_cif_row(row_text)
        while len(row) < len(tags) and idx + 1 < len(lines):
            idx += 1
            row_text += " " + lines[idx].strip()
            row = _split_cif_row(row_text)

        if len(row) >= len(tags):
            descriptors.append(
                {
                    "type": row[type_idx],
                    "program": row[program_idx],
                    "descriptor": row[descriptor_idx],
                }
            )
        idx += 1
    return idx


def _split_cif_row(row_text: str) -> list[str]:
    lexer = shlex.shlex(row_text, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _starts_new_block(line: str) -> bool:
    stripped = line.strip()
    return stripped == "loop_" or stripped.startswith("data_") or stripped.startswith("_")

