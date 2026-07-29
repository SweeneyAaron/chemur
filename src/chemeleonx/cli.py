from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

from .api import analyze, analyze_batch
from .core import USING_CPP_CORE
from .interaction_types import CUTOFF_HELP, INTERACTION_TYPES


def _rule_cutoff_flags() -> list[tuple[str, str, tuple[str, ...], str]]:
    """Build the --<rule>-<cutoff> flag table from the interaction-type registry.

    The primary spelling is derived from the rule name; ``aliases`` in the registry
    keeps the older hand-written spellings (``--hb-distance``, ``--pipi-offset``,
    ...) accepted so existing command lines do not break.
    """
    flags: list[tuple[str, str, tuple[str, ...], str]] = []
    for interaction in INTERACTION_TYPES:
        for cutoff in interaction.cutoffs:
            primary = f"--{interaction.name.replace('_', '-')}-{cutoff.replace('_', '-')}"
            spellings = (primary,) + tuple(interaction.aliases.get(cutoff, ()))
            help_text = f"{interaction.label} {CUTOFF_HELP.get(cutoff, cutoff + ' cutoff.')}"
            flags.append((interaction.name, cutoff, spellings, help_text))
    return flags


RULE_CUTOFF_FLAGS = _rule_cutoff_flags()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chemeleonx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze biomolecular interactions")
    analyze_parser.add_argument("structure", help="Input PDB or mmCIF structure")
    analyze_parser.add_argument(
        "--ligand-smiles",
        action="append",
        default=[],
        metavar="NAME=SMILES",
        help="Ligand residue/component SMILES template. May be repeated.",
    )
    analyze_parser.add_argument(
        "--ligand-smiles-file",
        help="JSON object mapping ligand residue/component names to SMILES strings.",
    )
    analyze_parser.add_argument(
        "--ligand-sdf",
        action="append",
        default=[],
        metavar="PATH",
        help="Ligand SDF file to add to the structure. May be repeated.",
    )
    analyze_parser.add_argument(
        "--ligand-sdf-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory containing .sdf ligand files to add to the structure. May be repeated.",
    )
    analyze_parser.add_argument(
        "--batch",
        action="store_true",
        help="Analyze each SDF ligand independently against the structure.",
    )
    analyze_parser.add_argument("--scope", default="all", help="Interaction scope. V1 supports 'all'.")
    analyze_parser.add_argument("--profile", default="default", help="Profile name or YAML file path.")
    analyze_parser.add_argument("--out", help="Write JSON output to this path.")
    analyze_parser.add_argument("--csv", help="Write CSV output to this path.")
    analyze_parser.add_argument(
        "--split-by-ligand-dir",
        default="ligand_outputs",
        help="Directory for default per-ligand outputs.",
    )
    analyze_parser.add_argument(
        "--split-by-ligand-format",
        choices=("json", "csv", "both"),
        default="both",
        help="Format for per-ligand outputs.",
    )
    analyze_parser.add_argument(
        "--no-split-by-ligand",
        action="store_true",
        help="Disable default per-ligand outputs.",
    )
    analyze_parser.add_argument("--include-raw", action="store_true", help="Include raw candidates in JSON/CSV.")
    analyze_parser.add_argument("--threads", default="auto", help="Reserved for threaded C++ kernels.")
    analyze_parser.add_argument(
        "--no-ccd-smiles",
        action="store_true",
        help="Disable automatic CCD SMILES lookup when no ligand SMILES are supplied.",
    )
    analyze_parser.add_argument(
        "--ccd-timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds for each CCD SMILES lookup.",
    )
    analyze_parser.add_argument(
        "--protonate",
        action="store_true",
        help="Use Dimorphite-DL to protonate ligand SMILES before chemistry mapping.",
    )
    analyze_parser.add_argument(
        "--protonate-ph-min",
        type=float,
        default=7.4,
        help="Minimum pH for Dimorphite-DL protonation.",
    )
    analyze_parser.add_argument(
        "--protonate-ph-max",
        type=float,
        default=7.4,
        help="Maximum pH for Dimorphite-DL protonation.",
    )
    analyze_parser.add_argument(
        "--protonate-precision",
        type=float,
        default=0.0,
        help="Dimorphite-DL pKa precision factor.",
    )
    analyze_parser.add_argument(
        "--protonate-max-variants",
        type=int,
        default=1,
        help="Maximum Dimorphite-DL variants per ligand; ChemeleonX uses the first returned variant.",
    )
    analyze_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug messages, including the ligand SMILES used for chemistry mapping.",
    )
    analyze_parser.add_argument(
        "--strict-ligand-templates",
        action="store_true",
        help="Fail if any ligand SMILES template cannot be mapped, instead of skipping that ligand.",
    )
    _add_rule_cutoff_args(analyze_parser)

    score_parser = subparsers.add_parser(
        "score",
        help="Rank docked ligand poses against a known 3D binder (CoSPLIF + IFM).",
    )
    score_parser.add_argument("structure", help="Receptor PDB or mmCIF structure.")
    score_parser.add_argument(
        "--ligand-sdf",
        action="append",
        default=[],
        metavar="PATH",
        help="Docked pose SDF to score. May be repeated.",
    )
    score_parser.add_argument(
        "--ligand-sdf-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory of docked pose .sdf files to score. May be repeated.",
    )
    score_parser.add_argument(
        "--reference-sdf",
        help="SDF of the known binder to score poses against (takes precedence).",
    )
    score_parser.add_argument(
        "--reference-id",
        help="Use one of the supplied poses (by its key / filename stem) as the reference.",
    )
    score_parser.add_argument(
        "--profile",
        default="cosplif_pose_v1",
        help="CoSPLIF scoring profile name (default: cosplif_pose_v1).",
    )
    score_parser.add_argument(
        "--cosplif-config",
        help="Optional CoSPLIF profile JSON defining custom named profiles.",
    )
    score_parser.add_argument(
        "--analysis-profile",
        default="default",
        help="ChemeleonX interaction-detection rule profile (default: default).",
    )
    score_parser.add_argument("--scope", default="all", help="Interaction scope. V1 supports 'all'.")
    score_parser.add_argument("--threads", default="auto", help="Reserved for threaded C++ kernels.")
    score_parser.add_argument(
        "--ifm-engine",
        choices=("auto", "cpp", "python"),
        default="python",
        help="IFM scoring engine for the hybrid blend (default: python).",
    )
    score_parser.add_argument("--ifm-workers", default="auto", help="Worker count for IFM scoring.")
    score_parser.add_argument(
        "--protonate",
        action="store_true",
        help="Protonate ligands with Dimorphite-DL before chemistry mapping.",
    )
    score_parser.add_argument("--protonate-ph-min", type=float, default=7.4)
    score_parser.add_argument("--protonate-ph-max", type=float, default=7.4)
    score_parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Keep the reference ligand in the ranked output instead of excluding it.",
    )
    score_parser.add_argument("--out", help="Write ranked results as JSON to this path.")
    score_parser.add_argument("--csv", help="Write ranked results as CSV to this path.")
    score_parser.add_argument("--report", help="Write a human-readable ranking to this path.")

    traj_parser = subparsers.add_parser(
        "trajectory",
        help="Analyse interactions per frame across an MD trajectory.",
    )
    traj_parser.add_argument("topology", help="Topology file (e.g. .gro, .pdb, .tpr).")
    traj_parser.add_argument("trajectory", help="Trajectory file (e.g. .xtc, .trr, .dcd).")
    traj_parser.add_argument(
        "--ligand-smiles",
        action="append",
        default=[],
        metavar="NAME=SMILES",
        help="Ligand residue/component SMILES template. May be repeated.",
    )
    traj_parser.add_argument(
        "--exclude-solvent",
        action="store_true",
        help="Drop solvent (water) atoms from the calculation.",
    )
    traj_parser.add_argument(
        "--exclude-ions",
        action="store_true",
        help="Drop monatomic ions from the calculation.",
    )
    traj_parser.add_argument("--start", type=int, default=0, help="First frame index (default: 0).")
    traj_parser.add_argument("--stop", type=int, default=None, help="Stop frame index (exclusive).")
    traj_parser.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1).")
    traj_parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Worker processes for parallel per-frame analysis (default: 1).",
    )
    traj_parser.add_argument(
        "--profile",
        default="default",
        help="Interaction-detection rule profile (default: default).",
    )
    traj_parser.add_argument(
        "--protonate",
        action="store_true",
        help="Protonate ligands with Dimorphite-DL before chemistry mapping.",
    )
    traj_parser.add_argument("--protonate-ph-min", type=float, default=7.4)
    traj_parser.add_argument("--protonate-ph-max", type=float, default=7.4)
    traj_parser.add_argument("--out", help="Write the full result as JSON to this path.")
    traj_parser.add_argument("--csv", help="Write per-frame interaction rows as CSV to this path.")
    _add_rule_cutoff_args(traj_parser)

    args = parser.parse_args(argv)

    if args.command == "analyze":
        if args.batch:
            return _run_batch_analyze(args, parser)
        return _run_single_analyze(args, parser)

    if args.command == "score":
        return _run_score(args, parser)

    if args.command == "trajectory":
        return _run_trajectory(args, parser)

    return 1


def _run_trajectory(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from .trajectory import MDAnalysisFrameSource, analyze_trajectory

    ligand_smiles = _parse_ligand_smiles(args.ligand_smiles, parser)
    source = MDAnalysisFrameSource(args.topology, args.trajectory)

    def _progress(done: int, total: int) -> None:
        sys.stderr.write(f"\rAnalysing frame {done}/{total}")
        sys.stderr.flush()
        if done == total:
            sys.stderr.write("\n")

    result = analyze_trajectory(
        source,
        profile=args.profile,
        rule_overrides=_rule_overrides_from_args(args) or None,
        ligand_smiles=ligand_smiles or None,
        protonate=args.protonate,
        protonate_ph_min=args.protonate_ph_min,
        protonate_ph_max=args.protonate_ph_max,
        exclude_solvent=args.exclude_solvent,
        exclude_ions=args.exclude_ions,
        frame_start=args.start,
        frame_stop=args.stop,
        frame_stride=args.stride,
        processes=args.processes,
        progress=_progress,
    )

    if args.out:
        result.to_json(args.out)
    if args.csv:
        result.to_csv(args.csv)

    if not args.out and not args.csv:
        sys.stdout.write(result.to_json() + "\n")
    else:
        occ = result.occupancy()
        sys.stdout.write(
            f"Analysed {result.n_frames} frames; found {len(occ)} distinct interactions "
            f"using {'C++' if USING_CPP_CORE else 'Python'} core\n"
        )
    return 0


def _run_single_analyze(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    ligand_smiles = _parse_ligand_smiles(args.ligand_smiles, parser)
    result = analyze(
        args.structure,
        ligand_smiles=ligand_smiles,
        ligand_smiles_file=args.ligand_smiles_file,
        ligand_sdf=args.ligand_sdf,
        ligand_sdf_dir=args.ligand_sdf_dir,
        fetch_ccd_smiles=not args.no_ccd_smiles,
        ccd_timeout=args.ccd_timeout,
        **_common_analysis_kwargs(args),
    )
    if args.out:
        result.to_json(args.out, include_raw=args.include_raw)
    if args.csv:
        result.to_csv(args.csv, include_raw=args.include_raw)

    ligand_paths = []
    split_by_ligand = not args.no_split_by_ligand
    if split_by_ligand:
        ligand_paths = result.write_ligand_outputs(
            args.split_by_ligand_dir,
            include_raw=args.include_raw,
            output_format=args.split_by_ligand_format,
        )

    if not args.out and not args.csv and not split_by_ligand:
        sys.stdout.write(result.to_json(include_raw=args.include_raw) + "\n")
    else:
        sys.stdout.write(
            f"Assigned {len(result.interactions)} interactions "
            f"using {'C++' if USING_CPP_CORE else 'Python'} core\n"
        )
        if split_by_ligand:
            sys.stdout.write(
                f"Wrote {len(ligand_paths)} per-ligand output files "
                f"to {args.split_by_ligand_dir}\n"
            )
    return 0


def _run_batch_analyze(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.ligand_sdf and not args.ligand_sdf_dir:
        parser.error("--batch requires --ligand-sdf or --ligand-sdf-dir")

    ligand_smiles = _parse_ligand_smiles(args.ligand_smiles, parser)
    results = analyze_batch(
        args.structure,
        ligand_sdf=args.ligand_sdf,
        ligand_sdf_dir=args.ligand_sdf_dir,
        ligand_smiles=ligand_smiles,
        ligand_smiles_file=args.ligand_smiles_file,
        **_common_analysis_kwargs(args),
    )

    if args.out:
        _write_batch_json(results, args.out, include_raw=args.include_raw)
    if args.csv:
        _write_batch_csv(results, args.csv, include_raw=args.include_raw)

    ligand_paths = []
    split_by_ligand = not args.no_split_by_ligand
    if split_by_ligand:
        output_root = Path(args.split_by_ligand_dir)
        for batch_id, result in results.items():
            ligand_paths.extend(
                result.write_ligand_outputs(
                    output_root / _safe_output_name(batch_id),
                    include_raw=args.include_raw,
                    output_format=args.split_by_ligand_format,
                )
            )

    if not args.out and not args.csv and not split_by_ligand:
        sys.stdout.write(json.dumps(_batch_payload(results, args.include_raw), indent=2, sort_keys=True) + "\n")
    else:
        total_interactions = sum(len(result.interactions) for result in results.values())
        sys.stdout.write(
            f"Assigned {total_interactions} interactions across {len(results)} "
            f"batch analyses using {'C++' if USING_CPP_CORE else 'Python'} core\n"
        )
        if split_by_ligand:
            sys.stdout.write(
                f"Wrote {len(ligand_paths)} per-ligand output files "
                f"to {args.split_by_ligand_dir}\n"
            )
    return 0


def _run_score(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.ligand_sdf and not args.ligand_sdf_dir:
        parser.error("chemeleonx score requires --ligand-sdf or --ligand-sdf-dir")
    if not args.reference_sdf and not args.reference_id:
        parser.error("chemeleonx score requires --reference-sdf or --reference-id")

    from .scoring import score_poses

    rows = score_poses(
        args.structure,
        ligand_sdf=args.ligand_sdf or None,
        ligand_sdf_dir=args.ligand_sdf_dir or None,
        reference_sdf=args.reference_sdf,
        reference_id=args.reference_id,
        profile=args.profile,
        cosplif_config=args.cosplif_config,
        analysis_profile=args.analysis_profile,
        protonate=args.protonate,
        protonate_ph_min=args.protonate_ph_min,
        protonate_ph_max=args.protonate_ph_max,
        scope=args.scope,
        threads=args.threads,
        ifm_engine=args.ifm_engine,
        ifm_workers=args.ifm_workers,
        include_reference=args.include_reference,
    )

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if args.csv:
        _write_score_csv(rows, args.csv)
    if args.report:
        Path(args.report).write_text(_render_score_report(rows, args.profile), encoding="utf-8")

    if not args.out and not args.csv:
        sys.stdout.write(json.dumps(rows, indent=2) + "\n")
    else:
        sys.stdout.write(
            f"Scored {len(rows)} pose(s) against the reference "
            f"using profile '{args.profile}'\n"
        )
    return 0


_SCORE_CSV_FIELDS = [
    "rank",
    "ligand_id",
    "source_component_id",
    "cosplif_score",
    "cosplif_base_score",
    "cosplif_ligand_block_score",
    "strict_ifm_similarity",
    "cosplif_containment",
    "cosplif_tanimoto",
    "cosplif_token_overlap",
]


def _write_score_csv(rows: list[dict], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SCORE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_score_report(rows: list[dict], profile: str) -> str:
    lines = [
        f"ChemeleonX pose ranking (profile: {profile})",
        f"{'rank':>4}  {'score':>9}  ligand",
    ]
    for row in rows:
        score = float(row.get("cosplif_score") or 0.0)
        lines.append(f"{row.get('rank', ''):>4}  {score:>9.4f}  {row.get('ligand_id', '')}")
    return "\n".join(lines) + "\n"


def _common_analysis_kwargs(args: argparse.Namespace) -> dict:
    return {
        "scope": args.scope,
        "profile": args.profile,
        "rule_overrides": _rule_overrides_from_args(args),
        "protonate": args.protonate,
        "protonate_ph_min": args.protonate_ph_min,
        "protonate_ph_max": args.protonate_ph_max,
        "protonate_precision": args.protonate_precision,
        "protonate_max_variants": args.protonate_max_variants,
        "debug": args.debug,
        "include_raw": args.include_raw,
        "threads": args.threads,
        "strict_ligand_templates": args.strict_ligand_templates,
    }


def _batch_payload(results: dict[str, object], include_raw: bool) -> dict:
    return {
        "batch": True,
        "results": {
            batch_id: result.to_dict(include_raw=include_raw)
            for batch_id, result in results.items()
        },
    }


def _write_batch_json(results: dict[str, object], path: str | Path, *, include_raw: bool) -> None:
    payload = json.dumps(_batch_payload(results, include_raw), indent=2, sort_keys=True)
    Path(path).write_text(payload + "\n", encoding="utf-8")


def _write_batch_csv(results: dict[str, object], path: str | Path, *, include_raw: bool) -> None:
    fieldnames = [
        "batch_id",
        "batch_ligand_component_ids",
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
        for batch_id, result in results.items():
            ligand_ids = [
                component.component_id
                for component in result.components
                if component.molecule_type == "ligand"
            ]
            for row in result.rows(include_raw=include_raw):
                writer.writerow(
                    {
                        "batch_id": batch_id,
                        "batch_ligand_component_ids": ";".join(ligand_ids),
                        **row,
                    }
                )


def _safe_output_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "batch"


def _add_rule_cutoff_args(parser: argparse.ArgumentParser) -> None:
    cutoff_group = parser.add_argument_group("interaction cutoff overrides")
    for rule_name, cutoff_name, flags, help_text in RULE_CUTOFF_FLAGS:
        cutoff_group.add_argument(
            *flags,
            dest=_override_dest(rule_name, cutoff_name),
            type=float,
            default=None,
            metavar="VALUE",
            help=help_text,
        )


def _rule_overrides_from_args(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    overrides: dict[str, dict[str, float]] = {}
    for rule_name, cutoff_name, _, _ in RULE_CUTOFF_FLAGS:
        value = getattr(args, _override_dest(rule_name, cutoff_name))
        if value is not None:
            overrides.setdefault(rule_name, {})[cutoff_name] = value
    return overrides


def _override_dest(rule_name: str, cutoff_name: str) -> str:
    return f"override_{rule_name}_{cutoff_name}"


def _parse_ligand_smiles(values: list[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            parser.error("--ligand-smiles values must be shaped as NAME=SMILES")
        name, smiles = value.split("=", 1)
        name = name.strip().upper()
        smiles = smiles.strip()
        if not name or not smiles:
            parser.error("--ligand-smiles values must include both NAME and SMILES")
        parsed[name] = smiles
    return parsed
