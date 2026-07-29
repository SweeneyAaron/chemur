# Chemur

[![PyPI](https://img.shields.io/pypi/v/chemur.svg)](https://pypi.org/project/chemur/)
[![Python](https://img.shields.io/pypi/pyversions/chemur.svg)](https://pypi.org/project/chemur/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Detect biomolecular interactions in PDB/mmCIF structures, and rescore docked
ligand poses against a known 3D binder.

- **Python API** — `chemur.analyze`, `chemur.analyze_batch`, `chemur.score_poses`
- **CLI** — `chemur analyze`, `chemur score`, `chemur trajectory`
- **22 interaction types, each with defined geometry** — hydrogen bonds (and
  weak/π/sulfur/low-barrier variants), salt bridges, metal coordination, amide and
  solvent bridges, π–π stacking, aliphatic–aromatic and aliphatic–aliphatic ring
  stacking, cation–π, anion–π, edgewise anion–aromatic, amide–π, halogen bonds,
  halogen–π, **chalcogen bonds**, chalcogen–π, **tetrel bonds**, **n→π\***, CH–π,
  and hydrophobic contacts. The σ-hole, stacking and n→π\* geometries follow
  [Adhav & Saikrishnan, *ACS Omega* **2023**, 8, 22268](https://doi.org/10.1021/acsomega.3c00205).
- **Template-driven ligand chemistry** — SMILES, SDF, or an automatic lookup in the
  RCSB Chemical Component Dictionary; the template is authoritative for protonation,
  charge, donor/acceptor status and aromaticity, while coordinates stay authoritative
  for geometry
- **Docked-pose rescoring** — a CoSPLIF + interaction-fingerprint hybrid scored
  against a reference binder
- **MD trajectories** — per-frame interaction analysis with occupancy and time series
- PDB/mmCIF parsing with [gemmi](https://gemmi.readthedocs.io); chemistry with
  [RDKit](https://www.rdkit.org)
- C++17/pybind11 geometry and neighbour-search hot loops, with a pure-Python fallback

A [ChimeraX plugin](https://github.com/SweeneyAaron/chimerax-chemur) builds on
this library for interactive 3D visualisation.

## Install

```bash
pip install chemur
```

Prebuilt wheels are published for CPython 3.10–3.13 on Linux x86_64, macOS
(x86_64 and Apple Silicon) and Windows x86_64. Any other platform installs from
the source distribution and needs a C++17 compiler and CMake ≥ 3.18.

### Optional extras

| Extra | Installs | Enables |
| --- | --- | --- |
| `protonation` | `dimorphite_dl` | pH-range ligand protonation (`--protonate`) |
| `trajectory` | `MDAnalysis` | reading trajectories in `chemur trajectory` |
| `dataframe` | `pandas` | `AnalysisResult.to_dataframe()` |
| `all` | all of the above | |

```bash
pip install "chemur[all]"
```

> `MDAnalysis` is GPL-2.0-or-later. Chemur itself is MIT and neither bundles
> nor links it, but installing the `trajectory` extra produces an environment
> subject to the GPL.

### Verify the compiled core

```bash
python -c "from chemur.core import USING_CPP_CORE; print('C++ core:', USING_CPP_CORE)"
```

Expected output is `C++ core: True`. If it prints `False`, the pure-Python
fallback loaded — correct results, but 10–100× slower. From a wheel this should
never happen; from an sdist build it means CMake or the compiler failed, so
re-run `pip install . -v` and read the build log.

Set `CHEMUR_FORCE_PYTHON_CORE=1` to select the fallback deliberately.

## Quick start

```python
import chemur

result = chemur.analyze("receptor.cif")
for interaction in result.interactions:
    print(interaction.interaction_type, interaction.atom_ids)

result.to_json("interactions.json")
```

```bash
chemur analyze receptor.cif --out interactions.json --csv interactions.csv
```

## Development

```bash
git clone https://github.com/SweeneyAaron/chemur
cd chemur

conda create -n chemur-dev -c conda-forge \
  python=3.11 rdkit gemmi numpy pyyaml pandas pytest \
  cmake ninja pybind11 scikit-build-core cxx-compiler
conda activate chemur-dev

pip install -e ".[test,all]" -v
pytest
```

On macOS you need the Apple command line tools (`xcode-select --install`). If
the build fails with `'cstddef' file not found` they are incomplete — repair
with `sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install`.

This project uses a `src/` layout and has no `conftest.py` on purpose, so
`pytest` always exercises the *installed* package rather than the source tree.
That is what makes the wheel-level checks in CI meaningful — run
`cd /tmp && python -m pytest <repo>/tests` if you want to prove it.

## CLI

```bash
chemur analyze structure.cif --ligand-smiles LIG='CC(=O)N' --out interactions.json
chemur analyze structure.pdb --ligand-smiles-file ligands.json --csv interactions.csv
```

By default, the CLI also writes one filtered result per ligand component to
`ligand_outputs/`:

```bash
chemur analyze structure.cif --ligand-smiles-file ligands.json
```

Choose a different split directory or output format with:

```bash
chemur analyze structure.cif \
  --ligand-smiles-file ligands.json \
  --split-by-ligand-dir ligand_outputs \
  --split-by-ligand-format both
```

Each per-ligand file contains interactions involving that ligand, plus the
partner components, atoms, and features needed to interpret those interactions.
Use `--include-raw` to include raw candidates in each split output too.

Disable per-ligand splitting when you only want aggregate JSON/CSV output:

```bash
chemur analyze structure.cif --no-split-by-ligand --out interactions.json
```

Use SDF ligands when the ligand coordinates should come from docked or external
ligand files rather than from the structure file:

```bash
chemur analyze receptor.pdb --ligand-sdf docked_ligand.sdf
chemur analyze receptor.pdb --ligand-sdf-dir docked_ligands/
```

SDF coordinates are used directly. Bond order, aromaticity, formal charge,
donor/acceptor perception, rings, and the starting SMILES are taken from the SDF
with RDKit. If SDF ligands are supplied, ligand residues already present in the
PDB/mmCIF are skipped so the SDF coordinates are authoritative.

Multiple SDF ligands are added to the receptor and analyzed together by default:

```bash
chemur analyze receptor.pdb \
  --ligand-sdf ligand_a.sdf \
  --ligand-sdf ligand_b.sdf
```

Use `--batch` to analyze each SDF ligand independently against the same
structure:

```bash
chemur analyze receptor.pdb \
  --ligand-sdf-dir docked_ligands/ \
  --batch
```

Batch mode writes per-ligand outputs under one subdirectory per batch ligand,
for example `ligand_outputs/ligand_a_1_LIG/S_LIG_1.json`. Aggregate batch JSON
or CSV can still be written with `--out` or `--csv`.

SDF ligands can still be protonated before chemistry mapping:

```bash
chemur analyze receptor.pdb \
  --ligand-sdf docked_ligand.sdf \
  --protonate \
  --debug
```

If no `--ligand-smiles` or `--ligand-smiles-file` is supplied, Chemur tries to
fetch ligand SMILES from the PDB Chemical Component Dictionary using each ligand
residue/component name. The CCD lookup uses RCSB's ligand definition files, for
example `https://files.rcsb.org/ligands/download/HEM.cif`. Ligands that cannot
be resolved are skipped with a warning, while resolved ligands continue through
the normal chemistry pipeline.

```bash
chemur analyze structure.cif --out interactions.json
```

Disable the automatic lookup if needed:

```bash
chemur analyze structure.cif --no-ccd-smiles --out interactions.json
```

Use `--debug` to print the final SMILES used for each ligand, including SMILES
found through the CCD fallback.

To protonate ligand SMILES automatically before chemistry mapping:

```bash
chemur analyze structure.cif \
  --ligand-smiles LIG='CC(=O)N' \
  --protonate \
  --protonate-ph-min 6.8 \
  --protonate-ph-max 7.4 \
  --out interactions.json
```

With `--protonate`, Chemur uses Dimorphite-DL and prints the final SMILES used
for each ligand to stderr, for example:

```text
[chemur] ligand LIG: using SMILES 'CC(=O)[NH3+]' (input 'CC(=O)N', variants=1)
```

CCD-derived SMILES are collected before protonation, so `--protonate` also works
when the ligand SMILES came from the CCD fallback.

The final SMILES is authoritative for ligand protonation, charge, donor status,
acceptor status, and aromaticity. Coordinates from the PDB/mmCIF remain
authoritative for atom positions. If the structure contains a ligand hydrogen
that is not present in the final SMILES, Chemur marks that hydrogen as ignored:
it will not create donor features, consume donor capacity, or act as an
occluding atom. If the final SMILES expects a donor hydrogen but the structure
does not contain coordinates for it, the donor interaction is kept only as a raw
candidate with `missing_donor_hydrogen_geometry`.

Cutoff values from the active profile can be overridden from the command line:

```bash
chemur analyze structure.cif \
  --ligand-smiles LIG='CC(=O)N' \
  --hb-distance 3.1 \
  --hb-angle 110 \
  --cation-pi-distance 5.5 \
  --out interactions.json
```

Useful cutoff flags include:

- `--hb-distance`, `--hb-angle`, `--hbond-sulfur-distance`, `--hbond-short-distance`
- `--weak-hb-distance`, `--weak-hb-angle`
- `--salt-bridge-distance`
- `--metal-distance`
- `--pipi-distance`, `--pipi-angle`, `--pipi-offset`
- `--aliphatic-pi-stack-distance`, `--aliphatic-pi-stack-angle`, `--aliphatic-pi-stack-offset`
- `--cation-pi-distance`, `--cation-pi-angle`, `--cation-pi-offset`
- `--anion-pi-distance`, `--anion-pi-angle`, `--anion-pi-offset`
- `--anion-aromatic-edge-distance`, `--anion-aromatic-edge-min-angle`
- `--hbond-pi-distance`, `--hbond-pi-angle`, `--hbond-pi-donor-angle`, `--hbond-pi-offset`
- `--n-pi-star-distance`, `--n-pi-star-min-angle`, `--n-pi-star-angle`, `--n-pi-star-approach-angle`
- `--halogen-bond-distance`, `--halogen-bond-angle`
- `--chalcogen-bond-distance`, `--chalcogen-bond-angle`
- `--halogen-pi-distance`, `--halogen-pi-offset`
- `--chalcogen-pi-distance`, `--chalcogen-pi-angle`, `--chalcogen-pi-offset`
- `--tetrel-bond-distance`, `--tetrel-bond-min-distance`, `--tetrel-bond-angle`
- `--ch-pi-distance`, `--ch-pi-angle`, `--ch-pi-donor-angle`, `--ch-pi-offset`
- `--hydrophobic-distance`

Every flag is generated from `chemur.interaction_types.INTERACTION_TYPES`, so
`chemur analyze --help` is always the complete list.

`chalcogen_bond` additionally requires the donor to carry a *positive* σ-hole, which
geometry cannot establish: at the C–S bond extension of an ordinary Met or Cys the
electrostatic potential is negative, so there is nothing to donate into. A sulfur
counts as activated when it is aromatic (thiophene, thiazole, thiadiazole), bonded to
N/O/halogen, cationic (sulfonium — S-adenosylmethionine), or conjugated into a C=S
bearing N or O (thioamide, thiourea). Plain thioethers, thiols, thioketones and
disulfides are not. Contacts that are geometrically perfect but electronically inert
are still reported, as raw candidates with
`rejection_reason = "sigma_hole_not_activated"`; set
`require_activated_sigma_hole: false` for pure-geometry behaviour.

`chalcogen_pi` is deliberately **not** gated this way — Met-S···π is dispersion driven
rather than a σ-hole interaction, and requires no activation.

Rules describing genuine π interactions carry `require_aromatic: true` in the
profile, so a saturated ligand ring (cyclohexyl, piperidine) is stacked as
`aliphatic_pi_stack` rather than counted as π–π. Set it to `false` to restore the
pre-0.2 behaviour. `aliphatic_stack` (saturated–saturated) ships disabled; enable
it with `enabled: true`.

## Python

```python
import chemur

result = chemur.analyze(
    "structure.cif",
    ligand_smiles={"LIG": "CC(=O)N"},
    scope="all",
    profile="default",
    rule_overrides={"hbond": {"distance": 3.1, "angle": 110.0}},
    protonate=True,
    protonate_ph_min=6.8,
    protonate_ph_max=7.4,
    include_raw=False,
)
```

Use SDF ligands from Python:

```python
result = chemur.analyze(
    "receptor.pdb",
    ligand_sdf=["ligand_a.sdf", "ligand_b.sdf"],
)

batch_results = chemur.analyze_batch(
    "receptor.pdb",
    ligand_sdf_dir="docked_ligands",
    protonate=True,
)
```

## Scoring docked poses against a known 3D binder

`chemur score` ranks a set of docked ligand poses by how closely each one
reproduces the protein–ligand interaction pattern (and 3D pharmacophore shape) of
a **known binder** — for example an experimental co-crystal ligand. It is the
practical "is this docked pose like the real thing?" tool.

Under the hood each pose and the reference are run through the normal Chemur
interaction analysis, then compared with the **`cosplif_pose_v1`** profile, a
hybrid scorer tuned on the CASF-2016 screening benchmark. The final score blends
three signals:

- **CoSPLIF tokens** — contextual interaction-fingerprint tokens (containment +
  Tanimoto, with background IDF weighting),
- **ligand blocks** — overlap of the ligand's 3D chemical/pharmacophore blocks
  with the reference (weight `0.40`), and
- **protein-IFM** — a protein-anchored interaction-fingerprint strict similarity
  (weight `0.30`).

Higher is more similar to the known binder (scores are roughly `0`–`1`). All poses
are scored against the same receptor, and the reference itself is excluded from the
ranking unless you ask to keep it.

### CLI

The minimal invocation needs a receptor, the docked poses, and the reference:

```bash
chemur score receptor.pdb \
  --ligand-sdf-dir docked_poses/ \
  --reference-sdf crystal_ligand.sdf \
  --out scores.json \
  --report scores.txt
```

Supply poses as a directory (`--ligand-sdf-dir`, repeatable) and/or individual
files (`--ligand-sdf`, repeatable). Specify the known binder in one of two ways
(one is required; `--reference-sdf` takes precedence if both are given):

- **`--reference-sdf ref.sdf`** — a separate SDF of the known binder (analyzed
  against the same receptor). Use this for a crystal/reference ligand.
- **`--reference-id POSE`** — use one of the supplied poses as the reference,
  identified by its pose key (the SDF file's stem). Mirrors the same-target
  fingerprint-matching workflow.

```bash
# Reference is one of the docked poses (e.g. the top-ranked or a manually chosen one)
chemur score receptor.pdb --ligand-sdf-dir docked_poses/ --reference-id pose_017 --out scores.json
```

Outputs:

- `--out scores.json` — ranked results as a JSON list (best first).
- `--csv scores.csv` — ranked results as a CSV table.
- `--report scores.txt` — a human-readable ranking (`rank  score  ligand`).
- With neither `--out` nor `--csv`, the JSON is printed to stdout.

Each result row contains:

| Field | Meaning |
|---|---|
| `rank` | 1 = most similar to the reference |
| `ligand_id` | pose key (SDF file stem) |
| `source_component_id` | the ligand component id inside that pose |
| `cosplif_score` | final blended score (the ranking key) |
| `cosplif_base_score` | CoSPLIF token + ligand-block score before the IFM blend |
| `cosplif_ligand_block_score` | 3D pharmacophore-block overlap component |
| `strict_ifm_similarity` | protein-anchored interaction-fingerprint component |
| `cosplif_containment`, `cosplif_tanimoto`, `cosplif_token_overlap` | token-level diagnostics |
| `is_reference` | `true` only for the reference row (present with `--include-reference`) |

Other useful flags:

- `--profile NAME` — scoring profile (default `cosplif_pose_v1`).
- `--cosplif-config profiles.json` — define custom scoring profiles (see below).
- `--analysis-profile NAME` — the Chemur interaction-detection rule profile used
  during analysis (default `default`); distinct from the scoring `--profile`.
- `--protonate`, `--protonate-ph-min`, `--protonate-ph-max` — protonate ligands
  before chemistry mapping (same as `analyze`).
- `--include-reference` — keep the reference in the ranked output.
- `--ifm-engine {python,auto,cpp}` (default `python`) and `--ifm-workers` —
  control the protein-IFM scoring engine. The Python engine needs no compiled
  extension; `auto` uses the C++ scorer when the engine exposes it and otherwise
  falls back to Python.

### Python API

One-call workflow (analyze + score + rank):

```python
import chemur

ranked = chemur.score_poses(
    "receptor.pdb",
    ligand_sdf_dir="docked_poses",     # or ligand_sdf=["a.sdf", "b.sdf"]
    reference_sdf="crystal_ligand.sdf", # or reference_id="pose_017"
    protonate=True,
)
for row in ranked[:5]:
    print(row["rank"], row["ligand_id"], round(row["cosplif_score"], 3))
```

If you have already run analysis, score directly from `AnalysisResult` objects:

```python
from chemur import analyze, analyze_batch, build_scoring_batch, score_pose_batch

results = analyze_batch("receptor.pdb", ligand_sdf_dir="docked_poses")
results["__reference__"] = analyze("receptor.pdb", ligand_sdf="crystal_ligand.sdf")

batch = build_scoring_batch(results)              # adds CoSPLIF + IFM payloads
ranked = score_pose_batch(batch, reference_ligand_id="__reference__")
```

`score_poses` accepts the same scoring controls as the CLI (`profile`,
`cosplif_config`, `ifm_engine`, `ifm_workers`, `include_reference`) plus the usual
analysis options (`analysis_profile`, `protonate`, `scope`, …).

### Profiles and tuning

The default `cosplif_pose_v1` works out of the box. To experiment with the scoring
weights without code changes, pass a JSON config that defines custom named
profiles, each starting from a built-in `base` and overriding fields:

```json
{
  "profiles": {
    "pose_bw050":   {"base": "cosplif_pose_v1", "ligand_block_weight": 0.50},
    "pose_ifm045":  {"base": "cosplif_pose_v1", "hybrid_ifm_weight": 0.45},
    "tokens_only":  {"base": "cosplif_pose_v1", "ligand_block_weight": 0.0, "hybrid_ifm_weight": 0.0}
  }
}
```

```bash
chemur score receptor.pdb --ligand-sdf-dir docked_poses/ \
  --reference-sdf crystal_ligand.sdf \
  --cosplif-config profiles.json --profile pose_bw050 --out scores.json
```

Tunable fields include `ligand_block_weight`, `hybrid_ifm_weight`,
`containment_weight`/`tanimoto_weight`, `idf_scale`, `min_multiplier`/`max_multiplier`,
`ligand_block_spatial_cutoff`, and the per-chemistry `sigma_*` spatial tolerances.

## ChimeraX plugin

Interactive 3D visualisation lives in a separate distribution,
[ChimeraX-Chemur](https://github.com/SweeneyAaron/chimerax-chemur):
interactions drawn as pseudobonds, a searchable results table, 2D interaction
diagrams, model and docked-pose comparison figures, and trajectory plots. It
declares `chemur` as a dependency, so installing the bundle from the
ChimeraX Toolshed pulls this library in automatically.

## License

MIT — see [LICENSE](LICENSE).

## Citation

A manuscript is in preparation. Until then, please cite the repository:

```
Sweeney, A. Chemur: biomolecular interaction detection and docked-pose
rescoring. https://github.com/SweeneyAaron/chemur
```
