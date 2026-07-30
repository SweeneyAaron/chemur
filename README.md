# Chemur

[![PyPI](https://img.shields.io/pypi/v/chemur.svg)](https://pypi.org/project/chemur/)
[![Python](https://img.shields.io/pypi/pyversions/chemur.svg)](https://pypi.org/project/chemur/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Find the non-covalent interactions in a PDB or mmCIF structure — between proteins,
nucleic acids, ligands, metals and solvent, in any combination.

```bash
pip install chemur
chemur analyze structure.cif --out interactions.json
```

- **22 interaction types, each with defined geometry** 
- **Any pair of components.** Protein–ligand, protein–protein, protein–DNA/RNA,
  nucleic acid–ligand, ligand–ligand. Nothing is hard-coded to expect a receptor and
  a small molecule.
- **Chemistry from templates, geometry from coordinates.** Ligands are typed from
  SMILES, an SDF, or an automatic Chemical Component Dictionary lookup.
- **MD trajectories** — per-frame analysis with occupancy and time series.

A [ChimeraX plugin](https://github.com/SweeneyAaron/chimerax-chemur) builds on this
library for interactive 3D visualisation.

## Install

```bash
pip install chemur
```

Wheels are published for CPython 3.10–3.13 on Linux x86_64, macOS (a single
universal2 build covering Intel and Apple Silicon) and Windows x86_64. Other
platforms build from the source distribution and need a C++17 compiler and CMake ≥ 3.18.

| Extra | Installs | Enables |
| --- | --- | --- |
| `trajectory` | `MDAnalysis` | reading trajectory files |
| `dataframe` | `pandas` | `AnalysisResult.to_dataframe()` |
| `all` | both | |

```bash
pip install "chemur[all]"
```

pH-range ligand protonation (`--protonate`) needs no extra — `dimorphite_dl` is a hard
dependency. It requires `rdkit<2026`, so a Chemur install is capped there too.

> `MDAnalysis` is LGPL-3.0-or-later (it relicensed from GPL at 2.8.0; the `>=2.4` floor
> still permits older GPL-licensed releases if you pin one). Chemur itself is MIT and
> neither bundles nor links it, but installing the `trajectory` extra adds
> copyleft-licensed code to your environment.

To confirm you got the compiled core rather than the fallback:

```bash
python -c "from chemur.core import USING_CPP_CORE; print('C++ core:', USING_CPP_CORE)"
```

`True` is expected. `False` means the pure-Python fallback loaded — correct results,
10–100× slower. From a wheel that should never happen; from a source build it means
CMake or the compiler failed. Set `CHEMUR_FORCE_PYTHON_CORE=1` to select it deliberately.

## Quick start

```python
import chemur

result = chemur.analyze("structure.cif")

for interaction in result.interactions:
    print(interaction.interaction_type, interaction.component_ids)

result.to_json("interactions.json")
```

```bash
chemur analyze structure.cif --out interactions.json --csv interactions.csv
```

## Any biomolecule, not just protein–ligand

`analyze` returns **every** interaction it finds in the structure. There is no receptor
or ligand role — you pick out the ones you want by component.

```python
import chemur
from collections import Counter

result = chemur.analyze("complex.cif")

# What was found, and between what?
print(Counter(i.interaction_type for i in result.interactions))
```

A protein–DNA interface, a protein–protein interface and a ligand binding site are all
the same query — filter on the components involved:

```python
def between(result, a, b):
    """Interactions with one atom in component `a` and the other in `b`."""
    return [
        i for i in result.interactions
        if {a, b} <= set(i.component_ids)
    ]

interface = between(result, "A:ARG:145", "B:DG:12")   # protein side chain to a base
```

Components are identified by `component_id`; `result.components` lists them all with
their names, so you can select a chain, a residue range, or a single ligand. Nucleic
acid phosphate backbones are perceived as anions, protein Arg/Lys as cations and
Asp/Glu as carboxylate anions, so salt bridges and cation–π across a protein–nucleic
acid interface are detected the same way as in a binding site.

## What it detects

| Family | Types |
| --- | --- |
| Hydrogen bonding | `hbond`, `weak_hbond`, `hbond_pi`, `solvent_bridge`, `amide_bridge` |
| Electrostatic | `salt_bridge`, `metal_coordination`, `cation_pi`, `anion_pi`, `anion_aromatic_edge` |
| Stacking | `pipi_stack`, `aliphatic_pi_stack`, `aliphatic_stack`, `amide_pi`, `ch_pi` |
| σ-hole | `halogen_bond`, `chalcogen_bond`, `tetrel_bond`, `halogen_pi`, `chalcogen_pi` |
| Other | `n_pi_star`, `hydrophobic` |

`aliphatic_stack` (saturated–saturated ring stacking) ships disabled; enable it in the
profile. The σ-hole, stacking and n→π\* geometries follow
[Adhav & Saikrishnan, *ACS Omega* **2023**, 8, 22268](https://doi.org/10.1021/acsomega.3c00205).

Each type has its own geometric criteria, all adjustable — see
[docs/cli.md](docs/cli.md) for overriding cutoffs and [docs/chemistry.md](docs/chemistry.md)
for the chemical gates (such as why an ordinary Met sulfur cannot donate a chalcogen bond).

## Ligand chemistry

Ligands need a chemical template so protonation, charge, aromaticity and
donor/acceptor status are correct. Chemur takes one from, in order of precedence:

```bash
chemur analyze structure.cif --ligand-smiles LIG='CC(=O)N'   # explicit SMILES
chemur analyze receptor.pdb --ligand-sdf docked.sdf          # an SDF (coordinates too)
chemur analyze structure.cif                                 # automatic CCD lookup
```

The automatic lookup resolves each ligand by its component name against the RCSB
Chemical Component Dictionary. Add `--protonate` to set the protonation state for a pH
range with Dimorphite-DL, and `--debug` to print the SMILES actually used.

Full detail in [docs/cli.md](docs/cli.md).

## Multiple ligands and docking output

Pass several ligands at once, or point at a directory of sdf files.

```bash
chemur analyze receptor.pdb --ligand-sdf a.sdf --ligand-sdf b.sdf   # repeatable
chemur analyze receptor.pdb --ligand-sdf-dir poses/                 # a whole directory
```

By default all supplied ligands are added to the structure and analysed **together**, as
one system — right for a cofactor plus a substrate, or two ligands sharing a pocket.

Docked poses are the opposite case: each is an alternative for the *same* site, so
analysing them together would stack overlapping copies into one pocket. Use `--batch` to
analyse each ligand independently against the same receptor:

```bash
chemur analyze receptor.pdb --ligand-sdf-dir poses/ --batch --out all_poses.json
```

Either way you also get one filtered file per ligand under `ligand_outputs/` — in batch
mode under a subdirectory per pose — while `--out` / `--csv` write the aggregate.

SDF coordinates are used directly, and ligand residues already present in the structure
are skipped so the SDF stays authoritative.

## Trajectories

```bash
pip install "chemur[trajectory]"
chemur trajectory topology.pdb trajectory.xtc --out frames.json
```

Analyses each frame and reports per-interaction occupancy and time series.

## Documentation

- [docs/cli.md](docs/cli.md) — full command-line reference
- [docs/chemistry.md](docs/chemistry.md) — templates, hydrogen policy, σ-hole activation, profiles

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

On macOS you need the Apple command line tools (`xcode-select --install`). If the build
fails with `'cstddef' file not found` they are incomplete — repair with
`sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install`.

This project uses a `src/` layout and has no `conftest.py` on purpose, so `pytest` always
exercises the *installed* package rather than the source tree. That is what makes the
wheel-level checks in CI meaningful.

## License

MIT — see [LICENSE](LICENSE).

## Citation

A manuscript is in preparation. Until then, please cite the repository:

```
Sweeney, A, Genz, L, Topf, M. Chemur: biomolecular interaction detection.
https://github.com/SweeneyAaron/chemur
```
