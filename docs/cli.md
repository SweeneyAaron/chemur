# CLI reference

`chemur analyze` is the main entry point. This page covers the options in depth;
`chemur analyze --help` is always the authoritative list.

```bash
chemur analyze structure.cif --out interactions.json --csv interactions.csv
```

## Per-ligand output

By default the CLI also writes one filtered result per ligand component to
`ligand_outputs/`:

```bash
chemur analyze structure.cif --ligand-smiles-file ligands.json
```

Each per-ligand file contains the interactions involving that ligand, plus the
partner components, atoms and features needed to interpret them. Add
`--include-raw` to include raw candidates as well.

Choose a different directory or format:

```bash
chemur analyze structure.cif \
  --ligand-smiles-file ligands.json \
  --split-by-ligand-dir ligand_outputs \
  --split-by-ligand-format both
```

Disable splitting when you only want aggregate output:

```bash
chemur analyze structure.cif --no-split-by-ligand --out interactions.json
```

## Ligand coordinates from SDF

Use SDF ligands when coordinates should come from docked or external files rather
than from the structure:

```bash
chemur analyze receptor.pdb --ligand-sdf ligand.sdf
chemur analyze receptor.pdb --ligand-sdf-dir ligands/
```

SDF coordinates are used directly. Bond order, aromaticity, formal charge,
donor/acceptor perception, rings and the starting SMILES are read from the SDF
with RDKit. If SDF ligands are supplied, ligand residues already present in the
PDB/mmCIF are skipped so the SDF coordinates stay authoritative.

Multiple SDF ligands are added to the structure and analysed together:

```bash
chemur analyze receptor.pdb --ligand-sdf ligand_a.sdf --ligand-sdf ligand_b.sdf
```

Use `--batch` to analyse each SDF ligand independently against the same structure:

```bash
chemur analyze receptor.pdb --ligand-sdf-dir ligands/ --batch
```

Batch mode writes per-ligand outputs under one subdirectory per batch ligand, for
example `ligand_outputs/ligand_a_1_LIG/S_LIG_1.json`. Aggregate JSON or CSV can
still be written with `--out` or `--csv`.

## Ligand chemistry templates

Supply SMILES directly, or from a JSON file mapping component names to SMILES:

```bash
chemur analyze structure.cif --ligand-smiles LIG='CC(=O)N' --out interactions.json
chemur analyze structure.pdb --ligand-smiles-file ligands.json --csv interactions.csv
```

If neither is given, Chemur looks each ligand up in the RCSB Chemical Component
Dictionary by component name, using files such as
`https://files.rcsb.org/ligands/download/HEM.cif`. Ligands that cannot be resolved
are skipped with a warning; the rest continue through the normal pipeline.

```bash
chemur analyze structure.cif --no-ccd-smiles --out interactions.json   # disable the lookup
chemur analyze structure.cif --debug                                   # print the SMILES used
```

## Protonation

```bash
chemur analyze structure.cif \
  --ligand-smiles LIG='CC(=O)N' \
  --protonate \
  --protonate-ph-min 6.8 \
  --protonate-ph-max 7.4 \
  --out interactions.json
```

`--protonate` uses Dimorphite-DL and prints the final SMILES for each ligand to
stderr:

```text
[chemur] ligand LIG: using SMILES 'CC(=O)[NH3+]' (input 'CC(=O)N', variants=1)
```

CCD-derived SMILES are collected before protonation, so `--protonate` also works
when the SMILES came from the CCD fallback. It applies to SDF ligands too:

```bash
chemur analyze receptor.pdb --ligand-sdf ligand.sdf --protonate --debug
```

See [chemistry.md](chemistry.md) for how the template interacts with structure
hydrogens.

## Overriding cutoffs

Any cutoff in the active profile can be overridden from the command line:

```bash
chemur analyze structure.cif \
  --hb-distance 3.1 \
  --hb-angle 110 \
  --cation-pi-distance 5.5 \
  --out interactions.json
```

Flags follow the pattern `--<rule-name>-<cutoff>`, with the older hand-written
spellings (`--hb-distance`, `--pipi-offset`, …) kept as aliases. Every flag is
generated from `chemur.interaction_types.INTERACTION_TYPES`, so **`chemur analyze
--help` is always the complete and current list** — it cannot drift out of date
the way a hand-maintained table here would.

## Other subcommands

- `chemur trajectory` — per-frame interaction analysis over an MD trajectory,
  with occupancy and time series. Needs the `trajectory` extra.
- `chemur --version`, `chemur --help`.
