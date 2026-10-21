# Explicit 1MHT chemical products

These documents are opt-in corrections for the deposited instances in **1MHT**.
They are never selected automatically from the residue name `C36`.

- `1MHT-B-C36-407.json`: the unreacted fluorinated nucleotide at B407, without a C5 methyl group.
- `1MHT-C-C36-427.json`: the methylated fluorinated nucleotide at C427, covalently attached from C6 to A:CYS:81 SG.
- `1MHT-C36-products.json`: both corrections in one document.

Each product retains the deposited fluorine, sugar atom identities and sugar
stereochemistry. The bases use neutral amino/oxo chemistry. C5/C6 stereochemistry
in the reacted product is deliberately unspecified. Each corrected phosphodiester
carries its normal DNA negative charge on O2P. The base remains neutral; this is
an explicit state of these two products, not a global protonation or tautomer
replacement.

Atom references include full component identities and coordinate model 1.
`source_entry_id: "1MHT"` prevents applying these files to another deposited entry.
The product includes real neighbouring polymer context. Dummy atoms represent
explicitly named boundary bonds outside that context, not additional ligand atoms.

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m chemur analyze test_data/structures/1MHT.cif \
  --assembly-overrides examples/assembly_overrides/1MHT-C36-products.json \
  --out test_data/1MHT-corrected.json
```

Both C36 instances remain nucleotides, and Cys81 remains protein. They do not
become ligands or eligible hydrophobic-enclosure centres. Preparation status,
product mappings and diagnostics are exported in `modified_polymer_chemistry`.
