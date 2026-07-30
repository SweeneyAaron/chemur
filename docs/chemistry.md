# Chemistry notes

How Chemur decides what an atom *is*, and why some geometrically valid contacts
are rejected.

## Template vs coordinates

For ligands, the final SMILES — whether supplied with `--ligand-smiles`, read from
an SDF, or fetched from the Chemical Component Dictionary — is authoritative for
**protonation, formal charge, donor status, acceptor status and aromaticity**.

Coordinates from the PDB/mmCIF remain authoritative for **atom positions**.

This split matters because deposited structures frequently disagree with the
chemistry: a ligand may be modelled without hydrogens, or with a protonation state
that does not hold at the pH you care about.

### Hydrogen policy

Two mismatches can arise, and each is handled explicitly rather than silently:

- **The structure has a hydrogen the template does not.** That hydrogen is marked
  ignored: it creates no donor feature, consumes no donor capacity, and does not
  act as an occluding atom.
- **The template expects a donor hydrogen the structure lacks coordinates for.**
  The donor interaction is kept only as a raw candidate, tagged
  `missing_donor_hydrogen_geometry`.

## σ-hole activation for chalcogen bonds

`chalcogen_bond` requires the donor to carry a *positive* σ-hole, which geometry
alone cannot establish. At the C–S bond extension of an ordinary Met or Cys the
electrostatic potential is negative — there is nothing to donate into.

A sulfur counts as activated when it is:

- aromatic (thiophene, thiazole, thiadiazole),
- bonded to N, O or a halogen,
- cationic (a sulfonium, such as S-adenosylmethionine), or
- conjugated into a C=S bearing N or O (thioamide, thiourea).

Plain thioethers, thiols, thioketones and disulfides are **not** activated.

Contacts that are geometrically perfect but electronically inert are still
reported, as raw candidates with `rejection_reason = "sigma_hole_not_activated"`.
Set `require_activated_sigma_hole: false` in the profile for pure-geometry
behaviour.

`chalcogen_pi` is deliberately **not** gated this way — Met-S···π is dispersion
driven rather than a σ-hole interaction, and requires no activation.

## Aromatic gating

Rules describing genuine π interactions carry `require_aromatic: true` in the
profile. A saturated ligand ring (cyclohexyl, piperidine) is therefore reported as
`aliphatic_pi_stack` rather than counted as π–π stacking. Set it to `false` to
restore the pre-0.2 behaviour.

`aliphatic_stack` (saturated–saturated) ships **disabled**; enable it with
`enabled: true` in the profile.

## Profiles

The active profile supplies every cutoff and gate described here. `default` ships
with the package and defines one rule per entry in
`chemur.interaction_types.INTERACTION_TYPES`. Pass `--profile` a name or a path to
your own YAML file, or override individual cutoffs on the command line — see
[cli.md](cli.md).
