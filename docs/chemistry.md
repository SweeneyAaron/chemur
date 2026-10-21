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

Hydrogen-dependent distances and angles must be satisfied by the same bonded
hydrogen. Conventional non-water bonds also require a real bonded acceptor
antecedent; see [geometry.md](geometry.md).

Two mismatches can arise, and each is handled explicitly rather than silently:

- **The structure has a hydrogen the template does not.** That hydrogen is marked
  ignored: it creates no donor feature, consumes no donor capacity, and does not
  act as an occluding atom.
- **The template expects a donor hydrogen the structure lacks coordinates for.**
  The donor interaction is kept only as a raw candidate, tagged
  `missing_donor_hydrogen_geometry`.

## Hydrogen-bond acceptors

Ligand acceptor eligibility uses RDKit's Lipinski acceptor SMARTS on a temporary
copy with explicit and isotope hydrogens removed. The deposited coordinates and
hydrogen attachments are preserved in the analysis. This gives the same
heavy-atom acceptor classification whether hydrogens were implicit in SMILES or
explicit in an SDF. Eligible nitrogens have capacity 1; eligible oxygens and
sulfurs have capacity 2. These capacities control assignment and are separate
from the decision that an atom can accept at all.

Protein and nucleotide acceptors use standard residue/atom tables. Peptide and
side-chain amide nitrogens, tryptophan's pyrrole nitrogen, Lys NZ and Arg
guanidinium nitrogens are not acceptors. Recognized carbonyl, alcohol and sulfur
sites and nucleotide ring nitrogens retain their supported roles. Deposited
positive charge excludes an acceptor. An explicit O–H bond excludes the
proton-bearing oxygen of a carboxyl or phosphate group; ordinary alcohol oxygen
can still accept.

Histidine requires enough evidence to distinguish its ring-nitrogen roles. An
explicit ring hydrogen can establish the neutral tautomer; two protonated ring
nitrogens cannot accept. Sites whose state remains unresolved are excluded
conservatively, with reasons in chemistry metadata and a warning aggregated by
component. This does not add hydrogen coordinates or infer a pH-dependent protein
protonation state. Atoms record `acceptor_source` and, for unresolved sites,
`acceptor_unresolved_reason`; components summarize these sites in
`acceptor_unresolved_sites`.

## Ligand atom correspondence

Template mapping uses the whole heavy-atom topology and evaluates competing
correspondences rather than choosing the first match. Evidence is ranked by
excess observed hydrogens, available bond-order mismatches, and deposited nonzero
charge mismatches, in that order. Missing hydrogen coordinates provide no evidence
against a mapping. The final template still determines chemistry: an authoritative
deprotonated template can deliberately exclude a deposited hydrogen.

Equally supported mappings must assign equivalent chemistry and feature
memberships at each physical atom. A proton-bearing oxygen cannot be exchanged
with a carbonyl oxygen simply because their heavy-atom connections match.
Unresolved mappings are skipped with a diagnostic; `strict_ligand_templates=True`
or `--strict-ligand-templates` raises a template error instead. The bounded search
allows at most 4,096 complete mappings and 100,000 search states; exhaustion also
produces a diagnostic rather than an arbitrary assignment. Search diagnostics
include `template_mapping_status` (`unique`, `equivalent`, `ambiguous`, or
`search_limit`) and the ranked evidence in `template_mapping_evidence`.

Hydrogen-free terminal oxygens on the same C/N/P/S centre may be equivalent under
resonance. Such equivalence preserves total charge and all other chemical roles;
a stable atom identity chooses the representative localized charge assignment.
Proton-bearing oxygens remain distinct. Consequently, an unhydrogenated neutral
phosphate with insufficient evidence to locate its protons may be skipped even
though a fully deprotonated, resonance-equivalent phosphate can be mapped.

Ordinary SDF inputs retain their direct atom correspondence. Remapping after
protonation or a SMILES override can use the SDF's bond-order information. Applying
a replacement template clears obsolete template-generated hydrogen exclusions.

## Connected ligands and covalent attachments

Chemur recognizes connected ligand residues before assigning chemistry. Named
atom/bond dictionaries and explicit covalent connections from mmCIF or PDB are
used to construct the bound product. Cross-residue bonds are never inferred from
proximity. Metal coordination and hydrogen bonds do not merge ligands.

For example, `1ANF.cif` describes maltose as two `GLC` residues. The declared
glycosidic linkage removes the second glucose's template O1/HO1 and the first
glucose's HO4, giving one 23-heavy-atom ligand. The oxygen connecting the sugars
has no donor hydrogen. Complete embedded chemistry allows this case to work with
CCD fetching disabled.

Residue names, atom IDs and coordinates remain unchanged. A connected ligand has
an additional assembly ID, `assembly:<smallest member component ID>`; a singleton
retains its original component ID. `result.split_by_ligand()` produces one result
per complete ligand. `result.select_ligand(id)` accepts the assembly ID or any of
its member component IDs. IFM and CoSPLIF use the whole ligand while retaining
the original residue labels, and exclude contacts internal to that ligand from
ligand–target scoring.

Assemblies and their chemistry are recorded in `metadata.ligand_assemblies`;
components and atoms carry `ligand_assembly_id`. Assembly records include member
IDs, source links, the serialized product and its atom mapping, status and
diagnostics. Source component records continue to partition the original atoms.

Only declared leaving atoms are removed from the chemical template. Unexplained
missing heavy atoms, contradictory deposited atoms, ambiguous links and
incomplete product chemistry skip the affected ligand with a diagnostic; strict
template mode raises an error. Uncertain attachment residues have their chemical
features suppressed. Their deposited coordinates remain available for occlusion.
Raw contacts within one or two covalent bonds across an imported link are tagged
`covalent_neighborhood`.

Protein/nucleotide attachments influence product chemistry without absorbing the
polymer into the ligand. Automatic support requires the resulting chemistry to
be specified by the dictionaries and link operations; Chemur does not infer an
unspecified covalent reaction. Grouped features spanning an attachment boundary
are not treated as wholly ligand-owned scoring features. Ligand fingerprints
cannot traverse into the attached polymer.

`--protonate` operates on the complete unattached assembly. Covalently attached
products, and explicit product overrides, retain their validated protonation and
charge state; this choice is recorded in diagnostics. Protonation does not add
hydrogen coordinates. SDF replacement continues to replace embedded ligand
coordinates and does not inherit attachments to the omitted original residues.

### Explicit product overrides

Pass `assembly_overrides=<dict or JSON path>` to the Python analysis API, or
`--assembly-overrides FILE` to the CLI. The version-1 document contains a
`products` list. Each product identifies its ligand `components`, supplies an
atom-mapped final `product_smiles`, and maps every real heavy atom through
`atom_map` entries containing `component` and `atom` (plus `model`/`altloc` when
needed). Product chemistry is authoritative, including changes to internal bond
orders. Coordinates are always those of the referenced deposited atoms.

For example, this declares ethene at two existing ligand carbon sites:

```json
{
  "version": 1,
  "products": [{
    "components": ["B:LIG:1"],
    "product_smiles": "[CH2:1]=[CH2:2]",
    "atom_map": {
      "1": {"component": "B:LIG:1", "atom": "C1"},
      "2": {"component": "B:LIG:1", "atom": "C2"}
    }
  }]
}
```

An attached product also includes real polymer context in its SMILES and mapping,
and lists map numbers to retype in `update_polymer_atoms`. The context must cover
complete affected rings/conjugated groups and at least three bonds around every
rewritten polymer site. A truncated context requires mapped wildcard atoms and
a `boundary_map` from each wildcard's map number to its omitted deposited
neighbor. Cuts must be known single, nonaromatic, nonconjugated, nonring bonds
outside that neighborhood. Boundary atoms have no coordinates or output IDs.
Overlapping products, duplicate mappings, omitted ligand heavy atoms, conflicting
links and incomplete context are rejected instead of generating artificial caps.

The same assembly preparation runs once per trajectory topology. Atom IDs and
assembly membership survive solvent/ion filtering and multiprocessing.

### Modified polymer residues

Deposited entity and sequence membership, together with CCD linking type,
determine polymer roles before ligand filtering. Author residue numbers are
labels: sequence indices determine neighbouring polymer residues, including
insertion codes. An explicitly nonpolymer component remains a ligand even when
its name also identifies a standard amino acid or nucleotide.

Validated chemical products are separate from ligand assemblies. Modified
polymer products supply their validated rings, carbonyls, amides, charges and
donor/acceptor features while retaining their protein or nucleotide role.
Ordinary peptide and phosphodiester links use explicit sequence-backed
transformations. Ambiguous reactions and unexplained missing atoms remain
unresolved, with diagnostics; unrelated canonical flanking residues retain their
chemistry.

The product-override format also accepts modified polymer components. An optional
`source_entry_id` restricts a product to its deposited structure; complete atom
maps, reaction context and boundary validation still apply. Native result metadata
contains `chemical_products` and `modified_polymer_chemistry`, including status,
product SMILES, transformations, source identities and diagnostics. Serialized
`boundary_template_mapping` identifies each diagram boundary anchor's external
atom without creating an additional ligand atom or contact.

The [1MHT correction documents](../examples/assembly_overrides/README.md) are
explicit, instance-scoped examples for B:C36:407 and C:C36:427. The former is an
unreacted fluorinated nucleotide; the latter is the methylated covalent
intermediate attached through C6 to CYS81 SG. They preserve fluorine, sugar
stereochemistry and anionic phosphodiester O2P; unsupported new stereochemistry
is left unspecified. They are never loaded automatically and never replace all
residues named C36. Pass the combined document with `--assembly-overrides` or
ChimeraX's `assemblyOverrides` option.

## Water bridges

A `solvent_bridge` describes two assigned hydrogen bonds sharing a solvent atom
and connecting distinct nonsolvent components. Both hydrogen bonds remain in the
output, and the bridge adds no resource consumption. Thus a total interaction
count includes the two bonds and their composite bridge; counts by interaction
type distinguish them.

Angle and occlusion checks run before bridge construction. A bridge is assigned
only if both legs also survive resource assignment. A raw bridge whose leg loses
assignment records `unassigned_hbond_leg`. Bridge metadata identifies its donor
and acceptor atoms, leg features, and accepted leg assignment orders, retaining
directionality for downstream scoring.

`solvent_bridge.distance` limits **each leg**, in addition to the hydrogen-bond
rule's own limits. The bridge's reported distance remains the **sum** of the leg
distances. Bridge priority orders derived bridges after ordinary interactions;
it cannot override leg acceptance. The default bridge `consume` map is empty,
and a supplied consume map does not charge its legs a second time. Water–water
chains are not represented by this interaction type.

## Activated carbon donors for weak hydrogen bonds

`weak_hbond` requires C-O distance ≤3.6 Å, C-H-O angle ≥130° and an
electronically activated C–H donor. Chemur uses a
structural heuristic: a carbon qualifies when it is:

- directly bonded to N, O, F, Cl, Br or I;
- directly bonded to the carbon of a carbonyl (C=O) or nitrile (C≡N);
- directly bonded to a sulfonyl sulfur with two oxo oxygens; or
- a terminal alkyne carbon (one heavy-atom neighbour, joined by a C≡C bond).

Aromaticity alone does not qualify. Ordinary alkyl and phenyl C–H groups, and
C–H groups next to a neutral thioether, are excluded unless another qualifying
group is present. Negatively charged donor carbons or immediate activating atoms
do not qualify; this restriction is local, so an anionic ligand may still contain
eligible donors. Charge-separated representations of sulfonyl groups are
recognised as equivalent to S(=O)₂.

Ligand activation comes from the authoritative RDKit template, including its bond
orders. For proteins and nucleotides, Chemur uses bonded neighbours and standard
residue carbonyl templates: backbone and side-chain carbonyls, plus the carbonyls
of cytosine, guanine, uracil and thymine. The necessary atoms and bonds must be
present. Deposited atom charges are also consulted for activation. A generic C–O
connection is not enough to infer a carbonyl. Unrecognised
environments are treated as unactivated; the rules are a documented approximation,
not a calculation of electronic structure or hydrogen-bond energy.

Activation and geometry are checked separately. A geometrically valid contact
from an unactivated donor appears only in raw output, with
`rejection_reason = "carbon_not_activated"`. It consumes no donor or acceptor
capacity. Existing missing-hydrogen and angle rejection reasons take precedence.
Hydrogens ignored by the ligand template cannot supply donor geometry.

`AtomRecord.weak_hbond_activated` records the chemical classification and defaults
to `False` for manually constructed atoms. The field is appended to the dataclass
to preserve existing positional arguments and is included in serialized atom
records. Weak hydrogen-bond candidate metadata also includes
`weak_hbond_activated`.

This requirement is always applied to `weak_hbond`; the profile still controls its
distance and angle cutoffs. C–H···π contacts keep their existing carbon-donor
typing and capacity. Compared with earlier versions, accepted weak hydrogen-bond
counts may decrease, and freed capacity may change other assigned interactions
and downstream scores. Raw output retains the rejected contacts for comparison.

The expanded ligand motifs include experimentally observed C–H donors adjacent
to [sulfonyl groups](https://journals.iucr.org/e/issues/2021/02/00/ey2004/) and in
[terminal alkynes](https://pubmed.ncbi.nlm.nih.gov/39930554/).

## Chalcogen and sulfur-aromatic contacts

Atom-atom `chalcogen_bond` now uses published S-O, Se-O and Se-S distance windows
and a minimum C-S/Se-acceptor angle of 130°. Methionine-like donors are included;
the former activated-sigma-hole gate is removed. A geometric classification does
not assert a calculated positive electrostatic potential or interaction energy.

`chalcogen_pi` separately accepts directional thioether/disulfide sulfur-pi and
broad thioether/sulfoxide sulfur-aromatic contacts. Each sulfur/ring pair is
counted once, with all matching definitions in `geometry_classifications`.
It consumes no chalcogen-donor resource. Sulfoxide resonance depictions are
normalised for feature perception so charge notation alone cannot add ionic
contacts. See [geometry.md](geometry.md) for the exact endpoints and sources.

## Aromatic gating

Rules describing genuine π interactions carry `require_aromatic: true` in the
profile. A saturated ligand ring (cyclohexyl, piperidine) is therefore reported as
`aliphatic_pi_stack` rather than counted as π–π stacking. Set it to `false` to
restore the pre-0.2 behaviour.

`aliphatic_stack` (saturated–saturated) ships **disabled**; enable it with
`enabled: true` in the profile.

## Tetrel donors and fitted planes

Tetrel donors require validated nonaromatic sp3 carbon and a supported C/N/O/S
root; acceptors are carbonyl oxygens. Both donor-side and acceptor-side angles
must fall in 160–180°, with C-O distance 2.5–3.6 Å. The former additional
hydrogen-angle exclusion is removed. Validated chemical bonds replace the
publication's additional crystallographic covalent-bond-length screening.

Python and C++ use the same scaled symmetric eigensolver for fitted planes. The
reported RMS deviation is measured from the returned plane, including when the
smallest covariance eigenvalue is repeated. In that case no unique plane normal
exists; Chemur chooses a deterministic unit normal from the minimum-eigenvalue
space. A tetrahedral point set therefore has nonzero deviation rather than being
reported as planar. The `plane_fit` return format remains `(normal, rmsd)`.

## Trajectory atom identity

Excluding solvent or ions preserves every retained atom's original topology ID.
Filtered IDs can have gaps: use an ID lookup rather than indexing the filtered
atom list by `atom_id`. Coordinate columns continue to follow the original
topology order, and ligand rings, carbonyls, amides and hydrogen references retain
their identities across frames and worker processes.

These corrections can change assigned interaction counts and downstream scores.
Acceptor and tetrel eligibility may remove previously accepted contacts; valid
water bridges may now appear alongside their constituent hydrogen bonds. Raw
candidate output and chemistry diagnostics explain exclusions.

## Hydrophobic enclosure

`hydrophobic_enclosure` reports one group per hydrophobic atom lying strictly
inside the three-dimensional convex hull of its contacting hydrophobic neighbours.
Only ligand atoms can be enclosure centres. Surrounding atoms may be eligible
ligand, protein or nucleotide atoms. It uses only contacts that pass the existing
entity, covalent-neighbourhood and occlusion checks. Contacts within one connected
ligand assembly cannot establish its enclosure.

Hydrogens directly bonded to either endpoint of a hydrophobic contact are exempt
from that contact's occlusion check. They belong to the participating group, so
adding those hydrogens alone does not remove its contact or an enclosure supported
by it. Unrelated hydrogens, hydrogens more than one bond away, and intervening heavy
atoms still undergo the existing occlusion check. This exemption applies to
ordinary hydrophobic contacts before enclosure construction; other interaction
types retain their existing occlusion rules.

At least four neighbours are needed. A tetrahedral or octahedral surround qualifies;
a planar ring, two opposing contacts, a one-sided shell or an atom on the hull
boundary does not. The test is independent of coordinate orientation: after
normalizing directions from the centre, every nonparallel pair defines a plane
through it, and the remaining directions must span both sides. Both geometry
backends use a cross-product tolerance of `1e-12` and a signed-projection tolerance
of `1e-9`. Zero-length vectors are ignored and nonfinite coordinates fail the test.
This measures geometric enclosure, not solvent-accessible surface area or a binding
free-energy contribution.

Enclosure is enabled by default with priority 135 and consumes no resources. It
inherits the `hydrophobic.distance` cutoff (4 Å by default), including CLI overrides.
Disabling `hydrophobic` disables its supporting contacts and therefore enclosure.
To retain ordinary pairs instead of enclosure groups, set:

```yaml
rules:
  hydrophobic_enclosure:
    enabled: false
```

All groups are constructed before assignment replaces their centre–neighbour pairs.
Overlapping enclosures remain separate groups; unrelated and neighbour–neighbour
contacts remain ordinary contacts unless they independently support an enclosure.
Replaced raw candidates have `rejection_reason = "subsumed_by_hydrophobic_enclosure"`
and `metadata.enclosure_center_atom_ids` identifies the responsible centres.
With CLI `--include-raw`, these candidates are written to separate `.raw.json`
and `.raw.csv` reports; the main files contain only final assignments.

Each group includes every participating atom, feature and component. Its metadata
contains `center_atom_id`, sorted `neighbor_atom_ids`, the subtype
`hydrophobic_enclosure`, and `legs`. Each leg records its `atom_ids`, `feature_ids`,
`component_ids` and `distance`. The group's reported distance is the maximum leg
distance; angle and offset are unset. Static, batch and trajectory CSV exports
include a JSON-encoded `metadata` column; Python rows and JSON retain dictionaries.
Trajectory occupancy follows the centre atom across changes in the surrounding
neighbours, with the current shell retained in frame and timeseries metadata.
Its mean distance is the mean of the per-frame maximum leg distances.

Fingerprints project groups back into their actual centre–neighbour contact pairs,
deduplicating contacts shared by overlapping groups before ligand filtering.
CoSPLIF tokens, ligand-block contact counts and IFM records therefore retain all
participating ligand–target contacts without inventing neighbour–neighbour links.
Enclosure has a distinct subtype within the existing hydrophobic family, with no
new energetic weight. IFM gives enclosure versus ordinary contacts its configured
subtype partial credit (default 0.75); ordinary versus ordinary and enclosure versus
enclosure receive full subtype credit. Missing hydrophobic subtypes are treated as
`hydrophobic_contact` only during matching, leaving ordinary serialized records and
CoSPLIF hashes unchanged. This also makes custom explicit hydrophobic subtypes versus
missing hydrophobic subtypes receive partial credit. Other families keep their
missing-subtype behaviour. IFM normalization is unchanged: lower raw match credit
does not necessarily lower normalized similarity when every contact still matches.

## Profiles

The active profile supplies geometric cutoffs and configurable gates; carbon
activation for `weak_hbond` is mandatory. `default` ships
with the package and defines one rule per entry in
`chemur.interaction_types.INTERACTION_TYPES`. Pass `--profile` a name or a path to
your own YAML file, or override individual cutoffs on the command line — see
[cli.md](cli.md).
