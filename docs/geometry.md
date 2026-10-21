# Interaction geometry and provenance

Geometry revision **2026-09-24.1** is shared by the source engine, CLI and ChimeraX
plugin. `chemur.get_geometry_catalogue()` describes each active rule's parameters,
distance endpoints, units, chemical scope, references and effective values.
`chemur.geometry_parameters()` provides the same data as a flat list for controls.
JSON results record `metadata.geometry_catalogue_revision` and a complete
`metadata.effective_profile`, including explicit overrides.

These are geometric contact definitions, not binding energies. A published
screening threshold is distinct from an energy minimum or a universal physical
limit. In particular, the [reported amide N-to-ring-centre energy minimum near
3.3 Å](https://mr.copernicus.org/articles/2/765/2021/) is not used as a maximum
hydrogen-pi detection distance.

## Hydrogen-mediated contacts

| Rule | Distance | Angles and eligibility |
| --- | --- | --- |
| `hbond`, non-water | D-A 2.4–3.5 Å; H-A 1.5–2.5 Å | D-H-A ≥120°; bonded antecedent-A-H ≥90° |
| `weak_hbond` | C-O ≤3.6 Å | C-H-O ≥130°; electronically activated carbon donor |
| `hbond_pi` | **H-centroid ≤3.5 Å** | N/O-H-centroid ≥100°; heavy donor-centroid within 40° of ring normal |
| `ch_pi` | **H-centroid ≤3.5 Å** | C-H-centroid ≥100°; C-centroid within 40° of ring normal |
| `amide_bridge` | Each reciprocal leg meets `hbond` geometry | Two independently valid, unoccluded amide N-H···O legs |
| `solvent_bridge` | Existing 3.5 Å maximum per water leg | Existing water H-bond geometry, grouping and assignment are retained |

Non-water hydrogen bonds follow [Hbind's complete geometric convention](https://github.com/psa-lab/Hbind).
Weak C-H···O distance/angle values come from [Ferreira de Freitas and Schapira,
Table S2](https://pmc.ncbi.nlm.nih.gov/articles/instance/6071916/bin/MD-008-C7MD00381A-s001.pdf).
Chemur additionally retains its documented carbon-activation typing. The two
hydrogen-pi conventions follow [Panigrahi and Desiraju](https://doi.org/10.1002/prot.21253).
Applying their protein aromatic-acceptor geometry to other chemically typed
aromatic rings, and applying reciprocal amide bridges to backbone and ligand
amides, are documented generalisations.

All hydrogen-dependent checks must pass for the **same bonded hydrogen**. Missing
or template-excluded hydrogens cannot supply geometry, and missing acceptor
antecedents are not invented. Water retains its existing donor/acceptor distance
and donor-angle checks instead of inheriting the new non-water antecedent check.

Hydrogen-pi rules have a `hydrogen_distance` parameter. The obsolete heavy-atom
`distance` and lateral `offset` gates are removed. Serialized interaction
`distance` still reports heavy atom-to-centroid distance for compatibility;
metadata supplies the selected hydrogen identity and actual hydrogen distance.
The plugin labels these measurements separately. Distance alone no longer labels
a hydrogen bond as strong or low-barrier.

## Rings, charged sites and halogens

| Rule | Default definition |
| --- | --- |
| `hydrophobic` | Valid hydrophobes ≤4 Å; endpoint-bonded hydrogens exempt from occlusion |
| `hydrophobic_enclosure` | Strict 3D enclosure centred on a **ligand atom**, using valid hydrophobic contacts |
| `salt_bridge` | Nearest actual opposite-charge-bearing atoms ≤4 Å |
| `pipi_stack` | Centroids ≤5.5 Å; plane angle ≤30° or ≥60°; minimum reciprocal lateral offset ≤2 Å |
| `cation_pi` | Cation-centroid ≤6 Å; offset ≤2 Å; tertiary-amine substituent-plane to ring-plane angle ≤30° |
| `amide_pi` | Amide C-N midpoint to ring centroid ≤6 Å; plane angle ≤30°; approach to amide normal ≤30° |
| `aliphatic_pi_stack` | Centroids **<7 Å**; acute mean-plane angle **<10°**; height above aromatic plane **<5 Å** |
| `halogen_bond` | Cl/Br/I to oxygen ≤4 Å; donor angle ≥135°; acceptor angle 90–150° |
| `halogen_pi` | Cl/Br/I-centroid ≤6 Å; C-halogen-centroid **>90°** |
| `n_pi_star` | Carbonyl O to carbonyl C ≤3.2 Å; O-C-O angle 99–119° |
| `metal_coordination` | Na/K: 2–2.9 Å; Ca/Co/Cu/Fe/Mg/Mn/Ni/Zn: 1.7–2.6 Å |

Aromatic stacking, cation-pi and halogen-bond conventions use
[PLIP](https://doi.org/10.1093/nar/gkv315) and its
[geometry implementation](https://github.com/pharmai/plip/blob/master/plip/structure/detection.py).
Chemur retains its own typing and assignment and uses the inclusive boundaries
listed above; this is not complete PLIP algorithm equivalence. The tertiary-amine
plane check applies to validated positively charged sp3 nitrogens with three
heavy neighbours. Amide-pi uses
[Arpeggio](https://github.com/harryjubb/arpeggio). The aliphatic-aromatic search
uses the published [cyclohexane-benzene convention](https://doi.org/10.1039/C6CP03734H),
generalised to Chemur's supported nonaromatic rings with meaningful fitted planes.
The halogen-pi search uses [de Vries et al.](https://doi.org/10.1002/pro.70321);
generalisation beyond their protein aromatic acceptors is recorded as such.
Chemur retains its Cl/Br/I donor scope, excluding fluorine, and does not reproduce
the paper's full HalBS ranking. The n→π* definition uses
[Bartlett et al.'s carbonyl approach convention](https://doi.org/10.1038/nchembio.406),
generalised from consecutive backbone amides to Chemur's other typed carbonyls.
Metal windows follow Hbind. Removed offset or approach gates are not silently
retained as additional requirements.

`aliphatic_stack` remains optional and disabled by default. `anion_aromatic_edge`
is also opt-in. Active `anion_pi` detection has been removed; its historical
records still load, render and score in the ionic family. Requesting the removed
detector produces an actionable error.

## Sulfur, selenium and carbon bonds

`chalcogen_pi` accepts the union of two sulfur-specific definitions:

- **Directional:** thioether/disulfide sulfur-centroid ≤3.8 Å and root-S-centroid
  ≥155°, following the [directional sulfur-pi survey](https://doi.org/10.1080/07391102.2022.2143427).
- **Broad aromatic contact:** thioether/sulfoxide sulfur-centroid <7 Å and approach
  to the ring normal <60°, following the [methionine/sulfoxide study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5060120/)
  and the [sulfur-aromatic distance distribution](https://doi.org/10.1074/jbc.M112.374504).

One sulfur/ring pair produces one record. `metadata.geometry_classifications`
records every matching definition; overlap does not double the count or change the
scoring subtype. This rule consumes no chalcogen-donor resource and has no extra
2.5 Å lateral-offset gate. Ligand analogues are a generalisation; selenium is not
silently included in sulfur-specific pi definitions.

Equivalent `S=O` and `[S+]–[O−]` sulfoxides have identical feature perception,
including charge-derived features. Source atom identities, coordinates and net
charge are preserved. Local resonance charges must not manufacture salt bridges
or cation-pi contacts.

Atom-atom `chalcogen_bond` uses the [published PDB survey](https://doi.org/10.1002/cbic.202100498):
S-O **>2 and ≤3.32 Å**, Se-O **>2 and ≤3.42 Å**, Se-S **>2.5 and ≤4.07 Å**,
with C-S/Se-acceptor ≥130°. Methionine-like donors are included without the former
activation gate. This is separate from sulfur-aromatic contact classification.

`tetrel_bond` uses sp3-carbon donors with a bonded C/N/O/S root and carbonyl-oxygen
acceptors at **2.5–3.6 Å**, with both root-C-O and C-O-carbonyl-C angles
**160–180°**, following
[Mundlapati et al.](https://doi.org/10.1002/anie.201811171). Chemur uses validated
chemical bonds in place of that survey's additional covalent bond-length filters.
The previous extra hydrogen-angle exclusion is removed.

## Compatibility

Geometry changes can change assignments and therefore scores, but family weights,
score normalization, enclosure projection and ordinary fingerprint serialization
are unchanged. Raw candidates remain optional. Explicit obsolete parameter names
are rejected rather than reinterpreted as distances with different endpoints.
