"""Versioned scientific parameter descriptions shared by the CLI and GUIs.

Values come from the effective profile, never a second set of GUI defaults.
The revision describes detection semantics independently of the package version.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

GEOMETRY_REVISION = "2026-09-24.1"
GEOMETRY_CATALOGUE_REVISION = GEOMETRY_REVISION

HBIND = "https://github.com/psa-lab/Hbind"
PLIP = "https://doi.org/10.1093/nar/gkv315"
PLIP_GEOMETRY = "https://github.com/pharmai/plip/blob/master/plip/structure/detection.py"
ARPEGGIO = "https://github.com/harryjubb/arpeggio"
WEAK_HBOND = "https://pmc.ncbi.nlm.nih.gov/articles/instance/6071916/bin/MD-008-C7MD00381A-s001.pdf"
HYDROGEN_PI = "https://doi.org/10.1002/prot.21253"

# Scope notes distinguish a published geometric convention from extensions to
# other chemical environments. These are geometric annotations, not energies.
_SOURCES = {
    "metal_coordination": ("Na/K and Ca/Co/Cu/Fe/Mg/Mn/Ni/Zn with typed lone-pair acceptors.", [HBIND]),
    "salt_bridge": ("Oppositely charged sites; nearest actual charge-bearing atoms, not group centroids.", [WEAK_HBOND]),
    "hbond": ("Hbind non-water geometry with Chemur donor/acceptor typing; water retains its existing distance/angle convention.", [HBIND]),
    "weak_hbond": ("C-H to oxygen; Chemur additionally requires electronic activation of the carbon donor.", [WEAK_HBOND]),
    "amide_bridge": ("Two reciprocal individually valid hydrogen bonds; extension to backbone and ligand amides is a Chemur generalisation.", [HBIND]),
    "solvent_bridge": ("Existing water-mediated contacts and grouping; distance is a per-leg limit.", []),
    "pipi_stack": ("PLIP-derived geometry for typed aromatic rings, with inclusive Chemur boundaries; parallel or perpendicular geometry and minimum reciprocal lateral offset.", [PLIP, PLIP_GEOMETRY]),
    "aliphatic_pi_stack": ("Published cyclohexane-benzene search, generalised to Chemur's typed nonaromatic/aromatic rings with valid mean planes.", ["https://doi.org/10.1039/C6CP03734H"]),
    "aliphatic_stack": ("Optional existing Chemur heuristic for two nonaromatic rings; disabled by default.", []),
    "cation_pi": ("Typed cations and aromatic rings; the amine/ring plane restriction applies only to validated positively charged sp3 nitrogens with three heavy neighbours.", [PLIP, PLIP_GEOMETRY]),
    "anion_aromatic_edge": ("Optional existing edge-contact heuristic, separate from the removed anion-pi face detector.", []),
    "hbond_pi": ("N/O-H donors to aromatic rings. Generalising protein aromatic acceptors to ligand and other typed aromatic rings is a Chemur extension.", [HYDROGEN_PI]),
    "ch_pi": ("C-H donors to aromatic rings. Generalising protein aromatic acceptors to ligand and other typed aromatic rings is a Chemur extension.", [HYDROGEN_PI]),
    "n_pi_star": ("Carbonyl oxygen to carbonyl carbon, using the Burgi-Dunitz window; extending the published consecutive backbone-amide survey to other typed carbonyls is a Chemur generalisation.", ["https://doi.org/10.1038/nchembio.406"]),
    "amide_pi": ("Arpeggio amide/aromatic convention using the amide C-N midpoint and the amide plane normal.", [ARPEGGIO]),
    "halogen_bond": ("PLIP-derived geometry for organic Cl/Br/I to typed oxygen acceptors, with donor- and acceptor-side orientation and inclusive Chemur boundaries.", [PLIP, PLIP_GEOMETRY]),
    "chalcogen_bond": ("Published S-O, Se-O and Se-S PDB search geometry; includes methionine-like donors without an activation gate.", ["https://doi.org/10.1002/cbic.202100498"]),
    "halogen_pi": ("Broad organic Cl/Br/I-aromatic contact geometry, extending the paper's protein rings to other typed aromatic rings. Chemur retains its Cl/Br/I donor scope rather than including fluorine; this is not the paper's full HalBS ranking or an energetic guarantee.", ["https://doi.org/10.1002/pro.70321"]),
    "chalcogen_pi": ("Union of directional thioether/disulfide sulfur-pi and broad thioether/sulfoxide sulfur-aromatic definitions; ligand analogues are a Chemur generalisation. No selenium extrapolation.", ["https://doi.org/10.1080/07391102.2022.2143427", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5060120/", "https://doi.org/10.1074/jbc.M112.374504"]),
    "tetrel_bond": ("sp3-carbon with a bonded C/N/O/S root to carbonyl oxygen, with both orientation angles. Validated chemical bonds replace the paper's additional covalent-bond-length screening.", ["https://doi.org/10.1002/anie.201811171"]),
    "hydrophobic_enclosure": ("Ligand-centred strict 3D enclosure by valid hydrophobic contacts; inherits the hydrophobic distance limit.", []),
    "hydrophobic": ("Chemur hydrophobe typing and occlusion with endpoint-bonded hydrogens exempted.", [WEAK_HBOND]),
}

# Labels are also the endpoint definitions used in command help and tooltips.
_LABELS = {
    "metal_coordination": {"min_distance": "Metal-acceptor minimum", "distance": "Metal-acceptor maximum", "alkali_min_distance": "Na/K-acceptor minimum", "alkali_distance": "Na/K-acceptor maximum"},
    "salt_bridge": {"distance": "Nearest charged-atom distance"},
    "hbond": {"min_distance": "Donor-acceptor minimum", "distance": "Donor-acceptor maximum", "hydrogen_min_distance": "H-acceptor minimum", "hydrogen_distance": "H-acceptor maximum", "angle": "Donor-H-acceptor minimum angle", "acceptor_angle": "Antecedent-acceptor-H minimum angle", "water_distance": "Water-leg donor-acceptor maximum", "water_angle": "Water-leg donor-H-acceptor minimum angle"},
    "weak_hbond": {"distance": "Carbon-oxygen maximum", "angle": "C-H-O minimum angle"},
    "amide_bridge": {"min_residue_separation": "Minimum same-polymer sequence separation"},
    "solvent_bridge": {"distance": "Maximum donor-acceptor distance per water leg"},
    "pipi_stack": {"distance": "Ring-centroid maximum", "angle": "Parallel planes maximum angle", "min_perpendicular_angle": "Perpendicular planes minimum angle", "offset": "Minimum reciprocal lateral offset maximum"},
    "aliphatic_pi_stack": {"distance": "Ring-centroid distance (strict maximum)", "angle": "Mean-plane angle (strict maximum)", "plane_separation": "Height above aromatic plane (strict maximum)"},
    "aliphatic_stack": {"distance": "Ring-centroid maximum", "angle": "Mean-plane maximum angle", "offset": "Minimum reciprocal lateral offset maximum"},
    "cation_pi": {"distance": "Cation-ring centroid maximum", "offset": "Cation lateral offset maximum", "tertiary_amine_angle": "Amine substituent-plane to aromatic-ring plane maximum angle"},
    "anion_aromatic_edge": {"distance": "Anion feature centre to ring-centroid maximum", "min_angle": "Anion approach to ring normal minimum angle", "min_offset": "Anion lateral offset minimum", "offset": "Anion lateral offset maximum"},
    "hbond_pi": {"hydrogen_distance": "H-ring centroid maximum", "angle": "Heavy donor-ring normal maximum angle", "donor_angle": "Donor-H-centroid minimum angle"},
    "ch_pi": {"hydrogen_distance": "H-ring centroid maximum", "angle": "Carbon-ring normal maximum angle", "donor_angle": "C-H-centroid minimum angle"},
    "n_pi_star": {"distance": "Carbonyl O to carbonyl C maximum", "min_angle": "O-C-O minimum angle", "angle": "O-C-O maximum angle", "min_residue_separation": "Minimum same-polymer sequence separation"},
    "amide_pi": {"distance": "Amide C-N midpoint to ring-centroid maximum", "angle": "Amide-ring plane maximum angle", "approach_angle": "Centroid approach to amide normal maximum angle"},
    "halogen_bond": {"distance": "Halogen-oxygen maximum", "angle": "C-halogen-O minimum angle", "acceptor_min_angle": "Halogen-O-antecedent minimum angle", "acceptor_angle": "Halogen-O-antecedent maximum angle"},
    "halogen_pi": {"distance": "Halogen-ring centroid maximum", "donor_angle": "C-halogen-centroid angle (strict minimum)"},
    "chalcogen_bond": {"s_o_min_distance": "S-O distance (strict minimum)", "s_o_distance": "S-O maximum", "se_o_min_distance": "Se-O distance (strict minimum)", "se_o_distance": "Se-O maximum", "se_s_min_distance": "Se-S distance (strict minimum)", "se_s_distance": "Se-S maximum", "angle": "C-chalcogen-acceptor minimum angle"},
    "chalcogen_pi": {"distance": "Broad S-centroid distance (strict maximum)", "angle": "Broad approach to ring normal (strict maximum)", "directional_distance": "Directional S-centroid maximum", "directional_angle": "Root-S-centroid minimum angle"},
    "tetrel_bond": {"min_distance": "sp3 C-carbonyl O minimum", "distance": "sp3 C-carbonyl O maximum", "angle": "Root-donor C-acceptor O minimum angle", "acceptor_angle": "Donor C-O-carbonyl C minimum angle"},
    "hydrophobic": {"distance": "Hydrophobe atom-atom maximum"},
}


def get_geometry_catalogue(profile: dict[str, Any] | str | None = None) -> dict[str, Any]:
    """Return independent, JSON-compatible descriptions with effective values."""
    from .interaction_types import INTERACTION_TYPES
    from .profile import DEFAULT_PROFILE, load_profile

    effective = load_profile("default" if profile is None else profile)
    rules = {}
    for interaction in INTERACTION_TYPES:
        name = interaction.name
        if name not in effective["rules"]:
            continue
        rule = effective["rules"][name]
        scope, references = _SOURCES[name]
        parameters = {}
        for key in interaction.cutoffs:
            if key not in rule:
                continue
            value = rule[key]
            angular = "angle" in key
            count = key == "min_residue_separation"
            label = _LABELS.get(name, {}).get(key, key.replace("_", " ").capitalize())
            units = "degrees" if angular else "residues" if count else "Å"
            parameters[key] = {
                "label": label, "endpoints": label,
                "description": f"{label}. {scope}", "units": units,
                "minimum": 0, "maximum": 180 if angular else 1000 if count else 100,
                "decimals": 0 if count else 1 if angular else 3,
                "step": 1 if angular or count else 0.1,
                "value": deepcopy(value),
                "default": deepcopy(DEFAULT_PROFILE["rules"][name].get(key)),
                "strict": "strict" in label,
            }
        rules[name] = {
            "label": interaction.label, "scope": scope,
            "references": list(references), "parameters": parameters,
            "enabled": bool(rule.get("enabled", True)),
        }
    return {"revision": GEOMETRY_REVISION, "rules": rules}


def geometry_parameters(profile: dict[str, Any] | str | None = None) -> list[dict[str, Any]]:
    """Flatten the catalogue for clients that create one control per parameter."""
    catalogue = get_geometry_catalogue(profile)
    return [
        {"rule": name, "key": key, "rule_label": body["label"],
         "scope": body["scope"], "references": body["references"], **parameter}
        for name, body in catalogue["rules"].items()
        for key, parameter in body["parameters"].items()
    ]


def geometry_result_metadata(profile: dict[str, Any]) -> dict[str, Any]:
    """Snapshot effective detection settings for reproducible native results."""
    return {"geometry_catalogue_revision": GEOMETRY_REVISION,
            "effective_profile": deepcopy(profile)}
