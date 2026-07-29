"""The interaction-type registry: one declaration per detected interaction.

Before this module the same list of interaction names was written out by hand in
six places -- the detector call list in :mod:`chemur.interactions`, the shipped
profile, the hard-coded profile fallback, the CLI cutoff flags, and two verbatim
copies of ``_canonical_family`` (one in :mod:`chemur.cosplif`, one in
:mod:`chemur.ifm_matcher`). Nothing tied them together, and an interaction that
was missing from the family maps was silently dropped from every score rather than
failing loudly.

Everything here is plain data with no intra-package imports, so any module can
depend on it. :mod:`chemur.interactions` owns the detector functions and binds
them to these names; ``tests/test_profile_consistency.py`` asserts that the
registry, the profiles and the detector table all agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class InteractionType:
    """One detectable interaction and everything the rest of the package needs.

    ``cutoffs`` names the numeric profile keys that the CLI exposes as flags.
    ``aliases`` carries the historical flag spellings so long-standing command
    lines keep working after flag generation moved to this table.
    """

    name: str
    family: str
    label: str
    cutoffs: tuple[str, ...] = ()
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)


# Canonical scoring families. The first eight predate this module; the rest arrived
# with the sigma-hole and stacking work and carry their own IFM weights.
FAMILY_HYDROGEN_BOND = "hydrogen_bond"
FAMILY_IONIC = "ionic"
FAMILY_HYDROPHOBIC = "hydrophobic"
FAMILY_AROMATIC_STACKING = "aromatic_stacking"
FAMILY_CATION_PI = "cation_pi"
FAMILY_HALOGEN_BOND = "halogen_bond"
FAMILY_METAL_COORDINATION = "metal_coordination"
FAMILY_WATER_BRIDGE = "water_bridge"
FAMILY_CHALCOGEN_BOND = "chalcogen_bond"
FAMILY_TETREL_BOND = "tetrel_bond"
FAMILY_N_PI_STAR = "n_pi_star"
FAMILY_RING_STACKING = "ring_stacking"


INTERACTION_TYPES: tuple[InteractionType, ...] = (
    InteractionType(
        name="metal_coordination",
        family=FAMILY_METAL_COORDINATION,
        label="Metal coordination",
        cutoffs=("distance",),
        aliases={"distance": ("--metal-distance",)},
    ),
    InteractionType(
        name="salt_bridge",
        family=FAMILY_IONIC,
        label="Salt bridge",
        cutoffs=("distance",),
        aliases={"distance": ("--sb-distance",)},
    ),
    InteractionType(
        name="amide_bridge",
        family=FAMILY_HYDROGEN_BOND,
        label="Amide bridge",
        cutoffs=("distance", "angle"),
    ),
    InteractionType(
        name="hbond",
        family=FAMILY_HYDROGEN_BOND,
        label="Hydrogen bond",
        cutoffs=("distance", "angle", "sulfur_distance", "short_distance"),
        aliases={"distance": ("--hb-distance",), "angle": ("--hb-angle",)},
    ),
    InteractionType(
        name="solvent_bridge",
        family=FAMILY_WATER_BRIDGE,
        label="Solvent bridge",
        cutoffs=("distance",),
    ),
    InteractionType(
        name="weak_hbond",
        family=FAMILY_HYDROGEN_BOND,
        label="Weak hydrogen bond",
        cutoffs=("distance", "angle"),
        aliases={"distance": ("--weak-hb-distance",), "angle": ("--weak-hb-angle",)},
    ),
    InteractionType(
        name="pipi_stack",
        family=FAMILY_AROMATIC_STACKING,
        label="Pi-pi stack",
        cutoffs=("distance", "angle", "offset"),
        aliases={
            "distance": ("--pipi-distance", "--pi-pi-distance"),
            "angle": ("--pipi-angle", "--pi-pi-angle"),
            "offset": ("--pipi-offset", "--pi-pi-offset"),
        },
    ),
    InteractionType(
        name="aliphatic_pi_stack",
        family=FAMILY_RING_STACKING,
        label="Aliphatic-aromatic ring stack",
        cutoffs=("distance", "angle", "offset"),
    ),
    InteractionType(
        name="aliphatic_stack",
        family=FAMILY_RING_STACKING,
        label="Aliphatic-aliphatic ring stack",
        cutoffs=("distance", "angle", "offset"),
    ),
    InteractionType(
        name="cation_pi",
        family=FAMILY_CATION_PI,
        label="Cation-pi",
        cutoffs=("distance", "angle", "offset"),
    ),
    InteractionType(
        name="anion_pi",
        family=FAMILY_IONIC,
        label="Anion-pi",
        cutoffs=("distance", "angle", "offset"),
    ),
    InteractionType(
        name="anion_aromatic_edge",
        family=FAMILY_IONIC,
        label="Edgewise anion-aromatic",
        cutoffs=("distance", "min_angle", "min_offset", "offset"),
    ),
    InteractionType(
        name="hbond_pi",
        family=FAMILY_HYDROGEN_BOND,
        label="H-bond-pi",
        cutoffs=("distance", "angle", "donor_angle", "offset"),
        aliases={
            "distance": ("--hb-pi-distance",),
            "angle": ("--hb-pi-angle",),
            "donor_angle": ("--hb-pi-donor-angle",),
            "offset": ("--hb-pi-offset",),
        },
    ),
    InteractionType(
        name="n_pi_star",
        family=FAMILY_N_PI_STAR,
        label="n->pi*",
        cutoffs=("distance", "min_angle", "angle", "approach_angle"),
    ),
    InteractionType(
        name="amide_pi",
        family=FAMILY_AROMATIC_STACKING,
        label="Amide-pi",
        cutoffs=("distance", "angle", "offset"),
    ),
    InteractionType(
        name="halogen_bond",
        family=FAMILY_HALOGEN_BOND,
        label="Halogen bond",
        cutoffs=("distance", "angle"),
    ),
    InteractionType(
        name="chalcogen_bond",
        family=FAMILY_CHALCOGEN_BOND,
        label="Chalcogen bond",
        cutoffs=("distance", "angle"),
    ),
    InteractionType(
        name="halogen_pi",
        family=FAMILY_HALOGEN_BOND,
        label="Halogen-pi",
        cutoffs=("distance", "offset"),
    ),
    InteractionType(
        name="chalcogen_pi",
        family=FAMILY_CHALCOGEN_BOND,
        label="Chalcogen-pi",
        cutoffs=("distance", "angle", "offset"),
    ),
    InteractionType(
        name="tetrel_bond",
        family=FAMILY_TETREL_BOND,
        label="Tetrel bond",
        cutoffs=("distance", "min_distance", "angle", "hydrogen_angle"),
    ),
    InteractionType(
        name="ch_pi",
        family=FAMILY_HYDROPHOBIC,
        label="C-H-pi",
        cutoffs=("distance", "angle", "donor_angle", "offset"),
    ),
    InteractionType(
        name="hydrophobic",
        family=FAMILY_HYDROPHOBIC,
        label="Hydrophobic contact",
        cutoffs=("distance",),
    ),
)

INTERACTION_TYPES_BY_NAME: dict[str, InteractionType] = {
    interaction.name: interaction for interaction in INTERACTION_TYPES
}

INTERACTION_FAMILIES: dict[str, str] = {
    interaction.name: interaction.family for interaction in INTERACTION_TYPES
}

# Ordered, de-duplicated family list; ifm_config validates a weight exists for each.
CANONICAL_FAMILIES: tuple[str, ...] = tuple(
    dict.fromkeys(interaction.family for interaction in INTERACTION_TYPES)
)

# Human-readable descriptions of what each cutoff key constrains, for CLI help.
CUTOFF_HELP = {
    "distance": "distance cutoff.",
    "min_distance": "minimum distance.",
    "angle": "angle cutoff.",
    "min_angle": "minimum angle.",
    "donor_angle": "donor-H-acceptor minimum angle.",
    "approach_angle": "maximum approach angle to the acceptor plane normal.",
    "elevation": "maximum elevation out of the donor substituent plane.",
    "hydrogen_angle": "minimum H-donor-acceptor angle required to rule out a C-H contact.",
    "offset": "offset cutoff.",
    "min_offset": "minimum offset.",
    "sulfur_distance": "distance cutoff when sulfur is the donor or acceptor.",
    "short_distance": "distance below which the bond is flagged short/low-barrier.",
}


# Types whose metadata["geometry"] carries a parallel/T-shaped label, so the
# scorers can subtype them the same way.
STACKING_TYPES = frozenset(
    {"pipi_stack", "aliphatic_pi_stack", "aliphatic_stack", "amide_pi"}
)


def canonical_family(interaction_type: str) -> str | None:
    """Scoring family for ``interaction_type``, or None if it is not registered."""
    return INTERACTION_FAMILIES.get(str(interaction_type))
