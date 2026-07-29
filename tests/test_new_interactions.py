"""Geometric acceptance/rejection tests for the interaction types added from
Adhav & Saikrishnan, ACS Omega 2023, 8, 22268.

Each type gets one positive case at ideal geometry and one negative per criterion,
so a cutoff that stops doing work fails loudly rather than silently widening.
"""

from math import cos, pi, radians, sin

import pytest

from chemeleonx.features import perceive_features
from chemeleonx.interactions import generate_candidates
from chemeleonx.models import AtomRecord, ComponentRecord
from chemeleonx.parser import _assign_default_atom_chemistry
from chemeleonx.profile import load_profile


def detect(atoms, components, profile=None):
    features = perceive_features(atoms, components)
    return generate_candidates(atoms, features, profile or load_profile())


def accepted_types(candidates):
    return {c.interaction_type for c in candidates if not c.rejection_reason}


def of_type(candidates, interaction_type):
    return [c for c in candidates if c.interaction_type == interaction_type]


def ring_atoms(start_id, radius, z, residue, component, molecule_type, *, aromatic, names=None):
    names = names or [f"C{i}" for i in range(6)]
    return [
        AtomRecord(
            start_id + i,
            names[i],
            "C",
            (radius * cos(2 * pi * i / 6), radius * sin(2 * pi * i / 6), z),
            residue,
            "1",
            component[0],
            component,
            molecule_type,
            is_aromatic=aromatic,
        )
        for i in range(6)
    ]


PHE_RING_NAMES = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]


def phenyl():
    return ring_atoms(0, 1.39, 0.0, "PHE", "A:PHE:1", "protein", aromatic=True, names=PHE_RING_NAMES)


def phenyl_component():
    return ComponentRecord("A:PHE:1", "PHE", "A", "1", "protein", tuple(range(6)))


def ligand_ring_component(aromatic):
    return ComponentRecord(
        "B:LIG:2",
        "LIG",
        "B",
        "2",
        "ligand",
        tuple(range(6, 12)),
        metadata={"template_rings": [tuple(range(6, 12))]},
    )


# ---------------------------------------------------------------- chalcogen bond


def sulfur_with_acceptor(acceptor_coord, *, substituent="C"):
    """A divalent sulfur at the origin, one substituent along -x so a sigma-hole
    points along +x.

    ``substituent="C"`` is an ordinary MET-like thioether -- geometrically a perfect
    donor, but electronically inert. ``substituent="N"`` puts a withdrawing atom on the
    sulfur and switches its sigma-hole on. The element must be chosen before typing,
    because activation is now decided there rather than at detection time.
    """
    atoms = [
        AtomRecord(0, "SD", "S", (0.0, 0.0, 0.0), "MET", "1", "A", "A:MET:1", "protein", bonds=(1, 2)),
        AtomRecord(1, "CG", "C", (-1.8, 0.0, 0.0), "MET", "1", "A", "A:MET:1", "protein", bonds=(0,)),
        AtomRecord(2, "CE", substituent, (0.31, -1.77, 0.0), "MET", "1", "A", "A:MET:1", "protein", bonds=(0,)),
    ]
    _assign_default_atom_chemistry(atoms)
    atoms.append(
        AtomRecord(3, "O1", "O", acceptor_coord, "LIG", "2", "B", "B:LIG:2", "ligand", acceptor_capacity=2)
    )
    components = [
        ComponentRecord("A:MET:1", "MET", "A", "1", "protein", (0, 1, 2)),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (3,)),
    ]
    return atoms, components


def test_chalcogen_bond_accepted_on_the_sigma_hole_axis():
    candidates = detect(*sulfur_with_acceptor((3.2, 0.0, 0.0), substituent="N"))
    bonds = of_type(candidates, "chalcogen_bond")

    assert len(bonds) == 1
    assert bonds[0].rejection_reason is None
    assert bonds[0].angle == pytest.approx(180.0)
    # Both sigma-holes lie in the R-S-R plane, so an axial acceptor is coplanar.
    assert bonds[0].metadata["elevation"] == pytest.approx(0.0, abs=1e-6)


def test_chalcogen_bond_rejected_on_the_lone_pair_belt():
    """Perpendicular to the R-S-R plane the sulfur is an H-bond acceptor, not a donor."""
    candidates = detect(*sulfur_with_acceptor((0.0, 0.0, 3.2), substituent="N"))
    bonds = of_type(candidates, "chalcogen_bond")

    assert len(bonds) == 1
    assert bonds[0].rejection_reason.startswith("angle_below_cutoff")


def test_chalcogen_bond_not_generated_beyond_the_distance_cutoff():
    candidates = detect(*sulfur_with_acceptor((4.5, 0.0, 0.0), substituent="N"))

    assert of_type(candidates, "chalcogen_bond") == []


def test_chalcogen_bond_uses_the_best_substituent_as_the_sigma_hole_axis():
    """The second R-S bond carries a sigma-hole too; taking only the first would miss it."""
    # The second substituent sits at (0.31, -1.77, 0), so its sigma-hole extends
    # along roughly (-0.17, 0.985).
    acceptor = (-0.17 * 3.2, 0.985 * 3.2, 0.0)
    candidates = detect(*sulfur_with_acceptor(acceptor, substituent="N"))
    bonds = of_type(candidates, "chalcogen_bond")

    assert len(bonds) == 1
    assert bonds[0].rejection_reason is None
    assert bonds[0].metadata["sigma_hole_root_atom_id"] == 2


def test_chalcogen_bond_rejected_when_the_sulfur_has_no_sigma_hole():
    """A plain MET thioether is perfect geometrically and inert electronically."""
    bonds = of_type(detect(*sulfur_with_acceptor((3.2, 0.0, 0.0))), "chalcogen_bond")

    assert len(bonds) == 1
    assert bonds[0].angle == pytest.approx(180.0)
    assert bonds[0].rejection_reason == "sigma_hole_not_activated"
    assert bonds[0].metadata["sigma_hole_activated"] is False


def test_activation_requirement_can_be_switched_off():
    atoms, components = sulfur_with_acceptor((3.2, 0.0, 0.0))
    profile = load_profile({"rules": {"chalcogen_bond": {"require_activated_sigma_hole": False}}})

    bonds = of_type(detect(atoms, components, profile), "chalcogen_bond")
    assert bonds[0].rejection_reason is None


# ------------------------------------------------------- aliphatic ring stacking


def stacked_rings(separation, *, ligand_aromatic, perpendicular=False):
    protein = phenyl()
    if perpendicular:
        ligand = [
            AtomRecord(
                6 + i,
                f"C{i}",
                "C",
                (1.46 * cos(2 * pi * i / 6), separation, 1.46 * sin(2 * pi * i / 6)),
                "LIG",
                "2",
                "B",
                "B:LIG:2",
                "ligand",
                is_aromatic=ligand_aromatic,
            )
            for i in range(6)
        ]
    else:
        ligand = ring_atoms(6, 1.46, separation, "LIG", "B:LIG:2", "ligand", aromatic=ligand_aromatic)
    return protein + ligand, [phenyl_component(), ligand_ring_component(ligand_aromatic)]


def test_saturated_ring_over_an_aromatic_ring_is_an_aliphatic_stack():
    candidates = detect(*stacked_rings(3.8, ligand_aromatic=False))
    stacks = accepted_types(candidates) & {"aliphatic_pi_stack", "pipi_stack"}

    assert stacks == {"aliphatic_pi_stack"}


def test_aromatic_pair_is_still_a_pipi_stack():
    candidates = detect(*stacked_rings(3.7, ligand_aromatic=True))
    stacks = accepted_types(candidates) & {"aliphatic_pi_stack", "pipi_stack"}

    assert stacks == {"pipi_stack"}


def test_aliphatic_stack_rejected_beyond_the_distance_cutoff():
    candidates = detect(*stacked_rings(6.5, ligand_aromatic=False))

    assert "aliphatic_pi_stack" not in accepted_types(candidates)


def test_aliphatic_stack_requires_near_parallel_faces():
    """Unlike pi-pi, a saturated ring has no quadrupole to stabilise an edge-on approach."""
    candidates = detect(*stacked_rings(3.8, ligand_aromatic=False, perpendicular=True))

    assert "aliphatic_pi_stack" not in accepted_types(candidates)


def test_require_aromatic_can_be_switched_off_to_reproduce_old_behaviour():
    atoms, components = stacked_rings(3.8, ligand_aromatic=False)
    profile = load_profile({"rules": {"pipi_stack": {"require_aromatic": False}}})

    assert "pipi_stack" in accepted_types(detect(atoms, components, profile))


def test_aliphatic_stack_between_two_saturated_rings_is_off_by_default():
    protein = ring_atoms(0, 1.46, 0.0, "PRO", "A:PRO:1", "protein", aromatic=False,
                         names=["N", "CA", "CB", "CG", "CD", "C"])
    protein[0].element = "N"
    ligand = ring_atoms(6, 1.46, 3.8, "LIG", "B:LIG:2", "ligand", aromatic=False)
    components = [
        ComponentRecord("A:PRO:1", "PRO", "A", "1", "protein", tuple(range(6))),
        ligand_ring_component(False),
    ]

    assert "aliphatic_stack" not in accepted_types(detect(protein + ligand, components))

    profile = load_profile({"rules": {"aliphatic_stack": {"enabled": True}}})
    assert "aliphatic_stack" in accepted_types(detect(protein + ligand, components, profile))


# -------------------------------------------------------------------- n -> pi*


def carbonyl_pair(distance, burgi_dunitz_angle, *, in_plane=False):
    """Acceptor C=O at the origin along -y; donor oxygen at the given approach.

    A real n->pi* donor sits above the sp2 plane, along pi*. ``in_plane`` puts the
    off-axis component in the plane instead, as a negative control.
    """
    theta = radians(burgi_dunitz_angle)
    if in_plane:
        donor_oxygen = (distance * sin(theta), -distance * cos(theta), 0.0)
    else:
        donor_oxygen = (0.0, -distance * cos(theta), distance * sin(theta))

    atoms = [
        AtomRecord(0, "C", "C", (0.0, 0.0, 0.0), "GLY", "1", "A", "A:GLY:1", "protein", bonds=(1, 2)),
        AtomRecord(1, "O", "O", (0.0, -1.23, 0.0), "GLY", "1", "A", "A:GLY:1", "protein", bonds=(0,)),
        AtomRecord(2, "CA", "C", (-1.5, 0.8, 0.0), "GLY", "1", "A", "A:GLY:1", "protein", bonds=(0,)),
        AtomRecord(3, "C", "C", tuple(donor_oxygen[i] + (0.6, 1.1, 0.0)[i] for i in range(3)),
                   "GLY", "2", "B", "B:GLY:2", "protein", bonds=(4, 5)),
        AtomRecord(4, "O", "O", donor_oxygen, "GLY", "2", "B", "B:GLY:2", "protein", bonds=(3,)),
        AtomRecord(5, "CA", "C", tuple(donor_oxygen[i] + (2.0, 1.6, 0.0)[i] for i in range(3)),
                   "GLY", "2", "B", "B:GLY:2", "protein", bonds=(3,)),
    ]
    components = [
        ComponentRecord("A:GLY:1", "GLY", "A", "1", "protein", (0, 1, 2)),
        ComponentRecord("B:GLY:2", "GLY", "B", "2", "protein", (3, 4, 5)),
    ]
    return atoms, components


@pytest.mark.parametrize("angle", [102.0, 109.0])
def test_n_pi_star_accepted_at_burgi_dunitz_geometry(angle):
    """109 deg is the Burgi-Dunitz angle; 102 deg is the paper's reported optimum."""
    candidates = detect(*carbonyl_pair(3.0, angle))
    interactions = of_type(candidates, "n_pi_star")

    assert interactions
    assert interactions[0].angle == pytest.approx(angle, abs=0.5)


def test_n_pi_star_rejected_outside_the_angle_window():
    assert of_type(detect(*carbonyl_pair(3.0, 150.0)), "n_pi_star") == []


def test_n_pi_star_rejected_beyond_the_van_der_waals_sum():
    assert of_type(detect(*carbonyl_pair(3.6, 109.0)), "n_pi_star") == []


def test_n_pi_star_rejected_for_an_in_plane_approach():
    """Correct angle but in the sp2 plane: that is not an approach along pi*."""
    assert of_type(detect(*carbonyl_pair(3.0, 109.0, in_plane=True)), "n_pi_star") == []


# ------------------------------------------------------------------ tetrel bond


def methyl_with_acceptor(hydrogen_coords):
    atoms = [
        AtomRecord(0, "CE", "C", (0.0, 0.0, 0.0), "MET", "1", "A", "A:MET:1", "protein", bonds=(1, 2, 3, 4)),
        AtomRecord(1, "SD", "S", (-1.8, 0.0, 0.0), "MET", "1", "A", "A:MET:1", "protein", bonds=(0,)),
    ] + [
        AtomRecord(2 + i, f"H{i}", "H", coord, "MET", "1", "A", "A:MET:1", "protein", bonds=(0,))
        for i, coord in enumerate(hydrogen_coords)
    ]
    _assign_default_atom_chemistry(atoms)
    atoms.append(
        AtomRecord(5, "O1", "O", (3.1, 0.0, 0.0), "LIG", "2", "B", "B:LIG:2", "ligand", acceptor_capacity=2)
    )
    components = [
        ComponentRecord("A:MET:1", "MET", "A", "1", "protein", (0, 1, 2, 3, 4)),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (5,)),
    ]
    return atoms, components


def test_tetrel_bond_accepted_when_hydrogens_point_away():
    atoms, components = methyl_with_acceptor(
        [(-0.4, 1.0, 0.0), (-0.4, -0.5, 0.87), (-0.4, -0.5, -0.87)]
    )
    bonds = of_type(detect(atoms, components), "tetrel_bond")

    assert len(bonds) == 1
    assert bonds[0].rejection_reason is None
    assert bonds[0].angle == pytest.approx(180.0)


def test_tetrel_bond_rejected_when_a_ch_points_at_the_acceptor():
    """That contact is a C-H...O, and weak_hbond outranks tetrel_bond anyway."""
    atoms, components = methyl_with_acceptor(
        [(0.9, 0.35, 0.0), (-0.4, -0.5, 0.87), (-0.4, -0.5, -0.87)]
    )
    bonds = of_type(detect(atoms, components), "tetrel_bond")

    assert len(bonds) == 1
    assert bonds[0].rejection_reason.startswith("hydrogen_contact")


# ----------------------------------------------------------------- amide bridge


def test_amide_bridge_detects_a_reciprocal_amide_pair():
    atoms = [
        AtomRecord(0, "C", "C", (0.0, 0.0, 0.0), "ALA", "1", "A", "A:ALA:1", "protein", bonds=(1, 2)),
        AtomRecord(1, "O", "O", (0.0, 1.23, 0.0), "ALA", "1", "A", "A:ALA:1", "protein", bonds=(0,)),
        AtomRecord(2, "CA", "C", (-1.5, -0.5, 0.0), "ALA", "1", "A", "A:ALA:1", "protein", bonds=(0,)),
        AtomRecord(3, "N", "N", (1.2, -0.7, 0.0), "GLY", "2", "A", "A:GLY:2", "protein", bonds=(4,)),
        AtomRecord(4, "H", "H", (1.2, 0.3, 0.0), "GLY", "2", "A", "A:GLY:2", "protein", bonds=(3,)),
        AtomRecord(5, "C", "C", (1.2, 3.4, 0.0), "ALA", "1", "B", "B:ALA:1", "protein", bonds=(6, 7)),
        AtomRecord(6, "O", "O", (1.2, 2.17, 0.0), "ALA", "1", "B", "B:ALA:1", "protein", bonds=(5,)),
        AtomRecord(7, "CA", "C", (2.7, 3.9, 0.0), "ALA", "1", "B", "B:ALA:1", "protein", bonds=(5,)),
        AtomRecord(8, "N", "N", (0.0, 4.1, 0.0), "GLY", "2", "B", "B:GLY:2", "protein", bonds=(9,)),
        AtomRecord(9, "H", "H", (0.0, 3.1, 0.0), "GLY", "2", "B", "B:GLY:2", "protein", bonds=(8,)),
    ]
    components = [
        ComponentRecord("A:ALA:1", "ALA", "A", "1", "protein", (0, 1, 2)),
        ComponentRecord("A:GLY:2", "GLY", "A", "2", "protein", (3, 4)),
        ComponentRecord("B:ALA:1", "ALA", "B", "1", "protein", (5, 6, 7)),
        ComponentRecord("B:GLY:2", "GLY", "B", "2", "protein", (8, 9)),
    ]

    # The backbone amide spans two residues, so it can only be built across components.
    amides = [f for f in perceive_features(atoms, components) if f.feature_type == "amide"]
    assert [f.atom_ids for f in amides] == [(0, 1, 3), (5, 6, 8)]

    bridges = of_type(detect(atoms, components), "amide_bridge")
    assert len(bridges) == 1
    assert bridges[0].rejection_reason is None
    assert bridges[0].distance == pytest.approx(2.87, abs=0.05)


# --------------------------------------------------- edgewise anion-aromatic


def anion_near_phenyl(coord):
    atoms = phenyl() + [
        AtomRecord(6, "O1", "O", coord, "LIG", "2", "B", "B:LIG:2", "ligand",
                   formal_charge=-1, acceptor_capacity=2)
    ]
    components = [
        phenyl_component(),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (6,)),
    ]
    return atoms, components


def test_anion_in_the_ring_plane_is_edgewise_not_face_on():
    accepted = accepted_types(detect(*anion_near_phenyl((4.0, 0.0, 0.0))))

    assert "anion_aromatic_edge" in accepted
    assert "anion_pi" not in accepted


def test_anion_on_the_ring_axis_is_face_on_not_edgewise():
    accepted = accepted_types(detect(*anion_near_phenyl((0.0, 0.0, 3.5))))

    assert "anion_pi" in accepted
    assert "anion_aromatic_edge" not in accepted


# ------------------------------------------------------- ch_pi from a ligand


def test_ch_pi_is_detected_from_a_templated_ligand_carbon():
    """Templated ligand carbons are not `donor` features, so this was unreachable."""
    ligand = [
        AtomRecord(6, "C1", "C", (0.0, 0.0, 3.6), "LIG", "2", "B", "B:LIG:2", "ligand",
                   carbon_donor_capacity=4, bonds=(7,)),
        AtomRecord(7, "H1", "H", (0.0, 0.0, 2.7), "LIG", "2", "B", "B:LIG:2", "ligand", bonds=(6,)),
    ]
    components = [
        phenyl_component(),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (6, 7)),
    ]

    assert "ch_pi" in accepted_types(detect(phenyl() + ligand, components))


def test_rdkit_template_gives_carbons_a_carbon_donor_capacity():
    chem = pytest.importorskip("rdkit.Chem")
    from chemeleonx.chemistry import rdkit_atom_annotations

    annotation = rdkit_atom_annotations(chem.MolFromSmiles("C"))[0]

    assert annotation.carbon_donor_capacity == 4
    # Still not a hydrogen-bond donor: ch_pi must not consume that capacity.
    assert annotation.donor_capacity == 0


# ------------------------------------------------------------- hbond changes


def hbond_pair(donor_element, acceptor_element, separation):
    atoms = [
        AtomRecord(0, "D", donor_element, (0.0, 0.0, 0.0), "CYS", "1", "A", "A:CYS:1", "protein", bonds=(1,)),
        AtomRecord(1, "H", "H", (1.0, 0.0, 0.0), "CYS", "1", "A", "A:CYS:1", "protein", bonds=(0,)),
        AtomRecord(2, "A", acceptor_element, (separation, 0.0, 0.0), "LIG", "2", "B", "B:LIG:2", "ligand"),
    ]
    _assign_default_atom_chemistry(atoms)
    components = [
        ComponentRecord("A:CYS:1", "CYS", "A", "1", "protein", (0, 1)),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (2,)),
    ]
    return atoms, components


def test_sulfur_hbond_uses_the_longer_cutoff():
    """d(S...A) ~ 3.5 A is normal for sulfur; a flat 3.5 A cutoff truncated it."""
    assert "hbond" in accepted_types(detect(*hbond_pair("S", "O", 3.8)))
    # The same distance between N and O is beyond the ordinary cutoff.
    assert "hbond" not in accepted_types(detect(*hbond_pair("N", "O", 3.8)))


def test_short_hydrogen_bond_is_flagged_low_barrier():
    bonds = of_type(detect(*hbond_pair("N", "O", 2.5)), "hbond")

    assert bonds
    assert bonds[0].metadata["short_strong"] is True
    assert bonds[0].metadata["interaction_subtype"] == "low_barrier"


def test_ordinary_hydrogen_bond_is_not_flagged_low_barrier():
    bonds = of_type(detect(*hbond_pair("N", "O", 3.0)), "hbond")

    assert bonds
    assert "short_strong" not in bonds[0].metadata


@pytest.mark.parametrize(
    "group, smiles, activated",
    [
        ("thiophene", "c1ccsc1", True),
        ("thiazole", "c1csc(n1)C", True),
        ("1,3,4-thiadiazole", "c1nnc(s1)C", True),
        ("S-nitroso", "CSN=O", True),
        ("thiourea", "NC(=S)N", True),
        ("thioamide", "CC(=S)N", True),
        ("sulfonium", "C[S+](C)CCC", True),
        ("thioether (MET SD)", "CSC", False),
        ("thiol (CYS SG)", "CCS", False),
        # No withdrawing substituent on the carbon, so nothing switches the S on.
        ("thioketone", "CC(=S)C", False),
        # Excluded by choice: the literature is unsettled and cystine is common.
        ("disulfide", "CSSC", False),
        ("selenoether (MSE)", "C[Se]C", False),
    ],
)
def test_sigma_hole_activation_matches_the_chemistry(group, smiles, activated):
    chem = pytest.importorskip("rdkit.Chem")
    from chemeleonx.chemistry import _sigma_hole_activated

    molecule = chem.MolFromSmiles(smiles)
    chalcogens = [a for a in molecule.GetAtoms() if a.GetSymbol() in {"S", "Se"}]

    assert chalcogens, group
    assert all(_sigma_hole_activated(a) is activated for a in chalcogens), group


@pytest.mark.parametrize(
    "group, smiles, capacity",
    [
        # Pyramidal, three substituents, three sigma-holes -- and the most strongly
        # activated sulfur in biology, so excluding charged chalcogens dropped SAM.
        ("sulfonium", "C[S+](C)CCC", 3),
        ("thioether", "CSC", 2),
        ("thiol", "CCS", 1),
        # A negatively charged sulfur is a nucleophile; it donates nothing.
        ("thiolate", "CC[S-]", 0),
        # Hypervalent sulfur: the oxygens do the accepting, not the S.
        ("sulfoxide", "CS(=O)C", 0),
        ("sulfonamide", "CS(=O)(=O)N", 0),
    ],
)
def test_chalcogen_donor_capacity_by_oxidation_state(group, smiles, capacity):
    chem = pytest.importorskip("rdkit.Chem")
    from chemeleonx.chemistry import _chalcogen_donor_capacity

    molecule = chem.MolFromSmiles(smiles)
    sulfur = [a for a in molecule.GetAtoms() if a.GetSymbol() in {"S", "Se"}][0]

    assert _chalcogen_donor_capacity(sulfur) == capacity, group


def test_sulfonium_forms_a_chalcogen_bond():
    """SAM's sulfonium was excluded outright by the old formal-charge test."""
    atoms = [
        AtomRecord(0, "SD", "S", (0.0, 0.0, 0.0), "SAM", "1", "A", "A:SAM:1", "ligand",
                   formal_charge=1, chalcogen_donor_capacity=3, sigma_hole_activated=True,
                   bonds=(1, 2, 3)),
        AtomRecord(1, "CG", "C", (-1.8, 0.0, 0.0), "SAM", "1", "A", "A:SAM:1", "ligand", bonds=(0,)),
        AtomRecord(2, "CE", "C", (0.6, -1.7, 0.0), "SAM", "1", "A", "A:SAM:1", "ligand", bonds=(0,)),
        AtomRecord(3, "C5", "C", (0.6, 0.85, -1.5), "SAM", "1", "A", "A:SAM:1", "ligand", bonds=(0,)),
        AtomRecord(4, "O1", "O", (3.2, 0.0, 0.0), "LIG", "2", "B", "B:LIG:2", "ligand", acceptor_capacity=2),
    ]
    components = [
        ComponentRecord("A:SAM:1", "SAM", "A", "1", "ligand", (0, 1, 2, 3)),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (4,)),
    ]
    bonds = of_type(detect(atoms, components), "chalcogen_bond")

    assert len(bonds) == 1
    assert bonds[0].rejection_reason is None
    # Pyramidal: there is no single substituent plane, so no elevation is reported.
    assert "elevation" not in bonds[0].metadata


def test_chalcogen_pi_does_not_require_an_activated_sigma_hole():
    """Met-S...pi is dispersion driven, not a sigma-hole interaction; gating it would
    delete the biotin-Trp contacts that motivated keeping the two types separate."""
    donor = AtomRecord(6, "S1", "S", (0.0, 0.0, 4.2), "LIG", "2", "B", "B:LIG:2", "ligand",
                       chalcogen_donor_capacity=2, sigma_hole_activated=False, bonds=(7,))
    substituent = AtomRecord(7, "C6", "C", (0.0, 1.6, 5.0), "LIG", "2", "B", "B:LIG:2", "ligand",
                             bonds=(6,))
    components = [
        phenyl_component(),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (6, 7)),
    ]

    assert "chalcogen_pi" in accepted_types(detect(phenyl() + [donor, substituent], components))


def test_multi_atom_occlusion_ignores_the_participants_own_substituents():
    """A centroid-to-centroid segment clips the bonded neighbours of the very atoms
    taking part; biotin's S...pi to Trp was lost to exactly that."""
    donor = AtomRecord(6, "S1", "S", (0.0, 0.0, 4.2), "LIG", "2", "B", "B:LIG:2", "ligand",
                       chalcogen_donor_capacity=2, bonds=(7,))
    # A substituent sitting directly on the S -> ring-centroid line.
    substituent = AtomRecord(7, "C6", "C", (0.0, 0.0, 2.6), "LIG", "2", "B", "B:LIG:2", "ligand",
                             bonds=(6,))
    components = [
        phenyl_component(),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (6, 7)),
    ]

    assert "chalcogen_pi" in accepted_types(detect(phenyl() + [donor, substituent], components))


# ------------------------------------------------------------ hydrogen isotopes


def test_deuterium_is_treated_as_hydrogen():
    """Neutron structures put D on every exchangeable site.

    Keying on element == "H" alone found zero hydrogen bonds in a neutron structure and
    reported it as missing_donor_hydrogen_geometry -- a silent wrong answer, not an error.
    """
    from chemeleonx.parser import normalize_element

    assert normalize_element("D") == "H"
    assert normalize_element("T") == "H"
    # Ordinary elements are untouched, including the two-letter ones.
    assert normalize_element("SE") == "Se"
    assert normalize_element("C") == "C"


def test_deuterated_donor_forms_a_hydrogen_bond():
    atoms = [
        AtomRecord(0, "N", "N", (0.0, 0.0, 0.0), "GLY", "1", "A", "A:GLY:1", "protein", bonds=(1,)),
        # As deposited by a neutron refinement: the exchangeable donor is a deuteron.
        AtomRecord(1, "D", "H", (1.0, 0.0, 0.0), "GLY", "1", "A", "A:GLY:1", "protein",
                   bonds=(0,), metadata={"isotope": "D"}),
        AtomRecord(2, "O1", "O", (2.9, 0.0, 0.0), "LIG", "2", "B", "B:LIG:2", "ligand"),
    ]
    _assign_default_atom_chemistry(atoms)
    components = [
        ComponentRecord("A:GLY:1", "GLY", "A", "1", "protein", (0, 1)),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (2,)),
    ]

    bonds = of_type(detect(atoms, components), "hbond")
    assert len(bonds) == 1
    assert bonds[0].rejection_reason is None
    assert bonds[0].angle == pytest.approx(180.0)
    # The deuteron is used as the hydrogen, and the deposited isotope is not lost.
    assert bonds[0].metadata["hydrogen_atom_id"] == 1
    assert atoms[1].metadata["isotope"] == "D"


# ------------------------------------------------- sequence-separation guards


def peptide_unit(start_id, residue_number, origin, *, chain="A"):
    """A backbone amide/carbonyl unit: C(=O) of residue i, N of residue i+1."""
    x, y, z = origin
    return [
        AtomRecord(start_id, "C", "C", (x, y, z), "ALA", str(residue_number), chain,
                   f"{chain}:ALA:{residue_number}", "protein", bonds=(start_id + 1, start_id + 2)),
        AtomRecord(start_id + 1, "O", "O", (x, y + 1.23, z), "ALA", str(residue_number), chain,
                   f"{chain}:ALA:{residue_number}", "protein", bonds=(start_id,)),
        AtomRecord(start_id + 2, "CA", "C", (x - 1.5, y - 0.5, z), "ALA", str(residue_number), chain,
                   f"{chain}:ALA:{residue_number}", "protein", bonds=(start_id,)),
        AtomRecord(start_id + 3, "N", "N", (x + 1.2, y - 0.7, z), "GLY", str(residue_number + 1),
                   chain, f"{chain}:GLY:{residue_number + 1}", "protein", bonds=(start_id + 4,)),
        AtomRecord(start_id + 4, "H", "H", (x + 1.2, y + 0.3, z), "GLY", str(residue_number + 1),
                   chain, f"{chain}:GLY:{residue_number + 1}", "protein", bonds=(start_id + 3,)),
    ]


def test_amide_bridge_ignores_consecutive_peptide_units():
    """Adjacent backbone amides overlap and always sit at a short N...O distance.

    Before the separation guard this was the *only* thing amide_bridge ever found: in
    1NTP the closest pair was Asp71/Asn72 against Asn72/Ile73 at a perfect 2.78 A, with
    the N-H and C=O nearly perpendicular. All of it was chain-local artefact.
    """
    # Two peptide units one residue apart, placed close enough to pass the distance test.
    atoms = peptide_unit(0, 71, (0.0, 0.0, 0.0)) + peptide_unit(5, 72, (1.2, 2.9, 0.0))
    components = [
        ComponentRecord("A:ALA:71", "ALA", "A", "71", "protein", (0, 1, 2)),
        ComponentRecord("A:GLY:72", "GLY", "A", "72", "protein", (3, 4)),
        ComponentRecord("A:ALA:72", "ALA", "A", "72", "protein", (5, 6, 7)),
        ComponentRecord("A:GLY:73", "GLY", "A", "73", "protein", (8, 9)),
    ]

    assert of_type(detect(atoms, components), "amide_bridge") == []


def test_n_pi_star_ignores_carbonyls_in_the_same_residue():
    """An ASP side-chain carboxylate reaches its own backbone carbonyl within the
    Burgi-Dunitz window because they are three bonds apart, not because they interact.
    These were ~4% of every n_pi_star hit on real structures."""
    # Backbone C=O and side-chain carboxylate of one ASP, at accepting geometry.
    theta = radians(109.0)
    donor_oxygen = (0.0, -3.0 * cos(theta), 3.0 * sin(theta))
    atoms = [
        AtomRecord(0, "C", "C", (0.0, 0.0, 0.0), "ASP", "67", "A", "A:ASP:67", "protein", bonds=(1, 2)),
        AtomRecord(1, "O", "O", (0.0, -1.23, 0.0), "ASP", "67", "A", "A:ASP:67", "protein", bonds=(0,)),
        AtomRecord(2, "CA", "C", (-1.5, 0.8, 0.0), "ASP", "67", "A", "A:ASP:67", "protein", bonds=(0,)),
        AtomRecord(3, "CG", "C", tuple(donor_oxygen[i] + (0.6, 1.1, 0.0)[i] for i in range(3)),
                   "ASP", "67", "A", "A:ASP:67", "protein", bonds=(4, 5)),
        AtomRecord(4, "OD1", "O", donor_oxygen, "ASP", "67", "A", "A:ASP:67", "protein", bonds=(3,)),
        AtomRecord(5, "CB", "C", tuple(donor_oxygen[i] + (2.0, 1.6, 0.0)[i] for i in range(3)),
                   "ASP", "67", "A", "A:ASP:67", "protein", bonds=(3,)),
    ]
    components = [ComponentRecord("A:ASP:67", "ASP", "A", "67", "protein", tuple(range(6)))]

    assert of_type(detect(atoms, components), "n_pi_star") == []

    # The same geometry across two residues is the canonical O(i)->C(i+1) case: kept.
    for atom in atoms[3:]:
        atom.residue_id = "68"
        atom.component_id = "A:ASP:68"
    components = [
        ComponentRecord("A:ASP:67", "ASP", "A", "67", "protein", (0, 1, 2)),
        ComponentRecord("A:ASP:68", "ASP", "A", "68", "protein", (3, 4, 5)),
    ]
    assert of_type(detect(atoms, components), "n_pi_star")
