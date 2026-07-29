from chemeleonx.features import perceive_features
from chemeleonx.interactions import generate_candidates
from chemeleonx.models import AtomRecord, ComponentRecord
from chemeleonx.profile import load_profile


def test_generates_basic_hbond_candidate():
    atoms = [
        AtomRecord(
            0,
            "N",
            "N",
            (0.0, 0.0, 0.0),
            "ASN",
            "1",
            "A",
            "A:ASN:1",
            "protein",
            donor_capacity=1,
        ),
        AtomRecord(
            1,
            "O",
            "O",
            (2.8, 0.0, 0.0),
            "LIG",
            "2",
            "B",
            "B:LIG:2",
            "ligand",
            acceptor_capacity=2,
        ),
    ]
    components = [
        ComponentRecord("A:ASN:1", "ASN", "A", "1", "protein", (0,)),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (1,)),
    ]

    features = perceive_features(atoms, components)
    candidates = generate_candidates(atoms, features, load_profile())

    assert any(candidate.interaction_type == "hbond" for candidate in candidates)


def test_hbond_candidate_requires_explicit_hydrogen_geometry():
    atoms = [
        AtomRecord(
            0,
            "N",
            "N",
            (0.0, 0.0, 0.0),
            "ASN",
            "1",
            "A",
            "A:ASN:1",
            "protein",
            donor_capacity=1,
            bonds=(1,),
        ),
        AtomRecord(
            1,
            "H",
            "H",
            (1.0, 0.0, 0.0),
            "ASN",
            "1",
            "A",
            "A:ASN:1",
            "protein",
            bonds=(0,),
        ),
        AtomRecord(
            2,
            "O",
            "O",
            (2.8, 0.0, 0.0),
            "LIG",
            "2",
            "B",
            "B:LIG:2",
            "ligand",
            acceptor_capacity=2,
        ),
    ]
    components = [
        ComponentRecord("A:ASN:1", "ASN", "A", "1", "protein", (0, 1)),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (2,)),
    ]

    features = perceive_features(atoms, components)
    candidates = generate_candidates(atoms, features, load_profile())
    hbond = next(candidate for candidate in candidates if candidate.interaction_type == "hbond")

    assert hbond.angle == 180.0
    assert hbond.rejection_reason is None


def test_hbond_distance_override_changes_candidate_gate():
    atoms = [
        AtomRecord(
            0,
            "N",
            "N",
            (0.0, 0.0, 0.0),
            "ASN",
            "1",
            "A",
            "A:ASN:1",
            "protein",
            donor_capacity=1,
            bonds=(1,),
        ),
        AtomRecord(
            1,
            "H",
            "H",
            (1.0, 0.0, 0.0),
            "ASN",
            "1",
            "A",
            "A:ASN:1",
            "protein",
            bonds=(0,),
        ),
        AtomRecord(
            2,
            "O",
            "O",
            (3.3, 0.0, 0.0),
            "LIG",
            "2",
            "B",
            "B:LIG:2",
            "ligand",
            acceptor_capacity=2,
        ),
    ]
    components = [
        ComponentRecord("A:ASN:1", "ASN", "A", "1", "protein", (0, 1)),
        ComponentRecord("B:LIG:2", "LIG", "B", "2", "ligand", (2,)),
    ]
    profile = load_profile({"rules": {"hbond": {"distance": 3.1}}})

    features = perceive_features(atoms, components)
    candidates = generate_candidates(atoms, features, profile)

    assert not any(candidate.interaction_type == "hbond" for candidate in candidates)


def test_carbon_donor_pi_contact_is_ch_pi_not_hbond_pi():
    atoms = [
        AtomRecord(0, "CG1", "C", (0.0, 0.0, 4.0), "VAL", "122", "R", "R:VAL:122", "protein", donor_capacity=1, carbon_donor_capacity=1, bonds=(1,)),
        AtomRecord(1, "HG1", "H", (0.0, 0.0, 3.0), "VAL", "122", "R", "R:VAL:122", "protein", bonds=(0,)),
        AtomRecord(2, "C1", "C", (1.0, 0.0, 0.0), "5FW", "401", "R", "R:5FW:401", "ligand", is_aromatic=True),
        AtomRecord(3, "C2", "C", (0.5, 0.866, 0.0), "5FW", "401", "R", "R:5FW:401", "ligand", is_aromatic=True),
        AtomRecord(4, "C3", "C", (-0.5, 0.866, 0.0), "5FW", "401", "R", "R:5FW:401", "ligand", is_aromatic=True),
        AtomRecord(5, "C4", "C", (-1.0, 0.0, 0.0), "5FW", "401", "R", "R:5FW:401", "ligand", is_aromatic=True),
        AtomRecord(6, "C5", "C", (-0.5, -0.866, 0.0), "5FW", "401", "R", "R:5FW:401", "ligand", is_aromatic=True),
        AtomRecord(7, "C6", "C", (0.5, -0.866, 0.0), "5FW", "401", "R", "R:5FW:401", "ligand", is_aromatic=True),
    ]
    components = [
        ComponentRecord("R:VAL:122", "VAL", "R", "122", "protein", (0, 1)),
        ComponentRecord(
            "R:5FW:401",
            "5FW",
            "R",
            "401",
            "ligand",
            (2, 3, 4, 5, 6, 7),
            metadata={"template_rings": [(2, 3, 4, 5, 6, 7)]},
        ),
    ]

    features = perceive_features(atoms, components)
    candidates = generate_candidates(atoms, features, load_profile())
    assigned_types = {candidate.interaction_type for candidate in candidates if not candidate.rejection_reason}

    assert "ch_pi" in assigned_types
    assert "hbond_pi" not in assigned_types
