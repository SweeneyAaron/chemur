from chemur.assignment import assign_interactions
from chemur.models import AtomRecord, CandidateInteraction, ResourceDemand
from chemur.profile import load_profile


def test_intra_hbond_consumes_capacity_before_inter_hbond():
    atoms = [
        AtomRecord(
            atom_id=0,
            name="N",
            element="N",
            coord=(0.0, 0.0, 0.0),
            residue_name="ASN",
            residue_id="1",
            chain_id="A",
            component_id="A:ASN:1",
            molecule_type="protein",
            donor_capacity=1,
        ),
        AtomRecord(
            atom_id=1,
            name="O",
            element="O",
            coord=(2.8, 0.0, 0.0),
            residue_name="ASN",
            residue_id="1",
            chain_id="A",
            component_id="A:ASN:1",
            molecule_type="protein",
            acceptor_capacity=2,
        ),
        AtomRecord(
            atom_id=2,
            name="O",
            element="O",
            coord=(2.9, 0.0, 0.0),
            residue_name="LIG",
            residue_id="2",
            chain_id="B",
            component_id="B:LIG:2",
            molecule_type="ligand",
            acceptor_capacity=2,
        ),
    ]
    intra = CandidateInteraction(
        interaction_type="hbond",
        feature_ids=(0, 1),
        atom_ids=(0, 1),
        component_ids=("A:ASN:1", "A:ASN:1"),
        distance=2.8,
        priority=30,
        stage="intra",
        resource_demands=(ResourceDemand("donor", 0), ResourceDemand("acceptor", 1)),
    )
    inter = CandidateInteraction(
        interaction_type="hbond",
        feature_ids=(0, 2),
        atom_ids=(0, 2),
        component_ids=("A:ASN:1", "B:LIG:2"),
        distance=2.7,
        priority=30,
        stage="inter",
        resource_demands=(ResourceDemand("donor", 0), ResourceDemand("acceptor", 2)),
    )

    assigned = assign_interactions(atoms, [inter, intra], load_profile())

    assert len(assigned) == 1
    assert assigned[0].component_ids == ("A:ASN:1", "A:ASN:1")
    assert inter.rejection_reason == "resource_exhausted:donor:0"
