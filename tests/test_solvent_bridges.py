"""A solvent bridge describes two assigned bonds without consuming them again."""

from dataclasses import asdict
import json

import pytest

from chemur.api import assign_for_atoms
from chemur.assignment import assign_interactions
from chemur.cosplif import _role_key
from chemur.features import perceive_features
from chemur.ifm_matcher import _directionality
from chemur.interactions import DetectorContext, _solvent_bridge_candidates, generate_candidates
from chemur.models import AtomRecord, CandidateInteraction, ComponentRecord, ResourceDemand
from chemur.profile import load_profile


def _network(left_donates=True, right_donates=True):
    def atom(atom_id, name, element, x, component, molecule_type, **kwargs):
        return AtomRecord(
            atom_id, name, element, (x, 0.0, 0.0), component.split(":")[1],
            component.split(":")[2], component[0], component, molecule_type, **kwargs,
        )

    atoms = [
        atom(0, "N" if left_donates else "O", "N" if left_donates else "O", -2.8,
             "A:ASN:1", "protein", donor_capacity=int(left_donates),
             acceptor_capacity=int(not left_donates), bonds=(1,) if left_donates else ()),
        atom(2, "O", "O", 0.0, "W:HOH:2", "solvent",
             donor_capacity=int(not left_donates) + int(not right_donates),
             acceptor_capacity=int(left_donates) + int(right_donates)),
        atom(3, "N" if right_donates else "O", "N" if right_donates else "O", 2.8,
             "L:LIG:3", "ligand", donor_capacity=int(right_donates),
             acceptor_capacity=int(not right_donates), bonds=(4,) if right_donates else ()),
    ]
    for donates, hydrogen_id, water_hydrogen_id, x, endpoint in (
        (left_donates, 1, 5, -1.8, atoms[0]),
        (right_donates, 4, 6, 1.8, atoms[2]),
    ):
        if donates:
            atoms.append(atom(hydrogen_id, "H", "H", x, endpoint.component_id,
                              endpoint.molecule_type, bonds=(endpoint.atom_id,)))
        else:
            water = atoms[1]
            water.bonds += (water_hydrogen_id,)
            atoms.append(atom(water_hydrogen_id, "H", "H", -1.0 if x < 0 else 1.0,
                              water.component_id, "solvent", bonds=(water.atom_id,)))
    components = [
        ComponentRecord(component_id, component_id.split(":")[1], component_id[0],
                        component_id.split(":")[2], molecule_type,
                        tuple(atom.atom_id for atom in atoms if atom.component_id == component_id))
        for component_id, molecule_type in (
            ("A:ASN:1", "protein"), ("W:HOH:2", "solvent"), ("L:LIG:3", "ligand")
        )
    ]
    return atoms, components


def _bridges(interactions):
    return [interaction for interaction in interactions if interaction.interaction_type == "solvent_bridge"]


@pytest.mark.parametrize("left_donates,right_donates", [(True, True), (True, False), (False, True), (False, False)])
def test_bridge_keeps_two_bonds_without_extra_capacity(left_donates, right_donates):
    atoms, components = _network(left_donates, right_donates)
    _, candidates, assigned = assign_for_atoms(atoms, components, load_profile())

    assert [interaction.interaction_type for interaction in assigned] == ["hbond", "hbond", "solvent_bridge"]
    bridge, = _bridges(assigned)
    assert bridge.resource_demands == ()
    assert bridge.distance == pytest.approx(5.6)
    assert set(bridge.metadata["leg_assignment_orders"]) == {1, 2}
    assert {tuple(pair) for pair in bridge.metadata["leg_feature_ids"]} == {
        interaction.feature_ids for interaction in assigned if interaction.interaction_type == "hbond"
    }
    assert all(candidate.rejection_reason is None for candidate in candidates)


@pytest.mark.parametrize("blocked_sides", [(-1,), (1,), (-1, 1)])
def test_occluded_legs_cannot_form_bridges(blocked_sides):
    atoms, components = _network()
    for index, side in enumerate(blocked_sides, start=7):
        atoms.append(AtomRecord(index, "C", "C", (side * 1.4, 0.5, 0.0),
                                "ALA", str(index), "B", f"B:ALA:{index}", "protein"))
    _, candidates, assigned = assign_for_atoms(atoms, components, load_profile())

    assert sum(candidate.rejection_reason == "occluded" for candidate in candidates) == len(blocked_sides)
    assert _bridges(candidates) == []
    assert _bridges(assigned) == []


@pytest.mark.parametrize("missing_hydrogen", [False, True])
def test_angle_or_missing_hydrogen_rejection_prevents_bridge(missing_hydrogen):
    atoms, components = _network()
    left = next(atom for atom in atoms if atom.atom_id == 0)
    if missing_hydrogen:
        left.bonds = ()
    else:
        next(atom for atom in atoms if atom.atom_id == 1).coord = (-3.8, 0.0, 0.0)
    _, candidates, assigned = assign_for_atoms(atoms, components, load_profile())

    assert any(candidate.rejection_reason for candidate in candidates)
    assert _bridges(candidates) == []
    assert _bridges(assigned) == []


def _competing_contact(priority=10):
    return CandidateInteraction(
        "hbond_pi", (100, 101), (0, 3), ("A:ASN:1", "L:LIG:3"),
        2.0, priority, "inter", resource_demands=(ResourceDemand("donor", 0),),
    )


@pytest.mark.parametrize("bridge_priority", [1, 40, 999])
def test_bridge_requires_assigned_legs_regardless_of_priority(bridge_priority):
    atoms, components = _network()
    profile = load_profile({"rules": {"solvent_bridge": {"priority": bridge_priority}}})
    candidates = generate_candidates(atoms, perceive_features(atoms, components), profile)
    candidates.append(_competing_contact())
    assigned = assign_interactions(atoms, list(reversed(candidates)), profile)

    bridge, = _bridges(candidates)
    assert bridge.rejection_reason == "unassigned_hbond_leg"
    assert _bridges(assigned) == []
    assert sum(interaction.interaction_type == "hbond" for interaction in assigned) == 1


def test_bridge_leaves_remaining_resources_for_other_contacts():
    atoms, components = _network()
    atoms[0].donor_capacity = 2
    profile = load_profile({"rules": {"solvent_bridge": {"consume": {"donor": 99, "acceptor": 99}}}})
    candidates = generate_candidates(atoms, perceive_features(atoms, components), profile)
    candidates.append(_competing_contact(priority=90))
    assigned = assign_interactions(atoms, candidates, profile)

    assert [interaction.interaction_type for interaction in assigned] == ["hbond", "hbond", "hbond_pi", "solvent_bridge"]
    assert _bridges(assigned)[0].resource_demands == ()


@pytest.mark.parametrize("cutoff,expected", [(2.79, 0), (2.8, 1), (3.0, 1)])
def test_bridge_cutoff_applies_to_each_leg(cutoff, expected):
    atoms, components = _network()
    profile = load_profile({"rules": {"solvent_bridge": {"distance": cutoff}}})
    _, _, assigned = assign_for_atoms(atoms, components, profile)

    assert sum(interaction.interaction_type == "hbond" for interaction in assigned) == 2
    assert len(_bridges(assigned)) == expected


def test_one_leg_beyond_bridge_cutoff_is_sufficient_to_reject_bridge():
    atoms, components = _network()
    for atom in atoms:
        if atom.atom_id in {3, 4}:
            atom.coord = (atom.coord[0] + 0.4, 0.0, 0.0)
    _, candidates, assigned = assign_for_atoms(
        atoms, components, load_profile({"rules": {"solvent_bridge": {"distance": 3.0}}})
    )

    assert sum(interaction.interaction_type == "hbond" for interaction in assigned) == 2
    assert _bridges(candidates) == []
    assert _bridges(assigned) == []


@pytest.mark.parametrize("rule", ["hbond", "solvent_bridge"])
def test_disabled_rule_prevents_bridge(rule):
    atoms, components = _network()
    _, candidates, assigned = assign_for_atoms(
        atoms, components, load_profile({"rules": {rule: {"enabled": False}}})
    )
    assert _bridges(candidates) == []
    assert _bridges(assigned) == []


@pytest.mark.parametrize("same_component", [False, True])
def test_bridge_requires_two_non_solvent_components(same_component):
    atoms, _ = _network()
    right = next(atom for atom in atoms if atom.atom_id == 3)
    if same_component:
        right.component_id = atoms[0].component_id
    else:
        right.molecule_type = "solvent"
    _, candidates, assigned = assign_for_atoms(atoms, [], load_profile())

    assert _bridges(candidates) == []
    assert _bridges(assigned) == []


def test_leg_pairing_is_deterministic_and_deduplicated():
    atoms, components = _network()
    profile = load_profile()
    features = perceive_features(atoms, components)
    legs = [candidate for candidate in generate_candidates(atoms, features, profile)
            if candidate.interaction_type == "hbond"]
    context = DetectorContext({atom.atom_id: atom for atom in atoms}, {}, profile,
                              candidates=list(reversed(legs)) + legs)
    bridges = _solvent_bridge_candidates(context, profile["rules"]["solvent_bridge"])

    assert len(bridges) == 1
    assert bridges[0].metadata["leg_feature_ids"] == [list(leg.feature_ids) for leg in sorted(legs, key=lambda leg: leg.feature_ids)]


@pytest.mark.parametrize("dependencies", [None, [], [[0, 1]], [[0, 1], [0, 1]], [[0, []], [2, 3]]])
def test_malformed_bridge_dependencies_are_rejected(dependencies):
    atoms, components = _network()
    profile = load_profile()
    candidates = generate_candidates(atoms, perceive_features(atoms, components), profile)
    bridge, = _bridges(candidates)
    bridge.metadata["leg_feature_ids"] = dependencies
    assigned = assign_interactions(atoms, candidates, profile)

    assert bridge.rejection_reason == "missing_hbond_leg_dependencies"
    assert _bridges(assigned) == []


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("left_donates,right_donates,role", [
    (True, True, "both_donor"), (True, False, "target_donor"),
    (False, True, "ligand_donor"), (False, False, "none"),
])
def test_bridge_donor_roles_survive_json_and_legacy_formats(left_donates, right_donates, role, legacy):
    atoms, components = _network(left_donates, right_donates)
    _, _, assigned = assign_for_atoms(atoms, components, load_profile())
    bridge, = _bridges(assigned)
    if legacy:
        bridge.resource_demands = tuple(demand for bond in assigned[:2] for demand in bond.resource_demands)
        del bridge.metadata["leg_donor_atom_ids"]
    payload = json.loads(json.dumps(asdict(bridge)))

    assert _directionality(payload, ligand_atom_ids={3}, target_atom_ids={0}) == {
        "ligand_is_donor": right_donates, "target_is_donor": left_donates,
    }
    assert _role_key(bridge, {3}, {0}) == role


def test_default_bridge_profile_has_no_additional_resource_consumption():
    assert load_profile()["rules"]["solvent_bridge"]["consume"] == {}
