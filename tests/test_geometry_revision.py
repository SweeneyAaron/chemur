"""Regression cases for the published geometry revision and explicit endpoints."""
from math import cos, pi, radians, sin

import pytest

from chemur.interactions import (
    DetectorContext, _amide_pi_candidates, _chalcogen_bond_candidates,
    _chalcogen_pi_candidates, _ch_pi_candidates, _halogen_bond_candidates,
    _halogen_pi_candidates, _hbond_candidates, _hbond_pi_candidates,
    _metal_candidates, _residue_separation_ok, _ring_stack_candidates,
    _salt_bridge_candidates, _tetrel_bond_candidates, _weak_hbond_candidates,
)
from chemur.models import AtomRecord, FeatureRecord
from chemur.profile import load_profile


def atom(i, element, coord, **kwargs):
    return AtomRecord(i, f'{element}{i}', element, coord, 'LIG', str(i), 'A', f'A:LIG:{i}', 'ligand', **kwargs)


def feature(i, kind, atoms, *, center=None, normal=None, metadata=None):
    return FeatureRecord(i, kind, atoms[0].component_id, tuple(a.atom_id for a in atoms),
                         center or atoms[0].coord, normal=normal, metadata=metadata or {})


def context(atoms, features):
    grouped = {}
    for f in features:
        grouped.setdefault(f.feature_type, []).append(f)
    return DetectorContext({a.atom_id: a for a in atoms}, grouped, load_profile())


def rule(ctx, name):
    return ctx.profile['rules'][name]


def pi_context(element='N', position=(0., 0., 4.55), hydrogens=((0., 0., 3.45),)):
    donor = atom(0, element, position, bonds=tuple(range(1, len(hydrogens) + 1)))
    hs = [atom(i, 'H', coord, bonds=(0,)) for i, coord in enumerate(hydrogens, 1)]
    ring_atom = atom(20, 'C', (0., 0., 0.))
    point = feature(0, 'carbon_donor' if element == 'C' else 'donor', [donor])
    ring = feature(1, 'ring', [ring_atom], normal=(0., 0., 1.), metadata={'aromatic': True})
    return context([donor, *hs, ring_atom], [point, ring])


@pytest.mark.parametrize('element,detector,name', [('N', _hbond_pi_candidates, 'hbond_pi'), ('O', _hbond_pi_candidates, 'hbond_pi'), ('C', _ch_pi_candidates, 'ch_pi')])
def test_pi_distance_is_from_hydrogen_without_obsolete_heavy_gate(element, detector, name):
    ctx = pi_context(element)
    result, = detector(ctx, rule(ctx, name))
    assert result.rejection_reason is None
    assert result.distance == pytest.approx(4.55)
    assert result.metadata['hydrogen_distance'] == pytest.approx(3.45)


def test_pi_criteria_use_one_hydrogen_and_choose_a_jointly_valid_one():
    ctx = pi_context(position=(0., 0., 4.), hydrogens=((0., 0., 3.6), (1., 0., 3.)))
    result, = _hbond_pi_candidates(ctx, rule(ctx, 'hbond_pi'))
    assert result.rejection_reason is None
    assert result.metadata['hydrogen_atom_id'] == 2
    ctx.atoms_by_id[2].coord = (2., 0., 4.)  # Close enough, but wrong angle.
    result, = _hbond_pi_candidates(ctx, rule(ctx, 'hbond_pi'))
    assert result.rejection_reason is not None


@pytest.mark.parametrize('hydrogen_z,accepted', [(3.5, True), (3.5001, False)])
def test_pi_hydrogen_distance_boundary(hydrogen_z, accepted):
    ctx = pi_context(position=(0., 0., 4.5), hydrogens=((0., 0., hydrogen_z),))
    results = _hbond_pi_candidates(ctx, rule(ctx, 'hbond_pi'))
    assert any(result.rejection_reason is None for result in results) == accepted


def test_degenerate_ring_normal_is_rejected():
    ctx = pi_context()
    ctx.features('ring')[0].normal = (0., 0., 0.)
    assert _hbond_pi_candidates(ctx, rule(ctx, 'hbond_pi')) == []


@pytest.mark.parametrize('distance,element,accepted', [(3.6, 'O', True), (3.6001, 'O', False), (3.2, 'N', False)])
def test_weak_hbond_uses_carbon_oxygen_distance(distance, element, accepted):
    donor = atom(0, 'C', (0., 0., 0.), bonds=(1,), weak_hbond_activated=True)
    hydrogen = atom(1, 'H', (1., 0., 0.))
    acceptor = atom(2, element, (distance, 0., 0.))
    ctx = context([donor, hydrogen, acceptor], [feature(0, 'carbon_donor', [donor]), feature(1, 'acceptor', [acceptor])])
    hits = _weak_hbond_candidates(ctx, rule(ctx, 'weak_hbond'))
    assert any(h.rejection_reason is None for h in hits) == accepted


def test_hbind_requires_real_acceptor_antecedent_and_same_hydrogen_geometry():
    donor = atom(0, 'N', (0., 0., 0.), bonds=(1,))
    hydrogen = atom(1, 'H', (1., 0., 0.))
    acceptor = atom(2, 'O', (2.8, 0., 0.), bonds=(3,))
    antecedent = atom(3, 'C', (4., 0., 0.))
    ctx = context([donor, hydrogen, acceptor, antecedent], [feature(0, 'donor', [donor]), feature(1, 'acceptor', [acceptor])])
    hit, = _hbond_candidates(ctx, rule(ctx, 'hbond'))
    assert hit.rejection_reason is None
    assert hit.metadata['acceptor_angle'] == 180.
    acceptor.bonds = ()
    hit, = _hbond_candidates(ctx, rule(ctx, 'hbond'))
    assert hit.rejection_reason == 'missing_acceptor_antecedent_geometry'
    acceptor.molecule_type = 'solvent'
    hit, = _hbond_candidates(ctx, rule(ctx, 'hbond'))
    assert hit.rejection_reason is None  # Existing water-leg definition is preserved.


@pytest.mark.parametrize('position,root,expected', [
    ((0., 0., 3.5), (0., 0., 5.3), ['directional', 'sulfur_aromatic']),
    ((3.5, 0., 0.), (5.3, 0., 0.), ['directional']),
    ((4.5, 0., 4.), (4.5, 1.8, 4.), ['sulfur_aromatic']),
])
def test_sulfur_pi_union_is_one_nonconsuming_contact(position, root, expected):
    sulfur = atom(0, 'S', position, bonds=(1,))
    carbon = atom(1, 'C', root)
    ring_atom = atom(20, 'C', (0., 0., 0.))
    ctx = context([sulfur, carbon, ring_atom], [feature(0, 'sulfur_pi', [sulfur], metadata={'sulfur_pi_types': ['thioether']}),
        feature(1, 'ring', [ring_atom], normal=(0., 0., 1.), metadata={'aromatic': True})])
    hit, = _chalcogen_pi_candidates(ctx, rule(ctx, 'chalcogen_pi'))
    assert hit.metadata['geometry_classifications'] == expected
    assert hit.resource_demands == ()
    if position[0] == 4.5:
        assert hit.offset > 2.5


@pytest.mark.parametrize('donor_element,acceptor_element,lower,upper', [('S', 'O', 2., 3.32), ('Se', 'O', 2., 3.42), ('Se', 'S', 2.5, 4.07)])
def test_chalcogen_pair_specific_windows_include_methionine(donor_element, acceptor_element, lower, upper):
    donor = atom(0, donor_element, (0., 0., 0.), bonds=(1,), sigma_hole_activated=False)
    root = atom(1, 'C', (-1.8, 0., 0.))
    acceptor = atom(2, acceptor_element, (upper, 0., 0.))
    ctx = context([donor, root, acceptor], [feature(0, 'chalcogen_donor', [donor]), feature(1, 'acceptor', [acceptor])])
    hit, = _chalcogen_bond_candidates(ctx, rule(ctx, 'chalcogen_bond'))
    assert hit.rejection_reason is None
    acceptor.coord = (lower, 0., 0.)
    ctx.features('acceptor')[0].center = acceptor.coord
    assert _chalcogen_bond_candidates(ctx, rule(ctx, 'chalcogen_bond')) == []


def test_tetrel_requires_both_angles_and_carbonyl_oxygen():
    donor = atom(0, 'C', (0., 0., 0.), bonds=(1,))
    root = atom(1, 'S', (-1.8, 0., 0.))
    oxygen = atom(2, 'O', (3.1, 0., 0.), bonds=(3,))
    carbon = atom(3, 'C', (4.33, 0., 0.))
    carbonyl = feature(2, 'carbonyl', [carbon, oxygen], metadata={'carbon_atom_id': 3, 'oxygen_atom_id': 2})
    ctx = context([donor, root, oxygen, carbon], [feature(0, 'tetrel_donor', [donor]), feature(1, 'acceptor', [oxygen]), carbonyl])
    hit, = _tetrel_bond_candidates(ctx, rule(ctx, 'tetrel_bond'))
    assert hit.rejection_reason is None
    assert hit.metadata['acceptor_angle'] == 180.
    carbon.coord = (3.1, 1.23, 0.)
    hit, = _tetrel_bond_candidates(ctx, rule(ctx, 'tetrel_bond'))
    assert hit.rejection_reason == 'acceptor_angle_outside_cutoff'
    ctx.features_by_type.pop('carbonyl')
    assert _tetrel_bond_candidates(ctx, rule(ctx, 'tetrel_bond')) == []


def test_aliphatic_pi_uses_strict_search_windows_without_offset_gate():
    a = atom(0, 'C', (4., 0., 4.))
    b = atom(1, 'C', (0., 0., 0.))
    first = feature(0, 'ring', [a], normal=(0., 0., 1.))
    second = feature(1, 'ring', [b], normal=(0., 0., 1.))
    geometry = load_profile()['rules']['aliphatic_pi_stack']
    hits = _ring_stack_candidates('aliphatic_pi_stack', geometry, [(first, second)], gate_angle=True)
    assert len(hits) == 1 and hits[0].offset == 4.
    first.center = (0., 0., 5.)
    assert _ring_stack_candidates('aliphatic_pi_stack', geometry, [(first, second)], gate_angle=True) == []


def test_same_author_chain_ligand_polymer_is_not_sequence_filtered():
    first = atom(0, 'C', (0., 0., 0.))
    second = atom(1, 'C', (1., 0., 0.))
    second.molecule_type = 'protein'
    first.residue_id = second.residue_id = '133'
    assert _residue_separation_ok(first, second, 2)
    first.molecule_type = 'protein'
    first.metadata.update(source_label_asym_id='X', source_polymer_sequence_index=10)
    second.metadata.update(source_label_asym_id='X', source_polymer_sequence_index=11)
    assert not _residue_separation_ok(first, second, 2)
    second.metadata['source_label_asym_id'] = 'Y'
    assert _residue_separation_ok(first, second, 2)


def test_removed_geometry_overrides_fail_actionably():
    with pytest.raises(ValueError, match='removed'):
        load_profile({'rules': {'anion_pi': {'enabled': False}}})
    with pytest.raises(ValueError, match='hydrogen_distance'):
        load_profile({'rules': {'hbond_pi': {'distance': 4.5}}})


@pytest.mark.parametrize('angle,accepted', [(129.9, False), (130.1, True)])
def test_weak_hbond_published_angle(angle, accepted):
    donor = atom(0, 'C', (0., 0., 0.), bonds=(1,), weak_hbond_activated=True)
    hydrogen = atom(1, 'H', (1., 0., 0.))
    acceptor = atom(2, 'O', (1. - 2. * cos(radians(angle)), 2. * sin(radians(angle)), 0.))
    ctx = context([donor, hydrogen, acceptor], [feature(0, 'carbon_donor', [donor]), feature(1, 'acceptor', [acceptor])])
    hit, = _weak_hbond_candidates(ctx, rule(ctx, 'weak_hbond'))
    assert (hit.rejection_reason is None) == accepted


@pytest.mark.parametrize('bad_coord', [(0., 0., 0.), (float('nan'), 0., 0.), (float('inf'), 0., 0.)])
def test_degenerate_hydrogen_never_passes_weak_geometry(bad_coord):
    donor = atom(0, 'C', (0., 0., 0.), bonds=(1,), weak_hbond_activated=True)
    hydrogen = atom(1, 'H', bad_coord)
    acceptor = atom(2, 'O', (3.2, 0., 0.))
    ctx = context([donor, hydrogen, acceptor], [feature(0, 'carbon_donor', [donor]), feature(1, 'acceptor', [acceptor])])
    hit, = _weak_hbond_candidates(ctx, rule(ctx, 'weak_hbond'))
    assert hit.rejection_reason is not None


@pytest.mark.parametrize('angle,accepted', [(29., True), (45., False), (61., True)])
def test_aromatic_stack_rejects_intermediate_plane_angles(angle, accepted):
    a, b = atom(0, 'C', (0., 0., 3.5)), atom(1, 'C', (0., 0., 0.))
    first = feature(0, 'ring', [a], normal=(sin(radians(angle)), 0., cos(radians(angle))))
    second = feature(1, 'ring', [b], normal=(0., 0., 1.))
    hits = _ring_stack_candidates('pipi_stack', load_profile()['rules']['pipi_stack'], [(first, second)], gate_angle=False)
    assert bool(hits) == accepted


def test_amide_pi_uses_cn_midpoint_for_geometry_and_occlusion():
    from chemur.interactions import _occlusion_endpoints
    carbon = atom(0, 'C', (-.5, 0., 0.))
    nitrogen = atom(1, 'N', (.5, 0., 0.))
    oxygen = atom(2, 'O', (-.5, 1.2, 0.))
    aromatic = atom(20, 'C', (0., 0., 5.99))
    amide = feature(0, 'amide', [carbon, nitrogen, oxygen], center=(-1/6, .4, 0.), normal=(0., 0., 1.),
                    metadata={'carbon_atom_id': 0, 'nitrogen_atom_id': 1, 'oxygen_atom_id': 2})
    ring = feature(1, 'ring', [aromatic], normal=(0., 0., 1.), metadata={'aromatic': True})
    atoms = [carbon, nitrogen, oxygen, aromatic]
    ctx = context(atoms, [amide, ring])
    hit, = _amide_pi_candidates(ctx, rule(ctx, 'amide_pi'))
    assert hit.distance == pytest.approx(5.99)
    start, end, ignored = _occlusion_endpoints(hit, {a.atom_id: i for i,a in enumerate(atoms)},
        [a.coord for a in atoms], {0: amide, 1: ring}, {}, True)
    assert start == (0., 0., 0.)
    assert end == aromatic.coord


@pytest.mark.parametrize('symbol,position,accepted', [('Na', 1.99, False), ('Na', 2., True), ('K', 2.9, True),
    ('K', 2.901, False), ('Zn', 1.7, True), ('Zn', 2.6, True), ('Zn', 2.601, False)])
def test_metal_coordination_element_windows(symbol, position, accepted):
    metal, oxygen = atom(0, symbol, (0., 0., 0.)), atom(1, 'O', (position, 0., 0.))
    ctx = context([metal, oxygen], [feature(0, 'metal', [metal]), feature(1, 'acceptor', [oxygen])])
    assert bool(_metal_candidates(ctx, rule(ctx, 'metal_coordination'))) == accepted


def test_salt_bridge_uses_nearest_charged_atoms_excluding_arg_carbon():
    carbon, nitrogen, oxygen = atom(0, 'C', (0., 0., 0.)), atom(1, 'N', (3., 0., 0.)), atom(2, 'O', (6.9, 0., 0.))
    cation = feature(0, 'cation', [carbon, nitrogen], center=(1.5, 0., 0.), metadata={'source': 'protein_arg'})
    ctx = context([carbon, nitrogen, oxygen], [cation, feature(1, 'anion', [oxygen])])
    hit, = _salt_bridge_candidates(ctx, rule(ctx, 'salt_bridge'))
    assert hit.distance == pytest.approx(3.9)
    assert hit.metadata['distance_atom_ids'] == [1, 2]
    oxygen.coord = (.1, 0., 0.)
    ctx.features('anion')[0].center = oxygen.coord
    hit, = _salt_bridge_candidates(ctx, rule(ctx, 'salt_bridge'))
    assert hit.distance == pytest.approx(2.9)


def test_halogen_bond_tests_both_donor_and_acceptor_angles():
    halogen = atom(0, 'Cl', (0., 0., 0.), bonds=(1,))
    root = atom(1, 'C', (-1.8, 0., 0.))
    oxygen = atom(2, 'O', (3.5, 0., 0.), bonds=(3,))
    carbon = atom(3, 'C', (4.1, 1.04, 0.))
    ctx = context([halogen, root, oxygen, carbon], [feature(0, 'halogen_donor', [halogen]), feature(1, 'acceptor', [oxygen])])
    hit, = _halogen_bond_candidates(ctx, rule(ctx, 'halogen_bond'))
    assert hit.rejection_reason is None
    carbon.coord = (4.73, 0., 0.)
    hit, = _halogen_bond_candidates(ctx, rule(ctx, 'halogen_bond'))
    assert hit.rejection_reason == 'acceptor_angle_outside_cutoff'
    oxygen.bonds = ()
    hit, = _halogen_bond_candidates(ctx, rule(ctx, 'halogen_bond'))
    assert hit.rejection_reason == 'missing_acceptor_antecedent_geometry'


def test_halogen_pi_requires_carbon_axis_and_strict_obtuse_angle():
    halogen = atom(0, 'Br', (4., 0., 4.), bonds=(1,))
    root = atom(1, 'C', (5., 0., 5.))
    aromatic = atom(20, 'C', (0., 0., 0.))
    ctx = context([halogen, root, aromatic], [feature(0, 'halogen_donor', [halogen]),
        feature(1, 'ring', [aromatic], normal=(0., 0., 1.), metadata={'aromatic': True})])
    hit, = _halogen_pi_candidates(ctx, rule(ctx, 'halogen_pi'))
    assert hit.rejection_reason is None and hit.offset == 4.
    root.coord = (3., 0., 5.)
    hit, = _halogen_pi_candidates(ctx, rule(ctx, 'halogen_pi'))
    assert hit.rejection_reason == 'donor_angle_not_above_cutoff'


@pytest.mark.parametrize('center,normal,accepted', [((1.5, 0., 1.), (0., 0., 1.), True),
                                                   ((0., 0., 1.), (1., 0., 0.), False)])
def test_tertiary_amine_pi_compares_planes_not_centroid_approach(center, normal, accepted):
    from chemur.interactions import _cation_pi_candidates
    nitrogen = atom(0, 'N', (0., 0., 0.), bonds=(1, 2, 3), formal_charge=1,
                    metadata={'tertiary_amine_cation': True})
    roots = [atom(i + 1, 'C', (cos(i * 2*pi/3), sin(i * 2*pi/3), -.5)) for i in range(3)]
    aromatic = atom(20, 'C', center)
    ctx = context([nitrogen, *roots, aromatic], [feature(0, 'cation', [nitrogen]),
        feature(1, 'ring', [aromatic], normal=normal, metadata={'aromatic': True})])
    assert bool(_cation_pi_candidates(ctx, rule(ctx, 'cation_pi'))) == accepted
    nitrogen.metadata['tertiary_amine_cation'] = False  # An iminium is not a tertiary amine.
    assert _cation_pi_candidates(ctx, rule(ctx, 'cation_pi'))
