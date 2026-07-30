from math import cos, pi, sin

from chemur import core


def _ring(radius, pucker):
    """A planar ring when pucker is 0, a chair when it is not."""
    return [
        (radius * cos(pi * i / 3), radius * sin(pi * i / 3), pucker if i % 2 else -pucker)
        for i in range(6)
    ]


def test_plane_fit_reports_planarity():
    _, flat = core.plane_fit(_ring(1.39, 0.0))
    _, puckered = core.plane_fit(_ring(1.46, 0.25))

    assert flat == 0.0
    assert puckered == 0.25


def test_plane_fit_normal_does_not_depend_on_atom_order():
    """A three-point cross product does; a puckered ring is exactly where it breaks."""
    chair = _ring(1.46, 0.25)
    reference, _ = core.plane_fit(chair)

    for rotation in range(1, 6):
        rotated = chair[rotation:] + chair[:rotation]
        normal, _ = core.plane_fit(rotated)
        assert max(abs(a - b) for a, b in zip(reference, normal)) < 1e-9


def test_plane_fit_survives_degenerate_input():
    for points in ([], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]):
        assert core.plane_fit(points) == ((0.0, 0.0, 1.0), 0.0)

    collinear = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    normal, rmsd = core.plane_fit(collinear)
    assert rmsd == 0.0
    assert abs(sum(v * v for v in normal) - 1.0) < 1e-9


def test_plane_normal_delegates_to_plane_fit():
    chair = _ring(1.46, 0.25)

    assert core.plane_normal(chair) == core.plane_fit(chair)[0]


def test_distance_angle_and_neighbor_pairs():
    assert core.distance((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == 1.0
    assert round(core.angle((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), 6) == 90.0

    pairs = core.neighbor_pairs(
        [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (3.0, 0.0, 0.0)],
        1.0,
    )
    assert pairs == [(0, 1, 0.5)]


def test_coordinate_returns_are_hashable_tuples():
    """The C++ and pure-Python cores must return the same types, not just the same numbers.

    pybind11 casts std::array to a list by default, which silently made the C++ core
    return lists where _fallback_core returns tuples. These values land on every
    FeatureRecord, so the mismatch made otherwise-identical results compare unequal
    across the two cores -- and unhashable under the C++ one.
    """
    ring = _ring(1.39, 0.0)

    for value in (core.centroid(ring), core.plane_normal(ring), core.plane_fit(ring)[0]):
        assert isinstance(value, tuple), f"expected tuple, got {type(value).__name__}"
        assert len(value) == 3
        hash(value)


def test_occlusion():
    assert core.is_occluded(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        [(0.0, 0.0, 0.0), (1.0, 0.2, 0.0), (2.0, 0.0, 0.0)],
        {0, 2},
        0.5,
    )

