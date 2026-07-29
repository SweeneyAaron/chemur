from chemeleonx import core


def test_distance_angle_and_neighbor_pairs():
    assert core.distance((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == 1.0
    assert round(core.angle((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), 6) == 90.0

    pairs = core.neighbor_pairs(
        [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (3.0, 0.0, 0.0)],
        1.0,
    )
    assert pairs == [(0, 1, 0.5)]


def test_occlusion():
    assert core.is_occluded(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        [(0.0, 0.0, 0.0), (1.0, 0.2, 0.0), (2.0, 0.0, 0.0)],
        {0, 2},
        0.5,
    )

