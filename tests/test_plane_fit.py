"""Independent numerical checks for both implementations of the plane fit."""

import importlib
from itertools import permutations

import numpy as np
import pytest


@pytest.fixture(params=["python", "cpp"])
def plane_core(request):
    if request.param == "python":
        return importlib.import_module("chemur._fallback_core")
    return pytest.importorskip("chemur._chemur_core")


def _rotation():
    # An arbitrary rotation mixes every covariance entry, unlike axis rotations.
    q, _ = np.linalg.qr(np.array([[1.0, 2.0, -1.0], [2.0, -1.0, 3.0], [3.0, 1.0, 2.0]]))
    return q


def _projection_rms(points, normal):
    centered = points - points.mean(axis=0)
    return np.linalg.norm(centered @ normal) / np.sqrt(len(points))


@pytest.mark.parametrize("scale", [1e-120, 1e-6, 1.0, 1e6, 1e120])
def test_tetrahedron_is_not_planar_after_transforms_and_permutations(plane_core, scale):
    tetrahedron = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                            [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    points = (tetrahedron @ _rotation() + [100.0, -200.0, 300.0]) * scale
    for order in permutations(range(4)):
        normal, rmsd = plane_core.plane_fit(points[list(order)].tolist())
        # Every normal is equally good for an isotropic point cloud. Our stable
        # tie convention chooses Z and must retain the nonzero deviation.
        assert normal == pytest.approx((0.0, 0.0, 1.0), abs=2e-12)
        assert rmsd / scale == pytest.approx(1.0, rel=2e-12)


@pytest.mark.parametrize("rotate", [False, True])
def test_repeated_nonzero_minimum_eigenvalue_has_a_valid_plane(plane_core, rotate):
    # Covariance eigenvalues are 1, 1, 9. The choice within the first two
    # eigendirections is free; the RMS deviation is nevertheless exactly 1.
    points = np.concatenate([np.eye(3), -np.eye(3)]) * np.sqrt(3.0) * [1.0, 1.0, 3.0]
    if rotate:
        points = points @ _rotation()
    points += [12.0, -9.0, 2.0]
    reference, rmsd = plane_core.plane_fit(points.tolist())
    assert np.linalg.norm(reference) == pytest.approx(1.0)
    assert rmsd == pytest.approx(1.0, abs=2e-12)
    assert _projection_rms(points, reference) == pytest.approx(rmsd, abs=2e-12)
    if not rotate:
        assert reference == pytest.approx((0.0, 1.0, 0.0), abs=2e-12)
    for order in ([5, 1, 3, 0, 4, 2], [3, 4, 5, 0, 1, 2], [2, 1, 0, 5, 4, 3]):
        normal, reordered_rmsd = plane_core.plane_fit(points[order].tolist())
        assert normal == pytest.approx(reference, abs=2e-12)
        assert reordered_rmsd == pytest.approx(rmsd, abs=2e-12)


@pytest.mark.parametrize("direction", [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)])
def test_collinear_points_use_a_normal_perpendicular_to_the_line(plane_core, direction):
    direction = np.array(direction)
    points = np.arange(-2.0, 3.0)[:, None] * direction + [7.0, 8.0, 9.0]
    normal, rmsd = plane_core.plane_fit(points.tolist())
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert np.dot(normal, direction) == pytest.approx(0.0, abs=2e-12)
    assert rmsd == pytest.approx(0.0, abs=2e-12)
    assert rmsd == pytest.approx(_projection_rms(points, normal), abs=2e-12)
    if np.all(direction == 1.0):
        # All projected axes have equal length. Z wins, then canonical sign
        # makes the first nonzero normal component positive.
        assert normal == pytest.approx(np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0), abs=2e-12)


def test_coincident_and_small_point_sets_keep_the_documented_default(plane_core):
    for points in ([], [(4.0, -2.0, 8.0)], [(1.0, 2.0, 3.0), (8.0, 9.0, 10.0)],
                   [(4.0, -2.0, 8.0)] * 5, [(0.1, 0.2, 0.3)] * 7):
        assert plane_core.plane_fit(points) == ((0.0, 0.0, 1.0), 0.0)


@pytest.mark.parametrize("scale", [1e-120, 1.0, 1e120])
def test_tilted_planar_fit_handles_coordinate_scale(plane_core, scale):
    points = np.array([[x, y, 2.0 * x - 3.0 * y] for x in (-2.0, 0.0, 2.0)
                       for y in (-1.0, 1.0)]) * scale
    normal, rmsd = plane_core.plane_fit(points.tolist())
    assert normal == pytest.approx(np.array([2.0, -3.0, -1.0]) / np.sqrt(14.0), abs=2e-12)
    assert rmsd / scale == pytest.approx(0.0, abs=2e-12)


def test_fit_matches_numpy_eigenvalue_oracle(plane_core):
    rng = np.random.default_rng(7031)
    for axes in ([1.0, 1.0, 1.0], [1.0, 2.0, 0.01], [0.01, 8.0, 0.3]):
        for _ in range(10):
            points = (rng.normal(size=(17, 3)) * axes) @ _rotation() + [9.0, -3.0, 15.0]
            centered = points - points.mean(axis=0)
            values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
            normal, rmsd = plane_core.plane_fit(points.tolist())
            assert np.linalg.norm(normal) == pytest.approx(1.0, abs=2e-12)
            assert abs(np.dot(normal, vectors[:, 0])) == pytest.approx(1.0, abs=2e-12)
            assert rmsd == pytest.approx(np.sqrt(max(0.0, values[0])), rel=2e-9, abs=2e-12)
            assert rmsd == pytest.approx(_projection_rms(points, normal), abs=2e-12)


def test_python_and_compiled_plane_fits_agree():
    python_core = importlib.import_module("chemur._fallback_core")
    cpp_core = pytest.importorskip("chemur._chemur_core")
    rng = np.random.default_rng(441)
    datasets = [
        np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]),
        np.arange(-2.0, 3.0)[:, None] * np.array([1.0, 1.0, 1.0]),
        np.concatenate([np.eye(3), -np.eye(3)]) * [1.0, 1.0, 3.0],
        np.zeros((4, 3)),
        rng.normal(size=(31, 3)) * [0.01, 0.5, 2.0],
    ]
    for dataset in datasets:
        for scale in (1e-120, 1.0, 1e120):
            points = ((dataset @ _rotation() + [2.0, 3.0, -4.0]) * scale).tolist()
            python_normal, python_rmsd = python_core.plane_fit(points)
            cpp_normal, cpp_rmsd = cpp_core.plane_fit(points)
            assert cpp_normal == pytest.approx(python_normal, abs=2e-12)
            assert cpp_rmsd / scale == pytest.approx(python_rmsd / scale, abs=2e-12)
