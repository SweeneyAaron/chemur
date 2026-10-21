"""Strict spatial enclosure, including degenerate and backend compatibility cases."""

import importlib
import importlib.util
from itertools import combinations, permutations
import sys
from types import ModuleType

import numpy as np
import pytest

from chemur import _fallback_core, core


TETRAHEDRON = np.array([
    [1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0],
])
OCTAHEDRON = np.concatenate([np.eye(3), -np.eye(3)])


@pytest.fixture(params=["python", "cpp"])
def enclosure_core(request):
    if request.param == "python":
        return _fallback_core
    compiled = pytest.importorskip("chemur._chemur_core")
    if not hasattr(compiled, "is_enclosed"):
        pytest.skip("compiled extension has not been rebuilt with is_enclosed")
    return compiled


def _rotation():
    rotation, _ = np.linalg.qr(np.array([[1.0, 2.0, -1.0], [2.0, -1.0, 3.0], [3.0, 1.0, 2.0]]))
    return rotation


@pytest.mark.parametrize("neighbors", [TETRAHEDRON, OCTAHEDRON], ids=["tetrahedron", "octahedron"])
def test_three_dimensional_shell_encloses_center(enclosure_core, neighbors):
    # The octahedron catches an incorrect shortcut: no tetrahedron drawn from
    # its six vertices contains the origin strictly, but the full shell does.
    assert enclosure_core.is_enclosed((0.0, 0.0, 0.0), neighbors.tolist()) is True


@pytest.mark.parametrize("scale", [1e-120, 1.0, 1e120])
def test_enclosure_is_invariant_to_rotation_translation_scale_and_order(enclosure_core, scale):
    center = np.array([12.0, -8.0, 25.0]) * scale
    # Independent radial changes must not bias the directional enclosure test.
    neighbors = (TETRAHEDRON * np.array([1.0, 2.0, 3.0, 4.0])[:, None]) @ _rotation()
    neighbors = neighbors * scale + center
    for order in permutations(range(4)):
        assert enclosure_core.is_enclosed(center.tolist(), neighbors[list(order)].tolist()) is True


@pytest.mark.parametrize("neighbors", [
    [],
    [[0.0, 0.0, 0.0]] * 5,
    TETRAHEDRON[:3].tolist(),
    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]] * 3,
    OCTAHEDRON[[0, 1, 3, 4]].tolist(),
    OCTAHEDRON[:-1].tolist(),
    [[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0]],
    (TETRAHEDRON * [1.0, 1.0, 1e-12]).tolist(),
], ids=["empty", "coincident", "three", "opposing", "planar", "boundary", "one-sided", "near-planar"])
def test_non_enclosures_remain_rejected_after_rotation(enclosure_core, neighbors):
    assert enclosure_core.is_enclosed((0.0, 0.0, 0.0), neighbors) is False
    if neighbors:
        rotated = np.array(neighbors) @ _rotation()
        assert enclosure_core.is_enclosed((0.0, 0.0, 0.0), rotated.tolist()) is False


def test_duplicate_and_zero_length_directions_do_not_change_enclosure(enclosure_core):
    neighbors = TETRAHEDRON.tolist()
    assert enclosure_core.is_enclosed((0.0, 0.0, 0.0), neighbors + neighbors + [[0.0, 0.0, 0.0]])
    assert not enclosure_core.is_enclosed((0.0, 0.0, 0.0), neighbors[:3] + [[0.0, 0.0, 0.0]])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_center_or_neighbor_rejects_geometry(enclosure_core, value):
    neighbors = TETRAHEDRON.tolist()
    assert enclosure_core.is_enclosed((value, 0.0, 0.0), neighbors) is False
    # Invalid additional neighbors cannot silently disappear from a valid shell.
    assert enclosure_core.is_enclosed((0.0, 0.0, 0.0), neighbors + [[0.0, value, 0.0]]) is False


def _supporting_plane_oracle(points):
    """Independent convex-hull facet enumeration for small general-position sets."""
    for i, j, k in combinations(range(len(points)), 3):
        normal = np.cross(points[j] - points[i], points[k] - points[i])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        offset = points[i] @ normal
        sides = points @ normal - offset
        if np.max(sides) <= 1e-10 and offset <= 1e-10:
            return False
        if np.min(sides) >= -1e-10 and offset >= -1e-10:
            return False
    return True


def test_enclosure_agrees_with_independent_hull_facets(enclosure_core):
    rng = np.random.default_rng(1671)
    for count in (4, 6, 10):
        for _ in range(12):
            points = rng.normal(size=(count, 3))
            expected = _supporting_plane_oracle(points)
            assert enclosure_core.is_enclosed((0.0, 0.0, 0.0), points.tolist()) is expected


def test_python_and_compiled_enclosures_agree():
    compiled = pytest.importorskip("chemur._chemur_core")
    if not hasattr(compiled, "is_enclosed"):
        pytest.skip("compiled extension has not been rebuilt with is_enclosed")
    rng = np.random.default_rng(8021)
    for count in (3, 4, 8, 16):
        for _ in range(10):
            points = rng.normal(size=(count, 3))
            for neighbors in (points, points * [1.0, 1.0, 1e-12], np.abs(points)):
                coords = neighbors.tolist()
                assert compiled.is_enclosed((0.0, 0.0, 0.0), coords) == _fallback_core.is_enclosed(
                    (0.0, 0.0, 0.0), coords
                )


def _load_isolated_core():
    spec = importlib.util.spec_from_file_location("chemur._enclosure_core_test", core.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_core_supports_extensions_without_enclosure(monkeypatch):
    import chemur

    stale = ModuleType("chemur._chemur_core")
    for name in core.__all__:
        if name not in {"USING_CPP_CORE", "is_enclosed"}:
            setattr(stale, name, getattr(_fallback_core, name))
    monkeypatch.delenv("CHEMUR_FORCE_PYTHON_CORE", raising=False)
    monkeypatch.setitem(sys.modules, "chemur._chemur_core", stale)
    monkeypatch.setattr(chemur, "_chemur_core", stale, raising=False)
    isolated = _load_isolated_core()
    assert isolated.USING_CPP_CORE is True
    assert isolated.is_enclosed is _fallback_core.is_enclosed
    assert isolated.is_enclosed((0.0, 0.0, 0.0), OCTAHEDRON.tolist()) is True


def test_public_core_honors_forced_python_backend(monkeypatch):
    monkeypatch.setenv("CHEMUR_FORCE_PYTHON_CORE", "1")
    isolated = _load_isolated_core()
    assert isolated.USING_CPP_CORE is False
    assert isolated.is_enclosed is _fallback_core.is_enclosed
