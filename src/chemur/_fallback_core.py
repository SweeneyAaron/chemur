from __future__ import annotations

from collections import defaultdict
from math import acos, cos, degrees, pi, sqrt
from typing import Iterable

Vec3 = tuple[float, float, float]


def distance(a: Vec3, b: Vec3) -> float:
    return sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def angle(a: Vec3, b: Vec3, c: Vec3) -> float:
    ba = tuple(a[i] - b[i] for i in range(3))
    bc = tuple(c[i] - b[i] for i in range(3))
    nba = sqrt(sum(v * v for v in ba))
    nbc = sqrt(sum(v * v for v in bc))
    if nba == 0.0 or nbc == 0.0:
        return 0.0
    dot = sum(ba[i] * bc[i] for i in range(3)) / (nba * nbc)
    dot = max(-1.0, min(1.0, dot))
    return degrees(acos(dot))


def centroid(points: Iterable[Vec3]) -> Vec3:
    pts = list(points)
    if not pts:
        return (0.0, 0.0, 0.0)
    return tuple(sum(point[i] for point in pts) / len(pts) for i in range(3))  # type: ignore[return-value]


def plane_fit(points: Iterable[Vec3]) -> tuple[Vec3, float]:
    """Best-fit plane through ``points``, as ``(unit normal, RMS deviation)``.

    The normal is the eigenvector of the smallest eigenvalue of the covariance
    matrix. Unlike a cross product of the first three points it therefore does
    not depend on which atoms happen to come first, which matters for puckered
    saturated rings and for the fused 9-atom purine feature -- in neither case
    do three atoms define the mean plane.

    The returned RMS deviation is the planarity of the point set: ~0 for an
    aromatic ring, distinctly non-zero for a chair cyclohexane.
    """
    pts = list(points)
    if len(pts) < 3:
        return (0.0, 0.0, 1.0), 0.0

    center = centroid(pts)
    xx = xy = xz = yy = yz = zz = 0.0
    for point in pts:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        dz = point[2] - center[2]
        xx += dx * dx
        xy += dx * dy
        xz += dx * dz
        yy += dy * dy
        yz += dy * dz
        zz += dz * dz

    covariance = (xx, xy, xz, yy, yz, zz)
    smallest = _smallest_eigenvalue(covariance)
    normal = _null_space_vector(covariance, smallest)
    if normal is None:
        # Collinear or coincident points: the plane is undefined, so fall back
        # to the three-point normal rather than returning an arbitrary axis.
        return _three_point_normal(pts), 0.0
    return normal, sqrt(max(0.0, smallest) / len(pts))


def plane_normal(points: Iterable[Vec3]) -> Vec3:
    return plane_fit(points)[0]


def _three_point_normal(pts: list[Vec3]) -> Vec3:
    if len(pts) < 3:
        return (0.0, 0.0, 1.0)
    p0, p1, p2 = pts[0], pts[1], pts[2]
    u = tuple(p1[i] - p0[i] for i in range(3))
    v = tuple(p2[i] - p0[i] for i in range(3))
    n = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    norm = sqrt(sum(x * x for x in n))
    if norm == 0.0:
        return (0.0, 0.0, 1.0)
    return tuple(x / norm for x in n)  # type: ignore[return-value]


def _smallest_eigenvalue(covariance: tuple[float, ...]) -> float:
    """Smallest eigenvalue of the symmetric 3x3 matrix (xx, xy, xz, yy, yz, zz).

    Closed-form trigonometric solution of the characteristic cubic -- no
    iteration, and no numpy dependency in the fallback path.
    """
    xx, xy, xz, yy, yz, zz = covariance
    off_diagonal = xy * xy + xz * xz + yz * yz
    if off_diagonal == 0.0:
        return min(xx, yy, zz)

    q = (xx + yy + zz) / 3.0
    p2 = (xx - q) ** 2 + (yy - q) ** 2 + (zz - q) ** 2 + 2.0 * off_diagonal
    p = sqrt(p2 / 6.0)
    if p == 0.0:
        return q

    b00 = (xx - q) / p
    b11 = (yy - q) / p
    b22 = (zz - q) / p
    b01 = xy / p
    b02 = xz / p
    b12 = yz / p
    determinant = (
        b00 * (b11 * b22 - b12 * b12)
        - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02)
    )
    r = max(-1.0, min(1.0, determinant / 2.0))
    phi = acos(r) / 3.0
    return q + 2.0 * p * cos(phi + 2.0 * pi / 3.0)


def _null_space_vector(covariance: tuple[float, ...], eigenvalue: float) -> Vec3 | None:
    """Unit eigenvector of ``covariance`` for ``eigenvalue``, or None if degenerate."""
    xx, xy, xz, yy, yz, zz = covariance
    rows = (
        (xx - eigenvalue, xy, xz),
        (xy, yy - eigenvalue, yz),
        (xz, yz, zz - eigenvalue),
    )
    best: Vec3 | None = None
    best_norm = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = rows[i], rows[j]
            candidate = (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )
            norm = sqrt(sum(v * v for v in candidate))
            if norm > best_norm:
                best_norm = norm
                best = candidate  # type: ignore[assignment]
    if best is None or best_norm < 1e-12:
        return None

    unit = tuple(v / best_norm for v in best)
    # The eigenvector sign is arbitrary. Every consumer folds it with abs(), but
    # pinning it keeps the stored normal reproducible across atom orderings.
    for value in unit:
        if abs(value) > 1e-9:
            if value < 0.0:
                unit = tuple(-v for v in unit)
            break
    return unit  # type: ignore[return-value]


def point_plane_offset(point: Vec3, center: Vec3, normal: Vec3) -> float:
    n_norm = sqrt(sum(x * x for x in normal))
    if n_norm == 0.0:
        return distance(point, center)
    unit = tuple(x / n_norm for x in normal)
    cp = tuple(point[i] - center[i] for i in range(3))
    signed = sum(cp[i] * unit[i] for i in range(3))
    projected = tuple(point[i] - signed * unit[i] for i in range(3))
    return distance(projected, center)


def neighbor_pairs(coords: list[Vec3], cutoff: float) -> list[tuple[int, int, float]]:
    if cutoff <= 0.0:
        return []
    cell_size = cutoff
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, coord in enumerate(coords):
        cells[_cell_key(coord, cell_size)].append(idx)

    pairs: list[tuple[int, int, float]] = []
    cutoff2 = cutoff * cutoff
    for key, indices in cells.items():
        for neighbor_key in _neighbor_keys(key):
            if neighbor_key not in cells:
                continue
            for i in indices:
                for j in cells[neighbor_key]:
                    if j <= i:
                        continue
                    d2 = sum((coords[i][axis] - coords[j][axis]) ** 2 for axis in range(3))
                    if d2 <= cutoff2:
                        pairs.append((i, j, sqrt(d2)))
    return pairs


def is_occluded(
    start: Vec3,
    end: Vec3,
    points: list[Vec3],
    ignore_indices: set[int] | frozenset[int] | None = None,
    radius: float = 1.0,
) -> bool:
    ignore = ignore_indices or set()
    ab = tuple(end[i] - start[i] for i in range(3))
    ab2 = sum(v * v for v in ab)
    if ab2 == 0.0:
        return False
    for idx, point in enumerate(points):
        if idx in ignore:
            continue
        ap = tuple(point[i] - start[i] for i in range(3))
        t = sum(ap[i] * ab[i] for i in range(3)) / ab2
        if t <= 0.0 or t >= 1.0:
            continue
        closest = tuple(start[i] + t * ab[i] for i in range(3))
        if distance(point, closest) <= radius:
            return True
    return False


def _cell_key(coord: Vec3, cell_size: float) -> tuple[int, int, int]:
    return (
        int(coord[0] // cell_size),
        int(coord[1] // cell_size),
        int(coord[2] // cell_size),
    )


def _neighbor_keys(key: tuple[int, int, int]):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (key[0] + dx, key[1] + dy, key[2] + dz)

