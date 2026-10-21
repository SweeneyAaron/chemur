from __future__ import annotations

from collections import defaultdict
from math import acos, degrees, isfinite, sqrt
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

    Repeated smallest covariance eigenvalues describe several equally good
    planes, not necessarily a planar point set. Select a reproducible normal
    from that eigenspace and measure the deviation from the returned plane.
    Keep the algorithm and tolerances in sync with cpp/chemur_core.cpp.
    """
    pts = list(points)
    if len(pts) < 3:
        return (0.0, 0.0, 1.0), 0.0
    if all(point[i] == pts[0][i] for point in pts[1:] for i in range(3)):
        # Averaging repeated decimal coordinates can introduce a rounding
        # residual even though the point set has exactly zero extent.
        return (0.0, 0.0, 1.0), 0.0

    center = centroid(pts)
    centered = [tuple(point[i] - center[i] for i in range(3)) for point in pts]
    coordinate_scale = max(abs(value) for point in centered for value in point)
    if coordinate_scale == 0.0:
        return (0.0, 0.0, 1.0), 0.0
    scaled = [tuple(value / coordinate_scale for value in point) for point in centered]
    covariance = [[0.0] * 3 for _ in range(3)]
    for point in scaled:
        for i in range(3):
            for j in range(i, 3):
                covariance[i][j] += point[i] * point[j]
    matrix_scale = max(covariance[i][i] for i in range(3))
    for i in range(3):
        for j in range(i, 3):
            covariance[i][j] /= matrix_scale
            covariance[j][i] = covariance[i][j]

    eigenvalues, eigenvectors = _symmetric_eigensystem(covariance)
    normal = _minimum_eigenspace_normal(eigenvalues, eigenvectors)
    squared_deviation = 0.0
    for point in scaled:
        projection = sum(point[i] * normal[i] for i in range(3))
        squared_deviation += projection * projection
    return normal, coordinate_scale * sqrt(squared_deviation / len(pts))


def plane_normal(points: Iterable[Vec3]) -> Vec3:
    return plane_fit(points)[0]


def _symmetric_eigensystem(matrix: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """Jacobi eigenvalues and column eigenvectors of a scaled symmetric 3x3 matrix."""
    vectors = [[float(i == j) for j in range(3)] for i in range(3)]
    for _ in range(32):
        if max(abs(matrix[i][j]) for i, j in ((0, 1), (0, 2), (1, 2))) <= 1e-14:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            off_diagonal = matrix[p][q]
            if abs(off_diagonal) <= 1e-14:
                continue
            tau = (matrix[q][q] - matrix[p][p]) / (2.0 * off_diagonal)
            tangent = (1.0 if tau >= 0.0 else -1.0) / (abs(tau) + sqrt(1.0 + tau * tau))
            cosine = 1.0 / sqrt(1.0 + tangent * tangent)
            sine = tangent * cosine
            matrix[p][p] -= tangent * off_diagonal
            matrix[q][q] += tangent * off_diagonal
            matrix[p][q] = matrix[q][p] = 0.0
            for i in range(3):
                if i != p and i != q:
                    ip, iq = matrix[i][p], matrix[i][q]
                    matrix[i][p] = matrix[p][i] = cosine * ip - sine * iq
                    matrix[i][q] = matrix[q][i] = sine * ip + cosine * iq
                ip, iq = vectors[i][p], vectors[i][q]
                vectors[i][p] = cosine * ip - sine * iq
                vectors[i][q] = sine * ip + cosine * iq
    return [matrix[i][i] for i in range(3)], vectors


def _minimum_eigenspace_normal(values: list[float], vectors: list[list[float]]) -> Vec3:
    smallest = min(values)
    tolerance = 1e-12 * max(1.0, max(abs(value) for value in values))
    columns = [i for i, value in enumerate(values) if value - smallest <= tolerance]
    # Project each coordinate axis into the minimum eigenspace. The largest
    # projection is stable even when an axis is almost perpendicular to it;
    # ties prefer Z, then Y, then X, including the fully isotropic case.
    best = (0.0, 0.0, 0.0)
    best_norm2 = -1.0
    for axis in (2, 1, 0):
        projected = tuple(sum(vectors[i][j] * vectors[axis][j] for j in columns) for i in range(3))
        norm2 = sum(value * value for value in projected)
        if norm2 > best_norm2 + 1e-12:
            best = projected
            best_norm2 = norm2
    normal = tuple(value / sqrt(best_norm2) for value in best)
    for value in normal:
        if abs(value) > 1e-12:
            if value < 0.0:
                normal = tuple(-component for component in normal)
            break
    return normal  # type: ignore[return-value]


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


def is_enclosed(center: Vec3, neighbor_coords: list[Vec3]) -> bool:
    """Whether neighbors surround ``center`` strictly in three dimensions.

    A closed hemisphere containing every contact direction rules out enclosure.
    Such a hemisphere has a boundary through two independent directions, so
    testing their cross products avoids constructing a full convex hull.
    Zero-length directions do not contribute; nonfinite coordinates reject the
    geometry. Keep this algorithm and its tolerances in sync with the C++ core.
    """
    if not all(isfinite(value) for value in center):
        return False
    directions: list[Vec3] = []
    for point in neighbor_coords:
        delta = tuple(point[i] - center[i] for i in range(3))
        if not all(isfinite(value) for value in delta):
            return False
        scale = max(abs(value) for value in delta)
        if scale == 0.0:
            continue
        scaled = tuple(value / scale for value in delta)
        norm = sqrt(sum(value * value for value in scaled))
        directions.append(tuple(value / norm for value in scaled))
    if len(directions) < 4:
        return False

    tested_pair = False
    for i, first in enumerate(directions):
        for second in directions[i + 1 :]:
            normal = (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            )
            norm = sqrt(sum(value * value for value in normal))
            if norm <= 1e-12:
                continue
            tested_pair = True
            # Comparing against the scaled tolerance is equivalent to using
            # a unit normal, without another normalization of every component.
            tolerance = 1e-9 * norm
            positive = negative = False
            for direction in directions:
                projection = sum(direction[axis] * normal[axis] for axis in range(3))
                positive = positive or projection > tolerance
                negative = negative or projection < -tolerance
                if positive and negative:
                    break
            if not (positive and negative):
                return False
    return tested_pair


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
