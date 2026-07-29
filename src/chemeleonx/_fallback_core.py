from __future__ import annotations

from collections import defaultdict
from math import acos, degrees, sqrt
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


def plane_normal(points: Iterable[Vec3]) -> Vec3:
    pts = list(points)
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

