#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

using Vec3 = std::array<double, 3>;
constexpr double PI = 3.14159265358979323846;

struct CellKey {
    int x;
    int y;
    int z;

    bool operator==(const CellKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct CellKeyHash {
    std::size_t operator()(const CellKey& key) const {
        std::size_t hx = std::hash<int>{}(key.x);
        std::size_t hy = std::hash<int>{}(key.y);
        std::size_t hz = std::hash<int>{}(key.z);
        return hx ^ (hy << 1) ^ (hz << 2);
    }
};

double distance(const Vec3& a, const Vec3& b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double angle(const Vec3& a, const Vec3& b, const Vec3& c) {
    const Vec3 ba{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    const Vec3 bc{c[0] - b[0], c[1] - b[1], c[2] - b[2]};
    const double nba = std::sqrt(ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]);
    const double nbc = std::sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);
    if (nba == 0.0 || nbc == 0.0) {
        return 0.0;
    }
    double dot = (ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]) / (nba * nbc);
    dot = std::max(-1.0, std::min(1.0, dot));
    return std::acos(dot) * 180.0 / PI;
}

Vec3 centroid(const std::vector<Vec3>& points) {
    if (points.empty()) {
        return Vec3{0.0, 0.0, 0.0};
    }
    Vec3 center{0.0, 0.0, 0.0};
    for (const auto& point : points) {
        center[0] += point[0];
        center[1] += point[1];
        center[2] += point[2];
    }
    center[0] /= static_cast<double>(points.size());
    center[1] /= static_cast<double>(points.size());
    center[2] /= static_cast<double>(points.size());
    return center;
}

Vec3 three_point_normal(const std::vector<Vec3>& points) {
    if (points.size() < 3) {
        return Vec3{0.0, 0.0, 1.0};
    }
    const Vec3& p0 = points[0];
    const Vec3& p1 = points[1];
    const Vec3& p2 = points[2];
    const Vec3 u{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    const Vec3 v{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    Vec3 n{
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    };
    const double norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (norm == 0.0) {
        return Vec3{0.0, 0.0, 1.0};
    }
    n[0] /= norm;
    n[1] /= norm;
    n[2] /= norm;
    return n;
}

// Smallest eigenvalue of the symmetric 3x3 matrix (xx, xy, xz, yy, yz, zz),
// via the closed-form trigonometric solution of the characteristic cubic.
double smallest_eigenvalue(double xx, double xy, double xz, double yy, double yz, double zz) {
    const double off_diagonal = xy * xy + xz * xz + yz * yz;
    if (off_diagonal == 0.0) {
        return std::min(xx, std::min(yy, zz));
    }

    const double q = (xx + yy + zz) / 3.0;
    const double p2 = (xx - q) * (xx - q) + (yy - q) * (yy - q) + (zz - q) * (zz - q) +
                      2.0 * off_diagonal;
    const double p = std::sqrt(p2 / 6.0);
    if (p == 0.0) {
        return q;
    }

    const double b00 = (xx - q) / p;
    const double b11 = (yy - q) / p;
    const double b22 = (zz - q) / p;
    const double b01 = xy / p;
    const double b02 = xz / p;
    const double b12 = yz / p;
    const double determinant = b00 * (b11 * b22 - b12 * b12) -
                               b01 * (b01 * b22 - b12 * b02) +
                               b02 * (b01 * b12 - b11 * b02);
    const double r = std::max(-1.0, std::min(1.0, determinant / 2.0));
    const double phi = std::acos(r) / 3.0;
    return q + 2.0 * p * std::cos(phi + 2.0 * PI / 3.0);
}

// Best-fit plane through `points` as (unit normal, RMS deviation). The normal is
// the eigenvector of the smallest covariance eigenvalue, so it does not depend on
// which atoms come first -- unlike a three-point cross product. The RMS deviation
// doubles as the planarity of the point set.
std::tuple<Vec3, double> plane_fit(const std::vector<Vec3>& points) {
    if (points.size() < 3) {
        return std::make_tuple(Vec3{0.0, 0.0, 1.0}, 0.0);
    }

    const Vec3 center = centroid(points);
    double xx = 0.0, xy = 0.0, xz = 0.0, yy = 0.0, yz = 0.0, zz = 0.0;
    for (const auto& point : points) {
        const double dx = point[0] - center[0];
        const double dy = point[1] - center[1];
        const double dz = point[2] - center[2];
        xx += dx * dx;
        xy += dx * dy;
        xz += dx * dz;
        yy += dy * dy;
        yz += dy * dz;
        zz += dz * dz;
    }

    const double smallest = smallest_eigenvalue(xx, xy, xz, yy, yz, zz);
    const std::array<Vec3, 3> rows{
        Vec3{xx - smallest, xy, xz},
        Vec3{xy, yy - smallest, yz},
        Vec3{xz, yz, zz - smallest},
    };

    Vec3 best{0.0, 0.0, 0.0};
    double best_norm = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = i + 1; j < 3; ++j) {
            const Vec3& a = rows[i];
            const Vec3& b = rows[j];
            const Vec3 candidate{
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            };
            const double norm = std::sqrt(candidate[0] * candidate[0] +
                                          candidate[1] * candidate[1] +
                                          candidate[2] * candidate[2]);
            if (norm > best_norm) {
                best_norm = norm;
                best = candidate;
            }
        }
    }

    if (best_norm < 1e-12) {
        // Collinear or coincident points: the plane is undefined.
        return std::make_tuple(three_point_normal(points), 0.0);
    }

    Vec3 unit{best[0] / best_norm, best[1] / best_norm, best[2] / best_norm};
    // The eigenvector sign is arbitrary; pin it so the stored normal is
    // reproducible across atom orderings.
    for (std::size_t i = 0; i < 3; ++i) {
        if (std::fabs(unit[i]) > 1e-9) {
            if (unit[i] < 0.0) {
                unit[0] = -unit[0];
                unit[1] = -unit[1];
                unit[2] = -unit[2];
            }
            break;
        }
    }

    const double rmsd = std::sqrt(std::max(0.0, smallest) / static_cast<double>(points.size()));
    return std::make_tuple(unit, rmsd);
}

Vec3 plane_normal(const std::vector<Vec3>& points) {
    return std::get<0>(plane_fit(points));
}

double point_plane_offset(const Vec3& point, const Vec3& center, const Vec3& normal) {
    const double norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (norm == 0.0) {
        return distance(point, center);
    }
    const Vec3 unit{normal[0] / norm, normal[1] / norm, normal[2] / norm};
    const Vec3 cp{point[0] - center[0], point[1] - center[1], point[2] - center[2]};
    const double signed_dist = cp[0] * unit[0] + cp[1] * unit[1] + cp[2] * unit[2];
    const Vec3 projected{
        point[0] - signed_dist * unit[0],
        point[1] - signed_dist * unit[1],
        point[2] - signed_dist * unit[2],
    };
    return distance(projected, center);
}

CellKey cell_key(const Vec3& coord, double cell_size) {
    return CellKey{
        static_cast<int>(std::floor(coord[0] / cell_size)),
        static_cast<int>(std::floor(coord[1] / cell_size)),
        static_cast<int>(std::floor(coord[2] / cell_size)),
    };
}

std::vector<std::tuple<int, int, double>> neighbor_pairs(const std::vector<Vec3>& coords, double cutoff) {
    std::vector<std::tuple<int, int, double>> pairs;
    if (cutoff <= 0.0) {
        return pairs;
    }

    std::unordered_map<CellKey, std::vector<int>, CellKeyHash> cells;
    for (int i = 0; i < static_cast<int>(coords.size()); ++i) {
        cells[cell_key(coords[i], cutoff)].push_back(i);
    }

    const double cutoff2 = cutoff * cutoff;
    for (const auto& item : cells) {
        const CellKey key = item.first;
        const auto& indices = item.second;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    const CellKey neighbor{key.x + dx, key.y + dy, key.z + dz};
                    auto found = cells.find(neighbor);
                    if (found == cells.end()) {
                        continue;
                    }
                    for (int i : indices) {
                        for (int j : found->second) {
                            if (j <= i) {
                                continue;
                            }
                            const double dxp = coords[i][0] - coords[j][0];
                            const double dyp = coords[i][1] - coords[j][1];
                            const double dzp = coords[i][2] - coords[j][2];
                            const double d2 = dxp * dxp + dyp * dyp + dzp * dzp;
                            if (d2 <= cutoff2) {
                                pairs.emplace_back(i, j, std::sqrt(d2));
                            }
                        }
                    }
                }
            }
        }
    }
    return pairs;
}

bool is_occluded(
    const Vec3& start,
    const Vec3& end,
    const std::vector<Vec3>& points,
    const std::unordered_set<int>& ignore_indices,
    double radius
) {
    const Vec3 ab{end[0] - start[0], end[1] - start[1], end[2] - start[2]};
    const double ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if (ab2 == 0.0) {
        return false;
    }
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        if (ignore_indices.find(i) != ignore_indices.end()) {
            continue;
        }
        const Vec3 ap{points[i][0] - start[0], points[i][1] - start[1], points[i][2] - start[2]};
        const double t = (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab2;
        if (t <= 0.0 || t >= 1.0) {
            continue;
        }
        const Vec3 closest{
            start[0] + t * ab[0],
            start[1] + t * ab[1],
            start[2] + t * ab[2],
        };
        if (distance(points[i], closest) <= radius) {
            return true;
        }
    }
    return false;
}

bool is_occluded_py(
    const Vec3& start,
    const Vec3& end,
    const std::vector<Vec3>& points,
    py::object ignore_indices,
    double radius
) {
    std::unordered_set<int> ignore;
    if (!ignore_indices.is_none()) {
        for (auto item : ignore_indices) {
            ignore.insert(py::cast<int>(item));
        }
    }
    return is_occluded(start, end, points, ignore, radius);
}

PYBIND11_MODULE(_chemeleonx_core, m) {
    m.doc() = "C++ geometry and neighbor-search core for standalone ChemeleonX";
    m.def("distance", &distance, py::arg("a"), py::arg("b"));
    m.def("angle", &angle, py::arg("a"), py::arg("b"), py::arg("c"));
    m.def("centroid", &centroid, py::arg("points"));
    m.def("plane_normal", &plane_normal, py::arg("points"));
    m.def("plane_fit", &plane_fit, py::arg("points"));
    m.def("point_plane_offset", &point_plane_offset, py::arg("point"), py::arg("center"), py::arg("normal"));
    m.def("neighbor_pairs", &neighbor_pairs, py::arg("coords"), py::arg("cutoff"));
    m.def(
        "is_occluded",
        &is_occluded_py,
        py::arg("start"),
        py::arg("end"),
        py::arg("points"),
        py::arg("ignore_indices") = py::none(),
        py::arg("radius") = 1.0
    );
}
