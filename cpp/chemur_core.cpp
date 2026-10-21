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

using Matrix3 = std::array<Vec3, 3>;

// Jacobi eigenvalues and column eigenvectors of a scaled symmetric 3x3 matrix.
// Keep the algorithm and tolerances in sync with _fallback_core.py.
std::tuple<Vec3, Matrix3> symmetric_eigensystem(Matrix3 matrix) {
    Matrix3 vectors{Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0}, Vec3{0.0, 0.0, 1.0}};
    constexpr std::array<std::array<int, 2>, 3> pairs{{{0, 1}, {0, 2}, {1, 2}}};
    for (int sweep = 0; sweep < 32; ++sweep) {
        if (std::max({std::fabs(matrix[0][1]), std::fabs(matrix[0][2]),
                      std::fabs(matrix[1][2])}) <= 1e-14) {
            break;
        }
        for (const auto& pair : pairs) {
            const int p = pair[0];
            const int q = pair[1];
            const double off_diagonal = matrix[p][q];
            if (std::fabs(off_diagonal) <= 1e-14) {
                continue;
            }
            const double tau = (matrix[q][q] - matrix[p][p]) / (2.0 * off_diagonal);
            const double tangent = (tau >= 0.0 ? 1.0 : -1.0) /
                                   (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
            const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
            const double sine = tangent * cosine;
            matrix[p][p] -= tangent * off_diagonal;
            matrix[q][q] += tangent * off_diagonal;
            matrix[p][q] = matrix[q][p] = 0.0;
            for (int i = 0; i < 3; ++i) {
                if (i != p && i != q) {
                    const double ip = matrix[i][p];
                    const double iq = matrix[i][q];
                    matrix[i][p] = matrix[p][i] = cosine * ip - sine * iq;
                    matrix[i][q] = matrix[q][i] = sine * ip + cosine * iq;
                }
                const double ip = vectors[i][p];
                const double iq = vectors[i][q];
                vectors[i][p] = cosine * ip - sine * iq;
                vectors[i][q] = sine * ip + cosine * iq;
            }
        }
    }
    return std::make_tuple(Vec3{matrix[0][0], matrix[1][1], matrix[2][2]}, vectors);
}

Vec3 minimum_eigenspace_normal(const Vec3& values, const Matrix3& vectors) {
    const double smallest = std::min({values[0], values[1], values[2]});
    const double tolerance = 1e-12 *
        std::max({1.0, std::fabs(values[0]), std::fabs(values[1]), std::fabs(values[2])});
    Vec3 best{0.0, 0.0, 0.0};
    double best_norm2 = -1.0;
    // Project coordinate axes into the minimum eigenspace, selecting the
    // largest projection. Ties prefer Z, then Y, then X.
    for (int axis : {2, 1, 0}) {
        Vec3 projected{0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (values[j] - smallest <= tolerance) {
                    projected[i] += vectors[i][j] * vectors[axis][j];
                }
            }
        }
        const double norm2 = projected[0] * projected[0] + projected[1] * projected[1] +
                             projected[2] * projected[2];
        if (norm2 > best_norm2 + 1e-12) {
            best = projected;
            best_norm2 = norm2;
        }
    }
    const double norm = std::sqrt(best_norm2);
    Vec3 normal{best[0] / norm, best[1] / norm, best[2] / norm};
    for (double value : normal) {
        if (std::fabs(value) > 1e-12) {
            if (value < 0.0) {
                for (double& component : normal) {
                    component = -component;
                }
            }
            break;
        }
    }
    return normal;
}

// Best-fit plane as (unit normal, RMS deviation). Repeated minimum eigenvalues
// define several equally good planes, not necessarily a planar point set.
std::tuple<Vec3, double> plane_fit(const std::vector<Vec3>& points) {
    if (points.size() < 3) {
        return std::make_tuple(Vec3{0.0, 0.0, 1.0}, 0.0);
    }
    // Averaging repeated decimal coordinates can otherwise introduce a
    // rounding residual despite the point set having exactly zero extent.
    if (std::all_of(points.begin() + 1, points.end(),
                    [&points](const Vec3& point) { return point == points.front(); })) {
        return std::make_tuple(Vec3{0.0, 0.0, 1.0}, 0.0);
    }

    const Vec3 center = centroid(points);
    std::vector<Vec3> scaled;
    scaled.reserve(points.size());
    double coordinate_scale = 0.0;
    for (const auto& point : points) {
        const Vec3 delta{point[0] - center[0], point[1] - center[1], point[2] - center[2]};
        scaled.push_back(delta);
        for (double value : delta) {
            coordinate_scale = std::max(coordinate_scale, std::fabs(value));
        }
    }
    if (coordinate_scale == 0.0) {
        return std::make_tuple(Vec3{0.0, 0.0, 1.0}, 0.0);
    }
    Matrix3 covariance{};
    for (auto& point : scaled) {
        for (double& value : point) {
            value /= coordinate_scale;
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                covariance[i][j] += point[i] * point[j];
            }
        }
    }
    const double matrix_scale = std::max({covariance[0][0], covariance[1][1], covariance[2][2]});
    for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
            covariance[i][j] /= matrix_scale;
            covariance[j][i] = covariance[i][j];
        }
    }
    const auto eigensystem = symmetric_eigensystem(covariance);
    const Vec3 normal = minimum_eigenspace_normal(std::get<0>(eigensystem), std::get<1>(eigensystem));
    double squared_deviation = 0.0;
    for (const auto& point : scaled) {
        const double projection = point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2];
        squared_deviation += projection * projection;
    }
    const double rmsd = coordinate_scale *
                       std::sqrt(squared_deviation / static_cast<double>(points.size()));
    return std::make_tuple(normal, rmsd);
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

// Strict 3D enclosure is equivalent to the absence of a closed hemisphere
// containing all contact directions. A separating hemisphere has a boundary
// through two independent directions, whose cross product gives its normal.
// Keep this algorithm and its tolerances in sync with _fallback_core.py.
bool is_enclosed(const Vec3& center, const std::vector<Vec3>& neighbor_coords) {
    for (double value : center) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    std::vector<Vec3> directions;
    directions.reserve(neighbor_coords.size());
    for (const auto& point : neighbor_coords) {
        Vec3 delta{point[0] - center[0], point[1] - center[1], point[2] - center[2]};
        double scale = 0.0;
        for (double value : delta) {
            if (!std::isfinite(value)) {
                return false;
            }
            scale = std::max(scale, std::fabs(value));
        }
        if (scale == 0.0) {
            continue;
        }
        for (double& value : delta) {
            value /= scale;
        }
        const double norm = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                                      delta[2] * delta[2]);
        for (double& value : delta) {
            value /= norm;
        }
        directions.push_back(delta);
    }
    if (directions.size() < 4) {
        return false;
    }

    bool tested_pair = false;
    for (std::size_t i = 0; i < directions.size(); ++i) {
        for (std::size_t j = i + 1; j < directions.size(); ++j) {
            const Vec3& first = directions[i];
            const Vec3& second = directions[j];
            const Vec3 normal{
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            };
            const double norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                                          normal[2] * normal[2]);
            if (norm <= 1e-12) {
                continue;
            }
            tested_pair = true;
            const double tolerance = 1e-9 * norm;
            bool positive = false;
            bool negative = false;
            for (const auto& direction : directions) {
                const double projection = direction[0] * normal[0] +
                                          direction[1] * normal[1] + direction[2] * normal[2];
                positive = positive || projection > tolerance;
                negative = negative || projection < -tolerance;
                if (positive && negative) {
                    break;
                }
            }
            if (!(positive && negative)) {
                return false;
            }
        }
    }
    return tested_pair;
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

// pybind11 casts std::array to a Python *list*, but _fallback_core returns tuples.
// The two cores have to be interchangeable -- these coordinates end up on every
// FeatureRecord (features.py), so a list would make results built by the C++ core
// compare unequal to the same results built by the Python core, and would raise on
// any attempt to hash them. Convert at the binding boundary and leave the C++
// signatures natural.
static py::tuple vec3_to_tuple(const Vec3& v) {
    return py::make_tuple(v[0], v[1], v[2]);
}

PYBIND11_MODULE(_chemur_core, m) {
    m.doc() = "C++ geometry and neighbor-search core for standalone Chemur";
    m.def("distance", &distance, py::arg("a"), py::arg("b"));
    m.def("angle", &angle, py::arg("a"), py::arg("b"), py::arg("c"));
    m.def(
        "centroid",
        [](const std::vector<Vec3>& points) { return vec3_to_tuple(centroid(points)); },
        py::arg("points")
    );
    m.def(
        "plane_normal",
        [](const std::vector<Vec3>& points) { return vec3_to_tuple(plane_normal(points)); },
        py::arg("points")
    );
    m.def(
        "plane_fit",
        [](const std::vector<Vec3>& points) {
            const auto fit = plane_fit(points);
            return py::make_tuple(vec3_to_tuple(std::get<0>(fit)), std::get<1>(fit));
        },
        py::arg("points")
    );
    m.def("point_plane_offset", &point_plane_offset, py::arg("point"), py::arg("center"), py::arg("normal"));
    m.def("neighbor_pairs", &neighbor_pairs, py::arg("coords"), py::arg("cutoff"));
    m.def("is_enclosed", &is_enclosed, py::arg("center"), py::arg("neighbor_coords"));
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
