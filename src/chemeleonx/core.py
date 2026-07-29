from __future__ import annotations

import os

# CHEMELEONX_FORCE_PYTHON_CORE makes the pure-Python fallback reachable without
# breaking the build, so CI can exercise it. Otherwise it is only reached when
# the compiled extension is genuinely missing -- exactly the case nobody tests
# deliberately.
if os.environ.get("CHEMELEONX_FORCE_PYTHON_CORE"):
    from . import _fallback_core as _core
    USING_CPP_CORE = False
else:
    try:
        from . import _chemeleonx_core as _core
        USING_CPP_CORE = True
    except ImportError:
        from . import _fallback_core as _core
        USING_CPP_CORE = False

distance = _core.distance
angle = _core.angle
centroid = _core.centroid
plane_normal = _core.plane_normal
point_plane_offset = _core.point_plane_offset
neighbor_pairs = _core.neighbor_pairs
is_occluded = _core.is_occluded

__all__ = [
    "USING_CPP_CORE",
    "angle",
    "centroid",
    "distance",
    "is_occluded",
    "neighbor_pairs",
    "plane_normal",
    "point_plane_offset",
]

