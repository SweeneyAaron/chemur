from importlib.metadata import PackageNotFoundError, version as _version

from .api import analyze, analyze_batch
from .models import (
    AnalysisResult,
    AssignedInteraction,
    AtomRecord,
    CandidateInteraction,
    ComponentRecord,
    FeatureRecord,
    ResourceDemand,
)
from .scoring import build_scoring_batch, score_pose_batch, score_poses

try:
    __version__ = _version("chemur")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "analyze",
    "analyze_batch",
    "score_poses",
    "score_pose_batch",
    "build_scoring_batch",
    "AnalysisResult",
    "AssignedInteraction",
    "AtomRecord",
    "CandidateInteraction",
    "ComponentRecord",
    "FeatureRecord",
    "ResourceDemand",
]
