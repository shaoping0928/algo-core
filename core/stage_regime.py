"""Compatibility facade re-exporting the modular ``core/stage/`` package.

All implementation now lives in focused, task-agnostic submodules under
``core/stage/``:
  - ``summarizer.py``          : drift stripping & StageRegimeSummarizer
  - ``transition.py``          : CalibratedRegimeTransitionModel, Evaluator & OnlineUpdater
  - ``failure_memory.py``      : TrunkPointFailureMemory
  - ``feature_demand.py``      : ControllableFeatureDemand & OnlineFeatureDemandUpdater
  - ``adaptive_controller.py`` : AdaptiveSearchController
"""
from __future__ import annotations

from _paths import ensure_core_paths

ensure_core_paths()

from adaptive_controller import AdaptiveSearchController
from failure_memory import TrunkPointFailureMemory
from feature_demand import ControllableFeatureDemand, OnlineFeatureDemandUpdater
from summarizer import (
    UNKNOWN,
    RegimeStage,
    StageRegimeSummarizer,
    _fit_deterministic_centroids,
    _fit_survival_tree_centroids,
    _native,
    detect_pure_drift_dims,
)
from transition import (
    CalibratedRegimeTransitionModel,
    OnlineRegimeContinuationUpdater,
    RegimeContinuationEvaluator,
    _add_scaled_pmf,
    _convolve_arrays,
    _dict_to_pmf_array,
)

__all__ = [
    "UNKNOWN",
    "RegimeStage",
    "StageRegimeSummarizer",
    "CalibratedRegimeTransitionModel",
    "RegimeContinuationEvaluator",
    "OnlineRegimeContinuationUpdater",
    "TrunkPointFailureMemory",
    "ControllableFeatureDemand",
    "OnlineFeatureDemandUpdater",
    "AdaptiveSearchController",
    "detect_pure_drift_dims",
]
