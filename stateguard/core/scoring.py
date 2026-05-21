"""Scoring — ScoreCard for computing weighted multi-dimension scores.

Provides :class:`ScoreCard` that aggregates per-dimension validation
scores into a composite ``EngineResult`` using configurable weights
and threshold logic.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from stateguard.config.settings import ConfigManager
from stateguard.models.enums import ValidationDimension
from stateguard.models.result import EngineResult

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS: dict[str, float] = {
    "structural": 0.25,
    "semantic": 0.25,
    "quantitative": 0.15,
    "behavioral": 0.20,
    "security": 0.15,
}

# Mapping from model enum value → weight key
_DIM_TO_KEY: dict[ValidationDimension, str] = {
    ValidationDimension.STRUCTURAL: "structural",
    ValidationDimension.SEMANTIC: "semantic",
    ValidationDimension.QUANTITATIVE: "quantitative",
    ValidationDimension.BEHAVIORAL: "behavioral",
    ValidationDimension.SECURITY: "security",
}


class ScoreCard:
    """Calculates and aggregates scores across multiple dimensions.

    Uses a weighted average formula defined in the project architecture:
    ``Total = (Structural × 0.25) + (Semantic × 0.25) + (Quantitative × 0.15)
             + (Behavioral × 0.20) + (Security × 0.15)``

    Weights and thresholds are configurable via constructor overrides
    or the project YAML configuration.

    Attributes:
        name: ``\"score-card\"``
    """

    name: str = "score-card"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initialise the ScoreCard with optional custom *weights*.

        Args:
            weights: Override dictionary mapping dimension keys
                     (``\"structural\"``, ``\"semantic\"``, …) to their
                     weight value.  Unspecified keys fall back to
                     defaults.  If ``None``, all weights come from
                     config (or built-in defaults).

        Raises:
            ValueError: If any custom weight is negative.
        """
        self._weights = dict(DEFAULT_WEIGHTS)

        # Merge config overrides
        try:
            cfg = ConfigManager()
            config = cfg.load()
            if hasattr(config, "scoring") and isinstance(config.scoring, dict):
                cfg_weights = config.scoring.get("weights")
                if isinstance(cfg_weights, dict):
                    self._weights.update(cfg_weights)
        except Exception:
            logger.warning("Failed to load scoring config from file, using defaults.")

        # Constructor override wins over everything
        if weights is not None:
            for k, v in weights.items():
                if v < 0:
                    raise ValueError(
                        f"Weight for '{k}' must be non-negative, got {v}"
                    )
            self._weights.update(weights)

    @property
    def weights(self) -> dict[str, float]:
        """Current weights (read-only snapshot)."""
        return dict(self._weights)

    def calculate(
        self,
        dimension_scores: dict[ValidationDimension, float],
    ) -> EngineResult:
        """Aggregate per-dimension scores into a composite ``EngineResult``.

        Args:
            dimension_scores: Mapping of dimension to a 0-100 score.

        Returns:
            An ``EngineResult`` with the weighted average, pass/fail
            decision, and detailed scoring metadata.
        """
        # --- Empty input guard ---
        if not dimension_scores:
            return EngineResult(
                overall_score=0.0,
                passed=False,
                dimension_scores={},
                details={
                    "scoring": {
                        "error": "No dimension scores provided.",
                        "threshold": {"pass": 75.0, "borderline": 50.0},
                    },
                },
            )

        # --- Separate valid keyed scores from unrecognised dimensions ---
        valid_scores: dict[ValidationDimension, float] = {}
        for dim, raw in dimension_scores.items():
            if dim not in _DIM_TO_KEY:
                continue
            # Guard against non-numeric and NaN/Inf
            try:
                score = float(raw)
            except (ValueError, TypeError):
                return EngineResult(
                    overall_score=0.0,
                    passed=False,
                    dimension_scores={},
                    details={
                        "scoring": {
                            "error": f"Non-numeric score for {dim}: {raw!r}.",
                            "threshold": {"pass": 75.0, "borderline": 50.0},
                        },
                    },
                )
            if math.isnan(score) or math.isinf(score):
                return EngineResult(
                    overall_score=0.0,
                    passed=False,
                    dimension_scores={},
                    details={
                        "scoring": {
                            "error": f"Invalid score value ({score}) for {dim}.",
                            "threshold": {"pass": 75.0, "borderline": 50.0},
                        },
                    },
                )
            # Clamp individual score to [0, 100]
            valid_scores[dim] = max(0.0, min(100.0, score))

        if not valid_scores:
            return EngineResult(
                overall_score=0.0,
                passed=False,
                dimension_scores={},
                details={
                    "scoring": {
                        "error": "No recognised dimension scores.",
                        "threshold": {"pass": 75.0, "borderline": 50.0},
                    },
                },
            )

        # --- Identify present vs missing dimensions ---
        present_keys = {_DIM_TO_KEY[d] for d in valid_scores}
        all_keys = set(DEFAULT_WEIGHTS.keys())
        missing_keys = all_keys - present_keys

        # --- Renormalisation ---
        used_weights: dict[str, float] = {}
        total_weight = sum(self._weights[k] for k in present_keys)

        if total_weight <= 0:
            logger.warning(
                "Total weight for present dimensions is %s, defaulting to 0.0 score.",
                total_weight,
            )
            for k in present_keys:
                used_weights[k] = 0.0
        else:
            for k in present_keys:
                used_weights[k] = self._weights[k] / total_weight

        # --- Weighted score ---
        weighted_sum = 0.0
        weighted_contributions: dict[str, float] = {}
        for dim, raw_score in valid_scores.items():
            key = _DIM_TO_KEY[dim]
            contrib = raw_score * used_weights[key]
            weighted_sum += contrib
            weighted_contributions[key] = round(contrib, 4)

        overall = round(weighted_sum, 2)

        # --- Threshold logic ---
        pass_threshold = 75.0
        borderline_threshold = 50.0
        try:
            cfg = ConfigManager()
            config = cfg.load()
            if hasattr(config, "scoring") and isinstance(config.scoring, dict):
                thresholds = config.scoring.get("thresholds", {})
                if isinstance(thresholds, dict):
                    pass_threshold = float(thresholds.get("pass", pass_threshold))
                    borderline_threshold = float(
                        thresholds.get("borderline", borderline_threshold)
                    )
        except Exception:
            logger.warning("Failed to load thresholds from config, using defaults.")

        passed = overall >= pass_threshold
        borderline = borderline_threshold <= overall < pass_threshold
        stop_pipeline = overall < borderline_threshold

        # --- Build EngineResult ---
        normalization_info: dict[str, Any] = {
            "applied": len(missing_keys) > 0,
        }
        if missing_keys:
            normalization_info["missing_dimensions"] = sorted(missing_keys)
            normalization_info["original_weights"] = {
                k: self._weights[k] for k in sorted(missing_keys | present_keys)
            }
            normalization_info["used_weights"] = used_weights

        return EngineResult(
            overall_score=overall,
            passed=passed,
            dimension_scores={k.name.lower(): v for k, v in valid_scores.items()},
            details={
                "scoring": {
                    "threshold": {
                        "pass": pass_threshold,
                        "borderline": borderline_threshold,
                    },
                    "weights_used": dict(self._weights),
                    "normalized_weights": used_weights,
                    "weighted_contributions": weighted_contributions,
                    "normalization": normalization_info,
                    "borderline": borderline,
                    "stop_pipeline": stop_pipeline,
                },
            },
        )
