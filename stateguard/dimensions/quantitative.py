"""Quantitative validation — Isolation Forest & Z-Score outlier detection.

Provides :class:`QuantitativeValidator` that treats validation as an
anomaly-detection problem.  Supports both unsupervised (Isolation Forest)
and statistical (Z-Score) methods to flag outputs that deviate from
expected distributions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models.enums import ValidationDimension, ValidationTier
from ..models.result import ValidationResult
from ..plugin.base import BaseValidator

# Optional scikit-learn dependency
try:
    from sklearn.ensemble import IsolationForest

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False


class QuantitativeValidator(BaseValidator):
    """Validator that flags outliers using statistical or ML methods.

    Two detection modes are available:

        * ``"zscore"`` — Classic Z-Score thresholding.  Values with
          |Z| > *z_threshold* are flagged.
        * ``"isolation_forest"`` — Unsupervised anomaly detection via
          :class:`sklearn.ensemble.IsolationForest`.
        * ``"both"`` — Average of both methods.

    The validator can be fitted on a reference dataset at initialisation
    time or incrementally over time.
    """

    name: str = "quantitative"
    dimension: ValidationDimension = ValidationDimension.QUANTITATIVE
    tier: ValidationTier = ValidationTier.TIER_2

    VALID_METHODS = ("zscore", "isolation_forest", "both")

    def __init__(
        self,
        method: str = "zscore",
        z_threshold: float = 3.0,
        contamination: float = 0.1,
    ) -> None:
        """Initialise the quantitative validator.

        Args:
            method:        Detection mode — ``"zscore"``, ``"isolation_forest"``,
                           or ``"both"``.
            z_threshold:   Z-Score cutoff (only used when *method* includes
                           ``"zscore"``). Must be > 0.
            contamination: Expected proportion of outliers in the data
                           (only used when *method* includes
                           ``"isolation_forest"``). Must be in (0, 0.5].

        Raises:
            ValueError: If *method* is not one of VALID_METHODS,
                        *z_threshold* ≤ 0, or *contamination* not in (0, 0.5].
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method {method!r}. Must be one of {self.VALID_METHODS}"
            )
        if z_threshold <= 0:
            raise ValueError(f"z_threshold must be > 0, got {z_threshold}")
        if not (0 < contamination <= 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5], got {contamination}"
            )
        super().__init__()
        self._method = method
        self._z_threshold = z_threshold
        self._contamination = contamination
        self._isolation_forest: IsolationForest | None = None
        self._if_fitted: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        """Current detection method."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        if value not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method {value!r}. Must be one of {self.VALID_METHODS}"
            )
        self._method = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Validate *output* by checking whether it is an outlier.

        Args:
            output:  Numeric value, list of values, or numpy array.
            context: May contain ``values``, ``method``, ``z_threshold``,
                     ``contamination``, ``reference_data`` overrides.

        Returns:
            A :class:`ValidationResult` where low scores indicate anomalies.
        """
        ctx = context if isinstance(context, dict) else {}
        details: dict[str, Any] = {}

        # --- Resolve method & parameters ------------------------------------
        method = ctx.get("method", self._method)
        if method not in self.VALID_METHODS:
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"error": f"Unknown method {method!r}"},
            )

        z_threshold = self._safe_float(ctx.get("z_threshold", self._z_threshold), 3.0)
        if z_threshold <= 0:
            z_threshold = self._z_threshold

        contamination = self._safe_float(
            ctx.get("contamination", self._contamination), 0.1
        )

        # --- Parse numeric values --------------------------------------------
        values = self._parse_values(output, ctx)
        if values is None:
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"error": "No numeric values provided"},
            )

        # --- Run selected detection method(s) --------------------------------
        score = 100.0

        if method in ("zscore", "both"):
            z_score, z_details = self._zscore_validate(values, z_threshold)
            if z_score is not None:
                score = z_score
                details.update(z_details)
            else:
                # Z-Score hesaplanamadı (std=0, tek değer) — warning ile geç
                details["warning"] = z_details.get("warning", "")

        if method in ("isolation_forest", "both"):
            if not HAS_SKLEARN and method == "isolation_forest":
                return ValidationResult(
                    score=0.0,
                    passed=False,
                    dimension=self.dimension,
                    details={
                        "error": (
                            "scikit-learn is not installed. "
                            "Install with: poetry add scikit-learn"
                        ),
                    },
                )

            if not HAS_SKLEARN and method == "both":
                # sklearn yoksa sadece Z-Score sonucu kullanılır
                pass
            else:
                reference_data = ctx.get("reference_data")
                ifo_score, if_details = self._isolation_forest_validate(
                    values, reference_data, contamination
                )
                if ifo_score is not None:
                    details.update(if_details)
                    if method == "both":
                        score = (score + ifo_score) / 2.0
                    else:
                        score = ifo_score
                else:
                    details["warning"] = if_details.get("warning", "")

        # --- Final scoring ----------------------------------------------------
        score = max(0.0, min(100.0, score))
        passed = score >= 50.0

        return ValidationResult(
            score=score,
            passed=passed,
            dimension=self.dimension,
            details=details,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_values(self, output: Any, ctx: dict) -> np.ndarray | None:
        """Extract numeric values from *output* or *ctx*.

        Priority:
            1. ``ctx["values"]`` — explicit list
            2. ``output`` — raw value(s)

        Returns ``None`` if no valid numeric values found.
        """
        raw = ctx.get("values", output)

        if raw is None:
            return None

        # Single numeric value → wrap in list
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return np.array([float(raw)], dtype=np.float64)

        # Already a numpy array
        if isinstance(raw, np.ndarray):
            if raw.ndim == 0:
                return np.array([float(raw)], dtype=np.float64)
            # Flatten multi-dim arrays
            flat = raw.ravel()
            numeric = flat[np.isfinite(flat)]
            if len(numeric) == 0:
                return None
            return numeric.astype(np.float64)

        # String → try single float (must be before iterable check)
        if isinstance(raw, str):
            try:
                return np.array([float(raw)], dtype=np.float64)
            except (ValueError, TypeError):
                return None

        # List or iterable
        if hasattr(raw, "__iter__"):
            try:
                arr = np.array(
                    [
                        float(v)
                        for v in raw
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    ],
                    dtype=np.float64,
                )
            except (TypeError, ValueError):
                return None
            # Filter NaN/Inf values (same as ndarray branch)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                return None
            return arr

        return None

    def _zscore_validate(
        self, values: np.ndarray, z_threshold: float
    ) -> tuple[float | None, dict[str, Any]]:
        """Compute Z-Score based validation.

        Returns ``(score, details)``.  If Z-Score cannot be computed
        (std=0, single value) returns ``(None, {warning})``.
        """
        details: dict[str, Any] = {}

        if z_threshold <= 0:
            return None, {"warning": "Z-Score threshold must be > 0"}

        if len(values) < 2:
            return None, {"warning": "Single value — Z-Score not computed"}

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))

        if std == 0.0:
            return None, {"warning": "Zero standard deviation — Z-Score not computed"}

        z_scores = (values - mean) / std
        max_z = float(np.max(np.abs(z_scores)))

        details["max_z_score"] = max_z
        details["mean"] = mean
        details["std"] = std
        details["z_scores"] = z_scores.tolist()
        details["outlier_count"] = int(np.sum(np.abs(z_scores) > z_threshold))

        score = max(0.0, 100.0 - (max_z / z_threshold) * 100.0)
        return score, details

    def _isolation_forest_validate(
        self,
        values: np.ndarray,
        reference_data: np.ndarray | None,
        contamination: float,
    ) -> tuple[float | None, dict[str, Any]]:
        """Compute Isolation Forest based validation.

        Returns ``(score, details)``.  If no reference data is available
        and model is not yet fitted, returns ``(None, {warning})``.
        """
        details: dict[str, Any] = {}

        # Fit or reuse the model
        if reference_data is not None and not self._if_fitted:
            try:
                ref = np.asarray(reference_data, dtype=np.float64)
                if ref.ndim == 1:
                    ref = ref.reshape(-1, 1)
                self._isolation_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_jobs=1,
                )
                self._isolation_forest.fit(ref)
                self._if_fitted = True
            except Exception as e:
                return None, {"warning": f"Isolation Forest fit failed: {e}"}

        if self._isolation_forest is None or not self._if_fitted:
            return None, {
                "warning": "Isolation Forest not fitted — provide reference_data",
            }

        # Predict anomaly
        try:
            X = values.reshape(-1, 1) if values.ndim == 1 else values
            preds = self._isolation_forest.predict(X)
            scores = self._isolation_forest.decision_function(X)

            # -1 = anomaly, 1 = normal (sklearn convention)
            n_anomalies = int(np.sum(preds == -1))
            details["n_anomalies"] = n_anomalies
            details["total_values"] = len(values)

            # Convert decision_function output (negative = anomaly) to 0-100 score
            # decision_function: higher = more normal, lower = more anomalous
            min_score = float(np.min(scores))
            max_score = float(np.max(scores))
            details["if_min_score"] = min_score
            details["if_max_score"] = max_score

            if n_anomalies == 0:
                score = 100.0
            elif n_anomalies >= len(values):
                score = 0.0
            else:
                # Scale: proportion of normal values
                normal_ratio = 1.0 - (n_anomalies / len(values))
                score = normal_ratio * 100.0

            return score, details

        except Exception as e:
            return None, {"warning": f"Isolation Forest prediction failed: {e}"}

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        """Safely convert a value to float, returning *default* on failure."""
        if isinstance(value, bool):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
