"""Tier 2 validation — Ensemble-based consensus checking.

Provides :class:`EnsembleValidator` as the second tier in the cascade
pipeline.  Uses three independent anomaly-detection methods
(Isolation Forest, One-Class SVM, Z-Score) and aggregates their
judgments via 2/3 majority voting.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stateguard.config.settings import ConfigManager
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.plugin.base import BaseValidator


# ── Individual Analyzers ─────────────────────────────────────────────


class ZScoreAnalyzer:
    """Z-Score based anomaly detector.

    Computes the absolute z-score for each feature and flags a sample
    as anomalous if **any** feature exceeds *threshold* standard
    deviations from the training mean.

    Attributes:
        threshold: Z-score threshold above which a value is anomalous.
    """

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        """Compute mean and standard deviation from training *data*.

        Args:
            data: Training array of shape ``(n_samples, n_features)``.
        """
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        # Guard against zero std (constant features)
        self.std_[self.std_ == 0] = 1.0

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict whether each sample is an anomaly.

        Returns:
            1 for normal, -1 for anomaly (sklearn convention).
        """
        z_scores = np.abs((data - self.mean_) / self.std_)
        max_z = np.max(z_scores, axis=1)
        return np.where(max_z > self.threshold, -1, 1).astype(np.int32)

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Return a score where positive means normal, negative anomaly.

        Uses negative max z-score as the anomaly score so that
        higher values mean more normal (like sklearn conventions).
        """
        z_scores = np.abs((data - self.mean_) / self.std_)
        max_z = np.max(z_scores, axis=1)
        # Positive score = normal, negative = anomaly
        return np.where(max_z > self.threshold, -max_z, self.threshold - max_z)


class IsolationForestAnalyzer:
    """Wrapper around ``sklearn.ensemble.IsolationForest``.

    Attributes:
        contamination: Expected proportion of outliers.
        random_state:  Seed for reproducibility.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sklearn.ensemble import IsolationForest

            self._model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
            )
        return self._model

    def fit(self, data: np.ndarray) -> None:
        """Fit the Isolation Forest model on *data*."""
        model = self._ensure_model()
        model.fit(data)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomaly labels for *data*.

        Returns:
            1 for normal, -1 for anomaly.
        """
        model = self._ensure_model()
        return model.predict(data)

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more normal)."""
        model = self._ensure_model()
        return model.decision_function(data)


class SVMAnomalyAnalyzer:
    """Wrapper around ``sklearn.svm.OneClassSVM``.

    Attributes:
        nu:     An upper bound on the fraction of training errors.
        kernel: Kernel type to use.
        gamma:  Kernel coefficient.
    """

    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "auto",
    ) -> None:
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sklearn.svm import OneClassSVM

            self._model = OneClassSVM(
                nu=self.nu,
                kernel=self.kernel,
                gamma=self.gamma,
            )
        return self._model

    def fit(self, data: np.ndarray) -> None:
        """Fit the One-Class SVM model on *data*."""
        model = self._ensure_model()
        model.fit(data)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomaly labels for *data*.

        Returns:
            1 for normal, -1 for anomaly.
        """
        model = self._ensure_model()
        return model.predict(data)

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more normal)."""
        model = self._ensure_model()
        return model.decision_function(data)


# ── Ensemble Validator ───────────────────────────────────────────────


class EnsembleValidator(BaseValidator):
    """Validator that runs multiple sub-validators and aggregates results.

    Tier 2 of the cascade validation pipeline.  Uses three independent
    anomaly-detection methods and applies 2/3 majority voting to
    produce a final verdict.

    Expects ``output`` as a dict with a ``"features"`` key containing
    a list (or list-of-lists) of numeric values, representing the
    feature vector to validate.

    Attributes:
        name:      ``\"ensemble-anomaly\"``
        dimension: :attr:`ValidationDimension.SEMANTIC`
        tier:      :attr:`ValidationTier.TIER_2`
    """

    name: str = "ensemble-anomaly"
    dimension: ValidationDimension = ValidationDimension.SEMANTIC
    tier: ValidationTier = ValidationTier.TIER_2

    def __init__(
        self,
        contamination: float | None = None,
        svm_nu: float | None = None,
        z_score_threshold: float | None = None,
    ) -> None:
        """Initialise the ensemble validator.

        Args:
            contamination:     Override IsolationForest contamination.
            svm_nu:            Override OneClassSVM nu.
            z_score_threshold: Override ZScoreAnalyzer threshold.
        """
        super().__init__()

        # Use explicit overrides when provided; fall back to defaults
        self._contamination = contamination if contamination is not None else 0.1
        self._svm_nu = svm_nu if svm_nu is not None else 0.1
        self._z_score_threshold = (
            z_score_threshold if z_score_threshold is not None else 3.0
        )

        # Try to read from config for config-managed overrides
        try:
            cfg = ConfigManager()
            config = cfg.load()
            if hasattr(config, "tier2_threshold"):
                self._tier2_threshold = config.tier2_threshold
            else:
                self._tier2_threshold = 50.0
        except Exception:
            self._tier2_threshold = 50.0

        # Lazy initialized — analyzers are created on first validate() call
        self._analyzers: list[Any] | None = None

    def _ensure_analyzers(self) -> list[Any]:
        """Lazy-initialize all three anomaly detectors on first call."""
        if self._analyzers is not None:
            return self._analyzers

        self._analyzers = [
            ("isolation_forest", IsolationForestAnalyzer(
                contamination=self._contamination,
                random_state=42,
            )),
            ("one_class_svm", SVMAnomalyAnalyzer(
                nu=self._svm_nu,
                kernel="rbf",
                gamma="auto",
            )),
            ("z_score", ZScoreAnalyzer(
                threshold=self._z_score_threshold,
            )),
        ]
        return self._analyzers

    def _extract_features(self, output: Any) -> np.ndarray | None:
        """Extract a 2-D feature array from *output*.

        Returns ``None`` if the output cannot be interpreted as
        numeric features.
        """
        if output is None:
            return None

        if isinstance(output, dict):
            features = output.get("features")
            if features is None:
                return None
            arr = np.asarray(features, dtype=np.float64)
        elif isinstance(output, (list, tuple)):
            arr = np.asarray(output, dtype=np.float64)
        elif isinstance(output, np.ndarray):
            arr = output.astype(np.float64)
        else:
            return None

        # Ensure 2-D shape
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim == 0:
            return None

        if arr.shape[1] == 0 or arr.shape[0] == 0:
            return None

        return arr

    def validate(
        self,
        output: Any,
        context: dict | None = None,
    ) -> ValidationResult:
        """Run ensemble validation on the given *output*.

        Args:
            output:  A dict with ``\"features\"`` key containing a list
                     of numeric values, or a list/ndarray directly.
            context: Optional context (currently unused by this tier).

        Returns:
            A :class:`ValidationResult` with the ensemble verdict.
        """
        # --- Input validation ---
        features = self._extract_features(output)
        if features is None:
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"error": "Output must contain numeric features."},
                error="Missing or invalid features. Expected dict with 'features' key containing numeric data.",
            )

        # --- Lazy init analyzers ---
        analyzers = self._ensure_analyzers()

        # --- Fit each analyzer on this data batch if not fitted ---
        method_results: dict[str, dict[str, Any]] = {}
        normal_count = 0
        anomaly_count = 0

        for name, analyzer in analyzers:
            # Fit on first use
            analyzer.fit(features)

            # Predict
            predictions = analyzer.predict(features)
            # Take the majority prediction across all samples
            n_normal = int(np.sum(predictions == 1))
            n_anomaly = int(np.sum(predictions == -1))
            is_anomaly = n_anomaly > n_normal

            # Score
            scores = analyzer.score_samples(features)
            avg_score = float(np.mean(scores))

            if is_anomaly:
                anomaly_count += 1
            else:
                normal_count += 1

            params: dict[str, Any] = {}
            if name == "isolation_forest":
                params = {"contamination": self._contamination}
            elif name == "one_class_svm":
                params = {"nu": self._svm_nu}
            elif name == "z_score":
                params = {"threshold": self._z_score_threshold}

            method_results[name] = {
                "is_anomaly": is_anomaly,
                "score": round(avg_score, 4),
                "normal_ratio": round(n_normal / (n_normal + n_anomaly), 4),
                **params,
            }

        # --- 2/3 Majority voting ---
        passed = normal_count >= 2
        ensemble_score = round((normal_count / len(analyzers)) * 100.0, 2)

        return ValidationResult(
            score=ensemble_score,
            passed=passed,
            dimension=self.dimension,
            details={
                **method_results,
                "ensemble": {
                    "vote": "passed" if passed else "failed",
                    "normal_count": normal_count,
                    "anomaly_count": anomaly_count,
                    "total_methods": len(analyzers),
                },
            },
            error=None,
        )
