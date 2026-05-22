"""Tier 2 validation — Ensemble-based consensus checking.

Provides :class:`EnsembleValidator` as the second tier in the cascade
pipeline.  Uses three independent anomaly-detection methods
(Isolation Forest, One-Class SVM, Z-Score) and aggregates their
judgments via 2/3 majority voting.

Usage::

    # Recommended: fit on reference data, then validate
    v = EnsembleValidator()
    v.fit(reference_features)
    result = v.validate(new_output)

    # Legacy: validate() auto-fits with a warning
    v = EnsembleValidator()
    result = v.validate(new_output)  # warns about auto-fit

    # Persistence
    v.save("/path/to/model.joblib")
    v2 = EnsembleValidator.load("/path/to/model.joblib")
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stateguard.config.settings import ConfigManager
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.plugin.base import BaseValidator

# Version tag embedded in saved models for compatibility checks.
_MODEL_VERSION = "1.0.0"


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

    **Usage:**

    1. Call :meth:`fit` on reference (normal) data to train all
       internal anomaly detectors.
    2. Call :meth:`validate` on new data — only prediction runs,
       no fitting (no data leakage).
    3. Persist with :meth:`save` and reload with :meth:`load`.

    For backward compatibility, :meth:`validate` auto-fits if
    :meth:`fit` was never called, but emits a warning.

    Attributes:
        name:      ``"ensemble-anomaly"``
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

        # Lazy initialized — analyzers are created on first validate() or fit() call
        self._analyzers: list[Any] | None = None
        self._lock = threading.Lock()

        # Training state
        self._is_fitted: bool = False
        self._fitted_at: str | None = None
        self._fit_version: str | None = None

    # ── Public API ───────────────────────────────────────────────────

    def fit(self, reference_data: Any) -> None:
        """Fit all internal anomaly detectors on reference (normal) data.

        After calling this method, :meth:`validate` will only predict
        — no further fitting occurs, eliminating data leakage.

        Args:
            reference_data: Reference data to train on.  Accepted
                formats are the same as :meth:`validate`:
                ``dict`` with ``"features"`` key, ``list`` of lists,
                or ``np.ndarray``.

        Raises:
            ValueError: If *reference_data* cannot be parsed as
                        numeric features.
        """
        features = self._extract_features(reference_data)
        if features is None:
            raise ValueError(
                "reference_data must contain numeric features. "
                "Expected dict with 'features' key, list of lists, or ndarray."
            )

        analyzers = self._ensure_analyzers()
        for _, analyzer in analyzers:
            analyzer.fit(features)

        self._is_fitted = True
        self._fitted_at = datetime.now(timezone.utc).isoformat()
        self._fit_version = _MODEL_VERSION

    def save(self, path: str | Path) -> None:
        """Persist the validator's state to disk.

        Uses ``joblib`` (bundled with scikit-learn) for serialisation.
        The saved file includes all trained analyzers, config
        parameters, and fitting metadata.

        Args:
            path: Destination file path.

        Raises:
            ValueError: If the path is empty or cannot be written.
        """
        import joblib as _joblib

        path = Path(path)
        if path.is_dir():
            raise ValueError(f"Path must be a file, not a directory: {path}")

        # Collect state for serialisation.
        state = {
            "_model_version": _MODEL_VERSION,
            "_contamination": self._contamination,
            "_svm_nu": self._svm_nu,
            "_z_score_threshold": self._z_score_threshold,
            "_tier2_threshold": self._tier2_threshold,
            "_is_fitted": self._is_fitted,
            "_fitted_at": self._fitted_at,
            "_fit_version": self._fit_version,
            "_analyzers": self._analyzers,
        }

        # Atomic write: temp file + rename to prevent partial writes.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.tmp_",
            delete=False,
            suffix=".joblib",
        )
        try:
            tmp.close()  # release fd before joblib writes by path
            _joblib.dump(state, tmp.name)
            os.replace(tmp.name, str(path))
        except Exception:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            raise

    @classmethod
    def load(cls, path: str | Path) -> EnsembleValidator:
        """Load a previously saved validator from disk.

        Uses ``joblib.load`` (backed by pickle) for deserialisation.
        **Security:** Only load ``.joblib`` files from trusted sources.
        Untrusted files can execute arbitrary code during deserialisation.

        Args:
            path: Path to a ``.joblib`` file created by :meth:`save`.

        Returns:
            A new :class:`EnsembleValidator` instance with the saved
            state restored (analyzers, config, fitting metadata).

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the saved model version is incompatible.
        """
        import joblib as _joblib

        logger = logging.getLogger(__name__)
        logger.info("Loading model from %s ...", path)

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = _joblib.load(str(path))

        # Version compatibility check.
        saved_version = state.get("_model_version", "0.1.0")
        if saved_version != _MODEL_VERSION:
            raise ValueError(
                f"Saved model version {saved_version!r} is incompatible with "
                f"current version {_MODEL_VERSION!r}. Please retrain."
            )

        # Reconstruct validator with saved config.
        v = cls(
            contamination=state["_contamination"],
            svm_nu=state["_svm_nu"],
            z_score_threshold=state["_z_score_threshold"],
        )
        v._tier2_threshold = state["_tier2_threshold"]
        v._is_fitted = state["_is_fitted"]
        v._fitted_at = state["_fitted_at"]
        v._fit_version = state["_fit_version"]
        v._analyzers = state["_analyzers"]

        return v

    def setup(self) -> None:
        """Lifecycle hook: load pre-trained model from config if set.

        Checks ``ConfigManager`` for ``tier2_model_path`` and loads
        the model if the path exists.  Logs a warning on failure
        rather than failing silently.
        """
        logger = logging.getLogger(__name__)
        try:
            cfg = ConfigManager()
            config = cfg.load()
            model_path = getattr(config, "tier2_model_path", None)
            if model_path:
                resolved = Path(model_path).expanduser().resolve()
                if resolved.exists():
                    loaded = self.__class__.load(str(resolved))
                    self.__dict__.update(loaded.__dict__)
                else:
                    logger.warning("tier2_model_path not found: %s", resolved)
        except FileNotFoundError:
            logger.info("No config file found — skipping setup model load.")
        except Exception as exc:
            logger.warning("setup() model load failed: %s", exc)

    # ── Internal helpers ─────────────────────────────────────────────

    def _ensure_analyzers(self) -> list[Any]:
        """Lazy-initialize all three anomaly detectors on first call (thread-safe)."""
        if self._analyzers is not None:
            return self._analyzers

        with self._lock:
            # Double-check locking pattern
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

        If :meth:`fit` was not called before the first ``validate()``,
        the model auto-fits on the input data (with a warning).  This
        preserves backward compatibility but introduces data leakage —
        call :meth:`fit` explicitly for production use.

        Args:
            output:  A dict with ``"features"`` key containing a list
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

        # --- Auto-fit if not fitted yet (backward compat) ---
        auto_fitted = False
        if not self._is_fitted:
            warnings.warn(
                f"{self.__class__.__name__} is not fitted. "
                "Auto-fitting on validation data — this introduces data leakage. "
                "Call .fit(reference_data) before .validate() for proper use.",
                UserWarning,
                stacklevel=2,
            )
            for _, analyzer in analyzers:
                analyzer.fit(features)
            self._is_fitted = True
            self._fitted_at = datetime.now(timezone.utc).isoformat()
            self._fit_version = _MODEL_VERSION
            auto_fitted = True

        # --- Predict (no fitting) ---
        method_results: dict[str, dict[str, Any]] = {}
        normal_count = 0
        anomaly_count = 0

        for name, analyzer in analyzers:
            predictions = analyzer.predict(features)
            n_normal = int(np.sum(predictions == 1))
            n_anomaly = int(np.sum(predictions == -1))
            is_anomaly = n_anomaly > n_normal

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

        details: dict[str, Any] = {
            **method_results,
            "ensemble": {
                "vote": "passed" if passed else "failed",
                "normal_count": normal_count,
                "anomaly_count": anomaly_count,
                "total_methods": len(analyzers),
            },
        }
        if auto_fitted:
            details["auto_fitted"] = True

        return ValidationResult(
            score=ensemble_score,
            passed=passed,
            dimension=self.dimension,
            details=details,
            error=None,
        )
