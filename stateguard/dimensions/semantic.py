"""Semantic validation — embedding cross-validation with configurable threshold.

Provides :class:`SemanticValidator` which compares the semantic similarity
between an output and a reference (or expected) embedding, flagging
results that fall below a configurable threshold.

Two independent cross-validation methods are used:
  1. Cosine similarity — via :meth:`EmbeddingModel.similarity`
  2. Normalised Euclidean distance — converted to a 0-1 score
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models.enums import ValidationDimension, ValidationTier
from ..models.result import ValidationResult
from ..plugin.base import BaseValidator
from ..utils.embedding import EmbeddingModel


class SemanticValidator(BaseValidator):
    """Validator that measures semantic similarity via embeddings.

    The validator encodes both the *output* and an optional *reference*
    into embedding vectors (using an :class:`~stateguard.utils.embedding.EmbeddingModel`)
    and computes a similarity score.  Scores below ``threshold`` cause
    the validation to fail.

    Two cross-validation methods are used (AC 3):
      1. Cosine similarity between output and reference embeddings.
      2. Normalised Euclidean distance (1 / (1 + distance)).
    The final score is the average of both methods.
    """

    name: str = "semantic"
    dimension: ValidationDimension = ValidationDimension.SEMANTIC
    tier: ValidationTier = ValidationTier.TIER_2

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        threshold: float = 0.70,
    ) -> None:
        """Initialise the semantic validator.

        Args:
            embedding_model: An ``EmbeddingModel`` instance.  ``None``
                creates a default model on first ``validate()`` call.
            threshold:       Minimum combined similarity to pass (0.0 – 1.0).
        """
        super().__init__()
        self._model = embedding_model
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Current similarity threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Update the similarity threshold."""
        self._threshold = value

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Validate *output* semantically against an optional reference.

        Args:
            output:  The produced output (text, dict, etc.).
            context: May contain ``reference`` (expected value),
                     ``threshold`` override, and ``embedding_model`` keys.

        Returns:
            A :class:`ValidationResult` with a score proportional to
            semantic similarity.
        """
        ctx = context if isinstance(context, dict) else {}
        details: dict[str, Any] = {}

        # --- Normalise output ----------------------------------------------------
        if output is None:
            output = ""

        if not isinstance(output, str):
            try:
                output = str(output)
            except Exception:
                output = ""

        if not output.strip():
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"error": "Empty output"},
            )

        # --- Resolve threshold & reference ---------------------------------------
        # Patch 1: guard non-numeric threshold types
        try:
            threshold = float(ctx.get("threshold", self._threshold))
        except (TypeError, ValueError):
            threshold = self._threshold

        reference = ctx.get("reference")

        # --- Resolve embedding model ---------------------------------------------
        model = ctx.get("embedding_model", self._model)
        if model is None:
            model = EmbeddingModel()
        # Patch 7: cache the model so subsequent calls reuse it
        if self._model is None:
            self._model = model

        # --- No reference — skip cross-validation, pass with warning -------------
        if reference is None:
            return ValidationResult(
                score=100.0,
                passed=True,
                dimension=self.dimension,
                details={
                    "warning": "No reference provided — semantic validation skipped",
                    "threshold": threshold,
                },
            )

        # --- Normalise reference to string ---------------------------------------
        # Patch 6: ensure reference is a string before encode
        if not isinstance(reference, str):
            reference = str(reference)

        # --- Encode output and reference -----------------------------------------
        try:
            out_vec = model.encode([output])
            ref_vec = model.encode([reference])
        # Patches 4+5: catch ImportError (missing sentence-transformers) and
        #              AttributeError (malformed model object) alongside ValueError
        except (ValueError, RuntimeError, ImportError, AttributeError) as e:
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"error": f"Embedding failed: {e}"},
            )

        # --- Method 1: Cosine similarity -----------------------------------------
        cos_sim_raw = model.similarity(out_vec, ref_vec)

        # similarity() returns a matrix; extract the single scalar
        cos_sim = float(cos_sim_raw[0, 0]) if cos_sim_raw.size > 0 else 0.0

        # Guard against NaN (zero-vector edge case — similarity() returns 0)
        if np.isnan(cos_sim):
            cos_sim = 0.0

        # --- Method 2: Normalised Euclidean distance -----------------------------
        diff = out_vec - ref_vec
        euclidean = float(np.linalg.norm(diff))
        euclidean_score = 1.0 / (1.0 + euclidean)  # 0.0 – 1.0

        if np.isnan(euclidean_score):
            euclidean_score = 0.0

        # --- Combined score ------------------------------------------------------
        avg = (cos_sim + euclidean_score) / 2.0
        # Patches 2+3: clamp score to [0, 100] to prevent Pydantic ge=0.0
        # violations (cosine can be negative) or overflow
        score_100 = max(0.0, min(100.0, avg * 100.0))

        passed = score_100 >= threshold * 100

        details.update({
            "cosine_similarity": cos_sim,
            "euclidean_distance": euclidean,
            "euclidean_score": euclidean_score,
            "combined_score": avg,
            "threshold": threshold,
        })

        return ValidationResult(
            score=score_100,
            passed=passed,
            dimension=self.dimension,
            details=details,
        )
