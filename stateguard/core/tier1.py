"""Tier 1 validation — Embedding-based similarity checking.

Provides :class:`EmbeddingValidator` as the first tier in the cascade
pipeline.  Uses cosine similarity between the output embedding and an
expected embedding to produce a score (0-100).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stateguard.config.settings import ConfigManager
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.plugin.base import BaseValidator
from stateguard.utils.embedding import EmbeddingModel


class EmbeddingValidator(BaseValidator):
    """Validator that uses embedding similarity to check outputs.

    Tier 1 of the cascade validation pipeline.  Fast semantic check
    using sentence-transformers embedding cosine similarity.

    Threshold logic (from config):
        * ``score >= tier1_threshold``  → ``passed=True``
        * ``score < tier1_threshold - 30`` → ``passed=False``
        * otherwise                     → ``passed=False`` (Tier 2 fallback)

    Attributes:
        name:      ``"embedding-similarity"``
        dimension: :attr:`ValidationDimension.SEMANTIC`
        tier:      :attr:`ValidationTier.TIER_1`
    """

    name: str = "embedding-similarity"
    dimension: ValidationDimension = ValidationDimension.SEMANTIC
    tier: ValidationTier = ValidationTier.TIER_1

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
    ) -> None:
        """Initialise the embedding validator.

        Args:
            model_name: Override the default embedding model name.
            device:     Override the device (``"cpu"`` / ``"cuda"``).
            threshold:  Override the Tier 1 pass threshold.
        """
        super().__init__()

        # Use explicit overrides when provided; fall back to config file
        if model_name is not None and device is not None and threshold is not None:
            self._model_name = model_name
            self._device = device
            self._threshold = threshold
        else:
            cfg = ConfigManager()
            config = cfg.load()
            self._model_name = model_name or config.default_embedding_model
            self._device = device or config.embedding_device
            self._threshold = threshold or config.tier1_threshold

        self._model = EmbeddingModel(
            model_name=self._model_name,
            device=self._device,
        )

    @property
    def threshold(self) -> float:
        """Current Tier 1 pass threshold."""
        return self._threshold

    def validate(
        self,
        output: str,
        context: dict | None = None,
    ) -> ValidationResult:
        """Run embedding-based validation on *output*.

        Encodes the output text and an optional expected embedding
        (from *context*), computes cosine similarity, and maps the
        result to a 0-100 score.

        Args:
            output:  The LLM output text to validate.
            context: Optional dict.  If it contains an ``"expected_output"``
                     key, that text is used as the reference.  Otherwise
                     the validator treats the output as self-referential
                     (compares output to itself).

        Returns:
            A :class:`ValidationResult` with the similarity score.
        """
        if not output or not isinstance(output, str):
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"error": "Output must be a non-empty string."},
                error="Empty or invalid output.",
            )

        # Determine reference text
        reference = output
        if context and isinstance(context, dict):
            expected = context.get("expected_output")
            if expected and isinstance(expected, str):
                reference = expected

        # Encode both texts
        try:
            output_vec = self._model.encode([output])  # (1, dim)
            reference_vec = self._model.encode([reference])  # (1, dim)
        except Exception as e:
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"error": f"Encoding failed: {e}"},
                error=str(e),
            )

        # Compute cosine similarity → score
        sim_matrix = self._model.similarity(output_vec, reference_vec)
        cosine_sim = float(sim_matrix[0, 0])  # scalar in [-1, 1]

        # Map [-1, 1] → [0, 100]
        score = round((cosine_sim + 1.0) * 50.0, 2)

        # Sanitize: ensure score is a valid number before clamping
        if np.isnan(score) or np.isinf(score):
            score = 0.0

        score = max(0.0, min(100.0, score))

        # Threshold logic
        # FAIL zone uses max(0, threshold - 30) so it works even for low thresholds
        fail_threshold = max(0.0, self._threshold - 30.0)

        if score >= self._threshold:  # >= 80 → PASS
            passed = True
        elif score < fail_threshold:  # < 50 (or lower for customized threshold) → FAIL
            passed = False
        else:  # 50-79 (or adjusted) → Tier 2
            passed = False

        return ValidationResult(
            score=score,
            passed=passed,
            dimension=self.dimension,
            details={
                "cosine_similarity": cosine_sim,
                "model": self._model_name,
                "threshold": self._threshold,
                "tier_hint": "tier_2" if (self._threshold - 30.0 <= score < self._threshold) else None,
            },
            error=None,
        )
