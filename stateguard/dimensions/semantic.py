"""Semantic validation — embedding cross-validation with configurable threshold.

Provides :class:`SemanticValidator` which compares the semantic similarity
between an output and a reference (or expected) embedding, flagging
results that fall below a configurable threshold.
"""

from typing import Any

from ..plugin.base import BaseValidator
from ..models.result import ValidationResult


class SemanticValidator(BaseValidator):
    """Validator that measures semantic similarity via embeddings.

    The validator encodes both the *output* and an optional *reference*
    into embedding vectors (using an :class:`~stateguard.utils.embedding.EmbeddingModel`)
    and computes a similarity score.  Scores below ``threshold`` cause
    the validation to fail.
    """

    def __init__(self, threshold: float = 0.75) -> None:
        """Initialise the semantic validator.

        Args:
            threshold: Minimum cosine similarity to pass (0.0 – 1.0).
        """
        super().__init__()
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Current similarity threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Update the similarity threshold."""
        self._threshold = value

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Validate *output* semantically against an optional reference.

        Args:
            output:  The produced output (text, dict, etc.).
            context: May contain ``reference`` (expected value) and
                     ``threshold`` override keys.

        Returns:
            A :class:`ValidationResult` with a score proportional to
            semantic similarity.
        """
        ...
