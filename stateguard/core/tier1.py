"""Tier 1 validation — Embedding-based similarity checking."""
from ..plugin.base import BaseValidator
from ..models.result import ValidationResult


class EmbeddingValidator(BaseValidator):
    """Validator that uses embedding similarity to check outputs."""

    def __init__(self):
        super().__init__()

    def validate(self, output, context) -> ValidationResult:
        """Run embedding-based validation on the given output."""
        pass
