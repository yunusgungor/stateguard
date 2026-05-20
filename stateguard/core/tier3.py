"""Tier 3 validation — LLM-as-judge checking."""
from ..plugin.base import BaseValidator
from ..models.result import ValidationResult


class LLMValidator(BaseValidator):
    """Validator that uses an LLM to judge output quality."""

    def __init__(self):
        super().__init__()

    def validate(self, output, context) -> ValidationResult:
        """Run LLM-based validation on the given output."""
        pass
