"""Validation engine — orchestrates the full validation pipeline."""
from ..plugin.base import BaseValidator
from ..models.result import ValidationResult


class ValidationEngine:
    """Orchestrates multi-tier validation across all registered validators."""

    def __init__(self):
        """Initialize the validation engine."""
        pass

    def validate(self, output, context) -> dict:
        """Run the full validation pipeline on the given output.

        Args:
            output: The output to validate.
            context: Contextual information for validation.

        Returns:
            Dictionary containing validation results from all tiers.
        """
        pass
