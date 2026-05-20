"""Keyword validator — checks for the presence or absence of keywords.

The :class:`KeywordValidator` scans the output for a configurable set of
required or forbidden keywords.
"""

from typing import Any

from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.plugin.base import BaseValidator


class KeywordValidator(BaseValidator):
    """Validator that checks for required or forbidden keywords.

    Class Attributes:
        name:      ``"keyword"``
        dimension: :attr:`ValidationDimension.SEMANTIC`
        tier:      :attr:`ValidationTier.TIER_1`
    """

    name: str = "keyword"
    dimension: ValidationDimension = ValidationDimension.SEMANTIC
    tier: ValidationTier = ValidationTier.TIER_1

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Scan *output* for required or forbidden keywords.

        Args:
            output:  The text or data to scan for keywords.
            context: Optional contextual information (may contain
                     ``required_keywords`` and/or ``forbidden_keywords`` lists).

        Returns:
            A :class:`ValidationResult` indicating keyword check outcome.
        """
        ...
