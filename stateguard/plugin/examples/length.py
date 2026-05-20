"""Length validator — validates output length constraints.

The :class:`LengthValidator` checks whether the length of the provided
*output* falls within a configurable minimum and maximum range.
"""

from typing import Any

from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.plugin.base import BaseValidator


class LengthValidator(BaseValidator):
    """Validator that enforces length constraints on the output.

    Class Attributes:
        name:      ``"length"``
        dimension: :attr:`ValidationDimension.QUANTITATIVE`
        tier:      :attr:`ValidationTier.TIER_1`
    """

    name: str = "length"
    dimension: ValidationDimension = ValidationDimension.QUANTITATIVE
    tier: ValidationTier = ValidationTier.TIER_1

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Check length of *output* against configured bounds.

        Args:
            output:  The data whose length is to be validated (string, list,
                     or any object with a ``__len__`` method).
            context: Optional contextual information (may contain ``min_length``
                     and/or ``max_length`` keys).

        Returns:
            A :class:`ValidationResult` indicating length compliance.
        """
        ...
