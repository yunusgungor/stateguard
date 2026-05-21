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
        ctx = context if isinstance(context, dict) else {}
        raw_min = ctx.get("min_length")
        raw_max = ctx.get("max_length")
        min_length: int | None = raw_min if isinstance(raw_min, (int, float)) and not isinstance(raw_min, bool) else None
        max_length: int | None = raw_max if isinstance(raw_max, (int, float)) and not isinstance(raw_max, bool) else None

        output_str = str(output)
        length = len(output_str)

        min_ok = True
        max_ok = True

        if min_length is not None:
            min_ok = length >= min_length
        if max_length is not None:
            max_ok = length <= max_length

        # Score: both ok = 100, one ok = 50, none ok = 0
        if min_ok and max_ok:
            score = 100.0
        elif not min_ok and not max_ok:
            score = 0.0
        else:
            score = 50.0

        details: dict[str, Any] = {
            "length": length,
            "min_length": min_length,
            "max_length": max_length,
            "min_ok": min_ok,
            "max_ok": max_ok,
        }

        return ValidationResult(
            score=score,
            passed=True,
            dimension=self.dimension,
            details=details,
        )
