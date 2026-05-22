"""Regex validator — validates output against a regular expression pattern.

The :class:`RegexValidator` checks whether the provided *output* matches
a configurable regular expression pattern.
"""

from __future__ import annotations

import re
from typing import Any

from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.plugin.base import BaseValidator


class RegexValidator(BaseValidator):
    """Validator that checks output against a regex pattern.

    Class Attributes:
        name:      ``"regex"``
        dimension: :attr:`ValidationDimension.STRUCTURAL`
        tier:      :attr:`ValidationTier.TIER_1`
    """

    name: str = "regex"
    dimension: ValidationDimension = ValidationDimension.STRUCTURAL
    tier: ValidationTier = ValidationTier.TIER_1

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Check *output* against a regex pattern from *context*.

        Args:
            output:  The data to validate.
            context: May contain a ``pattern`` key with a valid regex string.

        Returns:
            A :class:`ValidationResult` indicating pattern compliance.
        """
        ctx = context if isinstance(context, dict) else {}
        raw_pattern = ctx.get("pattern")

        # No pattern = no constraint
        if raw_pattern is None:
            return ValidationResult(
                score=100.0,
                passed=True,
                dimension=self.dimension,
                details={"pattern": None, "message": "No pattern specified"},
            )

        if not isinstance(raw_pattern, str):
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"pattern": str(raw_pattern), "error": "Pattern must be a string"},
                error="Pattern must be a string.",
            )

        # Compile and match
        try:
            compiled = re.compile(raw_pattern)
        except re.error as e:
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"pattern": raw_pattern, "error": str(e)},
                error=f"Invalid regex pattern: {e}",
            )

        output_str = str(output) if output is not None else ""
        match = compiled.search(output_str)

        if match:
            return ValidationResult(
                score=100.0,
                passed=True,
                dimension=self.dimension,
                details={
                    "pattern": raw_pattern,
                    "matched": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                },
            )

        return ValidationResult(
            score=0.0,
            passed=False,
            dimension=self.dimension,
            details={"pattern": raw_pattern, "matched": None},
            error=f"Output does not match pattern: {raw_pattern}",
        )
