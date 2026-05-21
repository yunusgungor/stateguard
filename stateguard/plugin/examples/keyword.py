"""Keyword validator — checks for the presence or absence of keywords.

The :class:`KeywordValidator` scans the output for a configurable set of
required or forbidden keywords.
"""

import math
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
        ctx = context if isinstance(context, dict) else {}
        raw_required = ctx.get("required_keywords", [])
        raw_forbidden = ctx.get("forbidden_keywords", [])
        required: list[str] = raw_required if isinstance(raw_required, list) else []
        forbidden: list[str] = raw_forbidden if isinstance(raw_forbidden, list) else []

        output_str = str(output) if output is not None else ""
        output_lower = output_str.lower()

        details: dict[str, Any] = {
            "required_found": [],
            "required_missing": [],
            "forbidden_found": [],
        }

        # Check forbidden first — any match => FAIL immediately
        for kw in forbidden:
            if isinstance(kw, str) and kw.lower() in output_lower:
                details["forbidden_found"].append(kw)

        if details["forbidden_found"]:
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details=details,
            )

        # Check required keywords
        for kw in required:
            if isinstance(kw, str) and kw.lower() in output_lower:
                details["required_found"].append(kw)
            elif isinstance(kw, str):
                details["required_missing"].append(kw)

        # Score calculation
        total_required = len(required)
        if total_required == 0:
            score = 100.0
        else:
            score = (len(details["required_found"]) / total_required) * 100.0

        # NaN/Inf guard
        if not math.isfinite(score):
            score = 0.0

        return ValidationResult(
            score=score,
            passed=True,
            dimension=self.dimension,
            details=details,
        )
