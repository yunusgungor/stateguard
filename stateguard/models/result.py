"""Pydantic v2 result models for the Stateguard validation engine.

Provides ``ValidationResult`` (single-dimension outcome) and
``EngineResult`` (aggregated multi-dimension outcome).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from stateguard.models.enums import ValidationDimension


class ValidationResult(BaseModel):
    """Outcome of validating a single dimension.

    Attributes:
        score:     Validation score between 0.0 (worst) and 100.0 (best).
        passed:    Whether the validation passed its threshold.
        dimension: The :class:`ValidationDimension` that was checked.
        details:   Arbitrary key-value metadata produced during validation.
        error:     Human-readable error message, if the validation failed.
    """

    score: float = Field(default=0.0, ge=0.0, le=100.0)
    passed: bool = False
    dimension: ValidationDimension
    details: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class EngineResult(BaseModel):
    """Aggregated result produced by the validation engine after executing
    a validation plan across one or more tiers.

    Attributes:
        overall_score:    Weighted composite score across all dimensions.
        passed:           Whether the overall validation succeeded.
        tier_path:        Ordered sequence of tier indices that were executed.
        dimension_scores: Per-step scores keyed by step name
                          (e.g. ``\"tier_1\"``, ``\"tier_2\"``).
        details:          Arbitrary key-value metadata (scoring breakdown,
                          normalization info, thresholds, etc.).
    """

    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    passed: bool = False
    tier_path: list[int] = Field(default_factory=list)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
