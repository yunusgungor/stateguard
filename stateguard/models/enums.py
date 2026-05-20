"""Enum definitions for the Stateguard validation engine.

Defines the core enumerated types used throughout the validation
pipeline to classify validation dimensions, tiers, and lifecycle states.
"""

from enum import Enum


class ValidationDimension(str, Enum):
    """The aspect of a response or output being validated.

    Attributes:
        STRUCTURAL:   Validates format, schema compliance, and structure.
        SEMANTIC:     Validates meaning, relevance, and factual correctness.
        QUANTITATIVE: Validates numerical ranges, metrics, and thresholds.
        BEHAVIORAL:   Validates behavioral constraints and usage patterns.
        SECURITY:     Validates safety, permissions, and harmful content.
    """

    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    QUANTITATIVE = "quantitative"
    BEHAVIORAL = "behavioral"
    SECURITY = "security"


class ValidationTier(str, Enum):
    """Progressive validation tier indicating the depth of analysis.

    Attributes:
        TIER_1: Fast, lightweight checks (e.g., schema, format).
        TIER_2: Intermediate checks (e.g., semantic, quantitative).
        TIER_3: Deep, expensive checks (e.g., behavioral, security).
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class ValidationState(str, Enum):
    """Lifecycle state of a validation run.

    Attributes:
        IDLE:          Validation has not started.
        VALIDATING:    Validation is in progress.
        PASSED:        Validation completed successfully.
        FAILED:        Validation failed.
        RETRY:         Validation will be retried.
        HITL_WAITING:  Awaiting human-in-the-loop approval.
        COMPLETED:     Final state after all retries / HITL resolution.
    """

    IDLE = "idle"
    VALIDATING = "validating"
    PASSED = "passed"
    FAILED = "failed"
    RETRY = "retry"
    HITL_WAITING = "hitl_waiting"
    COMPLETED = "completed"
