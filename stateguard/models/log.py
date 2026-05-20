"""Decision log model — structured record of every validation decision.

Each validation step (across all tiers, dimensions, and feedback stages)
produces a ``DecisionEntry`` that is persisted for audit, debugging,
and downstream consumption.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from stateguard.models.enums import ValidationDimension


class DecisionEntry(BaseModel):
    """A single decision produced by a validation step.

    Attributes:
        timestamp:  When the decision was made (UTC).
        agent_id:   Identifier of the agent whose output was validated.
        step_id:    Unique identifier of the pipeline step.
        dimension:  The :class:`ValidationDimension` that was checked.
        score:      Validation score (0.0 – 100.0).
        decision:   Outcome string (e.g. ``"pass"``, ``"fail"``, ``"retry"``, ``"escalate"``).
        details:    Arbitrary metadata produced during validation.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: str
    step_id: str
    dimension: ValidationDimension
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    decision: str
    details: dict[str, Any] = Field(default_factory=dict)
