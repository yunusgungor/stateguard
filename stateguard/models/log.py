"""Decision log model — structured record of every validation decision.

Each validation step (across all tiers, dimensions, and feedback stages)
produces a ``DecisionEntry`` that is persisted for audit, debugging,
and downstream consumption.
"""

from __future__ import annotations

import threading
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


class DecisionLogger:
    """Thread-safe in-memory decision log.

    Stores :class:`DecisionEntry` records and supports filtered queries
    by ``agent_id``, ``time_range``, and ``result`` (decision string).
    """

    def __init__(self) -> None:
        self._entries: list[DecisionEntry] = []
        self._lock = threading.Lock()

    def log(self, entry: DecisionEntry) -> None:
        """Record a decision entry.

        Args:
            entry: The :class:`DecisionEntry` to persist.
        """
        with self._lock:
            self._entries.append(entry)

    def query(
        self,
        agent_id: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        result: str | None = None,
    ) -> list[DecisionEntry]:
        """Query decision entries with optional filters.

        All supplied filters are combined with AND logic.
        Filters set to ``None`` are skipped.

        Args:
            agent_id:   If set, only entries with this ``agent_id``.
            time_range: If set, ``(start, end)`` inclusive range on
                        ``timestamp``.
            result:     If set, only entries whose ``decision`` field
                        matches (case-insensitive).

        Returns:
            A new list of matching :class:`DecisionEntry` objects.
        """
        with self._lock:
            entries = [e.model_copy(deep=True) for e in self._entries]

        if agent_id is not None:
            entries = [e for e in entries if e.agent_id == agent_id]

        if time_range is not None:
            start, end = time_range
            entries = [e for e in entries if start <= e.timestamp <= end]

        if result is not None:
            result_lower = result.lower()
            entries = [e for e in entries if e.decision.lower() == result_lower]

        return entries
