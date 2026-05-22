"""Finite State Machine — pipeline state tracking.

Tracks validation lifecycle through a pre-defined set of states and
valid transitions. Raises ``InvalidTransitionError`` on illegal moves.
"""

import copy
from datetime import datetime, timezone
from typing import Any


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self,
        current_state: str,
        requested_state: str,
        allowed_transitions: list[str],
    ) -> None:
        self.current_state = current_state
        self.requested_state = requested_state
        self.allowed_transitions = allowed_transitions
        super().__init__(
            f"Cannot transition from {current_state} to {requested_state}. "
            f"Allowed: {allowed_transitions}"
        )


class ValidationStateMachine:
    """Seven-state FSM for pipeline lifecycle.

    Valid transitions::

        IDLE → VALIDATING → PASSED  → COMPLETED
                          → FAILED  → RETRY → VALIDATING
                                            → HITL_WAITING → COMPLETED
                                                          → FAILED
        COMPLETED is terminal — no outgoing transitions.
    """

    VALID_TRANSITIONS: dict[str, list[str]] = {
        "IDLE": ["VALIDATING"],
        "VALIDATING": ["PASSED", "FAILED"],
        "FAILED": ["RETRY"],
        "RETRY": ["VALIDATING", "HITL_WAITING"],
        "HITL_WAITING": ["COMPLETED", "FAILED"],
        "PASSED": ["COMPLETED"],
        "COMPLETED": [],
    }

    def __init__(self) -> None:
        """Start in IDLE with empty history."""
        self._state: str = "IDLE"
        self._history: list[dict[str, Any]] = []

    @property
    def state(self) -> str:
        """Current state."""
        return self._state

    @property
    def history(self) -> list[dict[str, Any]]:
        """List of past transitions (deep copy — external callers cannot mutate)."""
        return copy.deepcopy(self._history)

    def transition(self, to_state: str) -> None:
        """Attempt a transition to *to_state*.

        Args:
            to_state: Target state name (UPPER_CASE).

        Raises:
            InvalidTransitionError: If the transition is not allowed.
        """
        # --- guard: empty / None to_state ---
        if not isinstance(to_state, str):
            raise TypeError(
                f"to_state must be a string, got {type(to_state).__name__}"
            )
        if not to_state:
            raise InvalidTransitionError(
                self._state, str(to_state), self.VALID_TRANSITIONS.get(self._state, [])
            )

        allowed = self.VALID_TRANSITIONS.get(self._state, [])

        if to_state not in allowed:
            raise InvalidTransitionError(self._state, to_state, allowed)

        # --- valid transition ---
        self._history.append({
            "from": self._state,
            "to": to_state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._state = to_state

    def get_state(self) -> dict[str, Any]:
        """Return current state and transition history.

        Returns:
            dict with keys ``state`` and ``history``.
        """
        return {"state": self._state, "history": copy.deepcopy(self._history)}
