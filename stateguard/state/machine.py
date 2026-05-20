"""Finite State Machine — pipeline state tracking.

Tracks validation lifecycle through a pre-defined set of states and
valid transitions. Raises ``InvalidTransitionError`` on illegal moves.
"""

from typing import Any


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    pass


class ValidationStateMachine:
    """Five-state FSM for pipeline lifecycle.

    Valid transitions::

        IDLE → VALIDATING → PASSED
                          → FAILED → RETRY → VALIDATING
                                           → HITL_WAITING → COMPLETED
                                                         → FAILED
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
        """List of past transitions."""
        return list(self._history)

    def transition(self, to_state: str) -> None:
        """Attempt a transition to *to_state*.

        Args:
            to_state: Target state name.

        Raises:
            InvalidTransitionError: If the transition is not allowed.
        """
        pass

    def get_state(self) -> dict[str, Any]:
        """Return current state and transition history.

        Returns:
            dict with keys ``state`` and ``history``.
        """
        return {"state": self._state, "history": self._history}
