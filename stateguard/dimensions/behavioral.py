"""Behavioral validation — Finite-state machine (FSM) state compliance.

Provides :class:`BehavioralValidator` that models the expected behaviour
of an agent or system as a finite-state machine and validates that
state transitions produced by the output are valid according to the
declared FSM graph.
"""

from typing import Any

from ..plugin.base import BaseValidator
from ..models.result import ValidationResult


class BehavioralValidator(BaseValidator):
    """Validator that checks FSM state-transition compliance.

    Usage::

        validator = BehavioralValidator(
            states={"idle", "running", "error"},
            transitions={("idle", "running"), ("running", "idle"), ("running", "error")},
        )
        result = validator.validate({"current": "running", "next": "error"})
    """

    def __init__(
        self,
        states: set[str] | None = None,
        transitions: set[tuple[str, str]] | None = None,
    ) -> None:
        """Initialise the behavioural validator.

        Args:
            states:      Allowed FSM states.
            transitions: Allowed (from → to) transitions.
        """
        super().__init__()
        self._states = states or set()
        self._transitions = transitions or set()

    @property
    def states(self) -> set[str]:
        """Allowed FSM states."""
        return self._states

    @property
    def transitions(self) -> set[tuple[str, str]]:
        """Allowed state transitions."""
        return self._transitions

    def add_transition(self, from_state: str, to_state: str) -> None:
        """Declare a new allowed transition."""
        self._transitions.add((from_state, to_state))

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Validate that *output* describes a valid state transition.

        Args:
            output:  Dict with ``"current"`` and ``"next"`` keys, or a
                     tuple ``(current, next)``.
            context: May contain ``states`` or ``transitions`` overrides.

        Returns:
            A :class:`ValidationResult` reflecting transition legality.
        """
        ...
