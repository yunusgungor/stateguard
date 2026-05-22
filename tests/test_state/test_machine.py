"""Tests for ValidationStateMachine — FSM transition, queries, edge cases."""

from __future__ import annotations

from datetime import datetime

import pytest

from stateguard.state.machine import (
    InvalidTransitionError,
    ValidationStateMachine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fsm() -> ValidationStateMachine:
    return ValidationStateMachine()


@pytest.fixture
def fsm_with_history(fsm: ValidationStateMachine) -> ValidationStateMachine:
    """FSM after a full valid path: IDLE → VALIDATING → FAILED → RETRY
    → VALIDATING → PASSED → COMPLETED."""
    fsm.transition("VALIDATING")
    fsm.transition("FAILED")
    fsm.transition("RETRY")
    fsm.transition("VALIDATING")
    fsm.transition("PASSED")
    fsm.transition("COMPLETED")
    return fsm


# ---------------------------------------------------------------------------
# Transition validation — AC-1, AC-2
# ---------------------------------------------------------------------------


class TestTransitionValidation:
    """Every state's valid and invalid transitions."""

    # --- IDLE ---

    def test_idle_to_validating(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        assert fsm.state == "VALIDATING"

    @pytest.mark.parametrize("invalid", [
        "PASSED", "FAILED", "RETRY", "HITL_WAITING", "COMPLETED",
    ])
    def test_idle_to_invalid_raises_error(
        self, invalid: str,
    ) -> None:
        m = ValidationStateMachine()
        with pytest.raises(InvalidTransitionError):
            m.transition(invalid)
        assert m.state == "IDLE"  # state unchanged

    # --- VALIDATING ---

    def test_validating_to_passed(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        fsm.transition("PASSED")
        assert fsm.state == "PASSED"

    def test_validating_to_failed(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        fsm.transition("FAILED")
        assert fsm.state == "FAILED"

    @pytest.mark.parametrize("invalid", ["IDLE", "RETRY", "HITL_WAITING", "COMPLETED"])
    def test_validating_to_invalid_raises_error(
        self, invalid: str,
    ) -> None:
        m = ValidationStateMachine()
        m.transition("VALIDATING")
        with pytest.raises(InvalidTransitionError):
            m.transition(invalid)
        assert m.state == "VALIDATING"

    # --- FAILED ---

    def test_failed_to_retry(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        fsm.transition("FAILED")
        fsm.transition("RETRY")
        assert fsm.state == "RETRY"

    @pytest.mark.parametrize("invalid", ["VALIDATING", "PASSED", "COMPLETED", "HITL_WAITING", "IDLE"])
    def test_failed_to_invalid_raises_error(
        self, invalid: str,
    ) -> None:
        m = ValidationStateMachine()
        m.transition("VALIDATING")
        m.transition("FAILED")
        with pytest.raises(InvalidTransitionError):
            m.transition(invalid)
        assert m.state == "FAILED"

    # --- RETRY ---

    def test_retry_to_validating(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        fsm.transition("FAILED")
        fsm.transition("RETRY")
        fsm.transition("VALIDATING")
        assert fsm.state == "VALIDATING"

    def test_retry_to_hitl_waiting(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        fsm.transition("FAILED")
        fsm.transition("RETRY")
        fsm.transition("HITL_WAITING")
        assert fsm.state == "HITL_WAITING"

    @pytest.mark.parametrize("invalid", ["PASSED", "FAILED", "COMPLETED", "IDLE"])
    def test_retry_to_invalid_raises_error(
        self, invalid: str,
    ) -> None:
        m = ValidationStateMachine()
        m.transition("VALIDATING")
        m.transition("FAILED")
        m.transition("RETRY")
        with pytest.raises(InvalidTransitionError):
            m.transition(invalid)
        assert m.state == "RETRY"

    # --- HITL_WAITING ---

    def test_hitl_waiting_to_completed(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        fsm.transition("FAILED")
        fsm.transition("RETRY")
        fsm.transition("HITL_WAITING")
        fsm.transition("COMPLETED")
        assert fsm.state == "COMPLETED"

    def test_hitl_waiting_to_failed(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        fsm.transition("FAILED")
        fsm.transition("RETRY")
        fsm.transition("HITL_WAITING")
        fsm.transition("FAILED")
        assert fsm.state == "FAILED"

    @pytest.mark.parametrize("invalid", ["IDLE", "VALIDATING", "PASSED", "RETRY"])
    def test_hitl_waiting_to_invalid_raises_error(
        self, invalid: str,
    ) -> None:
        m = ValidationStateMachine()
        m.transition("VALIDATING")
        m.transition("FAILED")
        m.transition("RETRY")
        m.transition("HITL_WAITING")
        with pytest.raises(InvalidTransitionError):
            m.transition(invalid)
        assert m.state == "HITL_WAITING"

    # --- PASSED ---

    def test_passed_to_completed(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        fsm.transition("PASSED")
        fsm.transition("COMPLETED")
        assert fsm.state == "COMPLETED"

    @pytest.mark.parametrize("invalid", ["IDLE", "VALIDATING", "FAILED", "RETRY", "HITL_WAITING"])
    def test_passed_to_invalid_raises_error(
        self, invalid: str,
    ) -> None:
        m = ValidationStateMachine()
        m.transition("VALIDATING")
        m.transition("PASSED")
        with pytest.raises(InvalidTransitionError):
            m.transition(invalid)
        assert m.state == "PASSED"

    # --- COMPLETED (terminal) ---

    @pytest.mark.parametrize("invalid", [
        "IDLE", "VALIDATING", "PASSED", "FAILED", "RETRY", "HITL_WAITING", "COMPLETED",
    ])
    def test_completed_is_terminal(self, fsm_with_history: ValidationStateMachine, invalid: str) -> None:
        with pytest.raises(InvalidTransitionError):
            fsm_with_history.transition(invalid)
        assert fsm_with_history.state == "COMPLETED"


# ---------------------------------------------------------------------------
# State queries — AC-4
# ---------------------------------------------------------------------------


class TestStateQuery:
    """get_state(), state property, history property."""

    def test_initial_state(self, fsm: ValidationStateMachine) -> None:
        assert fsm.state == "IDLE"
        assert fsm.history == []
        state_info = fsm.get_state()
        assert state_info == {"state": "IDLE", "history": []}

    def test_state_after_transitions(self, fsm_with_history: ValidationStateMachine) -> None:
        assert fsm_with_history.state == "COMPLETED"

    def test_get_state_returns_dict(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        state_info = fsm.get_state()
        assert isinstance(state_info, dict)
        assert "state" in state_info
        assert "history" in state_info

    def test_history_stores_transitions(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        fsm.transition("PASSED")
        history = fsm.history
        assert len(history) == 2
        assert history[0]["from"] == "IDLE"
        assert history[0]["to"] == "VALIDATING"
        assert history[1]["from"] == "VALIDATING"
        assert history[1]["to"] == "PASSED"

    def test_history_chronological_order(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        fsm.transition("FAILED")
        fsm.transition("RETRY")
        history = fsm.history
        assert len(history) == 3
        assert history[0]["to"] == "VALIDATING"
        assert history[1]["to"] == "FAILED"
        assert history[2]["to"] == "RETRY"

    def test_state_property_matches_get_state(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        assert fsm.state == fsm.get_state()["state"]

    def test_get_state_history_immutable(self, fsm: ValidationStateMachine) -> None:
        """Modifying get_state() return must not affect internal state."""
        fsm.transition("VALIDATING")
        state_info = fsm.get_state()
        state_info["history"].append("fake")
        # Internal history must be untouched
        assert len(fsm.history) == 1

    def test_history_property_immutable(self, fsm: ValidationStateMachine) -> None:
        """Modifying returned history list must not affect internal state."""
        fsm.transition("VALIDATING")
        h = fsm.history
        h.append("fake_entry")
        assert len(fsm.history) == 1

    def test_state_stays_same_after_multiple_get_state(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        fsm.get_state()
        fsm.get_state()
        assert fsm.state == "VALIDATING"


# ---------------------------------------------------------------------------
# Edge cases — AC-5
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """COMPLETED terminal, empty string, None, case sensitivity, history isolation."""

    def test_none_raises_error(self, fsm: ValidationStateMachine) -> None:
        with pytest.raises(TypeError, match="to_state must be a string"):
            fsm.transition(None)  # type: ignore[arg-type]
        assert fsm.state == "IDLE"

    def test_empty_string_raises_error(self, fsm: ValidationStateMachine) -> None:
        with pytest.raises(InvalidTransitionError):
            fsm.transition("")
        assert fsm.state == "IDLE"

    def test_false_raises_type_error(self, fsm: ValidationStateMachine) -> None:
        with pytest.raises(TypeError, match="to_state must be a string"):
            fsm.transition(False)  # type: ignore[arg-type]

    def test_case_sensitive_lowercase_invalid(self, fsm: ValidationStateMachine) -> None:
        with pytest.raises(InvalidTransitionError):
            fsm.transition("validating")
        assert fsm.state == "IDLE"

    def test_case_sensitive_mixed_case_invalid(self, fsm: ValidationStateMachine) -> None:
        m = ValidationStateMachine()
        m.transition("VALIDATING")
        with pytest.raises(InvalidTransitionError):
            m.transition("passed")
        assert m.state == "VALIDATING"

    def test_invalid_transition_does_not_modify_history(self, fsm: ValidationStateMachine) -> None:
        with pytest.raises(InvalidTransitionError):
            fsm.transition("PASSED")  # invalid from IDLE
        assert len(fsm.history) == 0

    def test_valid_transition_only_adds_one_history_entry(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        assert len(fsm.history) == 1

    def test_history_entries_have_timestamp(self, fsm: ValidationStateMachine) -> None:
        fsm.transition("VALIDATING")
        entry = fsm.history[0]
        assert "timestamp" in entry
        # Validate ISO 8601 format
        dt = datetime.fromisoformat(entry["timestamp"])
        from datetime import timezone as tz_mod
        assert dt.utcoffset() == tz_mod.utc.utcoffset(None)  # must be UTC

    def test_full_path_no_errors(self, fsm_with_history: ValidationStateMachine) -> None:
        """Full valid path from IDLE to COMPLETED with correct intermediate states."""
        # History must have 6 entries for the full path
        assert len(fsm_with_history.history) == 6
        assert fsm_with_history.state == "COMPLETED"
        # Verify the path chronologically
        path = [(e["from"], e["to"]) for e in fsm_with_history.history]
        assert path == [
            ("IDLE", "VALIDATING"),
            ("VALIDATING", "FAILED"),
            ("FAILED", "RETRY"),
            ("RETRY", "VALIDATING"),
            ("VALIDATING", "PASSED"),
            ("PASSED", "COMPLETED"),
        ]

    def test_only_valid_transitions_work(self) -> None:
        """Spot-check all 7 valid transitions from the spec."""
        valid_paths = [
            ("IDLE", "VALIDATING"),
            ("VALIDATING", "PASSED"),
            ("VALIDATING", "FAILED"),
            ("FAILED", "RETRY"),
            ("RETRY", "VALIDATING"),
            ("RETRY", "HITL_WAITING"),
            ("HITL_WAITING", "COMPLETED"),
            ("HITL_WAITING", "FAILED"),
            ("PASSED", "COMPLETED"),
        ]
        for start, target in valid_paths:
            m = ValidationStateMachine()
            # Navigate to start state
            if start == "IDLE":
                pass
            elif start == "VALIDATING":
                m.transition("VALIDATING")
            elif start == "FAILED":
                m.transition("VALIDATING")
                m.transition("FAILED")
            elif start == "RETRY":
                m.transition("VALIDATING")
                m.transition("FAILED")
                m.transition("RETRY")
            elif start == "HITL_WAITING":
                m.transition("VALIDATING")
                m.transition("FAILED")
                m.transition("RETRY")
                m.transition("HITL_WAITING")
            elif start == "PASSED":
                m.transition("VALIDATING")
                m.transition("PASSED")
            assert m.state == start
            m.transition(target)
            assert m.state == target


# ---------------------------------------------------------------------------
# InvalidTransitionError — AC-3
# ---------------------------------------------------------------------------


class TestInvalidTransitionError:
    """Exception type, message format, attributes."""

    def test_error_type(self) -> None:
        err = InvalidTransitionError("A", "B", ["C"])
        assert isinstance(err, Exception)
        assert isinstance(err, InvalidTransitionError)

    def test_error_attributes(self) -> None:
        err = InvalidTransitionError("A", "B", ["C"])
        assert err.current_state == "A"
        assert err.requested_state == "B"
        assert err.allowed_transitions == ["C"]

    def test_error_message_format(self, fsm: ValidationStateMachine) -> None:
        with pytest.raises(InvalidTransitionError) as exc_info:
            fsm.transition("PASSED")
        msg = str(exc_info.value)
        assert "IDLE" in msg
        assert "PASSED" in msg
        assert "VALIDATING" in msg  # allowed

    def test_error_message_from_completed(
        self, fsm_with_history: ValidationStateMachine,
    ) -> None:
        with pytest.raises(InvalidTransitionError) as exc_info:
            fsm_with_history.transition("IDLE")
        msg = str(exc_info.value)
        assert "COMPLETED" in msg
        assert "IDLE" in msg
        assert "Cannot transition" in msg

    def test_error_state_unchanged_property(self, fsm: ValidationStateMachine) -> None:
        """After InvalidTransitionError, .state still returns original state."""
        with pytest.raises(InvalidTransitionError):
            fsm.transition("PASSED")
        assert fsm.state == "IDLE"


class TestNonStringInput:
    """Non-string to_state type guard."""

    def test_non_string_int_raises_type_error(self, fsm: ValidationStateMachine) -> None:
        with pytest.raises(TypeError, match="to_state must be a string"):
            fsm.transition(0)  # type: ignore[arg-type]

    def test_non_string_list_raises_type_error(self, fsm: ValidationStateMachine) -> None:
        with pytest.raises(TypeError, match="to_state must be a string"):
            fsm.transition([1, 2, 3])  # type: ignore[arg-type]

    def test_non_string_dict_raises_type_error(self, fsm: ValidationStateMachine) -> None:
        with pytest.raises(TypeError, match="to_state must be a string"):
            fsm.transition({"a": 1})  # type: ignore[arg-type]


class TestIndependence:
    """FSM instances must not share state."""

    def test_fsms_are_independent(self) -> None:
        m1 = ValidationStateMachine()
        m2 = ValidationStateMachine()
        m1.transition("VALIDATING")
        m1.transition("FAILED")
        assert m1.state == "FAILED"
        assert m2.state == "IDLE"
        assert len(m2.history) == 0


class TestPerformance:
    """transition() performance < 0.1ms."""

    def test_transition_performance(self, fsm: ValidationStateMachine) -> None:
        import time
        start = time.perf_counter_ns()
        fsm.transition("VALIDATING")
        elapsed_ns = time.perf_counter_ns() - start
        assert elapsed_ns < 100_000  # 0.1ms in ns
