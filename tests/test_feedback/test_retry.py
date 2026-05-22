"""Tests for RetryHandler — auto-retry with feedback accumulation."""

from __future__ import annotations

from typing import Any

import pytest

from stateguard.feedback.retry import RetryHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler() -> RetryHandler:
    return RetryHandler()


@pytest.fixture
def zero_retry_handler() -> RetryHandler:
    return RetryHandler(max_retries=0)


@pytest.fixture
def two_retry_handler() -> RetryHandler:
    return RetryHandler(max_retries=2)


@pytest.fixture
def passing_validator() -> Any:
    def _validate(
        output: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {"passed": True, "score": 100.0, "details": {}}

    return _validate


@pytest.fixture
def failing_validator() -> Any:
    def _validate(
        output: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {"passed": False, "score": 20.0, "error": "validation failed"}

    return _validate


@pytest.fixture
def sample_output() -> str:
    return "test output"


@pytest.fixture
def sample_context() -> dict[str, Any]:
    return {"key": "value"}


@pytest.fixture
def error_validator() -> Any:
    """Validator that raises an exception."""

    def _validate(
        output: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        raise RuntimeError("unexpected crash")

    return _validate


# ---------------------------------------------------------------------------
# Execute — success cases (AC-1)
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    """execute() — passes on first try or after retry."""

    def test_passes_on_first_attempt(
        self,
        handler: RetryHandler,
        passing_validator: Any,
        sample_output: str,
    ) -> None:
        result = handler.execute(passing_validator, sample_output)
        assert result["result"]["passed"] is True
        assert result["attempts"] == 1
        assert result["escalated"] is False
        assert result["errors"] == []

    def test_passes_on_second_attempt(
        self,
        handler: RetryHandler,
        sample_output: str,
    ) -> None:
        """Flaky validator: fails first time, passes second time."""
        call_count: list[int] = [0]

        def flaky(
            output: str, ctx: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:
                return {"passed": False, "score": 30.0, "error": "first fail"}
            return {"passed": True, "score": 90.0}

        result = handler.execute(flaky, sample_output)
        assert result["result"]["passed"] is True
        assert result["attempts"] == 2
        assert result["escalated"] is False
        assert len(result["errors"]) == 1
        assert result["errors"][0]["attempt"] == 1

    def test_passes_on_third_attempt_last_retry(
        self,
        handler: RetryHandler,
        sample_output: str,
    ) -> None:
        """Max retries = 2, passes on 3rd attempt (1 initial + 2 retries)."""
        call_count: list[int] = [0]

        def flaky_triple(
            output: str, ctx: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            call_count[0] += 1
            if call_count[0] <= 2:
                return {"passed": False, "score": 20.0, "error": f"fail {call_count[0]}"}
            return {"passed": True, "score": 95.0}

        result = handler.execute(flaky_triple, sample_output)
        assert result["result"]["passed"] is True
        assert result["attempts"] == 3
        assert result["escalated"] is False
        assert len(result["errors"]) == 2

    def test_passes_with_none_context(
        self,
        handler: RetryHandler,
        passing_validator: Any,
        sample_output: str,
    ) -> None:
        result = handler.execute(passing_validator, sample_output, None)
        assert result["result"]["passed"] is True


# ---------------------------------------------------------------------------
# Execute — failure / escalation (AC-1, AC-4)
# ---------------------------------------------------------------------------


class TestExecuteFailure:
    """execute() — all attempts fail, escalated=True."""

    def test_all_attempts_fail_with_default_retries(
        self,
        handler: RetryHandler,
        failing_validator: Any,
        sample_output: str,
    ) -> None:
        """max_retries=2 → 3 total attempts, all fail."""
        result = handler.execute(failing_validator, sample_output)
        assert result["escalated"] is True
        assert result["attempts"] == 3  # 1 initial + 2 retries
        assert len(result["errors"]) == 3

    def test_zero_retries_escalates_immediately(
        self,
        zero_retry_handler: RetryHandler,
        failing_validator: Any,
        sample_output: str,
    ) -> None:
        result = zero_retry_handler.execute(failing_validator, sample_output)
        assert result["escalated"] is True
        assert result["attempts"] == 1  # only 1 attempt
        assert len(result["errors"]) == 1

    def test_one_retry_escalates_after_two_attempts(
        self,
        sample_output: str,
    ) -> None:
        handler = RetryHandler(max_retries=1)

        def always_fail(
            output: str, ctx: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            return {"passed": False, "score": 10.0}

        result = handler.execute(always_fail, sample_output)
        assert result["escalated"] is True
        assert result["attempts"] == 2
        assert len(result["errors"]) == 2

    def test_escalated_result_structure(
        self,
        handler: RetryHandler,
        failing_validator: Any,
        sample_output: str,
    ) -> None:
        result = handler.execute(failing_validator, sample_output)
        assert "result" in result
        assert "attempts" in result
        assert "errors" in result
        assert "escalated" in result
        assert isinstance(result["errors"], list)
        for err in result["errors"]:
            assert "attempt" in err
            assert "score" in err or "error" in err


# ---------------------------------------------------------------------------
# Input validation (AC-3)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """execute() / __init__ — input validation guards."""

    def test_non_callable_validator_raises_type_error(
        self,
        handler: RetryHandler,
        sample_output: str,
    ) -> None:
        with pytest.raises(TypeError, match="validator_func must be callable"):
            handler.execute("not_callable", sample_output)  # type: ignore[arg-type]

    def test_non_dict_context_raises_type_error(
        self,
        handler: RetryHandler,
        passing_validator: Any,
        sample_output: str,
    ) -> None:
        with pytest.raises(TypeError, match="context must be a dict or None"):
            handler.execute(passing_validator, sample_output, "bad_context")  # type: ignore[arg-type]

    def test_list_context_raises_type_error(
        self,
        handler: RetryHandler,
        passing_validator: Any,
        sample_output: str,
    ) -> None:
        with pytest.raises(TypeError, match="context must be a dict or None"):
            handler.execute(passing_validator, sample_output, [1, 2, 3])  # type: ignore[arg-type]

    def test_int_context_raises_type_error(
        self,
        handler: RetryHandler,
        passing_validator: Any,
        sample_output: str,
    ) -> None:
        with pytest.raises(TypeError, match="context must be a dict or None"):
            handler.execute(passing_validator, sample_output, 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Retry feedback format (AC-1)
# ---------------------------------------------------------------------------


class TestRetryFeedback:
    """execute() — _feedback format and error accumulation."""

    def test_feedback_added_on_retry(
        self,
        handler: RetryHandler,
        sample_output: str,
    ) -> None:
        """Validate _feedback structure in retried calls."""
        contexts: list[dict[str, Any]] = []

        def tracking_validator(
            output: str, ctx: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            contexts.append(ctx or {})
            attempt = ctx.get("_feedback", {}).get("attempt", 0)
            if attempt == 1:
                return {"passed": True, "score": 90.0}
            return {"passed": False, "score": 20.0, "error": "fail"}

        result = handler.execute(tracking_validator, sample_output)
        assert result["result"]["passed"] is True
        assert result["attempts"] == 2

        # First call should have no _feedback
        first_ctx = contexts[0]
        assert "_feedback" not in first_ctx

        # Second call should have _feedback
        assert len(contexts) >= 2
        second_ctx = contexts[1]
        assert "_feedback" in second_ctx
        fb = second_ctx["_feedback"]
        assert fb["attempt"] == 1
        assert fb["max_retries"] == 2
        assert fb["previous_error"] is not None

    def test_feedback_contains_previous_error_detail(
        self,
        handler: RetryHandler,
        sample_output: str,
    ) -> None:
        """The feedback dict contains the right previous error info."""
        contexts: list[dict[str, Any]] = []

        def track_ctx(
            output: str, ctx: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            contexts.append(ctx or {})
            # Always fail so we see feedback on next call
            return {
                "passed": False,
                "score": 25.0,
                "error": "score too low",
                "details": {"actual_score": 25.0},
            }

        handler.execute(track_ctx, sample_output, {"user": "test"})
        # Check 2nd call had feedback
        for ctx in contexts[1:]:
            fb = ctx["_feedback"]
            assert fb["previous_error"] == "score too low" or \
                   fb["previous_error"] == {"actual_score": 25.0}
            assert isinstance(fb["previous_score"], (int, float))

    def test_error_accumulation_contains_all_failures(
        self,
        handler: RetryHandler,
        failing_validator: Any,
        sample_output: str,
    ) -> None:
        result = handler.execute(failing_validator, sample_output)
        assert len(result["errors"]) == handler.max_retries + 1
        for i, err in enumerate(result["errors"]):
            assert err["attempt"] == i + 1
            assert err["score"] == 20.0


# ---------------------------------------------------------------------------
# Edge cases (AC-3)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases — max_retries=0, negative, bool, exception handling."""

    def test_max_retries_zero(
        self,
        zero_retry_handler: RetryHandler,
        failing_validator: Any,
        sample_output: str,
    ) -> None:
        result = zero_retry_handler.execute(failing_validator, sample_output)
        assert result["escalated"] is True
        assert result["attempts"] == 1

    def test_negative_max_retries_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            RetryHandler(max_retries=-1)

    def test_bool_max_retries_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="max_retries must be an int"):
            RetryHandler(max_retries=True)  # type: ignore[arg-type]

    def test_float_max_retries_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="max_retries must be an int"):
            RetryHandler(max_retries=2.5)  # type: ignore[arg-type]

    def test_string_max_retries_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="max_retries must be an int"):
            RetryHandler(max_retries="2")  # type: ignore[arg-type]

    def test_validator_exception_is_caught_and_continues(
        self,
        handler: RetryHandler,
        error_validator: Any,
        sample_output: str,
    ) -> None:
        """Exception in validator is caught and retry continues."""
        call_count: list[int] = [0]

        def mixed_validator(
            output: str, ctx: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("crash")
            return {"passed": True, "score": 100.0}

        result = handler.execute(mixed_validator, sample_output)
        assert result["result"]["passed"] is True
        assert result["attempts"] == 2
        assert len(result["errors"]) == 1
        assert result["errors"][0]["type"] == "RuntimeError"

    def test_all_exceptions_caught_in_loop(
        self,
        handler: RetryHandler,
        error_validator: Any,
        sample_output: str,
    ) -> None:
        """If validator always raises, all attempts are caught."""
        result = handler.execute(error_validator, sample_output)
        assert result["escalated"] is True
        assert len(result["errors"]) == 3
        for err in result["errors"]:
            assert err["type"] == "RuntimeError"
            assert err["error"] == "unexpected crash"

    def test_empty_output_string_is_valid(
        self,
        handler: RetryHandler,
        passing_validator: Any,
    ) -> None:
        result = handler.execute(passing_validator, "")
        assert result["result"]["passed"] is True

    def test_validator_returns_non_dict_raises_type_error(
        self,
        handler: RetryHandler,
        sample_output: str,
    ) -> None:
        def bad_return(
            output: str, ctx: dict[str, Any] | None = None
        ) -> str:
            return "not_a_dict"

        with pytest.raises(TypeError, match="validator_func must return a dict"):
            handler.execute(bad_return, sample_output)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reset (AC-2)
# ---------------------------------------------------------------------------


class TestReset:
    """reset() — attempt_count sıfırlama."""

    def test_reset_sets_attempt_count_to_zero(
        self,
        handler: RetryHandler,
        sample_output: str,
    ) -> None:
        def always_fail(
            output: str, ctx: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            return {"passed": False, "score": 10.0}

        handler.execute(always_fail, sample_output)
        assert handler.attempt_count > 0
        handler.reset()
        assert handler.attempt_count == 0

    def test_multiple_execute_cycles_with_reset(
        self,
        handler: RetryHandler,
        passing_validator: Any,
        failing_validator: Any,
        sample_output: str,
    ) -> None:
        """Multiple execute cycles: each cycle should start fresh."""
        # Cycle 1: fail
        result1 = handler.execute(failing_validator, sample_output)
        assert result1["escalated"] is True
        assert handler.attempt_count == 3

        handler.reset()
        assert handler.attempt_count == 0

        # Cycle 2: pass immediately
        result2 = handler.execute(passing_validator, sample_output)
        assert result2["escalated"] is False
        assert result2["attempts"] == 1

    def test_reset_does_not_affect_max_retries(
        self,
        handler: RetryHandler,
    ) -> None:
        handler.reset()
        assert handler.max_retries == 2


# ---------------------------------------------------------------------------
# Integration — HITL signal (AC-4)
# ---------------------------------------------------------------------------


class TestIntegration:
    """execute() — escalated format compatible with HITLHandler."""

    def test_escalated_result_contains_error_list_for_hitl(
        self,
        handler: RetryHandler,
        failing_validator: Any,
        sample_output: str,
    ) -> None:
        """Errors list is structured for HITL consumption."""
        result = handler.execute(failing_validator, sample_output)
        assert result["escalated"] is True
        for err in result["errors"]:
            assert "attempt" in err
            assert "score" in err or "error" in err
            assert isinstance(err["attempt"], int)

    def test_escalated_flag_true_when_max_retries_exceeded(
        self,
        handler: RetryHandler,
        failing_validator: Any,
        sample_output: str,
    ) -> None:
        result = handler.execute(failing_validator, sample_output)
        assert result["escalated"] is True
        assert result["attempts"] > handler.max_retries
