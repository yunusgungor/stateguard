"""Tests for HITLHandler — human-in-the-loop approval requests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from stateguard.feedback.hitl import HITLHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler() -> HITLHandler:
    return HITLHandler()


@pytest.fixture
def zero_timeout_handler() -> HITLHandler:
    return HITLHandler(timeout_minutes=0)


@pytest.fixture
def sample_step_info() -> dict[str, Any]:
    return {
        "step": 3,
        "dimension": "semantic",
        "attempts": 3,
        "errors": [
            {"attempt": 1, "score": 20.0, "error": "fail"},
            {"attempt": 2, "score": 25.0, "error": "fail"},
            {"attempt": 3, "score": 15.0, "error": "fail"},
        ],
    }


@pytest.fixture
def handler_with_request(
    handler: HITLHandler, sample_step_info: dict[str, Any]
) -> tuple[HITLHandler, str]:
    result = handler.request_approval(sample_step_info)
    return handler, result["request_id"]


# ---------------------------------------------------------------------------
# request_approval — AC-1
# ---------------------------------------------------------------------------


class TestRequestApproval:
    """request_approval() — request_id, status, timeout_at."""

    def test_returns_request_id(
        self,
        handler: HITLHandler,
        sample_step_info: dict[str, Any],
    ) -> None:
        result = handler.request_approval(sample_step_info)
        assert "request_id" in result
        assert result["request_id"].startswith("hitl_")

    def test_status_is_pending(
        self,
        handler: HITLHandler,
        sample_step_info: dict[str, Any],
    ) -> None:
        result = handler.request_approval(sample_step_info)
        assert result["status"] == "pending"

    def test_timeout_at_is_set(
        self,
        handler: HITLHandler,
        sample_step_info: dict[str, Any],
    ) -> None:
        result = handler.request_approval(sample_step_info)
        assert "timeout_at" in result
        timeout_dt = datetime.fromisoformat(result["timeout_at"])
        assert timeout_dt.tzinfo is not None

    def test_timeout_minutes_returned(
        self,
        handler: HITLHandler,
        sample_step_info: dict[str, Any],
    ) -> None:
        result = handler.request_approval(sample_step_info)
        assert result["timeout_minutes"] == 5

    def test_internal_record_created(
        self,
        handler: HITLHandler,
        sample_step_info: dict[str, Any],
    ) -> None:
        result = handler.request_approval(sample_step_info)
        rid = result["request_id"]
        record = handler._pending[rid]
        assert record["status"] == "pending"
        assert record["approved"] is None
        assert record["step_info"] == sample_step_info

    def test_non_dict_step_info_raises_type_error(
        self,
        handler: HITLHandler,
    ) -> None:
        with pytest.raises(TypeError, match="step_info must be a dict"):
            handler.request_approval("not_dict")  # type: ignore[arg-type]

    def test_none_step_info_raises_type_error(
        self,
        handler: HITLHandler,
    ) -> None:
        with pytest.raises(TypeError, match="step_info must be a dict"):
            handler.request_approval(None)  # type: ignore[arg-type]

    def test_empty_step_info_accepted(
        self,
        handler: HITLHandler,
    ) -> None:
        result = handler.request_approval({})
        assert result["status"] == "pending"


# ---------------------------------------------------------------------------
# check_approval — AC-2
# ---------------------------------------------------------------------------


class TestCheckApproval:
    """check_approval() — pending, decided, timeout detection."""

    def test_pending_returns_none_approved(
        self,
        handler_with_request: tuple[HITLHandler, str],
    ) -> None:
        handler, rid = handler_with_request
        result = handler.check_approval(rid)
        assert result["status"] == "pending"
        assert result["approved"] is None
        assert result["reason"] is None

    def test_invalid_request_id_raises_key_error(
        self,
        handler: HITLHandler,
    ) -> None:
        with pytest.raises(KeyError, match="not-found"):
            handler.check_approval("not-found")

    def test_immediate_timeout_with_zero_timeout(
        self,
        zero_timeout_handler: HITLHandler,
        sample_step_info: dict[str, Any],
    ) -> None:
        result = zero_timeout_handler.request_approval(sample_step_info)
        rid = result["request_id"]
        check = zero_timeout_handler.check_approval(rid)
        assert check["status"] == "decided"
        assert check["approved"] is False
        assert check["reason"] == "timeout"


# ---------------------------------------------------------------------------
# resolve_approval — AC-3
# ---------------------------------------------------------------------------


class TestResolveApproval:
    """resolve_approval() — approve/reject, idempotent, timeout guard."""

    def test_approve_request(
        self,
        handler_with_request: tuple[HITLHandler, str],
    ) -> None:
        handler, rid = handler_with_request
        result = handler.resolve_approval(rid, True)
        assert result["status"] == "decided"
        assert result["approved"] is True

    def test_reject_request(
        self,
        handler_with_request: tuple[HITLHandler, str],
    ) -> None:
        handler, rid = handler_with_request
        result = handler.resolve_approval(rid, False)
        assert result["status"] == "decided"
        assert result["approved"] is False

    def test_with_reason(
        self,
        handler_with_request: tuple[HITLHandler, str],
    ) -> None:
        handler, rid = handler_with_request
        result = handler.resolve_approval(rid, True, "looks good")
        assert result["reason"] == "looks good"

    def test_decision_time_set(
        self,
        handler_with_request: tuple[HITLHandler, str],
    ) -> None:
        handler, rid = handler_with_request
        result = handler.resolve_approval(rid, True)
        assert result["decision_time"] is not None
        dt = datetime.fromisoformat(result["decision_time"])
        # Should be recent
        assert datetime.now(timezone.utc) - dt < timedelta(seconds=5)

    def test_invalid_request_id_raises_key_error(
        self,
        handler: HITLHandler,
    ) -> None:
        with pytest.raises(KeyError, match="not-found"):
            handler.resolve_approval("not-found", True)

    def test_non_bool_approved_raises_type_error(
        self,
        handler_with_request: tuple[HITLHandler, str],
    ) -> None:
        handler, rid = handler_with_request
        with pytest.raises(TypeError, match="approved must be a bool"):
            handler.resolve_approval(rid, "yes")  # type: ignore[arg-type]

    def test_timeout_request_cannot_be_resolved(
        self,
        zero_timeout_handler: HITLHandler,
        sample_step_info: dict[str, Any],
    ) -> None:
        result = zero_timeout_handler.request_approval(sample_step_info)
        rid = result["request_id"]
        zero_timeout_handler.check_approval(rid)  # triggers timeout
        with pytest.raises(ValueError, match="already timed out"):
            zero_timeout_handler.resolve_approval(rid, True)

    def test_idempotent_resolve(
        self,
        handler_with_request: tuple[HITLHandler, str],
    ) -> None:
        handler, rid = handler_with_request
        handler.resolve_approval(rid, True)
        # Second resolve should not raise
        result = handler.resolve_approval(rid, True)
        assert result["status"] == "decided"


# ---------------------------------------------------------------------------
# on_timeout — AC-4
# ---------------------------------------------------------------------------


class TestOnTimeout:
    """on_timeout() — fail-close return format."""

    def test_returns_fail_close(self, handler: HITLHandler) -> None:
        result = handler.on_timeout()
        assert result["approved"] is False
        assert result["reason"] == "timeout"


# ---------------------------------------------------------------------------
# get_pending_requests — AC-5
# ---------------------------------------------------------------------------


class TestGetPending:
    """get_pending_requests() — pending list, remaining_seconds."""

    def test_empty_when_no_requests(
        self,
        handler: HITLHandler,
    ) -> None:
        result = handler.get_pending_requests()
        assert result["count"] == 0
        assert result["pending"] == []

    def test_contains_pending_request(
        self,
        handler_with_request: tuple[HITLHandler, str],
    ) -> None:
        handler, rid = handler_with_request
        result = handler.get_pending_requests()
        assert result["count"] == 1
        assert result["pending"][0]["request_id"] == rid

    def test_remaining_seconds_positive(
        self,
        handler_with_request: tuple[HITLHandler, str],
    ) -> None:
        handler, _ = handler_with_request
        result = handler.get_pending_requests()
        assert result["pending"][0]["remaining_seconds"] > 0.0

    def test_does_not_include_decided(
        self,
        handler_with_request: tuple[HITLHandler, str],
    ) -> None:
        handler, rid = handler_with_request
        handler.resolve_approval(rid, True)
        result = handler.get_pending_requests()
        assert result["count"] == 0

    def test_includes_step_info(
        self,
        handler_with_request: tuple[HITLHandler, str],
        sample_step_info: dict[str, Any],
    ) -> None:
        handler, _ = handler_with_request
        result = handler.get_pending_requests()
        assert result["pending"][0]["step_info"] == sample_step_info


# ---------------------------------------------------------------------------
# Edge cases — AC-6
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases — init validation, invalid IDs, boundary values."""

    def test_negative_timeout_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="timeout_minutes must be >= 0"):
            HITLHandler(timeout_minutes=-1)

    def test_bool_timeout_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="timeout_minutes must be an int"):
            HITLHandler(timeout_minutes=True)  # type: ignore[arg-type]

    def test_float_timeout_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="timeout_minutes must be an int"):
            HITLHandler(timeout_minutes=5.5)  # type: ignore[arg-type]

    def test_string_timeout_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="timeout_minutes must be an int"):
            HITLHandler(timeout_minutes="5")  # type: ignore[arg-type]

    def test_zero_timeout_no_timeout_at_in_past(
        self,
        zero_timeout_handler: HITLHandler,
        sample_step_info: dict[str, Any],
    ) -> None:
        result = zero_timeout_handler.request_approval(sample_step_info)
        # timeout_at is set to now (immediate timeout)
        timeout_dt = datetime.fromisoformat(result["timeout_at"])
        assert timeout_dt <= datetime.now(timezone.utc)

    def test_two_requests_unique_ids(
        self,
        handler: HITLHandler,
        sample_step_info: dict[str, Any],
    ) -> None:
        r1 = handler.request_approval(sample_step_info)
        r2 = handler.request_approval(sample_step_info)
        assert r1["request_id"] != r2["request_id"]

    def test_multiple_pending_requests(
        self,
        handler: HITLHandler,
        sample_step_info: dict[str, Any],
    ) -> None:
        for _ in range(3):
            handler.request_approval(sample_step_info)
        result = handler.get_pending_requests()
        assert result["count"] == 3


# ---------------------------------------------------------------------------
# Integration — Story 5-4 escalated signal (AC-8)
# ---------------------------------------------------------------------------


class TestIntegration:
    """Story 5-4 entegrasyonu — RetryHandler.escalated → request_approval."""

    def test_step_info_from_retry_errors_format(
        self,
        handler: HITLHandler,
    ) -> None:
        """RetryHandler.escalated error list formatı HITL step_info'ya uygun."""
        retry_errors = [
            {"attempt": 1, "score": 20.0, "error": "fail", "type": None},
            {"attempt": 2, "score": 25.0, "error": "fail again", "type": None},
        ]
        step_info = {
            "step": 2,
            "dimension": "structural",
            "attempts": 2,
            "errors": retry_errors,
            "last_result": {"score": 25.0, "passed": False},
        }
        result = handler.request_approval(step_info)
        rid = result["request_id"]
        check = handler.check_approval(rid)
        assert check["status"] == "pending"
        # Verify round-trip — step_info stored correctly
        record = handler._pending[rid]
        assert record["step_info"]["errors"] == retry_errors
