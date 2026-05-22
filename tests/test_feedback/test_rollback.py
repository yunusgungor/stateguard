"""Tests for RollbackHandler — pipeline rollback to safe snapshots."""

from __future__ import annotations

from typing import Any

import pytest

from stateguard.feedback.rollback import RollbackHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler() -> RollbackHandler:
    return RollbackHandler()


@pytest.fixture
def sample_snapshot_data() -> dict[str, Any]:
    return {
        "prompt": "test prompt",
        "context": {"key": "value"},
        "partial_output": "output",
        "validation_scores": {"structural": 95.0},
        "step_counter": 3,
        "_timestamp": "2026-05-21T12:00:00+00:00",
    }


@pytest.fixture
def handler_with_snapshots(
    handler: RollbackHandler,
    sample_snapshot_data: dict[str, Any],
) -> RollbackHandler:
    handler.register_snapshot("snap_1", sample_snapshot_data, is_safe=True)
    handler.register_snapshot("snap_2", {"step": 2}, is_safe=False)
    handler.register_snapshot("snap_3", {"step": 3}, is_safe=True)
    return handler


@pytest.fixture
def handler_with_rollback(
    handler_with_snapshots: RollbackHandler,
) -> RollbackHandler:
    handler_with_snapshots.rollback("snap_1")
    return handler_with_snapshots


# ---------------------------------------------------------------------------
# register_snapshot — AC-1
# ---------------------------------------------------------------------------


class TestRegisterSnapshot:
    """register_snapshot() — input validation, record structure."""

    def test_registers_with_default_not_safe(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        result = handler.register_snapshot("snap_1", sample_snapshot_data)
        assert result["snapshot_id"] == "snap_1"
        assert result["is_safe"] is False
        assert "registered_at" in result

    def test_registers_with_is_safe(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        result = handler.register_snapshot("snap_1", sample_snapshot_data, is_safe=True)
        assert result["is_safe"] is True

    def test_record_stored_internally(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        handler.register_snapshot("snap_1", sample_snapshot_data)
        assert "snap_1" in handler._snapshots

    def test_non_string_id_raises_type_error(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        with pytest.raises(TypeError, match="snapshot_id must be a str"):
            handler.register_snapshot(123, sample_snapshot_data)  # type: ignore[arg-type]

    def test_none_id_raises_type_error(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        with pytest.raises(TypeError, match="snapshot_id must be a str"):
            handler.register_snapshot(None, sample_snapshot_data)  # type: ignore[arg-type]

    def test_empty_id_raises_value_error(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        with pytest.raises(ValueError, match="snapshot_id must not be empty"):
            handler.register_snapshot("", sample_snapshot_data)

    def test_non_dict_data_raises_type_error(
        self,
        handler: RollbackHandler,
    ) -> None:
        with pytest.raises(TypeError, match="snapshot_data must be a dict"):
            handler.register_snapshot("s1", "bad_data")  # type: ignore[arg-type]

    def test_none_data_raises_type_error(
        self,
        handler: RollbackHandler,
    ) -> None:
        with pytest.raises(TypeError, match="snapshot_data must be a dict"):
            handler.register_snapshot("s1", None)  # type: ignore[arg-type]

    def test_non_bool_is_safe_raises_type_error(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        with pytest.raises(TypeError, match="is_safe must be a bool"):
            handler.register_snapshot("s1", sample_snapshot_data, is_safe="yes")  # type: ignore[arg-type]

    def test_overwrite_existing(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        handler.register_snapshot("s1", {"old": "data"})
        handler.register_snapshot("s1", sample_snapshot_data)
        assert handler._snapshots["s1"]["data"] == sample_snapshot_data

    def test_safe_snapshot_tracked(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        handler.register_snapshot("s1", sample_snapshot_data, is_safe=True)
        assert "s1" in handler._safe_snapshots

    def test_unsafe_snapshot_not_tracked(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        handler.register_snapshot("s1", sample_snapshot_data, is_safe=False)
        assert "s1" not in handler._safe_snapshots


# ---------------------------------------------------------------------------
# mark_safe — AC-2
# ---------------------------------------------------------------------------


class TestMarkSafe:
    """mark_safe() — mark existing, missing, duplicate."""

    def test_marks_existing_snapshot_safe(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        handler.register_snapshot("s1", sample_snapshot_data)
        result = handler.mark_safe("s1")
        assert result["is_safe"] is True
        assert handler._snapshots["s1"]["is_safe"] is True
        assert "s1" in handler._safe_snapshots

    def test_missing_snapshot_raises_key_error(
        self,
        handler: RollbackHandler,
    ) -> None:
        with pytest.raises(KeyError, match="not-found"):
            handler.mark_safe("not-found")

    def test_duplicate_mark_safe(
        self,
        handler: RollbackHandler,
        sample_snapshot_data: dict[str, Any],
    ) -> None:
        handler.register_snapshot("s1", sample_snapshot_data, is_safe=True)
        # Second mark_safe should not raise
        result = handler.mark_safe("s1")
        assert result["is_safe"] is True


# ---------------------------------------------------------------------------
# rollback — AC-3
# ---------------------------------------------------------------------------


class TestRollback:
    """rollback() — success, invalid ID, return format, history."""

    def test_rollback_returns_data(
        self,
        handler_with_snapshots: RollbackHandler,
    ) -> None:
        result = handler_with_snapshots.rollback("snap_1")
        assert result["success"] is True
        assert result["snapshot_id"] == "snap_1"
        assert "restored_data" in result
        assert "rolled_back_at" in result

    def test_rollback_invalid_id_raises_key_error(
        self,
        handler: RollbackHandler,
    ) -> None:
        with pytest.raises(KeyError, match="not-found"):
            handler.rollback("not-found")

    def test_rollback_adds_to_history(
        self,
        handler_with_rollback: RollbackHandler,
    ) -> None:
        history = handler_with_rollback.get_rollback_history()
        assert history["count"] == 1
        assert history["rollbacks"][0]["snapshot_id"] == "snap_1"

    def test_multiple_rollbacks(
        self,
        handler_with_snapshots: RollbackHandler,
    ) -> None:
        handler_with_snapshots.rollback("snap_1")
        handler_with_snapshots.rollback("snap_3")
        history = handler_with_snapshots.get_rollback_history()
        assert history["count"] == 2


# ---------------------------------------------------------------------------
# rollback_to_last_safe — AC-4
# ---------------------------------------------------------------------------


class TestRollbackToLastSafe:
    """rollback_to_last_safe() — success, empty safe list."""

    def test_rollback_to_last_safe(
        self,
        handler_with_snapshots: RollbackHandler,
    ) -> None:
        result = handler_with_snapshots.rollback_to_last_safe()
        assert result["success"] is True
        assert result["rolled_back_to_last_safe"] is True
        # Last safe snapshot is "snap_3"
        assert result["snapshot_id"] == "snap_3"

    def test_empty_safe_list_raises_value_error(
        self,
        handler: RollbackHandler,
    ) -> None:
        with pytest.raises(ValueError, match="No safe snapshots"):
            handler.rollback_to_last_safe()


# ---------------------------------------------------------------------------
# get_rollback_history — AC-5
# ---------------------------------------------------------------------------


class TestGetHistory:
    """get_rollback_history() — empty, after rollbacks."""

    def test_empty_history(
        self,
        handler: RollbackHandler,
    ) -> None:
        result = handler.get_rollback_history()
        assert result["count"] == 0
        assert result["rollbacks"] == []

    def test_after_rollback(
        self,
        handler_with_rollback: RollbackHandler,
    ) -> None:
        result = handler_with_rollback.get_rollback_history()
        assert result["count"] == 1


# ---------------------------------------------------------------------------
# list_snapshots — AC-6
# ---------------------------------------------------------------------------


class TestListSnapshots:
    """list_snapshots() — all, safe_only, empty."""

    def test_list_all(
        self,
        handler_with_snapshots: RollbackHandler,
    ) -> None:
        result = handler_with_snapshots.list_snapshots()
        assert result["count"] == 3

    def test_list_safe_only(
        self,
        handler_with_snapshots: RollbackHandler,
    ) -> None:
        result = handler_with_snapshots.list_snapshots(safe_only=True)
        assert result["count"] == 2

    def test_list_empty(
        self,
        handler: RollbackHandler,
    ) -> None:
        result = handler.list_snapshots()
        assert result["count"] == 0

    def test_snapshot_has_metadata(
        self,
        handler_with_snapshots: RollbackHandler,
    ) -> None:
        result = handler_with_snapshots.list_snapshots()
        snap = result["snapshots"][0]
        assert "snapshot_id" in snap
        assert "is_safe" in snap
        assert "registered_at" in snap
        assert "size_bytes" in snap


# ---------------------------------------------------------------------------
# Edge cases — AC-7
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases — None/non-string/empty ID, non-dict data, non-bool safe."""

    def test_rollback_nonexistent(self, handler: RollbackHandler) -> None:
        with pytest.raises(KeyError, match="not-found"):
            handler.rollback("not-found")

    def test_mark_safe_nonexistent(self, handler: RollbackHandler) -> None:
        with pytest.raises(KeyError):
            handler.mark_safe("ghost")


# ---------------------------------------------------------------------------
# Integration — Story 5-2 / 5-5 — AC-8
# ---------------------------------------------------------------------------


class TestIntegration:
    """Story 5-2 SnapshotManager + Story 5-5 HITL entegrasyonu."""

    def test_snapshot_round_trip(
        self,
        handler: RollbackHandler,
    ) -> None:
        """Simulate snapshot → validate → mark_safe → rollback flow."""
        snapshot_data = {
            "prompt": "test",
            "step_counter": 1,
            "validation_scores": {"structural": 95.0},
        }
        # Register as unsafe initially
        handler.register_snapshot("snap_1", snapshot_data)
        assert handler._snapshots["snap_1"]["is_safe"] is False

        # Mark safe after validation passes
        handler.mark_safe("snap_1")
        assert handler._snapshots["snap_1"]["is_safe"] is True

        # Rollback to last safe
        result = handler.rollback_to_last_safe()
        assert result["snapshot_id"] == "snap_1"
        assert result["restored_data"]["prompt"] == "test"

    def test_snapshot_data_immutable_after_register(
        self,
        handler: RollbackHandler,
    ) -> None:
        """Registering snapshot should deep-copy, so mutations don't affect stored data."""
        original = {"key": "value", "nested": {"inner": "data"}}
        handler.register_snapshot("s1", original)
        original["key"] = "mutated"
        original["nested"]["inner"] = "also_mutated"
        stored = handler._snapshots["s1"]["data"]
        assert stored["key"] == "value"
        assert stored["nested"]["inner"] == "data"
