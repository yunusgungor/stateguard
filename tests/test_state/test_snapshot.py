"""Tests for SnapshotManager — snapshot serialization and diff."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import pytest

from stateguard.state.snapshot import SnapshotManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager() -> SnapshotManager:
    return SnapshotManager()


@pytest.fixture
def sample_state() -> dict[str, Any]:
    return {
        "prompt": "What is the capital of France?",
        "context": {"difficulty": "easy", "tags": ["geography"]},
        "partial_output": "Paris is the capital of France.",
        "validation_scores": {"structural": 95.0, "semantic": 88.0},
        "step_counter": 3,
    }

# ---------------------------------------------------------------------------
# take_snapshot — AC-1
# ---------------------------------------------------------------------------


class TestTakeSnapshot:
    """take_snapshot() — timestamp, ID, deep copy, size guard."""

    def test_snapshot_contains_timestamp(
        self, manager: SnapshotManager, sample_state: dict[str, Any],
    ) -> None:
        result = manager.take_snapshot(sample_state)
        assert "_timestamp" in result["data"]
        dt = datetime.fromisoformat(result["data"]["_timestamp"])
        assert dt.tzinfo is not None  # timezone-aware

    def test_snapshot_returns_id_and_data(
        self, manager: SnapshotManager, sample_state: dict[str, Any],
    ) -> None:
        result = manager.take_snapshot(sample_state)
        assert "snapshot_id" in result
        assert isinstance(result["snapshot_id"], str)
        assert "data" in result
        assert isinstance(result["data"], dict)

    def test_snapshot_id_format(self, manager: SnapshotManager, sample_state: dict[str, Any]) -> None:
        result = manager.take_snapshot(sample_state)
        assert result["snapshot_id"].startswith("snap_")

    def test_snapshot_id_unique(
        self, manager: SnapshotManager, sample_state: dict[str, Any],
    ) -> None:
        r1 = manager.take_snapshot(sample_state)
        r2 = manager.take_snapshot(sample_state)
        assert r1["snapshot_id"] != r2["snapshot_id"]

    def test_snapshot_deep_copy(
        self, manager: SnapshotManager, sample_state: dict[str, Any],
    ) -> None:
        result = manager.take_snapshot(sample_state)
        original_prompt = sample_state["prompt"]
        sample_state["prompt"] = "changed"
        assert result["data"]["prompt"] == original_prompt  # unchanged

    def test_snapshot_stores_internally(
        self, manager: SnapshotManager, sample_state: dict[str, Any],
    ) -> None:
        result = manager.take_snapshot(sample_state)
        sid = result["snapshot_id"]
        # Access internal storage via take_snapshot + diff — no getter needed
        assert sid in manager._snapshots

    def test_snapshot_all_keys_preserved(
        self, manager: SnapshotManager, sample_state: dict[str, Any],
    ) -> None:
        result = manager.take_snapshot(sample_state)
        for key in sample_state:
            assert key in result["data"]
        assert "prompt" in result["data"]
        assert "step_counter" in result["data"]

    def test_snapshot_empty_dict(self, manager: SnapshotManager) -> None:
        result = manager.take_snapshot({})
        assert "_timestamp" in result["data"]
        assert result["snapshot_id"].startswith("snap_")

    def test_snapshot_none_input(self, manager: SnapshotManager) -> None:
        with pytest.raises(TypeError, match="state_data must be a dict"):
            manager.take_snapshot(None)  # type: ignore[arg-type]

    def test_snapshot_string_input(self, manager: SnapshotManager) -> None:
        with pytest.raises(TypeError, match="state_data must be a dict"):
            manager.take_snapshot("invalid")  # type: ignore[arg-type]

    def test_snapshot_list_input(self, manager: SnapshotManager) -> None:
        with pytest.raises(TypeError, match="state_data must be a dict"):
            manager.take_snapshot([1, 2, 3])  # type: ignore[arg-type]

    def test_snapshot_size_guard(self, manager: SnapshotManager) -> None:
        """Snapshot exceeding MAX_SNAPSHOT_BYTES raises ValueError."""
        # Create data > 1MB
        large = {"data": "x" * 2_000_000}
        with pytest.raises(ValueError, match="exceeds.*limit"):
            manager.take_snapshot(large)

    def test_snapshot_size_guard_near_limit(self, manager: SnapshotManager) -> None:
        """Snapshot just under 1MB should pass."""
        # ~500K chars ~500KB JSON, well under 1MB limit
        result = manager.take_snapshot({"data": "x" * 500_000})
        assert "snapshot_id" in result

    def test_snapshot_fifo_eviction(self, manager: SnapshotManager) -> None:
        """Max 1024 snapshots — oldest evicted on 1025th."""
        # Fill to 1024
        for i in range(1024):
            manager.take_snapshot({"i": i})
        assert len(manager._snapshots) == 1024

        # Get the oldest ID
        oldest_id = next(iter(manager._snapshots))

        # Add one more
        extra = manager.take_snapshot({"i": 2000})
        assert len(manager._snapshots) == 1024  # still 1024
        assert oldest_id not in manager._snapshots  # oldest evicted
        assert extra["snapshot_id"] in manager._snapshots

    def test_snapshot_id_collision(
        self, manager: SnapshotManager, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Same-millisecond snapshots get _1, _2 suffix."""
        frozen_ms = 1717000000000

        def frozen_time() -> int:
            return frozen_ms

        monkeypatch.setattr(time, "time", lambda: frozen_ms / 1000)
        r1 = manager.take_snapshot({"seq": 1})
        r2 = manager.take_snapshot({"seq": 2})
        assert r1["snapshot_id"] == f"snap_{frozen_ms}"
        assert r2["snapshot_id"] == f"snap_{frozen_ms}_1"

    def test_snapshot_no_mutation_on_error(
        self, manager: SnapshotManager,
    ) -> None:
        """On ValueError, snapshot must not be stored."""
        large = {"x": "y" * 2_000_000}
        with pytest.raises(ValueError):
            manager.take_snapshot(large)
        assert len(manager._snapshots) == 0


# ---------------------------------------------------------------------------
# diff — AC-2
# ---------------------------------------------------------------------------


class TestDiff:
    """diff() — key comparison, timestamp exclusion, type safety."""

    @pytest.fixture
    def snap_a(self) -> dict[str, Any]:
        return {
            "prompt": "hello",
            "result": "world",
            "score": 95.0,
            "_timestamp": "2026-01-01T00:00:00+00:00",
        }

    @pytest.fixture
    def snap_b(self) -> dict[str, Any]:
        return {
            "prompt": "hello",
            "result": "CHANGED",
            "step": 2,
            "_timestamp": "2026-01-02T00:00:00+00:00",
        }

    def test_diff_added(
        self, manager: SnapshotManager, snap_a: dict[str, Any], snap_b: dict[str, Any],
    ) -> None:
        diff = manager.diff(snap_a, snap_b)
        assert "step" in diff["added"]
        assert diff["added"]["step"] == 2

    def test_diff_removed(
        self, manager: SnapshotManager, snap_a: dict[str, Any], snap_b: dict[str, Any],
    ) -> None:
        diff = manager.diff(snap_a, snap_b)
        assert "score" in diff["removed"]
        assert diff["removed"]["score"] == 95.0

    def test_diff_changed(
        self, manager: SnapshotManager, snap_a: dict[str, Any], snap_b: dict[str, Any],
    ) -> None:
        diff = manager.diff(snap_a, snap_b)
        assert "result" in diff["changed"]
        assert diff["changed"]["result"]["old"] == "world"
        assert diff["changed"]["result"]["new"] == "CHANGED"
        assert diff["changed"]["result"]["type"] == "str"

    def test_diff_timestamp_excluded(
        self, manager: SnapshotManager, snap_a: dict[str, Any], snap_b: dict[str, Any],
    ) -> None:
        """_timestamp differences must NOT appear in diff output."""
        diff = manager.diff(snap_a, snap_b)
        assert "_timestamp" not in diff["added"]
        assert "_timestamp" not in diff["removed"]
        assert "_timestamp" not in diff["changed"]

    def test_diff_identical_snapshots(self, manager: SnapshotManager) -> None:
        a = {"x": 1, "y": "hello"}
        diff = manager.diff(a, a.copy())
        assert diff["added"] == {}
        assert diff["removed"] == {}
        assert diff["changed"] == {}
        assert diff["total_changes"] == 0

    def test_diff_total_changes(
        self, manager: SnapshotManager, snap_a: dict[str, Any], snap_b: dict[str, Any],
    ) -> None:
        diff = manager.diff(snap_a, snap_b)
        # added: step (1) + removed: score (1) + changed: result (1) = 3
        assert diff["total_changes"] == 3

    def test_diff_total_changes_empty(self, manager: SnapshotManager) -> None:
        diff = manager.diff({}, {})
        assert diff["total_changes"] == 0

    def test_diff_non_dict_a(self, manager: SnapshotManager) -> None:
        with pytest.raises(TypeError, match="snapshot_a must be a dict"):
            manager.diff("invalid", {})  # type: ignore[arg-type]

    def test_diff_non_dict_b(self, manager: SnapshotManager) -> None:
        with pytest.raises(TypeError, match="snapshot_b must be a dict"):
            manager.diff({}, 42)  # type: ignore[arg-type]

    def test_diff_changed_type_field(
        self, manager: SnapshotManager,
    ) -> None:
        a = {"val": 42}
        b = {"val": "forty-two"}
        diff = manager.diff(a, b)
        assert diff["changed"]["val"]["type"] == "int"  # type of old value


# ---------------------------------------------------------------------------
# Snapshot storage — AC-3
# ---------------------------------------------------------------------------


class TestSnapshotStorage:
    """Snapshot ID uniqueness, internal dict, chronological order."""

    def test_stored_in_internal_dict(
        self, manager: SnapshotManager, sample_state: dict[str, Any],
    ) -> None:
        result = manager.take_snapshot(sample_state)
        assert result["snapshot_id"] in manager._snapshots

    def test_multiple_snapshots_stored(
        self, manager: SnapshotManager,
    ) -> None:
        ids = []
        for i in range(5):
            r = manager.take_snapshot({"i": i})
            ids.append(r["snapshot_id"])
        assert len(manager._snapshots) == 5
        for sid in ids:
            assert sid in manager._snapshots

    def test_chronological_order(
        self, manager: SnapshotManager,
    ) -> None:
        """_snapshots insertion order matches chronological order (Python 3.7+)."""
        for i in range(5):
            manager.take_snapshot({"i": i})
        keys = list(manager._snapshots.keys())
        assert keys == sorted(keys)  # snap_<epoch_ms> timestamps are ascending


# ---------------------------------------------------------------------------
# Edge cases — AC-4
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """None/empty/non-dict, 1MB limit, FIFO eviction, ID collision."""

    def test_size_guard_exact_boundary(self, manager: SnapshotManager) -> None:
        """Verify ValueError at exact boundary."""
        # Each char = 1 byte in JSON, aim for just over 1MB
        big = {"payload": "x" * 1_100_000}
        with pytest.raises(ValueError):
            manager.take_snapshot(big)

    def test_size_guard_well_within(self, manager: SnapshotManager) -> None:
        result = manager.take_snapshot({"msg": "hello"})
        assert "snapshot_id" in result

    def test_fifo_eviction_exact(
        self, manager: SnapshotManager,
    ) -> None:
        """1025th snapshot evicts the 1st."""
        snap_ids = []
        for i in range(1025):
            r = manager.take_snapshot({"i": i})
            snap_ids.append(r["snapshot_id"])
        assert len(manager._snapshots) == 1024
        assert snap_ids[0] not in manager._snapshots  # 1st evicted
        assert snap_ids[-1] in manager._snapshots      # last still there

    def test_fifo_eviction_order_second_batch(
        self, manager: SnapshotManager,
    ) -> None:
        """After eviction, the NEW 1025th entry stays and next-oldest is evictable."""
        ids = []
        for i in range(1026):
            r = manager.take_snapshot({"i": i})
            ids.append(r["snapshot_id"])
        assert ids[0] not in manager._snapshots
        assert ids[1] not in manager._snapshots
        assert ids[-1] in manager._snapshots

    def test_id_collision_two_snapshots(
        self, manager: SnapshotManager, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Same timestamp yields _1 suffix."""
        monkeypatch.setattr(time, "time", lambda: 1717000000.0)
        r1 = manager.take_snapshot({"a": 1})
        r2 = manager.take_snapshot({"b": 2})
        assert r1["snapshot_id"] == "snap_1717000000000"
        assert r2["snapshot_id"] == "snap_1717000000000_1"

    def test_id_collision_three_snapshots(
        self, manager: SnapshotManager, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Three same-timestamp snapshots yield _0, _1, _2."""
        monkeypatch.setattr(time, "time", lambda: 1717000000.0)
        r1 = manager.take_snapshot({"a": 1})
        r2 = manager.take_snapshot({"b": 2})
        r3 = manager.take_snapshot({"c": 3})
        assert r1["snapshot_id"] == "snap_1717000000000"
        assert r2["snapshot_id"] == "snap_1717000000000_1"
        assert r3["snapshot_id"] == "snap_1717000000000_2"

    def test_diff_identical(self, manager: SnapshotManager) -> None:
        a = {"key": "val", "num": 10}
        diff = manager.diff(a, a.copy())
        assert diff["total_changes"] == 0

    def test_diff_both_none(self, manager: SnapshotManager) -> None:
        with pytest.raises(TypeError):
            manager.diff(None, {})  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            manager.diff({}, None)  # type: ignore[arg-type]

    def test_snapshot_mutation_after_error(
        self, manager: SnapshotManager,
    ) -> None:
        """After ValueError, internal storage must be unchanged."""
        manager.take_snapshot({"ok": 1})
        with pytest.raises(ValueError):
            manager.take_snapshot({"x": "y" * 2_000_000})
        # Original snapshot must still be there
        assert len(manager._snapshots) == 1

    def test_snapshot_timestamp_overwrite_guard(self, manager: SnapshotManager) -> None:
        """state_data with reserved '_timestamp' key must be rejected."""
        with pytest.raises(ValueError, match="reserved key.*_timestamp"):
            manager.take_snapshot({"_timestamp": "fake", "data": 1})

    def test_snapshot_circular_ref_handled(self, manager):
        """Circular reference ValueError'a donusur."""
        circular: dict[str, Any] = {"x": 1}
        circular["self"] = circular
        with pytest.raises(ValueError, match="cannot be serialized"):
            manager.take_snapshot(circular)

    def test_snapshot_non_json_key_typeerror(self, manager):
        """JSON-serializable olmayan dict key TypeError -> ValueError donusur."""
        with pytest.raises(ValueError, match="cannot be serialized"):
            manager.take_snapshot({(1, 2): "value"})

    def test_snapshot_frozenset_key_typeerror(self, manager):
        """Frozenset key de TypeError -> ValueError donusur."""
        with pytest.raises(ValueError, match="cannot be serialized"):
            manager.take_snapshot({frozenset([1, 2]): "value"})

    def test_snapshot_non_json_value_handled(self, manager):
        """Non-JSON-serializable value (datetime) ValueError'a donusur."""
        from datetime import datetime
        with pytest.raises(ValueError, match="cannot be serialized"):
            manager.take_snapshot({"date": datetime(2026, 5, 22)})

    def test_returned_data_mutation_does_not_corrupt_internal(
        self, manager: SnapshotManager, sample_state: dict[str, Any],
    ) -> None:
        """Mutating the returned 'data' must NOT affect internally stored snapshot."""
        result = manager.take_snapshot(sample_state)
        sid = result["snapshot_id"]
        # Mutate returned data
        result["data"]["prompt"] = "MUTATED"
        result["data"]["_timestamp"] = "FAKE"
        # Internal storage must remain pristine
        stored = manager._snapshots[sid]
        assert stored["prompt"] == "What is the capital of France?"
        assert stored["_timestamp"] != "FAKE"
