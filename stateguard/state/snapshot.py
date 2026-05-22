"""Snapshot management — JSON serialization and structural diff.

Captures pipeline state at each step and provides diffing utilities
for cumulative drift detection.
"""

import copy
import json
import time
from datetime import datetime, timezone
from typing import Any


class SnapshotManager:
    """Create and compare pipeline state snapshots."""

    MAX_SNAPSHOT_BYTES = 1_048_576  # 1 MB
    MAX_SNAPSHOTS = 1024

    def __init__(self) -> None:
        """Initialize snapshot storage."""
        self._snapshots: dict[str, dict[str, Any]] = {}

    def take_snapshot(self, state_data: dict[str, Any]) -> dict[str, Any]:
        """Serialize state data with timestamp.

        Args:
            state_data: Pipeline state to snapshot (must be a dict).

        Returns:
            dict with keys ``snapshot_id`` and ``data`` (includes
            ``_timestamp`` plus all original keys).

        Raises:
            TypeError: If *state_data* is not a dict.
            ValueError: If serialized JSON exceeds MAX_SNAPSHOT_BYTES.
        """
        if not isinstance(state_data, dict):
            raise TypeError(
                f"state_data must be a dict, got {type(state_data).__name__}"
            )

        # --- timestamp overwrite guard ---
        if "_timestamp" in state_data:
            raise ValueError(
                "state_data must not contain reserved key '_timestamp'"
            )

        data = copy.deepcopy(state_data)
        data["_timestamp"] = datetime.now(timezone.utc).isoformat()

        # --- size guard (catch circular refs and non-serializable gracefully) ---
        try:
            raw_size = len(json.dumps(data))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(
                f"state_data cannot be serialized: {exc}"
            ) from exc

        if raw_size > self.MAX_SNAPSHOT_BYTES:
            raise ValueError(
                f"Snapshot exceeds {self.MAX_SNAPSHOT_BYTES} byte limit "
                f"({raw_size} bytes)"
            )

        # --- unique ID with collision handling ---
        base_id = f"snap_{int(time.time() * 1000)}"
        snapshot_id = base_id
        counter = 0
        while snapshot_id in self._snapshots:
            counter += 1
            snapshot_id = f"{base_id}_{counter}"

        # --- FIFO eviction ---
        if len(self._snapshots) >= self.MAX_SNAPSHOTS:
            oldest = next(iter(self._snapshots))
            del self._snapshots[oldest]

        # Store a deep copy so mutation of returned data cannot corrupt internal state
        stored = copy.deepcopy(data)
        self._snapshots[snapshot_id] = stored
        return {"snapshot_id": snapshot_id, "data": data}

    def diff(
        self,
        snapshot_a: dict[str, Any],
        snapshot_b: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare two snapshots and produce a diff report.

        Args:
            snapshot_a: Earlier snapshot.
            snapshot_b: Later snapshot.

        Returns:
            dict with keys ``added``, ``removed``, ``changed``,
            and ``total_changes``.

        Raises:
            TypeError: If either argument is not a dict.
        """
        if not isinstance(snapshot_a, dict):
            raise TypeError(
                f"snapshot_a must be a dict, got {type(snapshot_a).__name__}"
            )
        if not isinstance(snapshot_b, dict):
            raise TypeError(
                f"snapshot_b must be a dict, got {type(snapshot_b).__name__}"
            )

        keys_a = set(snapshot_a.keys())
        keys_b = set(snapshot_b.keys())

        # Exclude timestamp from diff
        excluded = {"_timestamp"}
        keys_a -= excluded
        keys_b -= excluded

        added_keys = keys_b - keys_a
        removed_keys = keys_a - keys_b
        common_keys = keys_a & keys_b

        added = {k: snapshot_b[k] for k in added_keys}
        removed = {k: snapshot_a[k] for k in removed_keys}
        changed = {}
        for k in common_keys:
            if snapshot_a[k] != snapshot_b[k]:
                changed[k] = {
                    "old": snapshot_a[k],
                    "new": snapshot_b[k],
                    "type": type(snapshot_a[k]).__name__,
                }

        total = len(added) + len(removed) + len(changed)
        return {
            "added": added,
            "removed": removed,
            "changed": changed,
            "total_changes": total,
        }
