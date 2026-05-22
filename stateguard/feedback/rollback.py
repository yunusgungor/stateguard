"""Pipeline rollback — restore system to last safe snapshot.

When an error cascade is detected the pipeline can be rewound to a
previously recorded healthy snapshot.
"""

import copy
import json
from datetime import datetime, timezone
from typing import Any


class RollbackHandler:
    """Manage pipeline rollback to a safe snapshot."""

    def __init__(self) -> None:
        """Initialize rollback handler."""
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._safe_snapshots: list[str] = []
        self._rollback_history: list[dict[str, Any]] = []

    def register_snapshot(
        self,
        snapshot_id: str,
        snapshot_data: dict[str, Any],
        is_safe: bool = False,
    ) -> dict[str, Any]:
        """Register a snapshot for potential rollback.

        Args:
            snapshot_id: Unique snapshot identifier.
            snapshot_data: Pipeline state data.
            is_safe: Whether this snapshot passed validation.

        Returns:
            dict with keys ``snapshot_id``, ``is_safe``, ``registered_at``.

        Raises:
            TypeError: If any argument has an invalid type.
            ValueError: If *snapshot_id* is empty or whitespace-only.
        """
        if not isinstance(snapshot_id, str):
            raise TypeError(
                "snapshot_id must be a str, got "
                + type(snapshot_id).__name__
            )
        if not snapshot_id.strip():
            raise ValueError(
                "snapshot_id must not be empty or whitespace-only"
            )
        if not isinstance(snapshot_data, dict):
            raise TypeError(
                "snapshot_data must be a dict, got "
                + type(snapshot_data).__name__
            )
        if not isinstance(is_safe, bool):
            raise TypeError(
                "is_safe must be a bool, got "
                + type(is_safe).__name__
            )

        # Clean up stale safe tracking if this is an overwrite
        if snapshot_id in self._snapshots and not is_safe:
            self._safe_snapshots = [
                s for s in self._safe_snapshots if s != snapshot_id
            ]

        record = {
            "data": copy.deepcopy(snapshot_data),
            "is_safe": is_safe,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        self._snapshots[snapshot_id] = record

        if is_safe and snapshot_id not in self._safe_snapshots:
            self._safe_snapshots.append(snapshot_id)

        return {
            "snapshot_id": snapshot_id,
            "is_safe": is_safe,
            "registered_at": record["registered_at"],
        }

    def mark_safe(self, snapshot_id: str) -> dict[str, Any]:
        """Mark an existing snapshot as safe.

        Args:
            snapshot_id: The snapshot to mark as safe.

        Returns:
            dict with keys ``snapshot_id``, ``is_safe``.

        Raises:
            KeyError: If *snapshot_id* is not found.
            TypeError: If *snapshot_id* is not a str.
        """
        if not isinstance(snapshot_id, str):
            raise TypeError(
                "snapshot_id must be a str, got "
                + type(snapshot_id).__name__
            )
        if snapshot_id not in self._snapshots:
            raise KeyError("Snapshot not found: " + snapshot_id)

        self._snapshots[snapshot_id]["is_safe"] = True
        if snapshot_id not in self._safe_snapshots:
            self._safe_snapshots.append(snapshot_id)

        return {"snapshot_id": snapshot_id, "is_safe": True}

    def rollback(self, snapshot_id: str) -> dict[str, Any]:
        """Restore pipeline state to the given snapshot.

        Args:
            snapshot_id: Identifier of the snapshot to restore.

        Returns:
            dict with keys ``success``, ``snapshot_id``,
            ``restored_data``, ``rolled_back_at``.

        Raises:
            KeyError: If *snapshot_id* is not found.
            TypeError: If *snapshot_id* is not a str.
        """
        if not isinstance(snapshot_id, str):
            raise TypeError(
                "snapshot_id must be a str, got "
                + type(snapshot_id).__name__
            )
        if snapshot_id not in self._snapshots:
            raise KeyError("Snapshot not found: " + snapshot_id)

        record = self._snapshots[snapshot_id]
        restored_data = copy.deepcopy(record["data"])
        now = datetime.now(timezone.utc).isoformat()

        self._rollback_history.append({
            "snapshot_id": snapshot_id,
            "rolled_back_at": now,
            "snapshot_timestamp": record["data"].get("_timestamp"),
            "was_safe": record["is_safe"],
        })

        return {
            "success": True,
            "snapshot_id": snapshot_id,
            "restored_data": restored_data,
            "rolled_back_at": now,
        }

    def rollback_to_last_safe(self) -> dict[str, Any]:
        """Roll back to the most recent safe snapshot.

        Returns:
            Same format as ``rollback()`` with extra
            ``rolled_back_to_last_safe: True``.

        Raises:
            ValueError: If no safe snapshots are available.
        """
        if not self._safe_snapshots:
            raise ValueError("No safe snapshots available")

        last_safe = self._safe_snapshots[-1]
        result = self.rollback(last_safe)
        result["rolled_back_to_last_safe"] = True
        return result

    def get_rollback_history(self) -> dict[str, Any]:
        """Get the rollback history.

        Returns:
            dict with keys ``rollbacks`` (list) and ``count`` (int).
        """
        return {
            "rollbacks": copy.deepcopy(self._rollback_history),
            "count": len(self._rollback_history),
        }

    def list_snapshots(self, safe_only: bool = False) -> dict[str, Any]:
        """List registered snapshots.

        Args:
            safe_only: If True, only return safe snapshots.

        Returns:
            dict with keys ``snapshots`` (list) and ``count`` (int).
        """
        snapshot_list: list[dict[str, Any]] = []
        for sid, record in self._snapshots.items():
            if safe_only and not record["is_safe"]:
                continue
            data_size = len(json.dumps(record["data"], default=str))
            snapshot_list.append({
                "snapshot_id": sid,
                "is_safe": record["is_safe"],
                "registered_at": record["registered_at"],
                "size_bytes": data_size,
            })

        return {"snapshots": snapshot_list, "count": len(snapshot_list)}
