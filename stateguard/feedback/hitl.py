"""Human-in-the-loop — approval request with configurable timeout.

When automatic retries are exhausted the pipeline escalates to a
human operator who can approve, deny, or let the request time out.
"""

import copy
import time
from datetime import datetime, timedelta, timezone
from typing import Any


class HITLHandler:
    """Manage human-in-the-loop approval requests.

    Attributes:
        timeout_minutes: Max wait time for human response (default 5).
    """

    def __init__(self, timeout_minutes: int = 5) -> None:
        """Initialize with timeout setting.

        Args:
            timeout_minutes: Minutes before automatic fail-close.

        Raises:
            TypeError: If *timeout_minutes* is not an int or is a bool.
            ValueError: If *timeout_minutes* is negative.
        """
        if isinstance(timeout_minutes, bool):
            raise TypeError(
                f"timeout_minutes must be an int, got bool"
            )
        if not isinstance(timeout_minutes, int):
            raise TypeError(
                f"timeout_minutes must be an int, "
                f"got {type(timeout_minutes).__name__}"
            )
        if timeout_minutes < 0:
            raise ValueError(
                f"timeout_minutes must be >= 0, got {timeout_minutes}"
            )

        self.timeout_minutes = timeout_minutes
        self._pending: dict[str, dict[str, Any]] = {}
        self._request_counter: int = 0

    def request_approval(self, step_info: dict[str, Any]) -> dict[str, Any]:
        """Submit an approval request.

        Args:
            step_info: Dict with step, dimension, attempt count,
                error details.

        Returns:
            dict with keys ``request_id``, ``status``, ``timeout_at``,
            ``timeout_minutes``.

        Raises:
            TypeError: If *step_info* is not a dict.
        """
        if not isinstance(step_info, dict):
            raise TypeError(
                f"step_info must be a dict, "
                f"got {type(step_info).__name__}"
            )

        now = datetime.now(timezone.utc)

        # Build unique request ID
        base_id = f"hitl_{int(time.time() * 1000)}"
        request_id = base_id
        while request_id in self._pending:
            self._request_counter += 1
            request_id = f"{base_id}_{self._request_counter}"

        # Build timeout timestamp
        if self.timeout_minutes > 0:
            timeout_dt = now + timedelta(minutes=self.timeout_minutes)
        else:
            timeout_dt = now  # immediate timeout

        record = {
            "step_info": copy.deepcopy(step_info),
            "status": "pending",
            "created_at": now.isoformat(),
            "timeout_at": timeout_dt.isoformat(),
            "approved": None,
            "reason": None,
            "decision_time": None,
        }

        self._pending[request_id] = record
        return {
            "request_id": request_id,
            "status": "pending",
            "timeout_at": record["timeout_at"],
            "timeout_minutes": self.timeout_minutes,
        }

    def check_approval(self, request_id: str) -> dict[str, Any]:
        """Check the status of an approval request.

        Args:
            request_id: The request ID from request_approval().

        Returns:
            dict with keys ``request_id``, ``status``, ``approved``,
            ``reason``, ``decision_time``.

        Raises:
            KeyError: If *request_id* is not found.
        """
        if request_id not in self._pending:
            raise KeyError(f"Approval request not found: {request_id}")

        record = self._pending[request_id]

        # Auto-timeout check
        if record["status"] == "pending" and record["timeout_at"] is not None:
            timeout_dt = datetime.fromisoformat(record["timeout_at"])
            if datetime.now(timezone.utc) >= timeout_dt:
                record["approved"] = False
                record["reason"] = "timeout"
                record["status"] = "decided"
                record["decision_time"] = datetime.now(timezone.utc).isoformat()

        return {
            "request_id": request_id,
            "status": record["status"],
            "approved": record["approved"],
            "reason": record["reason"],
            "decision_time": record["decision_time"],
        }

    def resolve_approval(
        self,
        request_id: str,
        approved: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a pending approval request.

        Args:
            request_id: The request ID to resolve.
            approved: True to approve, False to reject.
            reason: Optional human-readable reason.

        Returns:
            dict with keys ``request_id``, ``status``, ``approved``,
            ``reason``, ``decision_time``.

        Raises:
            KeyError: If *request_id* is not found.
            TypeError: If *approved* is not a bool.
            ValueError: If the request has already timed out.
        """
        if request_id not in self._pending:
            raise KeyError(f"Approval request not found: {request_id}")

        if not isinstance(approved, bool):
            raise TypeError(
                f"approved must be a bool, got {type(approved).__name__}"
            )

        if not isinstance(reason, (str, type(None))):
            raise TypeError(
                f"reason must be a str or None, got {type(reason).__name__}"
            )

        record = self._pending[request_id]

        # Timeout guard — cannot override a timed-out request
        if record["status"] == "decided" and record.get("reason") == "timeout":
            raise ValueError(
                f"Cannot resolve request {request_id}: already timed out"
            )

        # Idempotent resolve — already decided
        if record["status"] == "decided":
            return self.check_approval(request_id)

        record["approved"] = approved
        record["reason"] = reason
        record["status"] = "decided"
        record["decision_time"] = datetime.now(timezone.utc).isoformat()

        return {
            "request_id": request_id,
            "status": "decided",
            "approved": approved,
            "reason": reason,
            "decision_time": record["decision_time"],
        }

    def on_timeout(self) -> dict[str, Any]:
        """Default behaviour when human does not respond in time.

        Returns:
            dict with ``approved=False`` (fail-close).
        """
        return {"approved": False, "reason": "timeout"}

    def get_pending_requests(self) -> dict[str, Any]:
        """List all pending approval requests.

        Returns:
            dict with keys ``pending`` (list) and ``count`` (int).
        """
        now = datetime.now(timezone.utc)
        pending_list: list[dict[str, Any]] = []

        for req_id, record in self._pending.items():
            if record["status"] != "pending":
                continue

            timeout_dt = datetime.fromisoformat(record["timeout_at"])

            # Auto-timeout expired requests
            if now >= timeout_dt:
                record["approved"] = False
                record["reason"] = "timeout"
                record["status"] = "decided"
                record["decision_time"] = now.isoformat()
                continue

            remaining = max(0.0, (timeout_dt - now).total_seconds())

            pending_list.append({
                "request_id": req_id,
                "step_info": copy.deepcopy(record["step_info"]),
                "created_at": record["created_at"],
                "timeout_at": record["timeout_at"],
                "remaining_seconds": remaining,
            })

        return {"pending": pending_list, "count": len(pending_list)}
