"""Cumulative drift detection — weighted structural + text diff.

Monitors cumulative drift across pipeline steps. If the accumulated
drift exceeds a configurable threshold the pipeline is blocked.
"""

import copy
import math
from typing import Any


class DriftDetector:
    """Detect cumulative drift across pipeline steps.

    Attributes:
        threshold: Max allowed cumulative drift percentage (default 15.0).
    """

    def __init__(self, threshold: float = 15.0) -> None:
        """Initialize with drift threshold.

        Args:
            threshold: Maximum cumulative drift before blocking (0-100).

        Raises:
            TypeError: If *threshold* is not a numeric type.
            ValueError: If *threshold* is NaN, Inf, or outside [0, 100].
        """
        self._validate_threshold(threshold)
        self.threshold = float(threshold)
        self._cumulative_drift: float = 0.0
        self._drift_history: list[dict[str, Any]] = []
        self._origin_step: int | None = None

    # ------------------------------------------------------------------
    # Threshold validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_threshold(value: object) -> None:
        """Validate threshold value (private helper)."""
        if isinstance(value, bool):
            raise TypeError(
                f"threshold must be a number, got bool"
            )
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"threshold must be a number, got {type(value).__name__}"
        )
        if not math.isfinite(value):
            raise ValueError(
                f"threshold must be finite, got {value}"
            )
        if value < 0.0 or value > 100.0:
            raise ValueError(
                f"threshold must be in [0, 100], got {value}"
            )

    # ------------------------------------------------------------------
    # Number validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_numeric(value: object, name: str) -> float:
        """Validate that *value* is a finite numeric type.

        Returns:
            The value as ``float``.

        Raises:
            TypeError: If *value* is not int or float (or is bool).
            ValueError: If *value* is NaN or Inf.
        """
        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be a number, got bool"
            )
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{name} must be a number, got {type(value).__name__}"
            )
        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite, got {value}"
            )
        return float(value)

    @staticmethod
    def _validate_diff_result(value: object) -> dict[str, Any]:
        """Validate that *value* is a dict with expected diff keys.

        Returns:
            The dict if valid.

        Raises:
            TypeError: If *value* is not a dict.
            ValueError: If required keys are missing or inner values are not dicts.
        """
        if not isinstance(value, dict):
            raise TypeError(
                f"diff_result must be a dict, got {type(value).__name__}"
            )
        required = {"total_changes", "added", "removed", "changed"}
        missing = required - set(value.keys())
        if missing:
            keys_str = ", ".join(sorted(missing))
            raise ValueError(
                f"diff_result missing required key(s): {keys_str}"
            )
        # Validate inner dict values
        for key in ("added", "removed", "changed"):
            if not isinstance(value.get(key), dict):
                raise TypeError(
                    f"diff_result['{key}'] must be a dict, "
                    f"got {type(value.get(key)).__name__}"
                )
        return value

    @staticmethod
    def _compute_text_dissimilarity(
        changed: dict[str, dict[str, Any]],
    ) -> float:
        """Compute average text dissimilarity for string changes.

        Uses token overlap ratio: 1.0 - (|old∩new| / max(|old|, |new|, 1))
        """
        dissimilarities: list[float] = []
        for key, change in changed.items():
            if change.get("type") == "str":
                old_text: str = str(change.get("old", ""))
                new_text: str = str(change.get("new", ""))
                old_tokens = set(old_text.split()) if old_text else set()
                new_tokens = set(new_text.split()) if new_text else set()
                if not old_tokens and not new_tokens:
                    dissimilarities.append(0.0)
                else:
                    overlap = len(old_tokens & new_tokens)
                    denom = max(len(old_tokens), len(new_tokens), 1)
                    dissimilarities.append(1.0 - (overlap / denom))
        if not dissimilarities:
            return 0.0
        return sum(dissimilarities) / len(dissimilarities)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, diff_result: dict[str, Any]) -> dict[str, Any]:
        """Calculate per-step drift from a SnapshotManager.diff() result.

        Args:
            diff_result: Dict with keys ``added``, ``removed``,
                ``changed``, ``total_changes``.

        Returns:
            dict with keys ``drift_score`` (0-100), ``structural_drift``,
            ``text_dissimilarity``, and ``total_changes``.

        Raises:
            TypeError: If *diff_result* is not a dict.
            ValueError: If required keys are missing.
        """
        dr = self._validate_diff_result(diff_result)

        # Structural drift
        n_added = len(dr.get("added", {}))
        n_removed = len(dr.get("removed", {}))
        n_changed = len(dr.get("changed", {}))
        structural = n_added * 2.5 + n_removed * 2.5 + n_changed * 5.0

        # Text dissimilarity
        text_dissim = self._compute_text_dissimilarity(dr.get("changed", {}))
        text_penalty = text_dissim * 10.0

        # Composite
        drift = structural + text_penalty

        # NaN/Inf guard before clamp
        if not math.isfinite(drift):
            drift = 0.0
        drift = max(0.0, min(100.0, drift))

        return {
            "drift_score": drift,
            "structural_drift": structural,
            "text_dissimilarity": text_penalty,
            "total_changes": dr.get("total_changes", 0),
        }

    def calculate_from_snapshots(
        self,
        prev_snapshot: dict[str, Any],
        curr_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate drift between two snapshot data dicts directly.

        Performs inline diff (same logic as SnapshotManager.diff —
        timestamp exclusion, key comparison) then delegates to
        ``calculate()``.

        Args:
            prev_snapshot: Earlier snapshot data.
            curr_snapshot: Later snapshot data.

        Returns:
            Same format as ``calculate()``.

        Raises:
            TypeError: If either snapshot is not a dict.
        """
        if not isinstance(prev_snapshot, dict):
            raise TypeError(
                f"prev_snapshot must be a dict, got {type(prev_snapshot).__name__}"
            )
        if not isinstance(curr_snapshot, dict):
            raise TypeError(
                f"curr_snapshot must be a dict, got {type(curr_snapshot).__name__}"
            )

        # Inline diff (mirrors SnapshotManager.diff logic)
        excluded = {"_timestamp"}
        keys_a = set(prev_snapshot.keys()) - excluded
        keys_b = set(curr_snapshot.keys()) - excluded

        added_keys = keys_b - keys_a
        removed_keys = keys_a - keys_b
        common_keys = keys_a & keys_b

        added = {k: curr_snapshot[k] for k in added_keys}
        removed = {k: prev_snapshot[k] for k in removed_keys}
        changed = {}
        for k in common_keys:
            if prev_snapshot[k] != curr_snapshot[k]:
                changed[k] = {
                    "old": prev_snapshot[k],
                    "new": curr_snapshot[k],
                    "type": type(prev_snapshot[k]).__name__,
                }

        total = len(added) + len(removed) + len(changed)
        diff_result = {
            "added": added,
            "removed": removed,
            "changed": changed,
            "total_changes": total,
        }
        return self.calculate(diff_result)

    def accumulate(self, step_drift: float) -> None:
        """Accumulate a step drift value into cumulative drift.

        Args:
            step_drift: Drift value for this step.

        Raises:
            TypeError: If *step_drift* is not numeric.
            ValueError: If *step_drift* is NaN or Inf.
        """
        validated = self._validate_numeric(step_drift, "step_drift")
        step = len(self._drift_history) + 1

        self._cumulative_drift += validated

        self._drift_history.append({
            "step": step,
            "drift": validated,
            "cumulative": self._cumulative_drift,
        })

        # Origin step tracking: first positive drift
        if self._origin_step is None and validated > 0:
            self._origin_step = step

    def check(self, cumulative_drift: float) -> dict[str, Any]:
        """Evaluate whether drift exceeds the threshold.

        Args:
            cumulative_drift: Accumulated drift value (0-100 expected).

        Returns:
            dict with keys ``drift_detected``, ``cumulative_drift``,
            ``threshold``, ``decision`` ("allow" | "block").

        Raises:
            TypeError: If *cumulative_drift* is not numeric.
            ValueError: If *cumulative_drift* is NaN or Inf.
        """
        validated = self._validate_numeric(cumulative_drift, "cumulative_drift")
        clamped = max(0.0, min(100.0, validated))
        drift_detected = clamped >= self.threshold

        return {
            "drift_detected": drift_detected,
            "cumulative_drift": clamped,
            "threshold": self.threshold,
            "decision": "block" if drift_detected else "allow",
        }

    def get_report(self) -> dict[str, Any]:
        """Return full drift report with history and origin step.

        Returns:
            dict with keys ``cumulative_drift``, ``threshold``,
            ``drift_detected``, ``origin_step``, ``drift_history``,
            ``threshold_exceeded``.
        """
        threshold_exceeded = self._cumulative_drift >= self.threshold
        return {
            "cumulative_drift": self._cumulative_drift,
            "threshold": self.threshold,
            "drift_detected": threshold_exceeded,
            "origin_step": self._origin_step,
            "drift_history": copy.deepcopy(self._drift_history),
            "threshold_exceeded": threshold_exceeded,
        }

    def reset(self) -> None:
        """Reset cumulative drift, history, and origin step."""
        self._cumulative_drift = 0.0
        self._drift_history = []
        self._origin_step = None
