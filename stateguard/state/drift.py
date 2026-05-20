"""Cumulative drift detection — weighted cosine + structural diff.

Monitors cumulative drift across pipeline steps. If the accumulated
drift exceeds a configurable threshold the pipeline is blocked.
"""

from typing import Any


class DriftDetector:
    """Detect cumulative drift across pipeline steps.

    Attributes:
        threshold: Max allowed cumulative drift percentage (default 15.0).
    """

    def __init__(self, threshold: float = 15.0) -> None:
        """Initialize with drift threshold.

        Args:
            threshold: Maximum cumulative drift before blocking.
        """
        self.threshold = threshold
        self._cumulative_drift: float = 0.0

    def check(self, cumulative_drift: float) -> dict[str, Any]:
        """Evaluate whether drift exceeds the threshold.

        Args:
            cumulative_drift: Accumulated drift value (0-100).

        Returns:
            dict with keys ``drift_detected``, ``cumulative_drift``,
            ``threshold``, ``decision`` ("allow" | "block").
        """
        pass
