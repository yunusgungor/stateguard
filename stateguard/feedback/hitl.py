"""Human-in-the-loop — approval request with configurable timeout.

When automatic retries are exhausted the pipeline escalates to a
human operator who can approve, deny, or let the request time out.
"""

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
        """
        self.timeout_minutes = timeout_minutes

    def request_approval(self, step_info: dict[str, Any]) -> dict[str, Any]:
        """Send approval request and wait for human decision.

        Args:
            step_info: Dict with step, dimension, attempt count, error details.

        Returns:
            dict with keys ``approved`` (bool), ``reason`` (str | None).
        """
        pass

    def on_timeout(self) -> dict[str, Any]:
        """Default behaviour when human does not respond in time.

        Returns:
            dict with ``approved=False`` (fail-close).
        """
        pass
