"""Retry mechanism — auto-retry with feedback accumulation.

Attempts validation up to ``max_retries + 1`` times. Each retry
incorporates the previous error context so the validator can adapt.
"""

from typing import Any, Callable


class RetryHandler:
    """Manage automatic retries for failed validations.

    Attributes:
        max_retries: Maximum number of retry attempts (default 2).
    """

    def __init__(self, max_retries: int = 2) -> None:
        """Initialize with a retry limit.

        Args:
            max_retries: Maximum retries before escalation.
        """
        self.max_retries = max_retries
        self.attempt_count: int = 0

    def execute(
        self,
        validator_func: Callable,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute validation with retries, feeding back error context.

        Args:
            validator_func: Callable that performs validation.
            output:        The output to validate.
            context:       Optional context dict.

        Returns:
            dict with keys ``result``, ``attempts``, ``errors``.
        """
        pass
