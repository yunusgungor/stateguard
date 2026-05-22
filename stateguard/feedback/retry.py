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

        Raises:
            TypeError: If *max_retries* is not an int or is a bool.
            ValueError: If *max_retries* is negative.
        """
        if isinstance(max_retries, bool):
            raise TypeError(
                f"max_retries must be an int, got bool"
            )
        if not isinstance(max_retries, int):
            raise TypeError(
                f"max_retries must be an int, got {type(max_retries).__name__}"
            )
        if max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {max_retries}"
            )

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
            dict with keys ``result``, ``attempts``, ``errors``,
            ``escalated``.

        Raises:
            TypeError: If *validator_func* is not callable.
            TypeError: If *context* is provided but not a dict or None.
        """
        if not callable(validator_func):
            raise TypeError(
                f"validator_func must be callable, "
                f"got {type(validator_func).__name__}"
            )

        if context is None:
            context = {}
        elif not isinstance(context, dict):
            raise TypeError(
                f"context must be a dict or None, "
                f"got {type(context).__name__}"
            )

        errors: list[dict[str, Any]] = []

        # --- auto-reset for fresh cycle ---
        self.attempt_count = 0

        # --- first attempt (shallow copy context to prevent mutation) ---
        self.attempt_count += 1
        result = self._call_validator(validator_func, output, dict(context))

        if result.get("passed", False):
            return {
                "result": result,
                "attempts": self.attempt_count,
                "errors": errors,
                "escalated": False,
            }

        errors.append(self._make_error_entry(
            self.attempt_count, result
        ))

        # --- retry loop ---
        while self.attempt_count <= self.max_retries:
            feedback_context = dict(context)
            feedback_context["_feedback"] = {
                "attempt": self.attempt_count,
                "max_retries": self.max_retries,
                "previous_error": (
                    errors[-1].get("error")
                    if errors[-1].get("error") is not None
                    else errors[-1].get("details", {})
                ),
                "previous_score": errors[-1].get("score"),
            }

            self.attempt_count += 1
            result = self._call_validator(
                validator_func, output, feedback_context
            )

            if result.get("passed", False):
                return {
                    "result": result,
                    "attempts": self.attempt_count,
                    "errors": errors,
                    "escalated": False,
                }

            errors.append(self._make_error_entry(
                self.attempt_count, result
            ))

        # --- all attempts exhausted ---
        return {
            "result": result,
            "attempts": self.attempt_count,
            "errors": errors,
            "escalated": True,
        }

    def reset(self) -> None:
        """Reset attempt counter for a new pipeline step."""
        self.attempt_count = 0

    def _call_validator(
        self,
        validator_func: Callable,
        output: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Call validator_func and validate return type.

        Args:
            validator_func: Callable to invoke.
            output: Output string to validate.
            context: Context dict.

        Returns:
            Result dict from validator_func.

        Raises:
            TypeError: If validator_func does not return a dict.
        """
        try:
            result = validator_func(output, context)
        except Exception as exc:
            return {
                "passed": False,
                "score": 0.0,
                "error": str(exc),
                "type": type(exc).__name__,
            }

        if not isinstance(result, dict):
            raise TypeError(
                f"validator_func must return a dict, "
                f"got {type(result).__name__}"
            )
        return result

    @staticmethod
    def _make_error_entry(
        attempt: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a standardised error entry from a validation result."""
        return {
            "attempt": attempt,
            "score": result.get("score"),
            "error": result.get("error"),
            "details": result.get("details") or {},
            "type": result.get("type"),
        }
