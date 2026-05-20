"""Structured-logging adapter — structlog-based logging utilities.

Provides :class:`StructLogAdapter` that configures *structlog* for the
StateGuard framework, adding context (dimension, tier, request ID) to
every log entry automatically.
"""

from typing import Any


class StructLogAdapter:
    """Central logging adapter for structured logging with *structlog*.

    Usage::

        log = StructLogAdapter("stateguard.dimensions.semantic")
        log.info("validation_complete", score=0.92, passed=True)

    By default the adapter outputs JSON-formatted records to stderr,
    making them suitable for ingestion by log aggregators (ELK,
    Datadog, etc.).
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialise the log adapter.

        Args:
            name: Logger name (typically ``__name__``).  If ``None`` the
                  root *structlog* logger is used.
        """
        self._name = name or "stateguard"
        self._logger: Any = None  # structlog.BoundLoggerLazyProxy

    @property
    def name(self) -> str:
        """Logger name."""
        return self._name

    @classmethod
    def configure(
        cls,
        level: str = "INFO",
        json_format: bool = True,
        **kwargs: Any,
    ) -> None:
        """Globally configure the *structlog* library.

        This should be called once at application startup.

        Args:
            level:       Minimum log level (``"DEBUG"``, ``"INFO"``, etc.).
            json_format: Whether to emit JSON or console-coloured output.
            **kwargs:    Additional *structlog* processor configuration.
        """
        ...

    def bind(self, **kwargs: Any) -> "StructLogAdapter":
        """Return a new adapter with *kwargs* bound as log context.

        Args:
            **kwargs: Key-value pairs to add to every subsequent log event.

        Returns:
            A new :class:`StructLogAdapter` with the additional context.
        """
        ...

    def debug(self, event: str, **kwargs: Any) -> None:
        """Emit a DEBUG-level log message."""
        ...

    def info(self, event: str, **kwargs: Any) -> None:
        """Emit an INFO-level log message."""
        ...

    def warning(self, event: str, **kwargs: Any) -> None:
        """Emit a WARNING-level log message."""
        ...

    def error(self, event: str, **kwargs: Any) -> None:
        """Emit an ERROR-level log message."""
        ...

    def critical(self, event: str, **kwargs: Any) -> None:
        """Emit a CRITICAL-level log message."""
        ...
