"""Structured-logging adapter — structlog-based logging utilities.

Provides :class:`StructLogAdapter` that configures *structlog* for the
StateGuard framework, adding context (dimension, tier, request ID) to
every log entry automatically.
"""

from __future__ import annotations

from typing import Any

# Optional structlog dependency
try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


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
        if not HAS_STRUCTLOG:
            import logging as _logging

            _logging.basicConfig(level=getattr(_logging, level.upper(), _logging.INFO))
            return

        processors: list[Any] = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
        ]

        if json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
            **kwargs,
        )

        # Apply log level on root stdlib logger (required by filter_by_level processor)
        import logging as _stdlib_logging

        _stdlib_logging.getLogger().setLevel(
            getattr(_stdlib_logging, level.upper(), _stdlib_logging.INFO)
        )

    def bind(self, **kwargs: Any) -> StructLogAdapter:
        """Return a new adapter with *kwargs* bound as log context.

        Args:
            **kwargs: Key-value pairs to add to every subsequent log event.

        Returns:
            A new :class:`StructLogAdapter` with the additional context.
        """
        if not HAS_STRUCTLOG:
            import warnings

            warnings.warn(
                "StructLogAdapter.bind() has no effect without structlog installed. "
                "Install structlog for structured context binding."
            )
        new = StructLogAdapter(self._name)
        if HAS_STRUCTLOG:
            new._logger = self._get_logger().bind(**kwargs)
        return new

    def _get_logger(self) -> Any:
        """Get or create the underlying logger instance."""
        if self._logger is None:
            if HAS_STRUCTLOG:
                self._logger = structlog.get_logger(self._name)
            else:
                import logging as _logging

                self._logger = _logging.getLogger(self._name)
        return self._logger

    def debug(self, event: str, **kwargs: Any) -> None:
        """Emit a DEBUG-level log message."""
        self._get_logger().debug(event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        """Emit an INFO-level log message."""
        self._get_logger().info(event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """Emit a WARNING-level log message."""
        self._get_logger().warning(event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Emit an ERROR-level log message."""
        self._get_logger().error(event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        """Emit a CRITICAL-level log message."""
        self._get_logger().critical(event, **kwargs)
