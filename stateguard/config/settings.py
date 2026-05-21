"""Configuration manager — YAML loading and runtime access.

Provides :class:`ConfigManager` that reads a YAML configuration file,
validates it against :class:`~stateguard.config.schema.StateGuardConfig`,
and exposes key-value access for the rest of the framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from stateguard.config.schema import StateGuardConfig

# Default config path resolved relative to this package
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "defaults.yaml"


class ConfigManager:
    """Loads, validates, and provides access to the StateGuard configuration.

    Usage::

        cfg = ConfigManager("stateguard/config/defaults.yaml")
        cfg.load()
        threshold = cfg.get("tier1_threshold", 80.0)
        logging_cfg = cfg.get("logging", {})
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialise the configuration manager.

        Args:
            path: Path to the YAML configuration file.
                  Defaults to ``stateguard/config/defaults.yaml`` resolved
                  relative to this package (CWD-independent).
        """
        self._path = Path(path) if path else _DEFAULT_CONFIG_PATH
        self._config: StateGuardConfig | None = None

    @property
    def path(self) -> Path:
        """Path to the configuration file."""
        return self._path

    @property
    def config(self) -> StateGuardConfig | None:
        """Loaded configuration model, or ``None`` before :meth:`load`."""
        return self._config

    def load(self) -> StateGuardConfig:
        """Load and validate the YAML configuration file.

        Returns:
            A fully populated :class:`StateGuardConfig` instance.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValidationError:   If the file contents fail Pydantic validation.
        """
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {self._path}"
            )

        if data is None:
            data = {}

        self._config = StateGuardConfig.model_validate(data)
        return self._config

    def reload(self) -> StateGuardConfig:
        """Re-read the configuration file from disk.

        Returns:
            The freshly loaded :class:`StateGuardConfig`.
        """
        return self.load()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dotted key path.

        Args:
            key:     Dotted key, e.g. ``"embedding.model_name"``.
            default: Value returned if the key is not found.

        Returns:
            The configuration value or *default*.

        Raises:
            RuntimeError: If :meth:`load` has not been called yet.
        """
        if self._config is None:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )

        parts = key.split(".")
        value: Any = self._config

        for part in parts:
            if isinstance(value, dict):
                if part not in value:
                    return default
                value = value[part]
            else:
                try:
                    value = getattr(value, part)
                except AttributeError:
                    return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value at runtime (in-memory only).

        Args:
            key:   Dotted key, e.g. ``"tier1_threshold"``.
            value: The new value.

        Raises:
            RuntimeError: If :meth:`load` has not been called yet.
        """
        if self._config is None:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )

        parts = key.split(".")
        obj: Any = self._config

        for i, part in enumerate(parts[:-1]):
            if isinstance(obj, dict):
                if part not in obj:
                    obj[part] = {}
                obj = obj[part]
            else:
                obj = getattr(obj, part)

        last_key = parts[-1]
        if isinstance(obj, dict):
            obj[last_key] = value
        else:
            setattr(obj, last_key, value)
