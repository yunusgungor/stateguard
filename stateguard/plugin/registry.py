"""Plugin registry — runtime discovery, registration, and management.

The :class:`PluginRegistry` provides a central registry for discovering,
registering, unregistering, and listing validator plugins.
"""

from __future__ import annotations

from typing import Any

from stateguard.plugin.base import BaseValidator


class PluginRegistry:
    """Central registry for validator plugins.

    Maintains an internal mapping of validator names to their
    :class:`BaseValidator` instances. Supports dynamic discovery,
    registration, and removal of validators.

    Attributes:
        _validators: Internal dict of ``{name: validator_instance}``.
    """

    def __init__(self) -> None:
        """Initialize an empty plugin registry."""
        self._validators: dict[str, BaseValidator] = {}

    def discover_plugins(self, path: str | None = None) -> list[str]:
        """Discover validator plugins from the filesystem.

        Args:
            path: Optional directory path to scan for plugins.
                  If ``None``, the default ``examples`` package is used.

        Returns:
            A list of validator names that were discovered and registered.
        """
        ...

    def register(self, validator: BaseValidator) -> None:
        """Register a validator instance.

        Args:
            validator: An instance of a :class:`BaseValidator` subclass.

        Raises:
            TypeError: If *validator* is not a :class:`BaseValidator` instance.
            ValueError: If a validator with the same name is already registered.
        """
        ...

    def unregister(self, name: str) -> None:
        """Unregister a validator by name.

        Args:
            name: The name of the validator to remove.

        Raises:
            KeyError: If no validator with the given *name* is registered.
        """
        ...

    def list_validators(self) -> list[dict[str, Any]]:
        """List all registered validators with their metadata.

        Returns:
            A list of dictionaries, each containing the validator's
            ``name``, ``dimension``, ``tier``, and class ``type``.
        """
        ...
