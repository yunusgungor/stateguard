"""Plugin registry — runtime discovery, registration, and management.

The :class:`PluginRegistry` provides a central registry for discovering,
registering, unregistering, and listing validator plugins.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any

from stateguard.models.enums import ValidationDimension
from stateguard.plugin.base import BaseValidator

logger = logging.getLogger(__name__)


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

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, validator: BaseValidator) -> None:
        """Register a validator instance.

        Args:
            validator: An instance of a :class:`BaseValidator` subclass.

        Raises:
            TypeError: If *validator* is not a :class:`BaseValidator` instance.
            ValueError: If a validator with the same name is already registered.
        """
        if not isinstance(validator, BaseValidator):
            raise TypeError(
                f"Expected a BaseValidator instance, got {type(validator).__name__}"
            )

        if validator.name in self._validators:
            raise ValueError(
                f"A validator named '{validator.name}' is already registered."
            )

        validator.setup()
        self._validators[validator.name] = validator

    # ------------------------------------------------------------------
    # Unregistration
    # ------------------------------------------------------------------

    def unregister(self, name: str) -> None:
        """Unregister a validator by name.

        Args:
            name: The name of the validator to remove.

        Raises:
            KeyError: If no validator with the given *name* is registered.
        """
        if name not in self._validators:
            raise KeyError(f"No validator named '{name}' is registered.")

        validator = self._validators.pop(name)
        try:
            validator.teardown()
        except Exception:
            logger.exception("Error during teardown of validator '%s'", name)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_validators(
        self,
        dimension: ValidationDimension | None = None,
    ) -> list[dict[str, Any]]:
        """List all registered validators with their metadata.

        Args:
            dimension: Optional :class:`ValidationDimension` filter.
                       If ``None``, all validators are returned.

        Returns:
            A list of dictionaries, sorted by registration order, each
            containing the validator's ``name``, ``dimension``, ``tier``,
            ``type`` (class name), ``description``, and ``version``.
        """
        result: list[dict[str, Any]] = []
        for validator in list(self._validators.values()):
            if dimension is not None and validator.dimension != dimension:
                continue
            result.append({
                "name": validator.name,
                "dimension": validator.dimension,
                "tier": validator.tier,
                "type": type(validator).__name__,
                "description": validator.description,
                "version": validator.version,
            })
        return result

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_plugins(self, path: str | list[str] | None = None) -> list[str]:
        """Discover validator plugins from the filesystem.

        Scans Python modules for :class:`BaseValidator` subclasses and
        automatically registers them.

        Args:
            path: Optional directory path (``str`` or ``list[str]``) to
                  scan for plugins. If ``None``, the default
                  ``stateguard.plugin.examples`` package is scanned.

        Returns:
            A list of validator names that were discovered and registered.
        """
        if path is None:
            # Default: scan the examples package
            from stateguard.plugin import examples as default_pkg

            scan_path: list[str] = list(default_pkg.__path__)
            prefix = default_pkg.__name__ + "."
        else:
            scan_path = [path] if isinstance(path, str) else list(path)
            prefix = ""

        discovered: list[str] = []

        # For custom paths, temporarily add to sys.path so modules can be imported
        import sys
        added_to_path: list[str] = []
        try:
            if scan_path:
                for sp in scan_path:
                    if sp not in sys.path:
                        sys.path.insert(0, sp)
                        added_to_path.append(sp)

            for importer, modname, is_pkg in pkgutil.walk_packages(
                path=scan_path,
                prefix=prefix,
                onerror=lambda name: logger.warning("Failed to scan module: %s", name),
            ):
                if is_pkg:
                    continue

                try:
                    module = importlib.import_module(modname)
                except Exception as e:
                    logger.warning("Failed to import plugin module '%s': %s", modname, e)
                    continue

                # Find BaseValidator subclasses defined in this module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseValidator)
                        and attr is not BaseValidator
                        and getattr(attr, "__module__", None) == module.__name__
                    ):
                        try:
                            instance = attr()
                            self.register(instance)
                            discovered.append(instance.name)
                            logger.info(
                                "Discovered and registered plugin '%s' from %s",
                                instance.name, modname,
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to instantiate validator '%s' from %s: %s",
                                attr_name, modname, e,
                            )
        finally:
            # Clean up added sys.path entries
            for sp in added_to_path:
                if sp in sys.path:
                    sys.path.remove(sp)

        # Apply config filter if available
        try:
            from stateguard.config.settings import ConfigManager

            cfg = ConfigManager().load()
            enabled_list = cfg.plugins.get("enabled", [])  # type: ignore[union-attr]
            if enabled_list:  # non-empty → filter
                to_remove = [
                    name for name in self._validators
                    if name not in enabled_list
                ]
                for name in to_remove:
                    removed = self._validators.pop(name)
                    try:
                        removed.teardown()
                    except Exception:
                        logger.exception(
                            "Error during teardown of disabled plugin '%s'", name,
                        )
                    logger.info("Plugin '%s' disabled by config", name)
        except Exception:
            logger.warning(
                "Could not load plugin config; using all discovered plugins."
            )

        # Ensure return value matches what's actually registered
        discovered = [name for name in discovered if name in self._validators]
        return discovered
