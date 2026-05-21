"""Tests for PluginRegistry — registration, unregistration, listing, and discovery."""

from __future__ import annotations

from typing import Any

import pytest

from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.plugin.base import BaseValidator
from stateguard.plugin.registry import PluginRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _ValidValidator(BaseValidator):
    """A minimal valid validator for registry tests."""
    name: str = "test-validator"
    dimension: ValidationDimension = ValidationDimension.STRUCTURAL
    tier: ValidationTier = ValidationTier.TIER_1

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        return ValidationResult(score=100.0, passed=True, dimension=self.dimension)


class _CallTrackingValidator(BaseValidator):
    """Validator that tracks setup/teardown calls."""
    name: str = "tracking-validator"
    dimension: ValidationDimension = ValidationDimension.SEMANTIC
    tier: ValidationTier = ValidationTier.TIER_2
    setup_called: bool = False
    teardown_called: bool = False

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        return ValidationResult(score=100.0, passed=True, dimension=self.dimension)

    def setup(self) -> None:
        self.setup_called = True

    def teardown(self) -> None:
        self.teardown_called = True


@pytest.fixture
def registry() -> PluginRegistry:
    return PluginRegistry()


@pytest.fixture
def valid_validator() -> _ValidValidator:
    return _ValidValidator()


# ---------------------------------------------------------------------------
# Task 1: register()
# ---------------------------------------------------------------------------

class TestRegister:
    """PluginRegistry.register — AC 2"""

    def test_register_validator_success(self, registry: PluginRegistry, valid_validator: _ValidValidator):
        """Geçerli bir validator kaydedilir."""
        registry.register(valid_validator)
        assert valid_validator.name in registry._validators
        assert registry._validators[valid_validator.name] is valid_validator

    def test_register_non_base_validator_raises_typeerror(self, registry: PluginRegistry):
        """BaseValidator instance'ı olmayan nesne TypeError fırlatır."""

        class NotAValidator:
            name = "not-valid"

        with pytest.raises(TypeError, match="BaseValidator"):
            registry.register(NotAValidator())  # type: ignore[arg-type]

    def test_register_duplicate_name_raises_valueerror(self, registry: PluginRegistry, valid_validator: _ValidValidator):
        """Aynı isimle ikinci kez kayıt ValueError fırlatır."""
        registry.register(valid_validator)

        dup = _ValidValidator()
        with pytest.raises(ValueError, match="already registered"):
            registry.register(dup)

    def test_register_calls_setup_hook(self, registry: PluginRegistry):
        """register sırasında setup() lifecycle hook'u çağrılır."""
        v = _CallTrackingValidator()
        assert v.setup_called is False

        registry.register(v)

        assert v.setup_called is True


# ---------------------------------------------------------------------------
# Task 2: unregister()
# ---------------------------------------------------------------------------

class TestUnregister:
    """PluginRegistry.unregister — AC 3"""

    def test_unregister_success(self, registry: PluginRegistry, valid_validator: _ValidValidator):
        """Validator başarıyla kaldırılır."""
        registry.register(valid_validator)
        assert valid_validator.name in registry._validators

        registry.unregister(valid_validator.name)

        assert valid_validator.name not in registry._validators

    def test_unregister_calls_teardown_hook(self, registry: PluginRegistry):
        """unregister sırasında teardown() lifecycle hook'u çağrılır."""
        v = _CallTrackingValidator()
        registry.register(v)
        assert v.teardown_called is False

        registry.unregister(v.name)

        assert v.teardown_called is True

    def test_unregister_nonexistent_raises_keyerror(self, registry: PluginRegistry):
        """Var olmayan bir validator adı KeyError fırlatır."""
        with pytest.raises(KeyError, match="nonexistent"):
            registry.unregister("nonexistent")


# ---------------------------------------------------------------------------
# Task 3: list_validators()
# ---------------------------------------------------------------------------

class TestListValidators:
    """PluginRegistry.list_validators — AC 4"""

    def test_list_empty_registry(self, registry: PluginRegistry):
        """Boş registry boş liste döndürür."""
        assert registry.list_validators() == []

    def test_list_all_validators(self, registry: PluginRegistry):
        """Tüm validatörler metadata ile listelenir."""
        v1 = _ValidValidator()
        registry.register(v1)

        result = registry.list_validators()
        assert len(result) == 1

        entry = result[0]
        assert entry["name"] == "test-validator"
        assert entry["dimension"] == ValidationDimension.STRUCTURAL
        assert entry["tier"] == ValidationTier.TIER_1
        assert entry["type"] == "_ValidValidator"
        assert entry["description"] == ""
        assert entry["version"] == "0.1.0"

    def test_list_filters_by_dimension(self, registry: PluginRegistry):
        """dimension parametresi ile filtreleme çalışır."""
        v1 = _ValidValidator()  # STRUCTURAL
        registry.register(v1)

        v2 = _CallTrackingValidator()  # SEMANTIC
        registry.register(v2)

        all_result = registry.list_validators()
        assert len(all_result) == 2

        structural_result = registry.list_validators(dimension=ValidationDimension.STRUCTURAL)
        assert len(structural_result) == 1
        assert structural_result[0]["name"] == "test-validator"

        semantic_result = registry.list_validators(dimension=ValidationDimension.SEMANTIC)
        assert len(semantic_result) == 1
        assert semantic_result[0]["name"] == "tracking-validator"

    def test_list_preserves_insertion_order(self, registry: PluginRegistry):
        """Liste kayıt sırasını korur."""
        names = ["alpha", "beta", "gamma"]
        for name in names:
            v = _ValidValidator()
            v.name = name
            registry.register(v)

        result = registry.list_validators()
        assert [e["name"] for e in result] == names

    def test_list_after_unregister_preserves_order(self, registry: PluginRegistry):
        """Unregister sonrası kalan validatörler sırasını korur."""
        for name in ["first", "second", "third"]:
            v = _ValidValidator()
            v.name = name
            registry.register(v)

        registry.unregister("second")

        result = registry.list_validators()
        assert [e["name"] for e in result] == ["first", "third"]


class TestRoundTrip:
    """Register → List → Unregister → List döngüsü."""

    def test_round_trip(self, registry: PluginRegistry, valid_validator: _ValidValidator):
        """Kayıt → Listele → Kaldır → Listele döngüsü tutarlıdır."""
        # Register
        registry.register(valid_validator)
        assert len(registry.list_validators()) == 1

        # Unregister
        registry.unregister(valid_validator.name)
        assert len(registry.list_validators()) == 0

        # Re-register (should work)
        registry.register(valid_validator)
        assert len(registry.list_validators()) == 1


# ---------------------------------------------------------------------------
# Task 4: discover_plugins()
# ---------------------------------------------------------------------------

VALID_PLUGIN_SRC = '''
from stateguard.plugin.base import BaseValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult

class TempValidator(BaseValidator):
    name = "temp-validator"
    dimension = ValidationDimension.STRUCTURAL
    tier = ValidationTier.TIER_1

    def validate(self, output, context=None):
        return ValidationResult(score=100.0, passed=True, dimension=self.dimension)
'''

INVALID_PLUGIN_SRC = '''
# This module has no BaseValidator subclass
NOT_A_PLUGIN = 42
'''


class TestDiscoverPlugins:
    """PluginRegistry.discover_plugins — AC 5"""

    def test_discover_default_examples(self, registry: PluginRegistry):
        """Varsayılan examples paketi taranır."""
        discovered = registry.discover_plugins()
        # stateguard.plugin.examples has 3 validators (json_schema, keyword, length)
        assert len(discovered) == 3
        assert all(v in registry._validators for v in discovered)

    def test_discover_plugins_custom_dir(self, registry: PluginRegistry, tmp_path):
        """Custom dizindeki validatörler keşfedilir."""
        plugin_file = tmp_path / "my_plugin.py"
        plugin_file.write_text(VALID_PLUGIN_SRC)

        discovered = registry.discover_plugins(path=str(tmp_path))
        assert "temp-validator" in discovered
        assert "temp-validator" in registry._validators

    def test_discover_plugins_skips_modules_without_validators(self, registry: PluginRegistry, tmp_path):
        """BaseValidator subclass'ı olmayan modüller atlanır."""
        plugin_file = tmp_path / "empty_module.py"
        plugin_file.write_text(INVALID_PLUGIN_SRC)

        discovered = registry.discover_plugins(path=str(tmp_path))
        # No valid validator found
        assert isinstance(discovered, list)
        assert len(discovered) == 0

    def test_discover_plugins_skips_packages(self, registry: PluginRegistry, tmp_path):
        """Paket dizinleri (__init__.py) atlanır, sadece modüller taranır."""
        pkg_dir = tmp_path / "mypackage"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("# package")

        plugin_file = pkg_dir / "sub_plugin.py"
        plugin_file.write_text(VALID_PLUGIN_SRC)

        discovered = registry.discover_plugins(path=str(tmp_path))
        assert "temp-validator" in discovered

    def test_discover_with_path_list(self, registry: PluginRegistry, tmp_path):
        """path parametresi list[str] olarak verilebilir."""
        plugin_file = tmp_path / "list_path_plugin.py"
        plugin_file.write_text(VALID_PLUGIN_SRC)

        discovered = registry.discover_plugins(path=[str(tmp_path)])
        assert "temp-validator" in discovered

    def test_discover_calls_register_for_each_validator(self, registry: PluginRegistry, tmp_path):
        """Keşfedilen her validator register() üzerinden eklenir."""
        # Two validators in one module
        src = '''
from stateguard.plugin.base import BaseValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult

class ValidatorA(BaseValidator):
    name = "validator-a"
    dimension = ValidationDimension.STRUCTURAL
    tier = ValidationTier.TIER_1
    def validate(self, output, context=None):
        return ValidationResult(score=100.0, passed=True, dimension=self.dimension)

class ValidatorB(BaseValidator):
    name = "validator-b"
    dimension = ValidationDimension.SEMANTIC
    tier = ValidationTier.TIER_2
    def validate(self, output, context=None):
        return ValidationResult(score=100.0, passed=True, dimension=self.dimension)
'''
        plugin_file = tmp_path / "dual_plugin.py"
        plugin_file.write_text(src)

        discovered = registry.discover_plugins(path=str(tmp_path))
        assert len(discovered) == 2
        assert "validator-a" in discovered
        assert "validator-b" in discovered

    def test_discover_handles_import_errors_gracefully(self, registry: PluginRegistry, tmp_path):
        """Import hatası alan modüller atlanır, diğerleri keşfedilir."""
        # A bad module
        bad_file = tmp_path / "bad_module.py"
        bad_file.write_text("import nonexistent_module_xyz\n")

        # A good module
        good_file = tmp_path / "good_plugin.py"
        good_file.write_text(VALID_PLUGIN_SRC)

        discovered = registry.discover_plugins(path=str(tmp_path))
        assert "temp-validator" in discovered


class TestConfigIntegration:
    """PluginRegistry config entegrasyonu — AC 6"""

    def test_discover_honors_config_enabled_filter(self, registry: PluginRegistry, tmp_path, monkeypatch):
        """plugins.enabled listesi doluysa sadece o validatörler kaydedilir."""
        src = '''
from stateguard.plugin.base import BaseValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult

class AllowedValidator(BaseValidator):
    name = "allowed-validator"
    dimension = ValidationDimension.STRUCTURAL
    tier = ValidationTier.TIER_1
    def validate(self, output, context=None):
        return ValidationResult(score=100.0, passed=True, dimension=self.dimension)

class BlockedValidator(BaseValidator):
    name = "blocked-validator"
    dimension = ValidationDimension.SEMANTIC
    tier = ValidationTier.TIER_2
    def validate(self, output, context=None):
        return ValidationResult(score=100.0, passed=True, dimension=self.dimension)
'''
        plugin_file = tmp_path / "filter_test.py"
        plugin_file.write_text(src)

        from stateguard.config.settings import ConfigManager
        monkeypatch.setattr(ConfigManager, "load", lambda self: type(
            "obj", (object,), {"plugins": {"enabled": ["allowed-validator"]}}
        )())

        discovered = registry.discover_plugins(path=str(tmp_path))
        assert "allowed-validator" in discovered
        assert "blocked-validator" not in registry._validators

    def test_empty_enabled_loads_all(self, registry: PluginRegistry, tmp_path, monkeypatch):
        """plugins.enabled boşsa tüm keşfedilen validatörler kaydedilir."""
        src = '''
from stateguard.plugin.base import BaseValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult

class MyValidator(BaseValidator):
    name = "my-validator"
    dimension = ValidationDimension.STRUCTURAL
    tier = ValidationTier.TIER_1
    def validate(self, output, context=None):
        return ValidationResult(score=100.0, passed=True, dimension=self.dimension)
'''
        plugin_file = tmp_path / "empty_filter_test.py"
        plugin_file.write_text(src)

        from stateguard.config.settings import ConfigManager
        monkeypatch.setattr(ConfigManager, "load", lambda self: type(
            "obj", (object,), {"plugins": {"enabled": []}}
        )())

        discovered = registry.discover_plugins(path=str(tmp_path))
        assert "my-validator" in discovered
