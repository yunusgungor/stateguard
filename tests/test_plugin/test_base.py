"""Tests for BaseValidator — abstract plugin base class."""

from __future__ import annotations

from typing import Any

import pytest

from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.plugin.base import BaseValidator


# ✅ Valid subclass
class GoodValidator(BaseValidator):
    name: str = "good-validator"
    dimension: ValidationDimension = ValidationDimension.STRUCTURAL
    tier: ValidationTier = ValidationTier.TIER_1

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        return ValidationResult(
            score=100.0,
            passed=True,
            dimension=self.dimension,
        )


class TestBaseValidatorEnforcement:
    """__init_subclass__ enforcement testleri."""

    def test_valid_subclass_works(self):
        """Geçerli subclass sorunsuz instantiate edilebilir."""
        v = GoodValidator()
        assert v.name == "good-validator"
        assert v.dimension == ValidationDimension.STRUCTURAL
        assert v.tier == ValidationTier.TIER_1

    def test_missing_name_raises_typeerror(self):
        """name class attr eksikse TypeError."""
        with pytest.raises(TypeError, match="name"):

            class _(BaseValidator):  # type: ignore
                dimension = ValidationDimension.STRUCTURAL
                tier = ValidationTier.TIER_1

                def validate(self, output, context=None):
                    return ValidationResult(score=0.0, passed=True, dimension=self.dimension)

    def test_missing_dimension_raises_typeerror(self):
        """dimension class attr eksikse TypeError."""
        with pytest.raises(TypeError, match="dimension"):

            class _(BaseValidator):
                name = "no-dim"
                tier = ValidationTier.TIER_1

                def validate(self, output, context=None):
                    return ValidationResult(score=0.0, passed=True, dimension=self.dimension)

    def test_missing_tier_raises_typeerror(self):
        """tier class attr eksikse TypeError."""
        with pytest.raises(TypeError, match="tier"):

            class _(BaseValidator):
                name = "no-tier"
                dimension = ValidationDimension.SEMANTIC

                def validate(self, output, context=None):
                    return ValidationResult(score=0.0, passed=True, dimension=self.dimension)


class TestExistingValidators:
    """Mevcut validatörler hala çalışıyor mu?"""

    @pytest.mark.usefixtures("auto_patch_embedding")
    def test_embedding_validator_compatible(self):
        """EmbeddingValidator mevcut haliyle çalışır."""
        from stateguard.core.tier1 import EmbeddingValidator

        v = EmbeddingValidator()
        assert v.name == "embedding-similarity"
        assert v.dimension == ValidationDimension.SEMANTIC
        assert v.tier == ValidationTier.TIER_1

    def test_ensemble_validator_compatible(self):
        """EnsembleValidator mevcut haliyle çalışır."""
        from stateguard.core.tier2 import EnsembleValidator

        v = EnsembleValidator()
        assert v.name == "ensemble-anomaly"
        assert v.dimension == ValidationDimension.SEMANTIC
        assert v.tier == ValidationTier.TIER_2

    def test_llm_validator_compatible(self):
        """LLMValidator mevcut haliyle çalışır."""
        from unittest.mock import MagicMock

        from stateguard.core.tier3 import LLMValidator

        mock_client = MagicMock()
        v = LLMValidator(llm_client=mock_client)
        assert v.name == "llm-validator"
        assert v.dimension == ValidationDimension.SEMANTIC
        assert v.tier == ValidationTier.TIER_3

    def test_example_plugins_compatible(self):
        """Örnek plugin validatörler mevcut haliyle çalışır."""
        from stateguard.plugin.examples.json_schema import JsonSchemaValidator
        from stateguard.plugin.examples.keyword import KeywordValidator
        from stateguard.plugin.examples.length import LengthValidator

        for ValidatorClass in [JsonSchemaValidator, KeywordValidator, LengthValidator]:
            v = ValidatorClass()
            assert v.name is not None
            assert v.dimension is not None
            assert v.tier is not None


class TestMetadata:
    """Opsiyonel metadata alanları."""

    def test_default_metadata_values(self):
        """Metadata alanları default değerlerini alır."""
        v = GoodValidator()
        assert v.description == ""
        assert v.version == "0.1.0"

    def test_custom_metadata_values(self):
        """Metadata alanları override edilebilir."""

        class CustomMetaValidator(BaseValidator):
            name: str = "custom-meta"
            dimension: ValidationDimension = ValidationDimension.SECURITY
            tier: ValidationTier = ValidationTier.TIER_3
            description: str = "Custom metadata test validator"
            version: str = "2.1.0"

            def validate(self, output, context=None):
                return ValidationResult(score=0.0, passed=True, dimension=self.dimension)

        v = CustomMetaValidator()
        assert v.description == "Custom metadata test validator"
        assert v.version == "2.1.0"


class TestLifecycleHooks:
    """Opsiyonel lifecycle hook'lar."""

    def test_setup_teardown_noop_by_default(self):
        """Varsayılan setup/teardown no-op'tur (çağrılabilir)."""
        v = GoodValidator()
        # should not raise
        v.setup()
        v.teardown()

    def test_setup_teardown_overridable(self):
        """Lifecycle hook'lar override edilebilir."""

        setup_called = False
        teardown_called = False

        class LifecycleValidator(BaseValidator):
            name: str = "lifecycle"
            dimension: ValidationDimension = ValidationDimension.STRUCTURAL
            tier: ValidationTier = ValidationTier.TIER_1

            def validate(self, output, context=None):
                return ValidationResult(score=0.0, passed=True, dimension=self.dimension)

            def setup(self):
                nonlocal setup_called
                setup_called = True

            def teardown(self):
                nonlocal teardown_called
                teardown_called = True

        v = LifecycleValidator()
        v.setup()
        assert setup_called is True
        v.teardown()
        assert teardown_called is True


class TestValidateEnforcement:
    """validate abstractmethod enforcement."""

    def test_validate_not_implemented_raises_typeerror(self):
        """validate override edilmezse instantiate ederken TypeError."""

        class IncompleteValidator(BaseValidator):  # type: ignore
            name: str = "incomplete"
            dimension = ValidationDimension.STRUCTURAL
            tier = ValidationTier.TIER_1
            # validate NOT overridden — ABC catches at instantiation

        with pytest.raises(TypeError):
            IncompleteValidator()
