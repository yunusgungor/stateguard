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


class TestDeepInheritance:
    """2+ seviyeli inheritance chain (W1 fix)."""

    def test_deep_inheritance_works(self):
        """Mid -> Child 2 seviye inheritance calisir."""

        class MidValidator(BaseValidator):
            name: str = "mid"
            dimension: ValidationDimension = ValidationDimension.STRUCTURAL
            tier: ValidationTier = ValidationTier.TIER_1

            def validate(self, output, context=None):
                return ValidationResult(score=100.0, passed=True, dimension=self.dimension)

        class ChildValidator(MidValidator):
            pass

        v = ChildValidator()
        assert v.name == "mid"
        assert v.dimension == ValidationDimension.STRUCTURAL
        assert v.tier == ValidationTier.TIER_1
        result = v.validate("test")
        assert result.passed is True

    def test_deep_three_level_inheritance_works(self):
        """Grand -> Mid -> Child 3 seviye inheritance calisir."""

        class GrandValidator(BaseValidator):
            name: str = "grand"
            dimension: ValidationDimension = ValidationDimension.QUANTITATIVE
            tier: ValidationTier = ValidationTier.TIER_1

            def validate(self, output, context=None):
                return ValidationResult(score=100.0, passed=True, dimension=self.dimension)

        class MidValidator(GrandValidator):
            pass

        class ChildValidator(MidValidator):
            pass

        v = ChildValidator()
        assert v.name == "grand"
        result = v.validate("test")
        assert result.passed is True

    def test_deep_inheritance_child_inherits_mid_attrs(self):
        """Mid'de tanimli attr'lar Child'a miras gecer."""

        class MidValidator(BaseValidator):
            name: str = "mid"
            dimension: ValidationDimension = ValidationDimension.STRUCTURAL
            tier: ValidationTier = ValidationTier.TIER_1

            def validate(self, output, context=None):
                return ValidationResult(score=100.0, passed=True, dimension=self.dimension)

        class ChildWithoutName(MidValidator):
            pass

        v = ChildWithoutName()
        assert v.name == "mid"

    def test_abstract_intermediate_still_skips_check(self):
        """Abstract intermediate sinif attr kontrolunden muaf (mevcut davranis)."""

        import abc

        class AbstractMid(BaseValidator, abc.ABC):
            # name/dimension/tier tanimlamadik - abstract oldugu icin sorun olmamali
            @abc.abstractmethod
            def validate(self, output, context=None):
                ...

        # Abstract sinifin kendisi instantiate edilemez
        with pytest.raises(TypeError):
            AbstractMid()

        # Concrete subclass attr'lari tanimlamali
        class ConcreteChild(AbstractMid):
            name: str = "concrete"
            dimension: ValidationDimension = ValidationDimension.SEMANTIC
            tier: ValidationTier = ValidationTier.TIER_2

            def validate(self, output, context=None):
                return ValidationResult(score=100.0, passed=True, dimension=self.dimension)

        v = ConcreteChild()
        assert v.name == "concrete"


class TestValidateCallableEnforcement:
    """validate attribute callable enforcement (W2 fix)."""

    def test_validate_must_be_callable(self):
        """validate non-callable ise class taniminda TypeError."""

        with pytest.raises(TypeError, match="validate"):

            class _(BaseValidator):
                name: str = "broken"
                dimension: ValidationDimension = ValidationDimension.STRUCTURAL
                tier: ValidationTier = ValidationTier.TIER_1
                validate = 42  # non-callable

    def test_validate_callable_passes(self):
        """validate callable ise sorunsuz gecer."""

        class Good2Validator(BaseValidator):
            name: str = "good2"
            dimension: ValidationDimension = ValidationDimension.STRUCTURAL
            tier: ValidationTier = ValidationTier.TIER_1

            def validate(self, output, context=None):
                return ValidationResult(score=100.0, passed=True, dimension=self.dimension)

        v = Good2Validator()
        assert v.name == "good2"

    def test_non_callable_validate_in_subclass_mid_chain(self):
        """Mid -> Child chain'inde Child validate callable degilse hata."""

        class MidValidator(BaseValidator):
            name: str = "mid"
            dimension: ValidationDimension = ValidationDimension.STRUCTURAL
            tier: ValidationTier = ValidationTier.TIER_1

            def validate(self, output, context=None):
                return ValidationResult(score=100.0, passed=True, dimension=self.dimension)

        # Child validate'i non-callable ile override ediyor
        with pytest.raises(TypeError, match="validate"):

            class ChildValidator(MidValidator):
                validate = "not-a-method"

    def test_staticmethod_and_classmethod_are_callable(self):
        """staticmethod ve classmethod da callable sayilir, hata vermez."""

        class StaticValidator(BaseValidator):
            name: str = "static"
            dimension: ValidationDimension = ValidationDimension.STRUCTURAL
            tier: ValidationTier = ValidationTier.TIER_1

            @staticmethod
            def validate(output, context=None):
                return ValidationResult(score=100.0, passed=True, dimension=ValidationDimension.STRUCTURAL)

        v = StaticValidator()
        result = v.validate("test")
        assert result.passed is True

    @pytest.mark.skip(reason="ABC abstractmethod callable kontrolu tartismali - simdilik skip")
    def test_abstract_intermediate_non_callable_validate_allowed(self):
        """Abstract intermediate sinifta non-callable validate sorun olmamali.
        
        Bu test tartismali - abstract intermediate sinif validate'i abstractmethod
        olarak bildirmis olsa bile, class attribute olarak non-callable deger
        atanmasi durumunda ne olmali? Su an icin skip.
        """
        pass
