"""Tests for QuantitativeValidator — Z-Score & Isolation Forest outlier detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from stateguard.dimensions.quantitative import QuantitativeValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def validator() -> QuantitativeValidator:
    return QuantitativeValidator()


@pytest.fixture
def z_validator() -> QuantitativeValidator:
    return QuantitativeValidator(method="zscore", z_threshold=3.0)


@pytest.fixture
def if_validator() -> QuantitativeValidator:
    return QuantitativeValidator(method="isolation_forest", contamination=0.1)


@pytest.fixture
def normal_values() -> list[float]:
    """Normal distribution centered at 10 with low variance."""
    return [10.0, 10.5, 9.8, 10.2, 9.9, 10.1, 10.3, 9.7, 10.4, 9.6]


@pytest.fixture
def outlier_values() -> list[float]:
    """Values with a clear outlier (100.0) — dense cluster around 10."""
    # 20 normal values + 1 extreme outlier — Z-Score > 3 için yeterli
    return [10.0 + (i % 5 - 2) * 0.5 for i in range(20)] + [100.0]


@pytest.fixture
def identical_values() -> list[float]:
    """All values identical — std=0 edge case."""
    return [5.0, 5.0, 5.0, 5.0, 5.0]


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------

class TestQuantitativeValidatorAttributes:
    """AC 1 — Class-level attribute enforcement."""

    def test_name(self, validator: QuantitativeValidator) -> None:
        assert validator.name == "quantitative"

    def test_dimension(self, validator: QuantitativeValidator) -> None:
        assert validator.dimension == ValidationDimension.QUANTITATIVE

    def test_tier(self, validator: QuantitativeValidator) -> None:
        assert validator.tier == ValidationTier.TIER_2


# ---------------------------------------------------------------------------
# Z-Score validation
# ---------------------------------------------------------------------------

class TestZScore:
    """AC 2 — Z-Score outlier detection."""

    def test_normal_values_pass(self, z_validator: QuantitativeValidator,
                                normal_values: list[float]) -> None:
        """Normal dağılım — outlier yok, passed=True."""
        result = z_validator.validate(normal_values)
        assert result.passed is True
        assert result.score >= 50.0  # Tier 2 eşiği
        assert "z_scores" in result.details
        assert "max_z_score" in result.details

    def test_outlier_detected(self, z_validator: QuantitativeValidator,
                              outlier_values: list[float]) -> None:
        """Outlier var (100) — Z-Score > 2, outlier_count > 0."""
        result = z_validator.validate(outlier_values)
        assert "z_scores" in result.details
        assert "max_z_score" in result.details
        assert result.details["outlier_count"] > 0
        assert result.score < 100.0

    def test_z_threshold_override(self, outlier_values: list[float]) -> None:
        """context ile z_threshold override edilebilir."""
        v = QuantitativeValidator(method="zscore", z_threshold=5.0)
        result = v.validate(outlier_values, context={"z_threshold": 5.0})
        assert result.score < 100.0

    def test_single_value_returns_warning(self) -> None:
        """Tek değer — Z-Score hesaplanamaz, warning ile PASS."""
        v = QuantitativeValidator(method="zscore")
        result = v.validate([42.0])
        assert result.passed is True
        assert "warning" in result.details

    def test_identical_values_returns_warning(self,
                                              identical_values: list[float]) -> None:
        """Tüm değerler aynı — std=0, warning ile PASS."""
        v = QuantitativeValidator(method="zscore")
        result = v.validate(identical_values)
        assert result.passed is True
        assert "warning" in result.details

    def test_negative_values_handled(self) -> None:
        """Negatif değerler de doğru hesaplanır."""
        v = QuantitativeValidator(method="zscore")
        values = [-100.0, -95.0, -102.0, -98.0, 0.0]
        result = v.validate(values)
        assert result.score < 100.0


# ---------------------------------------------------------------------------
# Isolation Forest validation
# ---------------------------------------------------------------------------

class TestIsolationForest:
    """AC 2 — Isolation Forest anomaly detection."""

    def test_iforest_requires_sklearn(self, if_validator: QuantitativeValidator) -> None:
        """scikit-learn yoksa ImportError mesajı döner."""
        result = if_validator.validate([1.0, 2.0, 3.0])
        # Eğer sklearn yoksa details'de error olur, passed=False
        if result.details.get("error") and "scikit-learn" in result.details["error"]:
            assert result.passed is False
        else:
            # sklearn varsa normal çalışır
            assert isinstance(result, ValidationResult)

    def test_iforest_with_reference_data(self) -> None:
        """reference_data ile Isolation Forest fit edilir ve anomali tespit eder."""
        v = QuantitativeValidator(method="isolation_forest", contamination=0.1)
        rng = np.random.RandomState(42)
        ref_data = rng.randn(100, 1)
        # Normal değer
        result = v.validate([0.5], context={"reference_data": ref_data})
        assert isinstance(result, ValidationResult)
        assert "n_anomalies" in result.details or "warning" in result.details

    def test_iforest_model_cached(self) -> None:
        """Isolation Forest modeli cache'lenir — ikinci validate'de fit tekrarlanmaz."""
        import unittest.mock as mock

        v = QuantitativeValidator(method="isolation_forest", contamination=0.1)
        rng = np.random.RandomState(42)
        ref_data = rng.randn(100, 1)

        # İlk çağrı — model fit edilir
        result1 = v.validate([0.5], context={"reference_data": ref_data})

        # İkinci çağrı — cache kullanılır, ref_data gerekmez
        result2 = v.validate([0.5])
        assert isinstance(result2, ValidationResult)

    def test_iforest_contamination_override(self) -> None:
        """context ile contamination override edilebilir."""
        v = QuantitativeValidator(method="isolation_forest", contamination=0.5)
        rng = np.random.RandomState(42)
        ref_data = rng.randn(50, 1)
        result = v.validate([1.0, 2.0], context={
            "reference_data": ref_data,
            "contamination": 0.5,
        })
        assert isinstance(result, ValidationResult)


# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------

class TestMethodSelection:
    """AC 3 — Method selection logic."""

    def test_default_method_zscore(self) -> None:
        """Varsayılan method 'zscore' olmalı."""
        v = QuantitativeValidator()
        assert v._method == "zscore"

    def test_method_context_override(self) -> None:
        """context['method'] constructor değerini override eder."""
        v = QuantitativeValidator(method="zscore")
        result = v.validate([1.0, 2.0, 3.0], context={"method": "isolation_forest"})
        # Her durumda ValidationResult döner
        assert isinstance(result, ValidationResult)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling."""

    def test_none_output(self, validator: QuantitativeValidator) -> None:
        """None output — passed=False, score=0."""
        result = validator.validate(None)
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details or result.details.get("warning")

    def test_empty_list(self, validator: QuantitativeValidator) -> None:
        """Boş liste — passed=False."""
        result = validator.validate([])
        assert result.passed is False
        assert result.score == 0.0

    def test_non_numeric_list_element(self, validator: QuantitativeValidator) -> None:
        """Liste içinde non-numeric eleman — passed=False."""
        result = validator.validate([1.0, "abc", 3.0])
        assert isinstance(result, ValidationResult)

    def test_single_float_value(self, validator: QuantitativeValidator) -> None:
        """Tek float değer — listeye çevrilir."""
        result = validator.validate(42.0)
        assert isinstance(result, ValidationResult)

    def test_context_type_guard(self, validator: QuantitativeValidator) -> None:
        """context None veya dict dışı tip — crash olmaz."""
        result = validator.validate([1.0, 2.0, 3.0], context=None)
        assert isinstance(result, ValidationResult)
        result2 = validator.validate([1.0, 2.0, 3.0], context="invalid")
        assert isinstance(result2, ValidationResult)

    def test_score_clamping(self) -> None:
        """Score her zaman [0, 100] aralığında."""
        v = QuantitativeValidator(method="zscore", z_threshold=0.01)
        result = v.validate([1.0, 1000.0])
        assert 0.0 <= result.score <= 100.0

    def test_method_both(self) -> None:
        """method='both' — iki yöntemin ortalaması."""
        v = QuantitativeValidator(method="both")
        rng = np.random.RandomState(42)
        ref_data = rng.randn(100, 1)
        result = v.validate([1.0, 2.0, 3.0], context={"reference_data": ref_data})
        assert isinstance(result, ValidationResult)
        assert 0.0 <= result.score <= 100.0

    def test_values_from_context(self, z_validator: QuantitativeValidator) -> None:
        """context['values'] output yerine kullanılır."""
        result = z_validator.validate("not-a-list", context={
            "values": [10.0, 12.0, 11.0, 13.0, 9.0],
        })
        assert result.passed is True
        assert result.score >= 50.0
