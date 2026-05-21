"""Tests for ValidationEngine — cascade pipeline orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from stateguard.core.engine import ValidationEngine
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import EngineResult, ValidationResult


@pytest.fixture
def mock_tier1():
    """Mock Tier 1 — default passes."""
    mock = MagicMock()
    mock.name = "embedding-similarity"
    mock.dimension = ValidationDimension.SEMANTIC
    mock.tier = ValidationTier.TIER_1
    mock.validate.return_value = ValidationResult(
        score=95.0,
        passed=True,
        dimension=ValidationDimension.SEMANTIC,
        details={"tier_hint": None},
    )
    return mock


@pytest.fixture
def mock_tier2():
    """Mock Tier 2 — default passes."""
    mock = MagicMock()
    mock.name = "ensemble-anomaly"
    mock.dimension = ValidationDimension.SEMANTIC
    mock.tier = ValidationTier.TIER_2
    mock.validate.return_value = ValidationResult(
        score=85.0,
        passed=True,
        dimension=ValidationDimension.SEMANTIC,
        details={
            "ensemble": {"vote": "passed", "normal_count": 3, "anomaly_count": 0},
        },
    )
    return mock


@pytest.fixture
def mock_tier3():
    """Mock Tier 3 — LLM stub."""
    mock = MagicMock()
    mock.name = "llm-validator"
    mock.dimension = ValidationDimension.SEMANTIC
    mock.tier = ValidationTier.TIER_3
    return mock


class TestValidationEngine:
    """ValidationEngine birim testleri."""

    def test_default_construction(self):
        """Varsayılan constructor çalışır."""
        engine = ValidationEngine()
        assert engine.name == "validation-engine"
        assert engine._agent_id == "default"

    def test_injectable_validators(self, mock_tier1, mock_tier2, mock_tier3):
        """Validator'lar constructor'da enjekte edilebilir."""
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
        )
        assert engine._embedding is mock_tier1
        assert engine._ensemble is mock_tier2
        assert engine._llm is mock_tier3

    def test_tier1_pass_stops_pipeline(self, mock_tier1, mock_tier2, mock_tier3):
        """Tier 1 PASS → pipeline durur, tier_path=[1]."""
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
        )
        result = engine.validate("test output")
        assert isinstance(result, EngineResult)
        assert result.passed is True
        assert result.tier_path == [1]
        assert result.overall_score == 95.0
        # Tier 2 ve Tier 3 çağrılmamalı
        mock_tier2.validate.assert_not_called()
        mock_tier3.validate.assert_not_called()

    def test_tier1_fail_stops_pipeline(self, mock_tier1, mock_tier2, mock_tier3):
        """Tier 1 FAIL (<50) → pipeline durur, tier_path=[1], passed=False."""
        mock_tier1.validate.return_value = ValidationResult(
            score=30.0,
            passed=False,
            dimension=ValidationDimension.SEMANTIC,
        )
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
        )
        result = engine.validate("bad output")
        assert result.passed is False
        assert result.tier_path == [1]
        assert result.overall_score == 30.0
        mock_tier2.validate.assert_not_called()

    def test_tier1_borderline_tier2_pass(self, mock_tier1, mock_tier2, mock_tier3):
        """Tier 1 borderline + Tier 2 PASS → tier_path=[1,2]."""
        mock_tier1.validate.return_value = ValidationResult(
            score=65.0,
            passed=False,
            dimension=ValidationDimension.SEMANTIC,
            details={"tier_hint": "tier_2"},
        )
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
        )
        result = engine.validate("borderline output")
        assert result.passed is True
        assert result.tier_path == [1, 2]
        mock_tier2.validate.assert_called_once()
        mock_tier3.validate.assert_not_called()

    def test_tier1_borderline_tier2_fail_tier3_disabled(
        self, mock_tier1, mock_tier2, mock_tier3
    ):
        """Tier 1 borderline + Tier 2 FAIL + tier3_disabled → passed=False, tier_path=[1,2]."""
        mock_tier1.validate.return_value = ValidationResult(
            score=65.0,
            passed=False,
            dimension=ValidationDimension.SEMANTIC,
            details={"tier_hint": "tier_2"},
        )
        mock_tier2.validate.return_value = ValidationResult(
            score=40.0,
            passed=False,
            dimension=ValidationDimension.SEMANTIC,
        )
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
            tier3_enabled=False,
        )
        result = engine.validate("borderline -> fail")
        assert result.passed is False
        assert result.tier_path == [1, 2]
        mock_tier3.validate.assert_not_called()

    def test_tier1_borderline_tier2_fail_tier3_enabled(
        self, mock_tier1, mock_tier2, mock_tier3
    ):
        """Tier 1 borderline + Tier 2 FAIL + tier3_enabled → tier_path=[1,2,3]."""
        mock_tier1.validate.return_value = ValidationResult(
            score=65.0,
            passed=False,
            dimension=ValidationDimension.SEMANTIC,
            details={"tier_hint": "tier_2"},
        )
        mock_tier2.validate.return_value = ValidationResult(
            score=40.0,
            passed=False,
            dimension=ValidationDimension.SEMANTIC,
        )
        # Tier 3 returns a result (not NotImplementedError)
        mock_tier3.validate.return_value = ValidationResult(
            score=100.0,
            passed=True,
            dimension=ValidationDimension.SEMANTIC,
            details={"raw_response": "EVET"},
        )
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
            tier3_enabled=True,
        )
        result = engine.validate("borderline -> fail -> tier3")
        assert result.tier_path == [1, 2, 3]
        assert result.passed is True
        assert result.overall_score == 100.0
        mock_tier3.validate.assert_called_once()

    def test_decision_log_present(self, mock_tier1, mock_tier2, mock_tier3):
        """Decision log EngineResult.details'te bulunur."""
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
            agent_id="test-agent",
            tier3_enabled=False,
        )
        result = engine.validate("test")
        assert "decision_log" in result.details
        assert len(result.details["decision_log"]) == 1  # Sadece Tier 1

    def test_decision_log_two_tiers(self, mock_tier1, mock_tier2, mock_tier3):
        """2 tier çalışınca decision_log 2 entry içerir."""
        mock_tier1.validate.return_value = ValidationResult(
            score=65.0,
            passed=False,
            dimension=ValidationDimension.SEMANTIC,
            details={"tier_hint": "tier_2"},
        )
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
            tier3_enabled=False,
        )
        result = engine.validate("test-2tier")
        assert len(result.details["decision_log"]) == 2

    def test_fail_close_on_exception(self, mock_tier1, mock_tier2, mock_tier3):
        """fail-close modunda exception → EngineResult(error)."""
        mock_tier1.validate.side_effect = RuntimeError("Model crashed")
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
            fail_mode="fail-close",
        )
        result = engine.validate("crash")
        assert result.passed is False
        assert "error" in result.details
        assert "Model crashed" in result.details["error"]

    def test_invalid_fail_mode_raises_error(self, mock_tier1, mock_tier2, mock_tier3):
        """Geçersiz fail_mode → ValueError."""
        with pytest.raises(ValueError, match="fail-close"):
            ValidationEngine(
                embedding_validator=mock_tier1,
                ensemble_validator=mock_tier2,
                llm_validator=mock_tier3,
                fail_mode="fail_close",
            )

    def test_fail_open_on_exception(self, mock_tier1, mock_tier2, mock_tier3):
        """fail-open modunda exception → passed=True ile devam eder."""
        mock_tier1.validate.side_effect = RuntimeError("Model crashed")
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
            fail_mode="fail-open",
        )
        result = engine.validate("crash")
        assert result.passed is True

    def test_different_agent_id(self, mock_tier1, mock_tier2, mock_tier3):
        """agent_id EngineResult'a yansır."""
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
            agent_id="content-os",
            tier3_enabled=False,
        )
        result = engine.validate("test")
        assert result.details["decision_log"][0].agent_id == "content-os"

    def test_tier1_low_score_returns_fail(self, mock_tier1, mock_tier2, mock_tier3):
        """Tier 1 düşük skor → passed=False, tier_path=[1]."""
        mock_tier1.validate.return_value = ValidationResult(
            score=0.0,
            passed=False,
            dimension=ValidationDimension.SEMANTIC,
            error="Empty output.",
        )
        engine = ValidationEngine(
            embedding_validator=mock_tier1,
            ensemble_validator=mock_tier2,
            llm_validator=mock_tier3,
        )
        result = engine.validate("")
        assert result.passed is False
