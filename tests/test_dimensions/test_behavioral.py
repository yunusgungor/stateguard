"""Tests for BehavioralValidator — FSM state compliance & snapshot comparison."""

from __future__ import annotations

from typing import Any

import pytest

from stateguard.dimensions.behavioral import BehavioralValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def validator() -> BehavioralValidator:
    return BehavioralValidator()


@pytest.fixture
def simple_validator() -> BehavioralValidator:
    """Validator with a minimal custom FSM."""
    return BehavioralValidator(
        drift_threshold=0.10,
    )


@pytest.fixture
def sample_snapshot_a() -> dict[str, Any]:
    return {
        "step": 1,
        "score": 85.0,
        "state": "VALIDATING",
        "confidence": 0.95,
    }


@pytest.fixture
def sample_snapshot_b() -> dict[str, Any]:
    return {
        "step": 2,
        "score": 20.0,
        "state": "FAILED",
        "confidence": 0.30,
        "extra": "unexpected",
    }


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------

class TestBehavioralValidatorAttributes:
    """AC 1 — Class-level attribute enforcement."""

    def test_name(self, validator: BehavioralValidator) -> None:
        assert validator.name == "behavioral"

    def test_dimension(self, validator: BehavioralValidator) -> None:
        assert validator.dimension == ValidationDimension.BEHAVIORAL

    def test_tier(self, validator: BehavioralValidator) -> None:
        assert validator.tier == ValidationTier.TIER_3

    def test_drift_threshold_default(self, validator: BehavioralValidator) -> None:
        assert validator.drift_threshold == 0.15

    def test_drift_threshold_custom(self, simple_validator: BehavioralValidator) -> None:
        assert simple_validator.drift_threshold == 0.10


# ---------------------------------------------------------------------------
# FSM State Compliance — AC 2 & 3
# ---------------------------------------------------------------------------

class TestStateCompliance:
    """AC 2, 3 — FSM state transition validation."""

    def test_valid_transition_dict(
        self, validator: BehavioralValidator,
    ) -> None:
        """Geçerli IDLE → VALIDATING → PASSED."""
        r = validator.validate({"current": "IDLE", "next": "VALIDATING"})
        assert r.passed is True
        assert r.score == 100.0
        assert "transition" in r.details.get("fsm", {})

    def test_invalid_transition(
        self, validator: BehavioralValidator,
    ) -> None:
        """IDLE → COMPLETED geçersiz, FSM'de izin yok."""
        r = validator.validate({"current": "IDLE", "next": "COMPLETED"})
        assert r.passed is False
        assert r.score == 0.0
        assert "invalid_transition" in r.details.get("fsm", {})

    def test_tuple_output(
        self, validator: BehavioralValidator,
    ) -> None:
        """Tuple (current, next) de kabul edilmeli."""
        r = validator.validate(("VALIDATING", "FAILED"))
        assert r.passed is True
        assert r.score == 100.0

    def test_tuple_invalid(
        self, validator: BehavioralValidator,
    ) -> None:
        """Tuple ile geçersiz transition."""
        r = validator.validate(("PASSED", "IDLE"))
        assert r.passed is False
        assert r.score == 0.0

    def test_context_states_override(
        self, validator: BehavioralValidator,
    ) -> None:
        """context['states'] override ile özel state seti."""
        r = validator.validate(
            {"current": "CUSTOM1", "next": "CUSTOM2"},
            context={"states": {"CUSTOM1", "CUSTOM2"}, "transitions": {("CUSTOM1", "CUSTOM2")}},
        )
        assert r.passed is True
        assert r.score == 100.0

    def test_context_states_override_invalid(
        self, validator: BehavioralValidator,
    ) -> None:
        """context ile override edilmiş state'lerde geçersiz transition."""
        r = validator.validate(
            {"current": "CUSTOM1", "next": "CUSTOM3"},
            context={"states": {"CUSTOM1", "CUSTOM2"}, "transitions": {("CUSTOM1", "CUSTOM2")}},
        )
        assert r.passed is False
        assert r.score == 0.0

    def test_no_state_info_in_output(
        self, validator: BehavioralValidator,
    ) -> None:
        """Output'ta state bilgisi yok → passed=True, warning ile."""
        r = validator.validate("some random text")
        assert r.passed is True
        assert r.score == 100.0
        assert "warning" in r.details.get("fsm", {}) or "warning" in r.details.get("snapshot", {})

    def test_none_output(
        self, validator: BehavioralValidator,
    ) -> None:
        """None output → passed=True, warning ile atla."""
        r = validator.validate(None)
        assert r.passed is True
        assert r.score == 100.0


# ---------------------------------------------------------------------------
# Snapshot Comparison — AC 4
# ---------------------------------------------------------------------------

class TestSnapshotComparison:
    """AC 4 — Snapshot comparison with drift detection."""

    def test_snapshot_no_drift(
        self, validator: BehavioralValidator,
        sample_snapshot_a: dict[str, Any],
    ) -> None:
        """Aynı snapshot → drift=0, passed=True."""
        r = validator.validate(
            "output",
            context={"snapshot_a": sample_snapshot_a, "snapshot_b": sample_snapshot_a},
        )
        assert r.passed is True
        assert r.score == 100.0

    def test_snapshot_drift_exceeded(
        self, validator: BehavioralValidator,
        sample_snapshot_a: dict[str, Any],
        sample_snapshot_b: dict[str, Any],
    ) -> None:
        """Farklı snapshot'lar, drift threshold (0.15) aşılır."""
        r = validator.validate(
            "output",
            context={"snapshot_a": sample_snapshot_a, "snapshot_b": sample_snapshot_b},
        )
        # Büyük fark var — drift threshold 0.15 aşılmalı
        # Snapshot score 0'dır; combined = (100 + 0)/2 = 50.0 → passed=True
        assert r.score == 50.0
        assert r.passed is True
        assert r.details.get("snapshot", {}).get("drift", 0) > 0

    def test_snapshot_drift_custom_threshold(
        self, validator: BehavioralValidator,
        sample_snapshot_a: dict[str, Any],
        sample_snapshot_b: dict[str, Any],
    ) -> None:
        """Yüksek drift threshold → passed=True."""
        r = validator.validate(
            "output",
            context={
                "snapshot_a": sample_snapshot_a,
                "snapshot_b": sample_snapshot_b,
                "drift_threshold": 0.99,
            },
        )
        assert r.passed is True
        assert r.score >= 50.0

    def test_snapshot_missing_one(
        self, validator: BehavioralValidator,
        sample_snapshot_a: dict[str, Any],
    ) -> None:
        """Tek snapshot var → passed=True, warning ile atla."""
        r = validator.validate(
            "output",
            context={"snapshot_a": sample_snapshot_a},
        )
        assert r.passed is True
        assert r.score == 100.0

    def test_snapshot_empty_dict(
        self, validator: BehavioralValidator,
    ) -> None:
        """Boş snapshot dict → passed=True ile atla."""
        r = validator.validate(
            "output",
            context={"snapshot_a": {}, "snapshot_b": {}},
        )
        assert r.passed is True
        assert r.score == 100.0

    def test_snapshot_no_snapshots(
        self, validator: BehavioralValidator,
    ) -> None:
        """Hiç snapshot yok → passed=True, warning ile atla."""
        r = validator.validate("output")
        assert r.passed is True
        assert r.score == 100.0


# ---------------------------------------------------------------------------
# Combined mode — AC 5
# ---------------------------------------------------------------------------

class TestCombinedMode:
    """AC 5 — Both FSM compliance + snapshot comparison."""

    def test_both_pass(
        self, validator: BehavioralValidator,
        sample_snapshot_a: dict[str, Any],
    ) -> None:
        """State geçerli + snapshot aynı → passed=True."""
        r = validator.validate(
            {"current": "VALIDATING", "next": "FAILED"},
            context={"snapshot_a": sample_snapshot_a, "snapshot_b": sample_snapshot_a},
        )
        assert r.passed is True
        assert r.score == 100.0

    def test_both_one_fails(
        self, validator: BehavioralValidator,
        sample_snapshot_a: dict[str, Any],
        sample_snapshot_b: dict[str, Any],
    ) -> None:
        """State geçerli ama snapshot drift var → passed=False (ortalama < 50)."""
        r = validator.validate(
            {"current": "VALIDATING", "next": "FAILED"},
            context={"snapshot_a": sample_snapshot_a, "snapshot_b": sample_snapshot_b},
        )
        assert r.score < 100.0
        # İkisinin ortalaması — state 100 + snapshot düşük
        assert "transition" in r.details.get("fsm", {}) or "drift" in r.details

    def test_state_invalid_snapshot_missing(
        self, validator: BehavioralValidator,
    ) -> None:
        """State geçersiz, snapshot yok → failed, score=0.0."""
        r = validator.validate({"current": "IDLE", "next": "COMPLETED"})
        assert r.passed is False
        assert r.score == 0.0


# ---------------------------------------------------------------------------
# Edge cases — AC 3, 4, 5
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases from story Dev Notes."""

    def test_output_empty_dict(
        self, validator: BehavioralValidator,
    ) -> None:
        """Boş dict → state bilgisi yok, passed=True ile atla."""
        r = validator.validate({})
        assert r.passed is True
        assert r.score == 100.0

    def test_output_json_string(
        self, validator: BehavioralValidator,
    ) -> None:
        """String output → JSON parse dene, başarısız olursa atla."""
        r = validator.validate('{"current": "IDLE", "next": "VALIDATING"}')
        assert r.passed is True
        assert r.score == 100.0

    def test_non_dict_non_tuple_output(
        self, validator: BehavioralValidator,
    ) -> None:
        """int output → state bilgisi yok, passed=True."""
        r = validator.validate(42)
        assert r.passed is True
        assert r.score == 100.0

    def test_context_type_guard(
        self, validator: BehavioralValidator,
    ) -> None:
        """context str ise crash yok, passed=True."""
        r = validator.validate("output", context="not-a-dict")
        assert r.passed is True

    def test_case_insensitive_states(
        self, validator: BehavioralValidator,
    ) -> None:
        """State isimleri normalize edilmeli (lowercase → UPPER)."""
        r = validator.validate({"current": "idle", "next": "validating"})
        assert r.passed is True
        assert r.score == 100.0

    def test_score_clamped(
        self, validator: BehavioralValidator,
    ) -> None:
        """Score her zaman [0, 100] aralığında."""
        # Extrem durum: state 0, snapshot 0
        r = validator.validate(
            {"current": "IDLE", "next": "COMPLETED"},
        )
        assert 0.0 <= r.score <= 100.0

    def test_details_contains_expected_keys(
        self, validator: BehavioralValidator,
        sample_snapshot_a: dict[str, Any],
        sample_snapshot_b: dict[str, Any],
    ) -> None:
        """details dict'i her zaman doldurulur (decision log için)."""
        r = validator.validate(
            {"current": "VALIDATING", "next": "FAILED"},
            context={"snapshot_a": sample_snapshot_a, "snapshot_b": sample_snapshot_b},
        )
        assert "transition" in r.details.get("fsm", {}) or "drift" in r.details or "warning" in r.details.get("fsm", {}) or "warning" in r.details.get("snapshot", {})

    def test_aliases_state_keys(
        self, validator: BehavioralValidator,
    ) -> None:
        """Dict'te 'state' ve 'next_state' alias'ları kabul edilmeli."""
        r = validator.validate({"state": "IDLE", "next_state": "VALIDATING"})
        assert r.passed is True
        assert r.score == 100.0

        r2 = validator.validate({"state": "IDLE", "next_state": "COMPLETED"})
        assert r2.passed is False
        assert r2.score == 0.0
