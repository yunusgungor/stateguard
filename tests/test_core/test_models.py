"""Tests for stateguard.models — enums, ValidationResult, EngineResult, DecisionEntry."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from stateguard.models import (
    DecisionEntry,
    EngineResult,
    ValidationDimension,
    ValidationResult,
    ValidationState,
    ValidationTier,
)


# ── Enum Tests ─────────────────────────────────────────────────────


class TestValidationDimension:
    def test_all_members_present(self):
        expected = {"STRUCTURAL", "SEMANTIC", "QUANTITATIVE", "BEHAVIORAL", "SECURITY"}
        assert {m.name for m in ValidationDimension} == expected

    def test_is_str_enum(self):
        assert isinstance(ValidationDimension.STRUCTURAL, str)
        assert ValidationDimension.STRUCTURAL == "structural"

    def test_values_are_unique(self):
        values = [m.value for m in ValidationDimension]
        assert len(values) == len(set(values))

    def test_serializable(self):
        d = {"dim": ValidationDimension.BEHAVIORAL}
        as_json = json.dumps(d, default=str)
        assert '"behavioral"' in as_json


class TestValidationTier:
    def test_all_members_present(self):
        expected = {"TIER_1", "TIER_2", "TIER_3"}
        assert {m.name for m in ValidationTier} == expected

    def test_is_str_enum(self):
        assert ValidationTier.TIER_1 == "tier_1"

    def test_ordering(self):
        tiers = list(ValidationTier)
        assert tiers == [ValidationTier.TIER_1, ValidationTier.TIER_2, ValidationTier.TIER_3]


class TestValidationState:
    def test_all_members_present(self):
        expected = {"IDLE", "VALIDATING", "PASSED", "FAILED", "RETRY", "HITL_WAITING", "COMPLETED"}
        assert {m.name for m in ValidationState} == expected

    def test_is_str_enum(self):
        assert ValidationState.IDLE == "idle"


# ── ValidationResult Tests ─────────────────────────────────────────


class TestValidationResult:
    def test_minimal_creation(self):
        vr = ValidationResult(dimension=ValidationDimension.STRUCTURAL)
        assert vr.score == 0.0
        assert vr.passed is False
        assert vr.details == {}
        assert vr.error is None

    def test_full_creation(self):
        vr = ValidationResult(
            score=92.5,
            passed=True,
            dimension=ValidationDimension.SEMANTIC,
            details={"cosine_similarity": 0.95},
            error=None,
        )
        assert vr.score == 92.5
        assert vr.passed is True
        assert vr.dimension == ValidationDimension.SEMANTIC

    def test_score_below_zero_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to"):
            ValidationResult(score=-1.0, dimension=ValidationDimension.STRUCTURAL)

    def test_score_above_100_raises(self):
        with pytest.raises(ValidationError, match="less than or equal to"):
            ValidationResult(score=101.0, dimension=ValidationDimension.STRUCTURAL)

    def test_score_zero_and_hundred_accepted(self):
        vr0 = ValidationResult(score=0.0, dimension=ValidationDimension.STRUCTURAL)
        vr100 = ValidationResult(score=100.0, dimension=ValidationDimension.STRUCTURAL)
        assert vr0.score == 0.0
        assert vr100.score == 100.0

    def test_dimension_required(self):
        with pytest.raises(ValidationError, match="Field required"):
            ValidationResult(score=50.0)

    def test_details_isolation(self):
        vr1 = ValidationResult(dimension=ValidationDimension.STRUCTURAL)
        vr2 = ValidationResult(dimension=ValidationDimension.SEMANTIC)
        vr1.details["key"] = "value"
        assert "key" not in vr2.details

    def test_model_dump(self):
        vr = ValidationResult(score=75.0, passed=True, dimension=ValidationDimension.SECURITY)
        dumped = vr.model_dump()
        assert dumped["score"] == 75.0
        assert dumped["dimension"] == "security"
        assert dumped["passed"] is True

    def test_model_dump_json(self):
        vr = ValidationResult(score=50.0, dimension=ValidationDimension.QUANTITATIVE)
        raw = vr.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["score"] == 50.0
        assert parsed["dimension"] == "quantitative"


# ── EngineResult Tests ─────────────────────────────────────────────


class TestEngineResult:
    def test_minimal_creation(self):
        er = EngineResult()
        assert er.overall_score == 0.0
        assert er.passed is False
        assert er.tier_path == []
        assert er.dimension_scores == {}

    def test_full_creation(self):
        er = EngineResult(
            overall_score=85.0,
            passed=True,
            tier_path=[1, 2],
            dimension_scores={
                ValidationDimension.STRUCTURAL: 90.0,
                ValidationDimension.SEMANTIC: 80.0,
            },
        )
        assert er.overall_score == 85.0
        assert er.tier_path == [1, 2]
        assert er.dimension_scores["structural"] == 90.0

    def test_score_below_zero_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to"):
            EngineResult(overall_score=-5.0)

    def test_score_above_100_raises(self):
        with pytest.raises(ValidationError, match="less than or equal to"):
            EngineResult(overall_score=200.0)

    def test_dimension_scores_type(self):
        er = EngineResult()
        # Should accept ValidationDimension keys
        er.dimension_scores["structural"] = 95.0
        assert er.dimension_scores["structural"] == 95.0

    def test_tier_path_isolation(self):
        er1 = EngineResult()
        er2 = EngineResult()
        er1.tier_path.append(1)
        assert len(er2.tier_path) == 0

    def test_model_dump_json_roundtrip(self):
        er = EngineResult(overall_score=77.0, passed=True, tier_path=[1, 3])
        raw = er.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["overall_score"] == 77.0
        assert parsed["tier_path"] == [1, 3]


# ── DecisionEntry Tests ────────────────────────────────────────────


class TestDecisionEntry:
    def test_minimal_creation(self):
        entry = DecisionEntry(
            agent_id="agent-1",
            step_id="s1",
            dimension=ValidationDimension.STRUCTURAL,
            score=85.0,
            decision="pass",
        )
        assert entry.agent_id == "agent-1"
        assert entry.decision == "pass"
        assert isinstance(entry.timestamp, datetime)

    def test_timestamp_default_is_utc(self):
        entry = DecisionEntry(
            agent_id="a", step_id="s", dimension=ValidationDimension.STRUCTURAL,
            score=50.0, decision="fail",
        )
        assert entry.timestamp.tzinfo is not None
        assert entry.timestamp.tzinfo.utcoffset(entry.timestamp) == timezone.utc.utcoffset(entry.timestamp)

    def test_score_constraint(self):
        with pytest.raises(ValidationError):
            DecisionEntry(
                agent_id="a", step_id="s", dimension=ValidationDimension.STRUCTURAL,
                score=150.0, decision="pass",
            )

    def test_details_default_factory(self):
        e1 = DecisionEntry(
            agent_id="a", step_id="s", dimension=ValidationDimension.STRUCTURAL,
            score=50.0, decision="pass",
        )
        e2 = DecisionEntry(
            agent_id="b", step_id="t", dimension=ValidationDimension.STRUCTURAL,
            score=60.0, decision="fail",
        )
        e1.details["key"] = "val"
        assert "key" not in e2.details
