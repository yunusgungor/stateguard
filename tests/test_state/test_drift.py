"""Tests for DriftDetector — cumulative drift detection across pipeline steps."""

from __future__ import annotations

from typing import Any

import pytest

from stateguard.state.drift import DriftDetector
from stateguard.state.snapshot import SnapshotManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> DriftDetector:
    return DriftDetector()


@pytest.fixture
def custom_detector() -> DriftDetector:
    return DriftDetector(threshold=50.0)


@pytest.fixture
def sample_diff() -> dict[str, Any]:
    return {
        "added": {"new_field": "hello"},
        "removed": {},
        "changed": {
            "prompt": {
                "old": "What is the capital of France?",
                "new": "What is the capital of Germany?",
                "type": "str",
            },
        },
        "total_changes": 2,
    }


@pytest.fixture
def sample_snapshot_a() -> dict[str, Any]:
    return {
        "prompt": "What is the capital of France?",
        "context": {"difficulty": "easy"},
        "partial_output": "Paris is the capital of France.",
        "validation_scores": {"structural": 95.0, "semantic": 88.0},
        "step_counter": 2,
        "_timestamp": "2026-05-21T12:00:00+00:00",
    }


@pytest.fixture
def sample_snapshot_b() -> dict[str, Any]:
    return {
        "prompt": "What is the capital of Germany?",
        "context": {"difficulty": "medium"},
        "partial_output": "Berlin is the capital of Germany.",
        "validation_scores": {"structural": 92.0, "semantic": 85.0},
        "step_counter": 3,
        "_timestamp": "2026-05-21T12:01:00+00:00",
    }


# ---------------------------------------------------------------------------
# __init__ and threshold validation
# ---------------------------------------------------------------------------


class TestInit:
    """Constructor — threshold validation (AC-5)."""

    def test_default_threshold(self, detector: DriftDetector) -> None:
        assert detector.threshold == 15.0

    def test_custom_threshold(self, custom_detector: DriftDetector) -> None:
        assert custom_detector.threshold == 50.0

    def test_threshold_float_conversion(self) -> None:
        detector = DriftDetector(threshold=20)
        assert detector.threshold == 20.0
        assert isinstance(detector.threshold, float)

    def test_threshold_zero(self) -> None:
        detector = DriftDetector(threshold=0.0)
        assert detector.threshold == 0.0

    def test_threshold_one_hundred(self) -> None:
        detector = DriftDetector(threshold=100.0)
        assert detector.threshold == 100.0

    def test_threshold_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            DriftDetector(threshold=-1.0)

    def test_threshold_above_hundred_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            DriftDetector(threshold=101.0)

    def test_threshold_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be finite"):
            DriftDetector(threshold=float("nan"))

    def test_threshold_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be finite"):
            DriftDetector(threshold=float("inf"))

    def test_threshold_bool_raises(self) -> None:
        with pytest.raises(TypeError, match="threshold must be a number"):
            DriftDetector(threshold=True)

    def test_threshold_string_raises(self) -> None:
        with pytest.raises(TypeError, match="threshold must be a number"):
            DriftDetector(threshold="15.0")

    def test_initial_state(self, detector: DriftDetector) -> None:
        assert detector._cumulative_drift == 0.0
        assert detector._drift_history == []
        assert detector._origin_step is None


# ---------------------------------------------------------------------------
# calculate() — AC-1
# ---------------------------------------------------------------------------


class TestCalculate:
    """calculate() — drift computation from SnapshotManager.diff() output."""

    def test_calculate_returns_all_keys(
        self, detector: DriftDetector, sample_diff: dict[str, Any],
    ) -> None:
        result = detector.calculate(sample_diff)
        assert "drift_score" in result
        assert "structural_drift" in result
        assert "text_dissimilarity" in result
        assert "total_changes" in result

    def test_calculate_empty_diff(self, detector: DriftDetector) -> None:
        diff = {"added": {}, "removed": {}, "changed": {}, "total_changes": 0}
        result = detector.calculate(diff)
        assert result["drift_score"] == 0.0
        assert result["structural_drift"] == 0.0
        assert result["text_dissimilarity"] == 0.0
        assert result["total_changes"] == 0

    def test_calculate_added_only(self, detector: DriftDetector) -> None:
        diff = {
            "added": {"a": 1, "b": 2},
            "removed": {},
            "changed": {},
            "total_changes": 2,
        }
        result = detector.calculate(diff)
        assert result["structural_drift"] == 5.0  # 2 * 2.5
        assert result["total_changes"] == 2
        assert 0.0 <= result["drift_score"] <= 100.0

    def test_calculate_removed_only(self, detector: DriftDetector) -> None:
        diff = {
            "added": {},
            "removed": {"x": "old"},
            "changed": {},
            "total_changes": 1,
        }
        result = detector.calculate(diff)
        assert result["structural_drift"] == 2.5  # 1 * 2.5

    def test_calculate_changed_only(self, detector: DriftDetector) -> None:
        diff = {
            "added": {},
            "removed": {},
            "changed": {
                "score": {"old": 50, "new": 80, "type": "int"},
            },
            "total_changes": 1,
        }
        result = detector.calculate(diff)
        assert result["structural_drift"] == 5.0  # 1 * 5.0

    def test_calculate_with_text_dissimilarity(
        self, detector: DriftDetector, sample_diff: dict[str, Any],
    ) -> None:
        result = detector.calculate(sample_diff)
        # 1 added (2.5) + 1 changed (5.0) = 7.5 structural
        assert result["structural_drift"] == 7.5
        # text dissimilarity penalty should be > 0 (different prompts)
        assert result["text_dissimilarity"] > 0.0
        assert result["text_dissimilarity"] <= 10.0
        assert result["total_changes"] == 2

    def test_calculate_non_string_changed_no_text_penalty(
        self, detector: DriftDetector,
    ) -> None:
        diff = {
            "added": {},
            "removed": {},
            "changed": {
                "score": {"old": 50, "new": 80, "type": "int"},
            },
            "total_changes": 1,
        }
        result = detector.calculate(diff)
        assert result["text_dissimilarity"] == 0.0
        assert result["drift_score"] == result["structural_drift"]

    def test_calculate_score_clamped_above(self, detector: DriftDetector) -> None:
        # 100 changes * 5.0 = 500 → should clamp to 100.0
        diff = {
            "added": {},
            "removed": {},
            "changed": {str(i): {"old": i, "new": i + 1, "type": "int"}
                        for i in range(100)},
            "total_changes": 100,
        }
        result = detector.calculate(diff)
        assert result["drift_score"] == 100.0
        assert result["structural_drift"] == 500.0

    def test_calculate_score_clamped_below(self, detector: DriftDetector) -> None:
        diff = {
            "added": {},
            "removed": {},
            "changed": {},
            "total_changes": 0,
        }
        result = detector.calculate(diff)
        assert result["drift_score"] == 0.0

    def test_calculate_none_raises_type_error(self, detector: DriftDetector) -> None:
        with pytest.raises(TypeError, match="diff_result must be a dict"):
            detector.calculate(None)  # type: ignore[arg-type]

    def test_calculate_non_dict_raises_type_error(
        self, detector: DriftDetector,
    ) -> None:
        with pytest.raises(TypeError, match="diff_result must be a dict"):
            detector.calculate("not a dict")  # type: ignore[arg-type]

    def test_calculate_missing_total_changes_raises(
        self, detector: DriftDetector,
    ) -> None:
        with pytest.raises(ValueError, match="missing required key"):
            detector.calculate({"added": {}, "removed": {}})

    def test_calculate_missing_added_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(ValueError, match="missing required key"):
            detector.calculate({"total_changes": 0, "removed": {}, "changed": {}})

    def test_calculate_missing_removed_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(ValueError, match="missing required key"):
            detector.calculate({"total_changes": 0, "added": {}, "changed": {}})

    def test_calculate_missing_changed_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(ValueError, match="missing required key"):
            detector.calculate({"total_changes": 0, "added": {}, "removed": {}})

    def test_calculate_non_dict_added_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(TypeError, match="diff_result\\['added'\\]"):
            detector.calculate({"added": "bad", "removed": {}, "changed": {}, "total_changes": 0})

    def test_calculate_non_dict_changed_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(TypeError, match="diff_result\\['changed'\\]"):
            detector.calculate({"added": {}, "removed": {}, "changed": 42, "total_changes": 0})

    def test_calculate_text_dissimilarity_multiple_strings(
        self, detector: DriftDetector,
    ) -> None:
        diff = {
            "added": {},
            "removed": {},
            "changed": {
                "prompt": {
                    "old": "Hello world",
                    "new": "Hello universe",
                    "type": "str",
                },
                "summary": {
                    "old": "Quick brown fox",
                    "new": "Quick brown dog",
                    "type": "str",
                },
            },
            "total_changes": 2,
        }
        result = detector.calculate(diff)
        assert result["structural_drift"] == 10.0  # 2 * 5.0
        assert result["text_dissimilarity"] > 0.0
        assert result["drift_score"] > result["structural_drift"]

    def test_calculate_text_dissimilarity_identical_string(
        self, detector: DriftDetector,
    ) -> None:
        diff = {
            "added": {},
            "removed": {},
            "changed": {
                "prompt": {
                    "old": "Hello world",
                    "new": "Hello world",
                    "type": "str",
                },
            },
            "total_changes": 1,
        }
        result = detector.calculate(diff)
        assert result["text_dissimilarity"] == 0.0
        # Identical strings → no text penalty despite being in changed

    def test_calculate_empty_string_changed(
        self, detector: DriftDetector,
    ) -> None:
        diff = {
            "added": {},
            "removed": {},
            "changed": {
                "note": {
                    "old": "",
                    "new": "new content",
                    "type": "str",
                },
            },
            "total_changes": 1,
        }
        result = detector.calculate(diff)
        assert result["structural_drift"] == 5.0
        # '' vs 'new content' → dissimilarity 1.0, penalty = 1.0 * 10.0 = 10.0
        assert result["text_dissimilarity"] == 10.0


# ---------------------------------------------------------------------------
# check() — AC-2
# ---------------------------------------------------------------------------


class TestCheck:
    """check() — threshold evaluation."""

    def test_check_drift_detected(self, detector: DriftDetector) -> None:
        result = detector.check(20.0)
        assert result["drift_detected"] is True
        assert result["decision"] == "block"

    def test_check_no_drift(self, detector: DriftDetector) -> None:
        result = detector.check(10.0)
        assert result["drift_detected"] is False
        assert result["decision"] == "allow"

    def test_check_exact_threshold(self, detector: DriftDetector) -> None:
        result = detector.check(15.0)
        assert result["drift_detected"] is True  # >= threshold

    def test_check_custom_threshold(self, custom_detector: DriftDetector) -> None:
        result = custom_detector.check(49.0)
        assert result["drift_detected"] is False
        result = custom_detector.check(50.0)
        assert result["drift_detected"] is True

    def test_check_clamp_negative(self, detector: DriftDetector) -> None:
        result = detector.check(-5.0)
        assert result["cumulative_drift"] == 0.0
        assert result["drift_detected"] is False

    def test_check_clamp_above(self, detector: DriftDetector) -> None:
        result = detector.check(200.0)
        assert result["cumulative_drift"] == 100.0
        assert result["drift_detected"] is True

    def test_check_zero_drift(self, detector: DriftDetector) -> None:
        result = detector.check(0.0)
        assert result["drift_detected"] is False
        assert result["cumulative_drift"] == 0.0

    def test_check_returns_all_keys(self, detector: DriftDetector) -> None:
        result = detector.check(10.0)
        assert "drift_detected" in result
        assert "cumulative_drift" in result
        assert "threshold" in result
        assert "decision" in result
        assert result["threshold"] == 15.0

    def test_check_none_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(TypeError, match="must be a number"):
            detector.check(None)  # type: ignore[arg-type]

    def test_check_string_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(TypeError, match="must be a number"):
            detector.check("15.0")  # type: ignore[arg-type]

    def test_check_bool_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(TypeError, match="must be a number"):
            detector.check(True)  # type: ignore[arg-type]

    def test_check_nan_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            detector.check(float("nan"))

    def test_check_inf_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            detector.check(float("inf"))

    def test_check_int_accepted(self, detector: DriftDetector) -> None:
        result = detector.check(20)
        assert result["drift_detected"] is True
        assert result["cumulative_drift"] == 20.0


# ---------------------------------------------------------------------------
# accumulate / get_report / reset — AC-3
# ---------------------------------------------------------------------------


class TestAccumulate:
    """accumulate() — cumulative drift tracking."""

    def test_accumulate_adds_to_cumulative(self, detector: DriftDetector) -> None:
        detector.accumulate(5.0)
        assert detector._cumulative_drift == 5.0

    def test_accumulate_multiple_steps(self, detector: DriftDetector) -> None:
        detector.accumulate(3.0)
        detector.accumulate(4.0)
        detector.accumulate(2.0)
        assert detector._cumulative_drift == 9.0

    def test_accumulate_tracks_history(self, detector: DriftDetector) -> None:
        detector.accumulate(3.0)
        detector.accumulate(4.0)
        assert len(detector._drift_history) == 2
        assert detector._drift_history[0] == {"step": 1, "drift": 3.0, "cumulative": 3.0}
        assert detector._drift_history[1] == {"step": 2, "drift": 4.0, "cumulative": 7.0}

    def test_accumulate_nan_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            detector.accumulate(float("nan"))
        # State unchanged after error
        assert detector._cumulative_drift == 0.0
        assert detector._drift_history == []

    def test_accumulate_inf_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            detector.accumulate(float("inf"))
        # State unchanged after error
        assert detector._cumulative_drift == 0.0
        assert detector._drift_history == []

    def test_accumulate_bool_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(TypeError, match="must be a number"):
            detector.accumulate(True)  # type: ignore[arg-type]
        # State unchanged after error
        assert detector._cumulative_drift == 0.0
        assert detector._drift_history == []

    def test_accumulate_none_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(TypeError, match="must be a number"):
            detector.accumulate(None)  # type: ignore[arg-type]
        # State unchanged after error
        assert detector._cumulative_drift == 0.0
        assert detector._drift_history == []

    def test_accumulate_string_raises(self, detector: DriftDetector) -> None:
        with pytest.raises(TypeError, match="must be a number"):
            detector.accumulate("5.0")  # type: ignore[arg-type]
        # State unchanged after error
        assert detector._cumulative_drift == 0.0
        assert detector._drift_history == []

    def test_accumulate_negative_allowed(self, detector: DriftDetector) -> None:
        detector.accumulate(10.0)
        detector.accumulate(-3.0)
        assert detector._cumulative_drift == 7.0


class TestOriginStep:
    """origin_step tracking."""

    def test_origin_step_none_initially(self, detector: DriftDetector) -> None:
        assert detector._origin_step is None

    def test_origin_step_set_on_first_positive(self, detector: DriftDetector) -> None:
        detector.accumulate(0.0)  # zero drift → no origin
        assert detector._origin_step is None
        detector.accumulate(5.0)  # positive drift → origin step 2
        assert detector._origin_step == 2

    def test_origin_step_persists(self, detector: DriftDetector) -> None:
        detector.accumulate(0.0)
        detector.accumulate(5.0)
        assert detector._origin_step == 2
        detector.accumulate(0.0)  # zero drift → origin unchanged
        assert detector._origin_step == 2

    def test_origin_step_resets_with_reset(self, detector: DriftDetector) -> None:
        detector.accumulate(5.0)
        detector.reset()
        assert detector._origin_step is None


class TestGetReport:
    """get_report() — full drift report."""

    def test_report_after_no_accumulate(self, detector: DriftDetector) -> None:
        report = detector.get_report()
        assert report["cumulative_drift"] == 0.0
        assert report["drift_history"] == []
        assert report["origin_step"] is None
        assert report["drift_detected"] is False
        assert report["threshold_exceeded"] is False

    def test_report_after_accumulate(self, detector: DriftDetector) -> None:
        detector.accumulate(5.0)
        detector.accumulate(10.0)
        report = detector.get_report()
        assert report["cumulative_drift"] == 15.0
        assert len(report["drift_history"]) == 2
        assert report["origin_step"] == 1
        assert report["drift_detected"] is True  # 15.0 >= 15.0
        assert report["threshold_exceeded"] is True

    def test_report_contains_all_keys(self, detector: DriftDetector) -> None:
        report = detector.get_report()
        assert "cumulative_drift" in report
        assert "threshold" in report
        assert "drift_detected" in report
        assert "origin_step" in report
        assert "drift_history" in report
        assert "threshold_exceeded" in report
        assert report["threshold"] == 15.0

    def test_report_mutation_safety(self, detector: DriftDetector) -> None:
        """Verify get_report() returns a deep copy — mutations don't affect internal state."""
        detector.accumulate(5.0)
        detector.accumulate(10.0)
        report = detector.get_report()

        # Mutate the returned report's history
        report["drift_history"][0]["drift"] = 999.0
        report["drift_history"].append({"step": 99, "drift": 99.0, "cumulative": 999.0})

        # Internal state must be unchanged
        assert detector._drift_history[0]["drift"] == 5.0
        assert len(detector._drift_history) == 2


class TestReset:
    """reset() — state cleanup."""

    def test_reset_clears_cumulative(self, detector: DriftDetector) -> None:
        detector.accumulate(10.0)
        detector.reset()
        assert detector._cumulative_drift == 0.0

    def test_reset_clears_history(self, detector: DriftDetector) -> None:
        detector.accumulate(5.0)
        detector.reset()
        assert detector._drift_history == []

    def test_reset_clears_origin(self, detector: DriftDetector) -> None:
        detector.accumulate(5.0)
        detector.reset()
        assert detector._origin_step is None

    def test_reset_idempotent(self, detector: DriftDetector) -> None:
        detector.reset()
        assert detector._cumulative_drift == 0.0
        assert detector._drift_history == []
        assert detector._origin_step is None


# ---------------------------------------------------------------------------
# calculate_from_snapshots — AC-4
# ---------------------------------------------------------------------------


class TestCalculateFromSnapshots:
    """calculate_from_snapshots() — direct snapshot pair drift."""

    def test_with_different_snapshots(
        self,
        detector: DriftDetector,
        sample_snapshot_a: dict[str, Any],
        sample_snapshot_b: dict[str, Any],
    ) -> None:
        result = detector.calculate_from_snapshots(sample_snapshot_a, sample_snapshot_b)
        assert "drift_score" in result
        assert "structural_drift" in result
        assert "total_changes" in result
        assert result["total_changes"] > 0
        assert result["drift_score"] > 0.0

    def test_identical_snapshots(
        self,
        detector: DriftDetector,
        sample_snapshot_a: dict[str, Any],
    ) -> None:
        result = detector.calculate_from_snapshots(sample_snapshot_a, sample_snapshot_a)
        assert result["drift_score"] == 0.0
        assert result["total_changes"] == 0

    def test_timestamp_excluded(
        self,
        detector: DriftDetector,
        sample_snapshot_a: dict[str, Any],
    ) -> None:
        # Snapshots with only timestamp difference
        snap_b = dict(sample_snapshot_a, _timestamp="2026-05-21T13:00:00+00:00")
        result = detector.calculate_from_snapshots(sample_snapshot_a, snap_b)
        assert result["drift_score"] == 0.0
        assert result["total_changes"] == 0

    def test_added_keys_detected(
        self,
        detector: DriftDetector,
        sample_snapshot_a: dict[str, Any],
    ) -> None:
        snap_b = dict(sample_snapshot_a, new_key="extra")
        snap_b["_timestamp"] = "2026-05-21T13:00:00+00:00"
        result = detector.calculate_from_snapshots(sample_snapshot_a, snap_b)
        assert result["total_changes"] == 1
        assert result["structural_drift"] == 2.5

    def test_removed_keys_detected(
        self,
        detector: DriftDetector,
        sample_snapshot_a: dict[str, Any],
    ) -> None:
        snap_b = {k: v for k, v in sample_snapshot_a.items() if k != "prompt"}
        snap_b["_timestamp"] = "2026-05-21T13:00:00+00:00"
        result = detector.calculate_from_snapshots(sample_snapshot_a, snap_b)
        assert result["total_changes"] == 1
        assert result["structural_drift"] == 2.5

    def test_changed_values_detected(
        self,
        detector: DriftDetector,
        sample_snapshot_a: dict[str, Any],
    ) -> None:
        """Values changing between snapshots detected as changed."""
        snap_b = dict(sample_snapshot_a)
        snap_b["step_counter"] = 99
        snap_b["_timestamp"] = "2026-05-21T13:00:00+00:00"
        result = detector.calculate_from_snapshots(sample_snapshot_a, snap_b)
        assert result["total_changes"] == 1
        assert result["structural_drift"] == 5.0  # 1 changed × 5.0

    def test_none_snapshot_raises(
        self,
        detector: DriftDetector,
        sample_snapshot_a: dict[str, Any],
    ) -> None:
        with pytest.raises(TypeError, match="must be a dict"):
            detector.calculate_from_snapshots(None, sample_snapshot_a)  # type: ignore[arg-type]

    def test_non_dict_snapshot_raises(
        self,
        detector: DriftDetector,
    ) -> None:
        with pytest.raises(TypeError, match="must be a dict"):
            detector.calculate_from_snapshots("bad", {})  # type: ignore[arg-type]

    def test_empty_snapshots(self, detector: DriftDetector) -> None:
        result = detector.calculate_from_snapshots({}, {})
        assert result["drift_score"] == 0.0
        assert result["total_changes"] == 0


# ---------------------------------------------------------------------------
# Edge cases — AC-6
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case coverage."""

    def test_drift_detector_negative_accumulate(
        self, detector: DriftDetector,
    ) -> None:
        # Negative drift means recovery
        detector.accumulate(10.0)
        detector.accumulate(-2.0)
        assert detector._cumulative_drift == 8.0

    def test_drift_detector_toggle_threshold(
        self, detector: DriftDetector,
    ) -> None:
        # Change threshold at runtime
        detector.threshold = 50.0
        assert detector.check(30.0)["drift_detected"] is False
        assert detector.check(50.0)["drift_detected"] is True

    def test_drift_detector_report_no_history(
        self, detector: DriftDetector,
    ) -> None:
        report = detector.get_report()
        assert report["cumulative_drift"] == 0.0
        assert report["origin_step"] is None
        assert report["drift_history"] == []

    def test_drift_detector_threshold_zero(
        self, detector: DriftDetector,
    ) -> None:
        detector.threshold = 0.0
        assert detector.check(0.0)["drift_detected"] is True

    def test_drift_detector_threshold_100(
        self, detector: DriftDetector,
    ) -> None:
        detector.threshold = 100.0
        assert detector.check(99.9)["drift_detected"] is False
        assert detector.check(100.0)["drift_detected"] is True

    def test_drift_detector_calculate_empty_added_removed(
        self, detector: DriftDetector,
    ) -> None:
        diff = {
            "added": {},
            "removed": {},
            "changed": {
                "field": {"old": "a", "new": "b", "type": "str"},
            },
            "total_changes": 1,
        }
        result = detector.calculate(diff)
        assert result["structural_drift"] == 5.0
        assert result["text_dissimilarity"] > 0.0

    def test_drift_detector_accumulate_zero_then_positive(
        self, detector: DriftDetector,
    ) -> None:
        detector.accumulate(0.0)
        assert detector._origin_step is None
        detector.accumulate(1.0)
        assert detector._origin_step == 2  # first positive is step 2


# ---------------------------------------------------------------------------
# Integration — SnapshotManager + DriftDetector
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end: SnapshotManager → DriftDetector."""

    def test_take_snapshot_then_calculate(
        self,
        detector: DriftDetector,
    ) -> None:
        sm = SnapshotManager()
        data_a = {"prompt": "Hello world", "score": 90}
        data_b = {"prompt": "Hello universe", "score": 95}

        snap_a = sm.take_snapshot(data_a)["data"]
        snap_b = sm.take_snapshot(data_b)["data"]

        diff_result = sm.diff(snap_a, snap_b)
        result = detector.calculate(diff_result)

        assert "drift_score" in result
        assert result["total_changes"] > 0
        assert result["structural_drift"] > 0.0

    def test_full_pipeline(
        self,
        detector: DriftDetector,
    ) -> None:
        sm = SnapshotManager()

        # Step 1
        snap1 = sm.take_snapshot({"prompt": "A", "score": 90})["data"]
        # Step 2
        snap2 = sm.take_snapshot({"prompt": "A", "score": 85})["data"]
        d1 = sm.diff(snap1, snap2)
        r1 = detector.calculate(d1)
        detector.accumulate(r1["drift_score"])

        # Step 3
        snap3 = sm.take_snapshot({"prompt": "B", "score": 80})["data"]
        d2 = sm.diff(snap2, snap3)
        r2 = detector.calculate(d2)
        detector.accumulate(r2["drift_score"])

        report = detector.get_report()
        assert report["cumulative_drift"] > 0.0
        assert len(report["drift_history"]) == 2

    def test_threshold_block_in_pipeline(
        self,
        detector: DriftDetector,
    ) -> None:
        sm = SnapshotManager()
        detector.threshold = 20.0

        snap_a = sm.take_snapshot({"prompt": "Hello world, this is a test"})["data"]
        snap_b = sm.take_snapshot({"prompt": "Goodbye world, completely different"})["data"]

        diff_result = sm.diff(snap_a, snap_b)
        step_drift = detector.calculate(diff_result)["drift_score"]
        detector.accumulate(step_drift)

        check = detector.check(detector._cumulative_drift)
        if check["drift_detected"]:
            assert check["decision"] == "block"
