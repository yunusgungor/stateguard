"""Tests for ScoreCard — weighted multi-dimension scoring."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from stateguard.core.scoring import DEFAULT_WEIGHTS, ScoreCard
from stateguard.models.enums import ValidationDimension
from stateguard.models.result import EngineResult


@pytest.fixture
def sample_scores() -> dict[ValidationDimension, float]:
    """All 5 dimensions with passing scores."""
    return {
        ValidationDimension.STRUCTURAL: 95.0,
        ValidationDimension.SEMANTIC: 88.0,
        ValidationDimension.QUANTITATIVE: 72.0,
        ValidationDimension.BEHAVIORAL: 90.0,
        ValidationDimension.SECURITY: 100.0,
    }


class TestScoreCard:
    """ScoreCard birim testleri."""

    def test_class_attributes(self):
        """Sınıf attribute'ları doğru set edilmiş."""
        card = ScoreCard()
        assert card.name == "score-card"

    def test_default_weights(self):
        """Varsayılan ağırlıklar doğru."""
        card = ScoreCard()
        for dim, w in DEFAULT_WEIGHTS.items():
            assert card.weights[dim] == pytest.approx(w, rel=1e-2)

    def test_weighted_average_full_set(self, sample_scores):
        """Tüm 5 boyut girilince doğru ağırlıklı ortalama.
        95*0.25 + 88*0.25 + 72*0.15 + 90*0.20 + 100*0.15
        = 23.75 + 22.0 + 10.8 + 18.0 + 15.0 = 89.55
        """
        card = ScoreCard()
        result = card.calculate(sample_scores)
        assert isinstance(result, EngineResult)
        assert result.overall_score == pytest.approx(89.55, rel=1e-2)
        assert result.passed is True

    def test_high_score_passes(self):
        """Score >= 75 → passed=True."""
        card = ScoreCard()
        scores = {ValidationDimension.STRUCTURAL: 80.0}
        result = card.calculate(scores)
        assert result.passed is True
        assert result.overall_score >= 75.0

    def test_borderline_score(self):
        """Score 50-74 → passed=False, borderline=True."""
        card = ScoreCard()
        scores = {ValidationDimension.STRUCTURAL: 60.0}
        result = card.calculate(scores)
        assert result.passed is False
        assert result.details["scoring"]["borderline"] is True

    def test_low_score_fails_with_stop(self):
        """Score < 50 → passed=False, stop_pipeline=True."""
        card = ScoreCard()
        scores = {ValidationDimension.STRUCTURAL: 30.0}
        result = card.calculate(scores)
        assert result.passed is False
        assert result.details["scoring"]["stop_pipeline"] is True

    def test_single_dimension_renormalization(self):
        """1 boyut eksik → yeniden normalizasyon, score=91.5."""
        card = ScoreCard()
        scores = {
            ValidationDimension.STRUCTURAL: 95.0,
            ValidationDimension.SEMANTIC: 88.0,
            # QUANTITATIVE, BEHAVIORAL, SECURITY eksik
        }
        result = card.calculate(scores)
        # Renormalized: structural weight = 0.25/0.5 = 0.5, semantic = 0.5
        # score = 95*0.5 + 88*0.5 = 47.5 + 44.0 = 91.5
        assert result.overall_score == pytest.approx(91.5, rel=1e-2)
        assert result.details["scoring"]["normalization"]["applied"] is True
        assert "quantitative" in result.details["scoring"]["normalization"]["missing_dimensions"]

    def test_three_dimensions_renormalization(self):
        """3 boyut eksik → yeniden normalizasyon."""
        card = ScoreCard()
        scores = {
            ValidationDimension.STRUCTURAL: 90.0,
            ValidationDimension.SEMANTIC: 85.0,
        }
        result = card.calculate(scores)
        # Sadece structural + semantic var → ağırlıklar 0.5/0.5
        assert result.overall_score == pytest.approx(87.5, rel=1e-2)

    def test_all_dimensions_missing_raises_error(self):
        """Tüm boyutlar eksik → error döner."""
        card = ScoreCard()
        result = card.calculate({})
        assert result.passed is False
        assert result.overall_score == 0.0
        assert "No dimension scores" in (
            result.details.get("scoring", {}).get("error") or ""
        )

    def test_custom_weights_override(self):
        """Custom weights constructor'dan geçilebilir."""
        custom = {
            "structural": 0.5,
            "semantic": 0.5,
        }
        card = ScoreCard(weights=custom)
        scores = {
            ValidationDimension.STRUCTURAL: 100.0,
            ValidationDimension.SEMANTIC: 0.0,
        }
        result = card.calculate(scores)
        assert result.overall_score == pytest.approx(50.0, rel=1e-2)

    def test_negative_weights_raises_error(self):
        """Negatif weight → ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            ScoreCard(weights={"structural": -0.5})

    def test_engine_result_format(self):
        """EngineResult formatı doğru."""
        card = ScoreCard()
        scores = {ValidationDimension.STRUCTURAL: 85.0}
        result = card.calculate(scores)
        assert isinstance(result, EngineResult)
        assert hasattr(result, "overall_score")
        assert hasattr(result, "passed")
        assert hasattr(result, "dimension_scores")
        assert "structural" in result.dimension_scores
        assert result.dimension_scores["structural"] == 85.0

    def test_weighted_contributions_in_details(self):
        """Her boyutun weighted contribution'ı details'te."""
        card = ScoreCard()
        scores = {
            ValidationDimension.STRUCTURAL: 100.0,
            ValidationDimension.SEMANTIC: 100.0,
        }
        result = card.calculate(scores)
        contributions = result.details["scoring"]["weighted_contributions"]
        assert "structural" in contributions
        assert "semantic" in contributions
        # structural: 100.0 * (0.25/0.5) = 50.0
        assert contributions["structural"] == pytest.approx(50.0, rel=1e-2)

    def test_score_bounds_clamped(self):
        """Score her zaman [0, 100] arasında kalır, negatif skor clamp edilir."""
        card = ScoreCard()
        # Negatif skor → clamp edilir
        scores = {ValidationDimension.STRUCTURAL: -10.0}
        result = card.calculate(scores)
        assert 0.0 <= result.overall_score <= 100.0
        # dimension_score da clamp edilmiş olmalı
        assert result.dimension_scores["structural"] == 0.0

    def test_returns_engine_result_not_dict(self):
        """Dönüş tipi EngineResult, dict değil."""
        card = ScoreCard()
        result = card.calculate({ValidationDimension.STRUCTURAL: 80.0})
        assert isinstance(result, EngineResult)
        assert not isinstance(result, dict)

    def test_dimension_scores_preserved(self):
        """dimension_scores alanı gelen skorları korur."""
        card = ScoreCard()
        scores = {
            ValidationDimension.STRUCTURAL: 92.0,
            ValidationDimension.SEMANTIC: 85.0,
        }
        result = card.calculate(scores)
        assert result.dimension_scores["structural"] == 92.0
        assert result.dimension_scores["semantic"] == 85.0

    def test_nan_score_returns_error(self):
        """NaN skor → error döner."""
        card = ScoreCard()
        with patch("stateguard.core.scoring.DEFAULT_WEIGHTS", {"structural": 1.0}):
            result = card.calculate({ValidationDimension.STRUCTURAL: float("nan")})
        assert result.passed is False
        assert result.overall_score == 0.0
        assert "Invalid score" in (result.details.get("scoring", {}).get("error") or "")

    def test_inf_score_returns_error(self):
        """Inf skor → error döner."""
        card = ScoreCard()
        with patch("stateguard.core.scoring.DEFAULT_WEIGHTS", {"structural": 1.0}):
            result = card.calculate({ValidationDimension.STRUCTURAL: float("inf")})
        assert result.passed is False
        assert result.overall_score == 0.0

    def test_non_numeric_score_returns_error(self):
        """String skor → error döner."""
        card = ScoreCard()
        result = card.calculate({ValidationDimension.STRUCTURAL: "abc"})  # type: ignore
        assert result.passed is False
        assert result.overall_score == 0.0

    def test_unrecognised_dimension_keys_ignored(self):
        """Tanınmayan dimension key'leri sessizce atlanır."""
        card = ScoreCard()
        # Geçersiz bir enum değeri kullan
        result = card.calculate({"invalid_dim": 80.0})  # type: ignore
        assert result.passed is False
        assert "No recognised dimension scores" in (
            result.details.get("scoring", {}).get("error") or ""
        )

    def test_zero_total_weight_defaults_zero(self):
        """Tüm weight'ler 0 ise score=0."""
        card = ScoreCard(weights={"structural": 0.0, "semantic": 0.0})
        scores = {
            ValidationDimension.STRUCTURAL: 100.0,
            ValidationDimension.SEMANTIC: 100.0,
        }
        result = card.calculate(scores)
        assert result.overall_score == 0.0
        assert result.passed is False

    def test_config_weight_reading(self):
        """Config'den weight okuma."""
        mock_scoring_cfg = {
            "weights": {"structural": 1.0, "semantic": 0.0},
            "thresholds": {"pass": 75.0, "borderline": 50.0},
        }
        with patch(
            "stateguard.core.scoring.ConfigManager"
        ) as mock_cfg_mgr:
            mock_cfg = mock_cfg_mgr.return_value.load.return_value
            mock_cfg.scoring = mock_scoring_cfg
            card = ScoreCard()
            result = card.calculate({ValidationDimension.STRUCTURAL: 50.0})
            assert result.overall_score == pytest.approx(50.0, rel=1e-2)
