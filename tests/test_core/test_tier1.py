"""Tests for EmbeddingValidator — Tier 1 cascade validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stateguard.core.tier1 import EmbeddingValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult


@pytest.fixture(autouse=True)
def patch_sentence_transformers():
    """Auto-patch SentenceTransformer for all tests in this module."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        instance = mock_st.return_value
        # Return a fixed 384-dim vector
        instance.encode.return_value = np.array([[0.1] * 384], dtype=np.float32)
        yield


class TestEmbeddingValidator:
    """EmbeddingValidator birim testleri."""

    def test_class_attributes(self):
        """Sınıf attribute'ları doğru set edilmiş."""
        v = EmbeddingValidator(model_name="test", device="cpu", threshold=80.0)
        assert v.name == "embedding-similarity"
        assert v.dimension == ValidationDimension.SEMANTIC
        assert v.tier == ValidationTier.TIER_1

    def test_default_threshold(self):
        """Varsayılan threshold config'den gelir (80.0)."""
        with patch("stateguard.core.tier1.ConfigManager") as mock_cfg_mgr:
            mock_cfg = mock_cfg_mgr.return_value.load.return_value
            mock_cfg.tier1_threshold = 80.0
            mock_cfg.default_embedding_model = "all-MiniLM-L6-v2"
            mock_cfg.embedding_device = "cpu"
            v = EmbeddingValidator()
        assert v.threshold == 80.0

    def test_valid_output_passes(self):
        """Yüksek benzerlik → passed=True, score >= 80."""
        v = EmbeddingValidator(model_name="test", device="cpu", threshold=80.0)
        # Mock encode to return identical vectors → cosine=1.0 → score=100
        with patch.object(v._model, "encode") as mock_encode:
            mock_encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
            with patch.object(v._model, "similarity") as mock_sim:
                mock_sim.return_value = np.array([[1.0]])  # identical
                result = v.validate("Merhaba dünya")
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.score >= 80.0
        assert result.dimension == ValidationDimension.SEMANTIC

    def test_invalid_output_fails(self):
        """Düşük benzerlik → passed=False, score < 50."""
        v = EmbeddingValidator(model_name="test", device="cpu", threshold=80.0)
        with patch.object(v._model, "encode") as mock_encode:
            mock_encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
            with patch.object(v._model, "similarity") as mock_sim:
                mock_sim.return_value = np.array([[-0.5]])  # opposite
                result = v.validate("Tamamen farklı çıktı")
        assert result.passed is False
        assert result.score < 50.0

    def test_borderline_routes_to_tier2(self):
        """Sınırda benzerlik → passed=False, 50-79 arası (Tier 2)."""
        v = EmbeddingValidator(model_name="test", device="cpu", threshold=80.0)
        with patch.object(v._model, "encode") as mock_encode:
            mock_encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
            with patch.object(v._model, "similarity") as mock_sim:
                # cosine=0.3 → score=65 → borderline
                mock_sim.return_value = np.array([[0.3]])
                result = v.validate("Kısmen benzer çıktı")
        assert result.passed is False
        assert 50.0 <= result.score < 80.0
        assert result.details.get("tier_hint") == "tier_2"

    def test_empty_output_returns_error(self):
        """Boş output → passed=False, error mesajı."""
        v = EmbeddingValidator(model_name="test", device="cpu", threshold=80.0)
        result = v.validate("")
        assert result.passed is False
        assert result.score == 0.0
        assert "Empty" in (result.error or "")

    def test_context_expected_output_used_as_reference(self):
        """context['expected_output'] referans olarak kullanılır."""
        v = EmbeddingValidator(model_name="test", device="cpu", threshold=80.0)
        with patch.object(v._model, "encode") as mock_encode:
            mock_encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
            with patch.object(v._model, "similarity") as mock_sim:
                mock_sim.return_value = np.array([[1.0]])
                result = v.validate(
                    "merhaba",
                    context={"expected_output": "merhaba dünya"},
                )
        assert result.passed is True

    def test_none_context_does_not_crash(self):
        """context=None → sorunsuz çalışır (output kendine referans)."""
        v = EmbeddingValidator(model_name="test", device="cpu", threshold=80.0)
        with patch.object(v._model, "encode") as mock_encode:
            mock_encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
            with patch.object(v._model, "similarity") as mock_sim:
                mock_sim.return_value = np.array([[1.0]])
                result = v.validate("test", context=None)
        assert isinstance(result, ValidationResult)

    def test_details_contains_cosine_similarity(self):
        """details alanı cosine_similarity ve model bilgisi içerir."""
        v = EmbeddingValidator(model_name="test-model", device="cpu", threshold=80.0)
        with patch.object(v._model, "encode") as mock_encode:
            mock_encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
            with patch.object(v._model, "similarity") as mock_sim:
                mock_sim.return_value = np.array([[0.75]])
                result = v.validate("test")
        assert "cosine_similarity" in result.details
        assert result.details["model"] == "test-model"
        assert result.details["threshold"] == 80.0

    def test_score_bounds_clamped(self):
        """Score her zaman [0, 100] arasında kalır."""
        v = EmbeddingValidator(model_name="test", device="cpu", threshold=80.0)
        with patch.object(v._model, "encode") as mock_encode:
            mock_encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
            with patch.object(v._model, "similarity") as mock_sim:
                mock_sim.return_value = np.array([[-5.0]])  # extreme negative
                result = v.validate("test")
        assert 0.0 <= result.score <= 100.0

    def test_non_string_output_fails_gracefully(self):
        """String olmayan output → passed=False."""
        v = EmbeddingValidator(model_name="test", device="cpu", threshold=80.0)
        result = v.validate(12345)  # type: ignore[arg-type]
        assert result.passed is False
        assert result.score == 0.0

    def test_embedding_model_lazy_load(self):
        """EmbeddingModel sadece encode çağrılınca yüklenir."""
        v = EmbeddingValidator(model_name="test", device="cpu", threshold=80.0)
        # Model instance var ama yüklenmemiş
        assert v._model.is_loaded is False
        # encode çağrısı load'u tetikler
        with patch.object(v._model, "encode") as mock_encode:
            mock_encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
            with patch.object(v._model, "similarity") as mock_sim:
                mock_sim.return_value = np.array([[1.0]])
                v.validate("test")
        # Mock'lanmış modelde is_loaded False kalır (SentenceTransformer instantiate edilmedi)
