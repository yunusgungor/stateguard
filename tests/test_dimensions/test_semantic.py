"""Tests for SemanticValidator — embedding similarity cross-validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from stateguard.dimensions.semantic import SemanticValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.utils.embedding import EmbeddingModel


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_encoder(
    *,
    output_vec: np.ndarray | None = None,
    ref_vec: np.ndarray | None = None,
) -> Any:
    """Build a duck-typed mock object that acts like an EmbeddingModel.

    The mock returns controlled vectors so tests are deterministic and
    do not need sentence-transformers installed.
    """
    _default_vec = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
    _out = output_vec if output_vec is not None else _default_vec
    _ref = ref_vec if ref_vec is not None else _default_vec + 0.01

    class _MockModel:
        _call_count: int = 0

        def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
            # Return different vectors for output (first call) vs reference
            # (second call) regardless of the actual string content.
            self.__class__._call_count += 1
            if self.__class__._call_count % 2 == 1:  # first → output
                return _out
            return _ref

        def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            # Normalise like the real EmbeddingModel.similarity()
            norm_a = np.linalg.norm(a, axis=1, keepdims=True)
            norm_b = np.linalg.norm(b, axis=1, keepdims=True)
            if np.any(norm_a == 0) or np.any(norm_b == 0):
                return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
            a_norm = a / norm_a
            b_norm = b / norm_b
            return np.dot(a_norm, b_norm.T)

    return _MockModel()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def validator() -> SemanticValidator:
    return SemanticValidator()


@pytest.fixture
def mock_model() -> Any:
    return _make_mock_encoder()


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------

class TestSemanticValidatorAttributes:
    """AC 1 — Class-level attribute enforcement."""

    def test_name(self, validator: SemanticValidator) -> None:
        assert validator.name == "semantic"

    def test_dimension(self, validator: SemanticValidator) -> None:
        assert validator.dimension == ValidationDimension.SEMANTIC

    def test_tier(self, validator: SemanticValidator) -> None:
        assert validator.tier == ValidationTier.TIER_2

    def test_default_threshold(self, validator: SemanticValidator) -> None:
        assert validator.threshold == 0.70


# ---------------------------------------------------------------------------
# Embedding similarity — AC 2
# ---------------------------------------------------------------------------

class TestEmbeddingSimilarity:
    """AC 2 — Cosine similarity with reference."""

    def test_passes_above_threshold(
        self, mock_model: Any,
    ) -> None:
        """Cosine similarity threshold üstünde → PASS."""
        v = SemanticValidator(embedding_model=mock_model, threshold=0.5)
        result = v.validate("output", context={"reference": "expected"})
        assert result.passed is True
        assert result.score > 50.0
        assert "cosine_similarity" in result.details

    def test_fails_below_threshold(
        self,
    ) -> None:
        """Cosine similarity threshold altında → FAIL."""
        # Create vectors that are far apart
        out = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        ref = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        model = _make_mock_encoder(output_vec=out, ref_vec=ref)
        v = SemanticValidator(embedding_model=model, threshold=0.9)
        result = v.validate("output", context={"reference": "expected"})
        assert result.passed is False
        assert result.score < 50.0

    def test_no_reference_returns_warning(
        self, validator: SemanticValidator,
    ) -> None:
        """Reference yok → warning ile PASS."""
        result = validator.validate("some text")
        assert result.passed is True
        assert result.score == 100.0
        assert "warning" in result.details
        assert "No reference provided" in result.details["warning"]

    def test_no_reference_with_model(
        self, mock_model: Any,
    ) -> None:
        """Model var ama reference yok → yine warning ile PASS."""
        v = SemanticValidator(embedding_model=mock_model)
        result = v.validate("some text")
        assert result.passed is True
        assert "warning" in result.details

    def test_none_output(self, mock_model: Any) -> None:
        """None output → FAIL."""
        v = SemanticValidator(embedding_model=mock_model)
        result = v.validate(None, context={"reference": "x"})  # type: ignore[arg-type]
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_string(self, mock_model: Any) -> None:
        """Boş string → FAIL."""
        v = SemanticValidator(embedding_model=mock_model)
        result = v.validate("", context={"reference": "x"})
        assert result.passed is False
        assert result.score == 0.0

    def test_details_structure(self, mock_model: Any) -> None:
        """details dict doğru anahtarları içerir."""
        v = SemanticValidator(embedding_model=mock_model, threshold=0.4)
        result = v.validate("output", context={"reference": "expected"})
        assert "cosine_similarity" in result.details
        assert "euclidean_distance" in result.details
        assert "euclidean_score" in result.details
        assert "combined_score" in result.details
        assert "threshold" in result.details
        assert result.details["threshold"] == 0.4

    def test_dimension_preserved(self, validator: SemanticValidator) -> None:
        """Tüm sonuçlarda dimension=SEMANTIC."""
        result = validator.validate("anything")
        assert result.dimension == ValidationDimension.SEMANTIC
        result2 = validator.validate("")
        assert result2.dimension == ValidationDimension.SEMANTIC


# ---------------------------------------------------------------------------
# Cross-validation — AC 3
# ---------------------------------------------------------------------------

class TestCrossValidation:
    """AC 3 — Two independent methods."""

    def test_both_methods_contribute(
        self,
    ) -> None:
        """Hem cosine hem euclidean score ayrı ayrı raporlanır."""
        out = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float64)
        ref = np.array([[0.6, 0.4, 0.5, 0.5]], dtype=np.float64)
        model = _make_mock_encoder(output_vec=out, ref_vec=ref)
        v = SemanticValidator(embedding_model=model, threshold=0.0)
        result = v.validate("output", context={"reference": "expected"})
        assert result.details["cosine_similarity"] >= 0.0
        assert result.details["euclidean_score"] >= 0.0
        assert result.details["combined_score"] > 0.0

    def test_identical_vectors_give_max_score(
        self,
    ) -> None:
        """Aynı vektörler → cosine=1.0, euclidean=1.0 (ideal)."""
        vec = np.array([[0.3, 0.4, 0.5, 0.6]], dtype=np.float64)
        model = _make_mock_encoder(output_vec=vec, ref_vec=vec)
        v = SemanticValidator(embedding_model=model, threshold=0.9)
        result = v.validate("output", context={"reference": "expected"})
        assert result.passed is True
        assert result.score == pytest.approx(100.0, abs=0.01)
        # cosine = 1.0 exactly (normalized dot product)
        assert result.details["cosine_similarity"] == pytest.approx(1.0, abs=0.01)
        # distance = 0 → normalized = 1.0
        assert result.details["euclidean_score"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Threshold management — AC 4
# ---------------------------------------------------------------------------

class TestThreshold:
    """AC 4 — Configurable and overridable."""

    def test_constructor_threshold(self) -> None:
        """Constructor'dan threshold ayarlanabilir."""
        v = SemanticValidator(threshold=0.85)
        assert v.threshold == 0.85

    def test_context_overrides_constructor(
        self, mock_model: Any,
    ) -> None:
        """Context threshold constructor değerini override eder."""
        v = SemanticValidator(embedding_model=mock_model, threshold=0.99)
        # context threshold lower so it passes
        result = v.validate("output", context={
            "reference": "expected",
            "threshold": 0.01,
        })
        assert result.passed is True

    def test_setter_updates_threshold(self) -> None:
        """Property setter threshold'u günceller."""
        v = SemanticValidator()
        v.threshold = 0.5
        assert v.threshold == 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case'ler: None, boş, embedding failure."""

    def test_none_output_no_context(
        self, mock_model: Any,
    ) -> None:
        """None output, reference var → FAIL."""
        v = SemanticValidator(embedding_model=mock_model)
        result = v.validate(None, context={"reference": "x"})  # type: ignore[arg-type]
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_string_no_context(
        self, validator: SemanticValidator,
    ) -> None:
        """Boş string, hiç context yok → FAIL."""
        result = validator.validate("")
        assert result.passed is False
        assert result.score == 0.0

    def test_non_string_output(
        self, mock_model: Any,
    ) -> None:
        """Non-string output → str() ile normalize edilir."""
        v = SemanticValidator(embedding_model=mock_model, threshold=0.0)
        result = v.validate(42, context={"reference": "expected"})
        # Should encode str(42) and compare
        assert result.score > 0.0

    def test_context_not_dict(
        self, validator: SemanticValidator,
    ) -> None:
        """Non-dict context → {} gibi davranır, warning ile PASS."""
        result = validator.validate("text", context="bad-context")  # type: ignore[arg-type]
        assert result.passed is True
        assert "warning" in result.details

    def test_model_from_context(
        self, mock_model: Any,
    ) -> None:
        """Embedding model context'ten geçilebilir."""
        v = SemanticValidator()  # no model
        result = v.validate("output", context={
            "reference": "expected",
            "embedding_model": mock_model,
        })
        # Should work because model comes from context
        assert result.score > 0.0

    def test_invalid_type_checks(
        self, mock_model: Any,
    ) -> None:
        """Yanlış tipte context değerleri crash yapmaz — reference int'e cast edilir."""
        v = SemanticValidator(embedding_model=mock_model)
        result = v.validate(
            "any text",
            context={"reference": 12345},  # non-string reference → str(12345)
        )
        # reference str(12345)="12345" is encoded and compared; should not crash
        assert result.score >= 0.0
        assert isinstance(result.passed, bool)
