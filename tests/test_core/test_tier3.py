"""Tests for LLMValidator — Tier 3 LLM-as-judge validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from stateguard.core.tier3 import LLMValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult


@pytest.fixture
def mock_llm_client():
    """Mock LLM client — default returns EVET."""
    client = MagicMock()
    client.ask.return_value = "EVET"
    return client


class TestLLMValidator:
    """LLMValidator birim testleri."""

    def test_class_attributes(self, mock_llm_client):
        """Sınıf attribute'ları doğru set edilmiş."""
        v = LLMValidator(llm_client=mock_llm_client)
        assert v.name == "llm-validator"
        assert v.dimension == ValidationDimension.SEMANTIC
        assert v.tier == ValidationTier.TIER_3

    def test_llm_passes(self, mock_llm_client):
        """EVET → passed=True, score=100."""
        v = LLMValidator(llm_client=mock_llm_client)
        result = v.validate("Test output")
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.score == 100.0

    def test_llm_fails(self, mock_llm_client):
        """HAYIR → passed=False, score=0."""
        mock_llm_client.ask.return_value = "HAYIR"
        v = LLMValidator(llm_client=mock_llm_client)
        result = v.validate("Test output")
        assert result.passed is False
        assert result.score == 0.0

    def test_llm_response_case_insensitive(self, mock_llm_client):
        """Küçük harf yanıt da çalışır."""
        mock_llm_client.ask.return_value = "evet"
        v = LLMValidator(llm_client=mock_llm_client)
        result = v.validate("Test")
        assert result.passed is True

    def test_llm_response_partial_match(self, mock_llm_client):
        """Kısmi eşleşme de çalışır (içinde EVET geçiyorsa)."""
        mock_llm_client.ask.return_value = "Bence EVET, çünkü mantıklı."
        v = LLMValidator(llm_client=mock_llm_client)
        result = v.validate("Test")
        assert result.passed is True

    def test_unrecognised_response(self, mock_llm_client):
        """Tanınmayan yanıt → error."""
        mock_llm_client.ask.return_value = "BELKİ"
        v = LLMValidator(llm_client=mock_llm_client)
        result = v.validate("Test")
        assert result.passed is False
        assert result.error is not None

    def test_empty_output(self, mock_llm_client):
        """Boş output → error."""
        v = LLMValidator(llm_client=mock_llm_client)
        result = v.validate("")
        assert result.passed is False
        # LLM çağrılmamalı
        mock_llm_client.ask.assert_not_called()

    def test_none_output(self, mock_llm_client):
        """None output → error."""
        v = LLMValidator(llm_client=mock_llm_client)
        result = v.validate(None)
        assert result.passed is False
        mock_llm_client.ask.assert_not_called()

    def test_llm_client_exception(self, mock_llm_client):
        """LLM client exception → error ValidationResult."""
        from stateguard.core.llm_client import LLMError

        mock_llm_client.ask.side_effect = LLMError("API down")
        v = LLMValidator(llm_client=mock_llm_client)
        result = v.validate("Test")
        assert result.passed is False
        assert "API down" in (result.error or "")

    def test_injectable_llm_client(self, mock_llm_client):
        """LLMClient constructor'dan enjekte edilebilir."""
        v = LLMValidator(llm_client=mock_llm_client)
        assert v._llm_client is mock_llm_client

    def test_default_llm_client_created(self):
        """LLMClient verilmezse varsayılan HTTPLLMClient oluşturulur."""
        v = LLMValidator()
        assert v._llm_client is not None
        from stateguard.core.llm_client import HTTPLLMClient

        assert isinstance(v._llm_client, HTTPLLMClient)

    def test_validation_result_format(self, mock_llm_client):
        """ValidationResult formatı doğru."""
        v = LLMValidator(llm_client=mock_llm_client)
        result = v.validate("Test output")
        assert isinstance(result, ValidationResult)
        assert result.dimension == ValidationDimension.SEMANTIC
        assert "raw_response" in result.details
        assert "EVET" in result.details["raw_response"]
