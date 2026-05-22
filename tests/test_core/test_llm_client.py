"""Tests for LLMClient — HTTP LLM client for Tier 3."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stateguard.core.llm_client import HTTPLLMClient, LLMClient, LLMError


class TestLLMClientProtocol:
    """LLMClient protocol testleri."""

    def test_protocol_ask_method(self):
        """Protocol ask metodu çağrılabilir."""
        client = MagicMock(spec=LLMClient)
        client.ask.return_value = "EVET"
        result = client.ask("Test prompt")
        assert result == "EVET"
        client.ask.assert_called_once_with("Test prompt")


class TestHTTPLLMClient:
    """HTTPLLMClient birim testleri."""

    def test_default_constructor(self):
        """Varsayılan constructor çalışır."""
        client = HTTPLLMClient()
        assert client.endpoint == "http://localhost:8000"
        assert client.model == "llama-3.2-1b"
        assert client.timeout_seconds == 5.0

    def test_custom_constructor(self):
        """Custom parametreler constructor'dan geçilebilir."""
        client = HTTPLLMClient(
            endpoint="http://custom:8080",
            model="gpt-4o-mini",
            api_key="test-key",
            timeout_seconds=10.0,
        )
        assert client.endpoint == "http://custom:8080"
        assert client.model == "gpt-4o-mini"
        assert client.api_key == "test-key"
        assert client.timeout_seconds == 10.0

    @patch("httpx.Client")
    def test_ask_success(self, mock_client_cls):
        """Başarılı HTTP yanıtı → response text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "EVET"}}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        client = HTTPLLMClient()
        result = client.ask("Test prompt")
        assert result == "EVET"

    @patch("httpx.Client")
    def test_ask_http_error(self, mock_client_cls):
        """HTTP hata kodu → LLMError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        client = HTTPLLMClient()
        with pytest.raises(LLMError, match="500"):
            client.ask("Test prompt")

    @patch("httpx.Client")
    def test_ask_connection_error(self, mock_client_cls):
        """Bağlantı hatası → LLMError."""
        mock_client = MagicMock()
        mock_client.post.side_effect = ConnectionError("Connection refused")
        mock_client_cls.return_value.__enter__.return_value = mock_client

        client = HTTPLLMClient(timeout_seconds=0.001)
        with pytest.raises(LLMError, match="Connection refused"):
            client.ask("Test prompt")

    @patch("httpx.Client")
    def test_ask_empty_response(self, mock_client_cls):
        """Boş choices → LLMError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        client = HTTPLLMClient()
        with pytest.raises(LLMError, match="no choices"):
            client.ask("Test prompt")

    def test_prompt_format(self):
        """Prompt doğru formatta oluşturulur."""
        prompt = HTTPLLMClient._build_prompt("Merhaba dünya")
        assert "Merhaba dünya" in prompt
        assert "EVET" in prompt
        assert "HAYIR" in prompt

    def test_public_build_prompt(self):
        """build_prompt public method olarak calisir."""
        client = HTTPLLMClient()
        prompt = client.build_prompt("Test output")
        assert "Test output" in prompt
        assert "EVET" in prompt or "EVET" in prompt

    def test_protocol_build_prompt(self):
        """LLMClient protocol build_prompt methoduna sahip."""

        class CustomClient:
            model: str = "test-model"

            def ask(self, prompt: str) -> str:
                return "EVET"

            def build_prompt(self, output: str) -> str:
                return f"Analyze: {output}"

        client = CustomClient()
        # Protocol uyumlulugu kontrolu
        from stateguard.core.llm_client import LLMClient
        assert isinstance(client, LLMClient)

    def test_custom_client_build_prompt(self):
        """Custom client farkli prompt kullanabilir."""

        class CustomClient:
            model: str = "custom"
            ask_called_with: str | None = None

            def ask(self, prompt: str) -> str:
                self.ask_called_with = prompt
                return "EVET"

            def build_prompt(self, output: str) -> str:
                return f"CUSTOM_PROMPT:{output}"

        client = CustomClient()
        from stateguard.core.tier3 import LLMValidator
        v = LLMValidator(llm_client=client)
        result = v.validate("hello")
        assert result.passed is True
        # Custom prompt kullanildigini dogrula
        assert client.ask_called_with == "CUSTOM_PROMPT:hello"
