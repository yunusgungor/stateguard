"""LLM client abstraction for Tier 3 validation.

Provides :class:`LLMClient` (protocol) and :class:`HTTPLLMClient`
(default implementation) for calling small LLMs to make binary
judgements about output quality.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when LLM communication fails."""


class LLMClient(Protocol):
    """Protocol for LLM-based validation clients.

    Implementations must provide an ``ask`` method that sends a
    prompt to an LLM and returns the raw text response, and a
    ``model`` attribute identifying the model name.
    """

    model: str

    def ask(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the response text."""
        ...


class HTTPLLMClient:
    """Default LLM client that calls an OpenAI-compatible HTTP endpoint.

    Expects a ``/v1/chat/completions`` style API (OpenAI, llama.cpp
    server, vLLM, etc.).  Customisable via constructor parameters.

    Attributes:
        endpoint:         Base URL of the LLM API server.
        model:            Model name to use for completions.
        api_key:          API key (sent as ``Authorization: Bearer``).
        timeout_seconds:  Maximum wait time per request.
    """

    JUDGE_PROMPT_TEMPLATE: str = (
        "Çıktı: {output}\n\n"
        "Bu çıktı mantıklı ve tutarlı mı? "
        "Sadece EVET veya HAYIR cevabı ver."
    )

    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        model: str = "llama-3.2-1b",
        api_key: str = "",
        timeout_seconds: float = 5.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {timeout_seconds}"
            )
        if not endpoint:
            raise ValueError(
                f"endpoint must not be empty, got {endpoint!r}"
            )
        if not model:
            raise ValueError(
                f"model must not be empty, got {model!r}"
            )
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @classmethod
    def _build_prompt(cls, output: str) -> str:
        """Build the judgement prompt for a given *output*."""
        return cls.JUDGE_PROMPT_TEMPLATE.replace("{output}", output)

    def ask(self, prompt: str) -> str:
        """Send *prompt* via HTTP POST and return the text response.

        Raises:
            LLMError: If the HTTP request fails or the response
                      cannot be parsed.
        """

        url = f"{self.endpoint.rstrip('/')}/v1/chat/completions"

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,  # deterministic
            "max_tokens": 50,
        }

        try:
            import httpx  # lazy import — httpx is optional
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
        except Exception as e:
            raise LLMError(f"LLM request failed: {e}") from e

        if response.status_code != 200:
            raise LLMError(
                f"LLM API returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM response is not valid JSON: {e}") from e

        choices = data.get("choices", [])
        if not choices:
            raise LLMError(
                f"Unexpected LLM response format (no choices): {data}"
            )

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise LLMError("LLM returned empty content.")

        return content.strip()
