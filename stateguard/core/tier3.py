"""Tier 3 validation — LLM-as-judge checking.

Provides :class:`LLMValidator` as the third tier in the cascade
pipeline.  Uses a small LLM (via an injected or default client) to
make a binary judgment about whether the output is coherent and
consistent.
"""

from __future__ import annotations

from typing import Any

from stateguard.core.llm_client import HTTPLLMClient, LLMClient, LLMError
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.plugin.base import BaseValidator


class LLMValidator(BaseValidator):
    """Validator that uses an LLM to judge output quality.

    Tier 3 of the cascade validation pipeline.  Sends the output to
    a small LLM and asks whether it is coherent and consistent.
    The LLM's binary answer (EVET / HAYIR) determines the result.

    Attributes:
        name:      ``\"llm-validator\"``
        dimension: :attr:`ValidationDimension.SEMANTIC`
        tier:      :attr:`ValidationTier.TIER_3`
    """

    name: str = "llm-validator"
    dimension: ValidationDimension = ValidationDimension.SEMANTIC
    tier: ValidationTier = ValidationTier.TIER_3

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """Initialise the LLM validator.

        Args:
            llm_client: An :class:`LLMClient` instance.  If ``None``,
                        a default :class:`HTTPLLMClient` is created.
        """
        super().__init__()
        self._llm_client = llm_client or HTTPLLMClient()

    def validate(
        self,
        output: Any,
        context: dict | None = None,
    ) -> ValidationResult:
        """Run LLM-based validation on the given *output*.

        Args:
            output:  The LLM output text to validate.
            context: Optional context (currently unused by this tier).

        Returns:
            A :class:`ValidationResult` with the LLM's binary verdict.
        """
        # --- Input validation ---
        if not output or not isinstance(output, str):
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                error="Output must be a non-empty string.",
            )

        # --- Build prompt ---
        prompt = HTTPLLMClient._build_prompt(output)

        # --- Call LLM ---
        try:
            response = self._llm_client.ask(prompt)
        except LLMError as e:
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"prompt": prompt},
                error=str(e),
            )

        # --- Parse binary response ---
        cleaned = response.strip().upper()

        if "EVET" in cleaned:
            passed = True
            score = 100.0
        elif "HAYIR" in cleaned:
            passed = False
            score = 0.0
        else:
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"prompt": prompt, "raw_response": response},
                error=f"Unrecognised LLM response: {response}",
            )

        return ValidationResult(
            score=score,
            passed=passed,
            dimension=self.dimension,
            details={
                "prompt": prompt,
                "raw_response": response,
            },
            error=None,
        )
