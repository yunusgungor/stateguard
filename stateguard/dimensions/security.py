"""Security validation — prompt injection detection.

Provides :class:`SecurityValidator` that inspects outputs for common
prompt-injection patterns, jailbreak attempts, and other LLM-security
violations using heuristic rules, regex signatures, and optional
classifier-based detection.
"""

from typing import Any

from ..plugin.base import BaseValidator
from ..models.result import ValidationResult


class SecurityValidator(BaseValidator):
    """Validator that detects prompt-injection and jailbreak attempts.

    Detection strategies (configurable):

        * **Pattern-based** — Regex signatures for known attack strings
          (e.g., ``"ignore previous instructions"``, ``"DAN"``, etc.).
        * **Heuristic scoring** — Weighted combination of suspicious
          n-grams, special-character ratios, and instruction-override
          keywords.
        * **Classifier** — (Optional) ML classifier trained on
          injection-vs-benign examples.
    """

    def __init__(
        self,
        use_patterns: bool = True,
        use_heuristics: bool = True,
        use_classifier: bool = False,
        threshold: float = 0.7,
    ) -> None:
        """Initialise the security validator.

        Args:
            use_patterns:   Enable regex-based pattern detection.
            use_heuristics: Enable heuristic scoring.
            use_classifier: Enable ML classifier (requires a trained model).
            threshold:      Score above which the output is flagged as
                            an injection (0.0 – 1.0).
        """
        super().__init__()
        self._use_patterns = use_patterns
        self._use_heuristics = use_heuristics
        self._use_classifier = use_classifier
        self._threshold = threshold

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Validate *output* for prompt-injection indicators.

        Args:
            output:  The text or message to inspect.
            context: May contain ``threshold`` override or additional
                     patterns to check.

        Returns:
            A :class:`ValidationResult` where a low score indicates a
            likely injection attempt.
        """
        ...
