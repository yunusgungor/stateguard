"""Security validation — prompt injection & jailbreak detection.

Provides :class:`SecurityValidator` that inspects outputs for common
prompt-injection patterns, jailbreak attempts, and other LLM-security
violations using:

* **Pattern-based** — Regex signatures for known attack strings.
* **Heuristic scoring** — Weighted combination of suspicious n-grams,
  special-character ratios, and instruction-override keywords.
* **Classifier** — (Optional) ML classifier trained on injection-vs-benign
  examples (placeholder for future use).
"""

from __future__ import annotations

import math
import re
from typing import Any, ClassVar

from ..models.enums import ValidationDimension, ValidationTier
from ..models.result import ValidationResult
from ..plugin.base import BaseValidator


class SecurityValidator(BaseValidator):
    """Validator that detects prompt-injection and jailbreak attempts.

    Detection strategies (configurable via constructor flags):

        * **Pattern-based** — Regex signatures for known attack strings
          (e.g. ``"ignore previous instructions"``, ``"DAN"``, etc.).
        * **Heuristic scoring** — Weighted combination of suspicious
          n-grams, special-character ratios, and instruction-override
          keywords.
        * **Classifier** — (Optional) ML classifier trained on
          injection-vs-benign examples (placeholder that returns 0.5).

    Configuration is passed through the *context* dict (modes can be
    combined — the final score is the average of all active modes).
    """

    # --- Required class attributes (BaseValidator __init_subclass__) ---
    name: str = "security"
    dimension: ValidationDimension = ValidationDimension.SECURITY
    tier: ValidationTier = ValidationTier.TIER_1

    # ------------------------------------------------------------------
    # Regex pattern library — 8 categories, class-level, compiled once
    # ------------------------------------------------------------------
    PATTERNS: ClassVar[dict[str, list[re.Pattern]]] = {
        "IGNORE_PREVIOUS": [
            re.compile(r"ignore\s+(all\s+)?(previous|prior)", re.I),
            re.compile(r"forget\s+(all\s+)?(your\s+)?(instructions|rules|prompt)", re.I),
            re.compile(r"disregard\s+(all\s+)?(previous|prior)", re.I),
            re.compile(r"don'?t\s+(follow|obey)\s+(the\s+)?(rules|instructions)", re.I),
        ],
        "JAILBREAK": [
            re.compile(r"\bDAN\b", re.I),
            re.compile(r"do\s+anything\s+now", re.I),
            re.compile(r"(hypothetical|fictional)\s+(scenario|setting)", re.I),
            re.compile(r"role(-|\s*)play\s+as\s+(an?\s+)?(unrestricted|unfiltered)", re.I),
            re.compile(r"act\s+as\s+if\s+you\s+(have\s+)?no\s+(restrictions|limits|rules)", re.I),
            re.compile(r"you\s+(are|were)\s+(now\s+)?(free|unleashed|unconstrained)", re.I),
            re.compile(r"you\s+(will|must)\s+(now\s+)?(act|behave)\s+(as|without)", re.I),
        ],
        "SYSTEM_PROMPT_EXTRACTION": [
            re.compile(r"(print|show|reveal|output|display|tell)\s+(me\s+)?(your\s+)?(instructions|prompt|rules|system\s+prompt)", re.I),
            re.compile(r"what\s+(are|were)\s+your\s+(instructions|rules|prompt)", re.I),
            re.compile(r"how\s+(are\s+)?you\s+(instructed|programmed|configured)", re.I),
            re.compile(r"repeat\s+(everything|all)\s+(above|before)", re.I),
            re.compile(r"ignore\s+(the\s+)?(above|previous)\s+(and|instructions)", re.I),
        ],
        "ROLE_OVERRIDE": [
            re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.I),
            re.compile(r"from\s+now\s+on\s+you\s+(will|are|shall)", re.I),
            re.compile(r"pretend\s+to\s+be", re.I),
            re.compile(r"act\s+as\s+if", re.I),
            re.compile(r"you\s+(will|must|need\s+to)\s+(act|behave|respond)\s+as", re.I),
        ],
        "SPECIAL_CHAR_ABUSE": [
            re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]{5,}"),
            re.compile(r"(?:[A-Fa-f0-9]{2,}\s*){10,}"),  # hex-like sequence
            re.compile(r"(?:[A-Za-z0-9+/]{40,}={0,2})"),  # base64-like block
            re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}"),  # \x escape sequence
        ],
        "TOKEN_SMUGGLING": [
            re.compile(r"\\\\(['\"])"),  # escaped quotes
            re.compile(r"<\|[a-z]+\|>"),  # model-specific tokens
            re.compile(r"\[/?(INST|SYS|ASSISTANT)\]", re.I),  # chat template tokens
            re.compile(r"-{10,}"),  # delimiter lines
        ],
        "HALLUCINATION_MARKERS": [
            re.compile(r"(?:as\s+(an?\s+)?AI|as\s+a\s+language\s+model)", re.I),
            re.compile(r"I\s+(don'?t|cannot|can'?t)\s+(have\s+)?access", re.I),
            re.compile(r"I\s+am\s+(not\s+)?(able|capable)\s+to", re.I),
            re.compile(r"I\s+do\s+not\s+(have|possess|maintain)", re.I),
        ],
        "COMMAND_INJECTION": [
            re.compile(r"[;|&]\s*(rm|del|shutdown|reboot|format|mkfs|dd)", re.I),
            re.compile(r"`[^`]+`"),
            re.compile(r"\$\([^)]+\)"),
            re.compile(r"(sudo|su\s+-)\s+", re.I),
            re.compile(r"(exec|eval|system|popen|subprocess)\s*\(", re.I),
        ],
    }

    # ------------------------------------------------------------------
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
                            an injection (0.0 – 1.0).  Lower values make
                            detection more aggressive.

        Raises:
            ValueError: If *threshold* is not positive, or if any of the
                        boolean flags is not actually a bool.
        """
        super().__init__()

        # Type guards
        if not isinstance(use_patterns, bool):
            raise TypeError(f"use_patterns must be bool, got {type(use_patterns).__name__}")
        if not isinstance(use_heuristics, bool):
            raise TypeError(f"use_heuristics must be bool, got {type(use_heuristics).__name__}")
        if not isinstance(use_classifier, bool):
            raise TypeError(f"use_classifier must be bool, got {type(use_classifier).__name__}")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError(f"threshold must be a number, got {type(threshold).__name__}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold!r}")

        self._use_patterns = use_patterns
        self._use_heuristics = use_heuristics
        self._use_classifier = use_classifier
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def use_patterns(self) -> bool:
        """Whether regex-based pattern detection is enabled."""
        return self._use_patterns

    @property
    def use_heuristics(self) -> bool:
        """Whether heuristic scoring is enabled."""
        return self._use_heuristics

    @property
    def use_classifier(self) -> bool:
        """Whether ML classifier is enabled."""
        return self._use_classifier

    @property
    def threshold(self) -> float:
        """Current detection threshold (0.0 – 1.0)."""
        return self._threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        output: Any,
        context: dict | None = None,
    ) -> ValidationResult:
        """Validate *output* for prompt-injection indicators.

        Args:
            output:  The text or message to inspect.
            context: May contain ``threshold`` override, ``input`` for
                     input-side validation, or ``extra_patterns`` dict.

        Returns:
            A :class:`ValidationResult` where a low score indicates a
            likely injection attempt.
        """
        # Context type guard
        ctx = context if isinstance(context, dict) else {}
        details: dict[str, Any] = {}
        scores: list[float] = []

        # Resolve effective threshold
        effective_threshold = ctx.get("threshold", self._threshold)
        if not isinstance(effective_threshold, (int, float)) or isinstance(effective_threshold, bool):
            effective_threshold = self._threshold
        if effective_threshold <= 0:
            effective_threshold = self._threshold

        # --- 1. Output validation --------------------------------------
        output_score, output_details = self._validate_output(output, ctx)
        scores.append(output_score)
        details["output"] = output_details

        # --- 2. Input validation (optional) ---------------------------
        raw_input = ctx.get("input")
        if raw_input is not None and isinstance(raw_input, str) and raw_input.strip():
            input_score, input_details = self._validate_input(raw_input, ctx)
            details["input"] = input_details
            # If input is suspicious, tighten threshold
            # Note: effective_threshold is 0-1, input_score is 0-100, so convert
            threshold_100 = effective_threshold * 100
            if input_score < threshold_100:
                details["input_suspicious"] = True
                details["input_score"] = input_score
                effective_threshold = effective_threshold * 1.5  # require higher score
                if output_score < min(100.0, effective_threshold * 100):
                    details["threshold_tightened"] = True

        # --- 3. Combined score ----------------------------------------
        final_score = sum(scores) / len(scores) if scores else 100.0
        if not math.isfinite(final_score):
            final_score = 0.0
        final_score = max(0.0, min(100.0, final_score))
        passed = final_score >= 50.0

        return ValidationResult(
            score=final_score,
            passed=passed,
            dimension=self.dimension,
            details=details,
        )

    # ------------------------------------------------------------------
    # Internal helpers — Output validation
    # ------------------------------------------------------------------

    def _validate_output(self, output: Any, ctx: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """Run all active detection modes on *output*.

        Returns ``(score, details)``.
        """
        # Non-string output guard
        if output is None:
            return (100.0, {"warning": "None output — security check skipped"})
        if not isinstance(output, str):
            return (100.0, {"warning": f"Non-string output ({type(output).__name__}) — security check skipped"})
        if not output.strip():
            return (100.0, {"warning": "Empty output — security check skipped"})

        # Truncate very long output for performance
        if len(output) > 10_000:
            text = output[:10_000]
            truncated = True
        else:
            text = output
            truncated = False

        details: dict[str, Any] = {}
        if truncated:
            details["truncated"] = True
            details["original_length"] = len(output)
        scores: list[float] = []

        # Pattern-based detection
        if self._use_patterns:
            pat_score, pat_details = self._check_patterns(text, ctx)
            scores.append(pat_score)
            details["pattern"] = pat_details

        # Heuristic scoring
        if self._use_heuristics:
            heur_score, heur_details = self._check_heuristics(text)
            scores.append(heur_score)
            details["heuristic"] = heur_details

        # Classifier (optional, placeholder)
        if self._use_classifier:
            cls_score, cls_details = self._check_classifier(text)
            scores.append(cls_score)
            details["classifier"] = cls_details

        if not scores:
            return (100.0, {"warning": "No detection modes active — all disabled"})

        avg_score = sum(scores) / len(scores)
        if not math.isfinite(avg_score):
            avg_score = 0.0
        avg_score = max(0.0, min(100.0, avg_score))
        details["active_modes"] = len(scores)
        return (avg_score, details)

    def _check_patterns(self, text: str, ctx: dict[str, Any] | None = None) -> tuple[float, dict[str, Any]]:
        """Run all 8 regex categories on *text*.

        Returns ``(score, details)`` where *score* is the proportion of
        clean categories (100 = no matches found, 0 = all categories matched).
        """
        patterns = self._resolve_extra_patterns(ctx) if ctx else self.PATTERNS
        matched: dict[str, list[str]] = {}
        total_categories = len(self.PATTERNS)
        matched_count = 0

        for category, patterns in patterns.items():
            category_matches: list[str] = []
            for compiled in patterns:
                try:
                    if compiled.search(text):
                        category_matches.append(compiled.pattern)
                except re.error:
                    continue  # skip broken pattern
            if category_matches:
                matched[category] = category_matches
                matched_count += 1

        # Score: proportion of clean categories, mapped to 0-100
        clean_ratio = (total_categories - matched_count) / total_categories if total_categories > 0 else 1.0
        score = clean_ratio * 100.0

        result: dict[str, Any] = {
            "matched_categories": matched_count,
            "total_categories": total_categories,
            "matches": matched,
        }
        if matched_count > 0:
            result["matched_category_names"] = list(matched.keys())

        return (score, result)

    def _check_heuristics(self, text: str) -> tuple[float, dict[str, Any]]:
        """Compute heuristic suspicion score for *text*.

        Evaluates five independent signals and returns a combined
        suspicion score mapped to the 0–100 range (100 = clean).

        Returns ``(score, details)``.
        """
        words = text.split()
        chars = list(text)
        total_words = len(words)
        total_chars = len(chars) or 1

        # 1. Suspicious n-gram ratio
        suspicious_ngrams = {
            "ignore", "forget", "disregard", "override", "bypass",
            "unrestricted", "unfiltered", "jailbreak", "hypothetical",
            "dan", "evil", "malicious", "hack", "exploit", "inject",
        }
        ngram_matches = sum(1 for w in words if w.strip(".,!?;:'\"").lower() in suspicious_ngrams)
        ngram_ratio = min(1.0, ngram_matches / max(1, total_words) * 10)  # scale: 10% words = 1.0

        # 2. Special character ratio
        special_chars = sum(1 for c in chars if c in "!@#$%^&*()_+-=[]{}|;:',.<>?/`~\"\\")
        special_ratio = min(1.0, special_chars / total_chars * 3)  # scale: ~33% special = 1.0

        # 3. Instruction-override keyword density
        override_keywords = {
            "you", "your", "now", "must", "will", "shall", "need",
            "act", "behave", "respond", "answer", "ignore", "remember",
            "forget", "pretend", "new", "different", "instead",
        }
        override_matches = sum(1 for w in words if w.strip(".,!?;:'\"").lower() in override_keywords)
        override_density = min(1.0, override_matches / max(1, total_words) * 5)  # scale: 20% = 1.0

        # 4. Uppercase ratio anomaly
        upper_chars = sum(1 for c in chars if c.isupper())
        upper_ratio = upper_chars / total_chars
        # Normal: < 0.5, Abnormal: > 0.8
        if upper_ratio < 0.5:
            upper_score = 0.0
        elif upper_ratio > 0.8:
            upper_score = 1.0
        else:
            upper_score = (upper_ratio - 0.5) / 0.3  # linear 0.5→0.8

        # 5. Repetition score
        if total_words >= 4:
            unique_ratio = len(set(w.lower() for w in words)) / total_words
            repetition_score = max(0.0, 1.0 - unique_ratio)
        else:
            repetition_score = 0.0

        # Weighted average (equal weights)
        components = [ngram_ratio, special_ratio, override_density, upper_score, repetition_score]
        final_heuristic = sum(components) / len(components)
        final_heuristic = max(0.0, min(1.0, final_heuristic))

        # Map to score: clean = 100, fully suspicious = 0
        score = (1.0 - final_heuristic) * 100.0

        details = {
            "suspicious_ngram_ratio": round(ngram_ratio, 4),
            "special_char_ratio": round(special_ratio, 4),
            "override_density": round(override_density, 4),
            "uppercase_ratio": round(upper_ratio, 4),
            "repetition_score": round(repetition_score, 4),
            "final_heuristic_score": round(final_heuristic, 4),
        }

        return (score, details)

    def _check_classifier(self, text: str) -> tuple[float, dict[str, Any]]:
        """Run ML classifier (placeholder).

        Returns a neutral score of 50 (no decision) until a real
        classifier is configured.

        Returns ``(score, details)``.
        """
        return (50.0, {"note": "Classifier not configured — returning neutral score (50.0)"})

    # ------------------------------------------------------------------
    # Internal helpers — Input validation
    # ------------------------------------------------------------------

    def _validate_input(self, input_text: str, ctx: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """Validate user input for injection patterns.

        Returns ``(score, details)`` — same 0-100 scale as output
        validation.  Input validation is intentionally stricter:
        pattern matches apply a 50 % penalty to the score.
        """
        score, details = self._validate_output(input_text, ctx)
        details["input_checked"] = True
        # Penalty: if any pattern matched, slash the score
        pattern_details = details.get("pattern", {})
        if isinstance(pattern_details, dict):
            matched = pattern_details.get("matched_categories", 0)
            if matched > 0:
                score *= 0.5  # 50 % penalty for input-side pattern match
        return (score, details)

    # ------------------------------------------------------------------
    # Extra patterns support
    # ------------------------------------------------------------------

    def _resolve_extra_patterns(self, ctx: dict[str, Any]) -> dict[str, list[re.Pattern]]:
        """Merge context extra patterns into the standard pattern library.

        *extra_patterns* should be a dict mapping category names to
        lists of regex strings.

        Returns a new dict that includes both built-in and extra patterns.
        """
        extra = ctx.get("extra_patterns", {})
        if not isinstance(extra, dict):
            return dict(self.PATTERNS)

        merged = dict(self.PATTERNS)
        for category_name, patterns_list in extra.items():
            if not isinstance(category_name, str):
                continue
            if not isinstance(patterns_list, (list, tuple)):
                continue
            compiled: list[re.Pattern] = []
            for pat_str in patterns_list:
                if isinstance(pat_str, str):
                    try:
                        compiled.append(re.compile(pat_str))
                    except re.error:
                        continue
            if compiled:
                existing = merged.get(category_name, [])
                merged[category_name] = existing + compiled
        return merged
