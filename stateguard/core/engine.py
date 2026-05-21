"""Validation engine — orchestrates the full cascade validation pipeline.

Provides :class:`ValidationEngine` that runs Tier 1 (Embedding) →
Tier 2 (Ensemble) → Tier 3 (LLM) in sequence, stopping early when a
tier passes or the configuration dictates a fail-fast behaviour.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from stateguard.config.settings import ConfigManager
from stateguard.core.tier1 import EmbeddingValidator
from stateguard.core.tier2 import EnsembleValidator
from stateguard.core.tier3 import LLMValidator
from stateguard.models.enums import ValidationDimension
from stateguard.models.log import DecisionEntry
from stateguard.models.result import EngineResult, ValidationResult

logger = logging.getLogger(__name__)

VALID_FAIL_MODES: tuple[str, ...] = ("fail-close", "fail-open")


class ValidationEngine:
    """Orchestrates multi-tier validation across all registered validators.

    Implements the cascade pipeline:
    ``Tier 1 (Embedding) → (gerekirse) Tier 2 (Ensemble) → (gerekirse) Tier 3 (LLM)``

    Attributes:
        name: ``\"validation-engine\"``
    """

    name: str = "validation-engine"

    def __init__(
        self,
        embedding_validator: EmbeddingValidator | None = None,
        ensemble_validator: EnsembleValidator | None = None,
        llm_validator: LLMValidator | None = None,
        agent_id: str | None = None,
        tier3_enabled: bool | None = None,
        fail_mode: str | None = None,
    ) -> None:
        """Initialise the validation engine.

        Args:
            embedding_validator: Injected Tier 1 validator.
            ensemble_validator:  Injected Tier 2 validator.
            llm_validator:       Injected Tier 3 validator.
            agent_id:            Identifier for decision-log entries.
            tier3_enabled:       Override config's Tier 3 enable flag.
            fail_mode:           Override config's fail mode
                                 (``\"fail-close\"`` or ``\"fail-open\"``).

        Raises:
            ValueError: If *fail_mode* is not one of ``\"fail-close\"``
                        or ``\"fail-open\"``.
        """
        self._embedding = embedding_validator or EmbeddingValidator()
        self._ensemble = ensemble_validator or EnsembleValidator()
        self._llm = llm_validator or LLMValidator()
        self._agent_id = agent_id or "default"

        # Read config
        try:
            cfg = ConfigManager()
            config = cfg.load()
            self._tier1_threshold = float(config.tier1_threshold)
            self._tier2_threshold = float(config.tier2_threshold)
            self._tier3_enabled = (
                bool(tier3_enabled) if tier3_enabled is not None
                else bool(config.tier3_enabled)
            )
            resolved_fail_mode = (
                fail_mode if fail_mode is not None
                else config.fail_mode
            )
        except Exception:
            logger.warning("Failed to load config, using defaults.")
            self._tier1_threshold = 80.0
            self._tier2_threshold = 50.0
            self._tier3_enabled = (
                bool(tier3_enabled) if tier3_enabled is not None else True
            )
            resolved_fail_mode = fail_mode if fail_mode is not None else "fail-close"

        if resolved_fail_mode not in VALID_FAIL_MODES:
            raise ValueError(
                f"Invalid fail_mode: {resolved_fail_mode!r}. "
                f"Expected one of: {', '.join(VALID_FAIL_MODES)}"
            )
        self._fail_mode = resolved_fail_mode

    def validate(
        self,
        output: Any,
        context: dict | None = None,
    ) -> EngineResult:
        """Run the full cascade validation pipeline on *output*.

        Args:
            output:  The data to validate.
            context: Optional contextual information.

        Returns:
            An ``EngineResult`` summarising the pipeline outcome.
        """
        decision_log: list[DecisionEntry] = []
        current_context: dict[str, Any] = dict(context or {})

        # ── Tier 1 ────────────────────────────────────────────────────
        tier1_result = self._run_tier(
            self._embedding, "tier_1", output, current_context,
            decision_log,
        )
        if isinstance(tier1_result, EngineResult):
            return self._build_error_result(
                tier1_result, decision_log,
            )

        fail_borderline = max(0.0, self._tier1_threshold - 30.0)  # 50.0

        if tier1_result.score >= self._tier1_threshold:
            return self._build_result(
                overall_score=tier1_result.score,
                passed=True,
                tier_path=[1],
                decision_log=decision_log,
                tier_details={"tier_1": tier1_result.model_dump()},
            )

        if tier1_result.score < fail_borderline:
            return self._build_result(
                overall_score=tier1_result.score,
                passed=False,
                tier_path=[1],
                decision_log=decision_log,
                tier_details={"tier_1": tier1_result.model_dump()},
            )

        # Borderline — escalate to Tier 2
        current_context["tier_1_result"] = tier1_result.model_dump()

        # ── Tier 2 ────────────────────────────────────────────────────
        tier2_result = self._run_tier(
            self._ensemble, "tier_2", output, current_context,
            decision_log,
        )
        if isinstance(tier2_result, EngineResult):
            return self._build_error_result(
                tier2_result, decision_log,
            )

        if tier2_result.passed:
            return self._build_result(
                overall_score=tier2_result.score,
                passed=True,
                tier_path=[1, 2],
                decision_log=decision_log,
                tier_details={
                    "tier_1": tier1_result.model_dump(),
                    "tier_2": tier2_result.model_dump(),
                },
            )

        # Tier 2 failed — check if Tier 3 should run
        if not self._tier3_enabled:
            return self._build_result(
                overall_score=tier2_result.score,
                passed=False,
                tier_path=[1, 2],
                decision_log=decision_log,
                tier_details={
                    "tier_1": tier1_result.model_dump(),
                    "tier_2": tier2_result.model_dump(),
                },
            )

        # ── Tier 3 ────────────────────────────────────────────────────
        current_context["tier_1_result"] = tier1_result.model_dump()
        current_context["tier_2_result"] = tier2_result.model_dump()

        tier3_result = self._run_tier(
            self._llm, "tier_3", output, current_context,
            decision_log,
        )
        if isinstance(tier3_result, EngineResult):
            return self._build_error_result(
                tier3_result, decision_log,
            )

        # Tier 3 gave us a real result (will be valid after Story 2.5)
        return self._build_result(
            overall_score=tier3_result.score,
            passed=tier3_result.passed,
            tier_path=[1, 2, 3],
            decision_log=decision_log,
            tier_details={
                "tier_1": tier1_result.model_dump(),
                "tier_2": tier2_result.model_dump(),
                "tier_3": tier3_result.model_dump(),
            },
        )

    # ── Internals ─────────────────────────────────────────────────────

    def _run_tier(
        self,
        validator: Any,
        step_id: str,
        output: Any,
        context: dict[str, Any],
        decision_log: list[DecisionEntry],
    ) -> ValidationResult | EngineResult:
        """Run a single tier and record its decision.

        Returns:
            A ``ValidationResult`` on success, or an ``EngineResult``
            when an exception was caught and handled (fail-open/close).
        """
        try:
            result = validator.validate(output, context)
        except NotImplementedError:
            # Tier 3 stub — gracefully degrade to pass
            logger.warning("%s not yet implemented, falling back to pass.", step_id)
            self._append_decision(
                decision_log, step_id,
                ValidationDimension.SEMANTIC, 0.0, "fallback",
                {"error": f"{step_id} not implemented"},
            )
            return ValidationResult(
                score=0.0,
                passed=True,
                dimension=ValidationDimension.SEMANTIC,
                details={"error": f"{step_id} not implemented, fallback pass"},
            )
        except Exception as e:
            logger.exception("%s validation failed: %s", step_id, e)
            if self._fail_mode == "fail-close":
                self._append_decision(
                    decision_log, step_id,
                    ValidationDimension.SEMANTIC, 0.0, "fail-close",
                    {"error": str(e)},
                )
                return EngineResult(
                    overall_score=0.0,
                    passed=False,
                    tier_path=[],
                    details={"error": f"{step_id} failed: {e}"},
                )
            else:
                self._append_decision(
                    decision_log, step_id,
                    ValidationDimension.SEMANTIC, 0.0, "fail-open",
                    {"error": str(e)},
                )
                return ValidationResult(
                    score=100.0,
                    passed=True,
                    dimension=ValidationDimension.SEMANTIC,
                    details={"error": f"{step_id} failed (fail-open): {e}"},
                )

        # Record the decision
        decision = "pass" if result.passed else "escalate"
        if step_id == "tier_3":
            decision = "pass" if result.passed else "fail"

        self._append_decision(
            decision_log, step_id,
            result.dimension, result.score, decision,
            result.details,
        )

        return result

    def _append_decision(
        self,
        log: list[DecisionEntry],
        step_id: str,
        dimension: ValidationDimension,
        score: float,
        decision: str,
        details: dict[str, Any],
    ) -> None:
        """Append a decision entry to the log."""
        log.append(DecisionEntry(
            timestamp=datetime.now(timezone.utc),
            agent_id=self._agent_id,
            step_id=step_id,
            dimension=dimension,
            score=score,
            decision=decision,
            details=details,
        ))

    def _build_result(
        self,
        overall_score: float,
        passed: bool,
        tier_path: list[int],
        decision_log: list[DecisionEntry],
        tier_details: dict[str, Any],
    ) -> EngineResult:
        """Build the final ``EngineResult`` with details and log."""
        return EngineResult(
            overall_score=overall_score,
            passed=passed,
            tier_path=tier_path,
            dimension_scores={
                step: details.get("score", 0.0)
                for step, details in tier_details.items()
            },
            details={
                "decision_log": decision_log,
                "tier_results": tier_details,
                "config": {
                    "tier1_threshold": self._tier1_threshold,
                    "tier2_threshold": self._tier2_threshold,
                    "tier3_enabled": self._tier3_enabled,
                    "fail_mode": self._fail_mode,
                },
            },
        )

    def _build_error_result(
        self,
        error_result: EngineResult,
        decision_log: list[DecisionEntry],
    ) -> EngineResult:
        """Wrap an error ``EngineResult`` to preserve the decision log."""
        error_result.details.setdefault("decision_log", list(decision_log))
        return error_result
