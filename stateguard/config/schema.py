"""Pydantic v2 configuration schema for StateGuard.

Provides :class:`StateGuardConfig` — the canonical configuration model
that drives all validation tiers, dimensions, plugins, and output
behaviours.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError


class Tier3Config(BaseModel):
    """Structured configuration for the Tier 3 (LLM) validator.

    Attributes:
        endpoint:        Base URL of the LLM API server.
        model:           Model name to use for completions.
        timeout_seconds: Maximum wait time per request (positive).
        usage_limit:     Maximum Tier 3 calls (0 = unlimited).
    """

    endpoint: str = Field(
        default="http://localhost:8000",
        min_length=1,
        description="LLM API endpoint URL.",
    )
    model: str = Field(
        default="llama-3.2-1b",
        min_length=1,
        description="Model name for completions.",
    )
    timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Request timeout in seconds (must be positive).",
    )
    usage_limit: int = Field(
        default=10,
        ge=0,
        description="Maximum Tier 3 calls (0 = unlimited).",
    )


class StateGuardConfig(BaseModel):
    """Top-level configuration model for the StateGuard validation engine.

    This model is loaded from a YAML file (via :class:`ConfigManager`)
    and consumed by every tier and dimension validator.

    Attributes:
        version:               Schema version for forward-compatibility checks.
        tier1_threshold:       Tier 1 (Embedding) pass/fail eşiği (0-100).
                               score >= tier1_threshold → PASS,
                               score < tier1_threshold - 30 → FAIL,
                               aradaki → Tier 2.
        tier2_threshold:       Tier 2 (Ensemble) geçer/kal eşiği (0-100).
        tier3_enabled:         Tier 3 (Küçük LLM) aktif/pasif.
        default_embedding_model: Varsayılan sentence-transformers model adı.
        embedding_device:      Embedding model device (cpu/cuda).
        hitl_timeout_seconds:  HITL insan yanıtı bekleme süresi (saniye).
        fail_mode:             fail-close (pipeline durur) veya fail-open.
        default_threshold:     Global pass/fail threshold (0.0 – 1.0).
        dimensions:            Per-dimension configuration overrides.
        plugins:               Plugin registration and settings.
        logging:               Structured-logging configuration.
        security:              Security-validator specific options.
    """

    version: str = Field(
        "1.0",
        description="Configuration schema version.",
    )
    tier1_threshold: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Tier 1 (Embedding) eşiği: >=80 PASS, <50 FAIL, aradaki → Tier 2.",
    )
    tier2_threshold: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Tier 2 (Ensemble) geçer/kal eşiği.",
    )
    tier3_enabled: bool = Field(
        default=True,
        description="Tier 3 (Küçük LLM) aktif/pasif.",
    )
    tier3: Tier3Config = Field(
        default_factory=lambda: Tier3Config(),
        description="Tier 3 (LLM Validator) config: endpoint, model, timeout, usage_limit.",
    )
    default_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Varsayılan sentence-transformers model adı.",
    )
    embedding_device: str = Field(
        default="cpu",
        pattern=r"^(cpu|cuda)$",
        description="Embedding model device (cpu/cuda).",
    )
    hitl_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="HITL insan yanıtı bekleme süresi (saniye).",
    )
    fail_mode: str = Field(
        default="fail-close",
        pattern=r"^(fail-close|fail-open)$",
        description="fail-close (pipeline durur) veya fail-open (devam eder).",
    )
    default_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Global pass/fail threshold for validation.",
    )
    dimensions: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-dimension configuration overrides.",
    )
    plugins: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin registration and settings.",
    )
    logging: dict[str, Any] = Field(
        default_factory=lambda: {
            "level": "INFO",
            "format": "json",
        },
        description="Structured-logging configuration.",
    )
    security: dict[str, Any] = Field(
        default_factory=dict,
        description="Security-validator specific options.",
    )
    scoring: dict[str, Any] = Field(
        default_factory=lambda: {
            "weights": {
                "structural": 0.25,
                "semantic": 0.25,
                "quantitative": 0.15,
                "behavioral": 0.20,
                "security": 0.15,
            },
            "thresholds": {
                "pass": 75.0,
                "borderline": 50.0,
            },
        },
        description="ScoreCard weight and threshold configuration.",
    )
