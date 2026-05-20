"""Core module — Cascade validation engine and tier validators.

Modules:
    engine:   ValidationEngine — cascade pipeline orchestrator
    tier1:    EmbeddingValidator — embedding similarity (Tier 1)
    tier2:    EnsembleValidator — ML ensemble anomaly detection (Tier 2)
    tier3:    LLMValidator — small LLM binary validator (Tier 3)
    scoring:  ScoreCard — multi-dimension weighted score aggregation
"""
