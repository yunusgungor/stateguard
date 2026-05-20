"""Tests for StateGuardConfig Pydantic schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stateguard.config.schema import StateGuardConfig


class TestStateGuardConfig:
    """StateGuardConfig schema validation tests."""

    def test_default_config(self):
        """Varsayılan config tüm field'ları doğru değerlerle oluşturur."""
        config = StateGuardConfig()
        assert config.version == "1.0"
        assert config.tier1_threshold == 80.0
        assert config.tier2_threshold == 50.0
        assert config.tier3_enabled is True
        assert config.default_embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_device == "cpu"
        assert config.hitl_timeout_seconds == 300
        assert config.fail_mode == "fail-close"
        assert config.default_threshold == 0.7

    def test_custom_config(self):
        """Özel değerlerle config oluşturulabilir."""
        config = StateGuardConfig(
            tier1_threshold=90.0,
            tier2_threshold=60.0,
            tier3_enabled=False,
            default_embedding_model="all-mpnet-base-v2",
            hitl_timeout_seconds=600,
            fail_mode="fail-open",
        )
        assert config.tier1_threshold == 90.0
        assert config.tier2_threshold == 60.0
        assert config.tier3_enabled is False
        assert config.default_embedding_model == "all-mpnet-base-v2"
        assert config.hitl_timeout_seconds == 600
        assert config.fail_mode == "fail-open"

    def test_tier1_threshold_bounds(self):
        """tier1_threshold 0-100 arasında olmalıdır."""
        StateGuardConfig(tier1_threshold=0.0)
        StateGuardConfig(tier1_threshold=100.0)
        StateGuardConfig(tier1_threshold=50.0)

        with pytest.raises(ValidationError):
            StateGuardConfig(tier1_threshold=-1.0)
        with pytest.raises(ValidationError):
            StateGuardConfig(tier1_threshold=101.0)

    def test_tier2_threshold_bounds(self):
        """tier2_threshold 0-100 arasında olmalıdır."""
        StateGuardConfig(tier2_threshold=0.0)
        StateGuardConfig(tier2_threshold=100.0)

        with pytest.raises(ValidationError):
            StateGuardConfig(tier2_threshold=-0.1)
        with pytest.raises(ValidationError):
            StateGuardConfig(tier2_threshold=100.1)

    def test_hitl_timeout_bounds(self):
        """hitl_timeout_seconds 30-3600 arasında olmalıdır."""
        StateGuardConfig(hitl_timeout_seconds=30)
        StateGuardConfig(hitl_timeout_seconds=3600)

        with pytest.raises(ValidationError):
            StateGuardConfig(hitl_timeout_seconds=29)
        with pytest.raises(ValidationError):
            StateGuardConfig(hitl_timeout_seconds=3601)

    def test_fail_mode_validation(self):
        """fail_mode sadece fail-close veya fail-open olabilir."""
        StateGuardConfig(fail_mode="fail-close")
        StateGuardConfig(fail_mode="fail-open")

        with pytest.raises(ValidationError):
            StateGuardConfig(fail_mode="fail-silent")
        with pytest.raises(ValidationError):
            StateGuardConfig(fail_mode="")

    def test_embedding_device_validation(self):
        """embedding_device sadece cpu veya cuda olabilir."""
        StateGuardConfig(embedding_device="cpu")
        StateGuardConfig(embedding_device="cuda")

        with pytest.raises(ValidationError):
            StateGuardConfig(embedding_device="mps")
        with pytest.raises(ValidationError):
            StateGuardConfig(embedding_device="gpu")

    def test_dict_like_dimensions(self):
        """dimensions ve plugins dict olarak saklanır."""
        config = StateGuardConfig(
            dimensions={"structural": {"schema": {"type": "object"}}},
            plugins={"custom": {"enabled": True}},
        )
        assert config.dimensions["structural"]["schema"]["type"] == "object"
        assert config.plugins["custom"]["enabled"] is True

    def test_model_validate_from_dict(self):
        """model_validate ile dict'ten config oluşturulabilir."""
        data = {
            "tier1_threshold": 85.0,
            "tier3_enabled": False,
            "logging": {"level": "DEBUG"},
        }
        config = StateGuardConfig.model_validate(data)
        assert config.tier1_threshold == 85.0
        assert config.tier3_enabled is False
        assert config.logging["level"] == "DEBUG"
        # Varsayılanlar korunur
        assert config.tier2_threshold == 50.0
        assert config.fail_mode == "fail-close"

    def test_invalid_type_raises_error(self):
        """Yanlış tip gönderilince ValidationError fırlar."""
        with pytest.raises(ValidationError):
            StateGuardConfig(tier1_threshold="seksen")
        # Boolean coercion: "yes"/1 coerces to True — sadece bozuk tipler hata verir
        with pytest.raises(ValidationError):
            StateGuardConfig(tier3_enabled="belki")
        with pytest.raises(ValidationError):
            StateGuardConfig(hitl_timeout_seconds="beş")
        # None değil, tip hatası olmalı
        with pytest.raises(ValidationError):
            StateGuardConfig(embedding_device=123)
