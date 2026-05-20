"""Pytest fixtures for config module tests."""

from __future__ import annotations

import pytest
import yaml

from stateguard.config.settings import ConfigManager


@pytest.fixture
def defaults_data() -> dict:
    """Varsayılan config değerleri."""
    return {
        "tier1_threshold": 80.0,
        "tier2_threshold": 50.0,
        "tier3_enabled": True,
        "default_embedding_model": "all-MiniLM-L6-v2",
        "embedding_device": "cpu",
        "hitl_timeout_seconds": 300,
        "fail_mode": "fail-close",
        "logging": {"level": "INFO", "format": "json"},
        "default_threshold": 0.7,
    }


@pytest.fixture
def loaded_cfg(tmp_path, defaults_data) -> ConfigManager:
    """Yüklenmiş ConfigManager fixture'ı (geçici dosya, CWD bağımsız)."""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(defaults_data))
    cfg = ConfigManager(str(config_path))
    cfg.load()
    return cfg
