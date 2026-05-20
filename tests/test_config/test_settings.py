"""Tests for ConfigManager settings loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from stateguard.config.settings import ConfigManager


class TestConfigManager:
    """ConfigManager yükleme ve erişim testleri."""

    def test_load_defaults(self):
        """defaults.yaml başarıyla yüklenir ve StateGuardConfig döner."""
        cfg = ConfigManager("stateguard/config/defaults.yaml")
        config = cfg.load()
        assert config.tier1_threshold == 80.0
        assert config.tier2_threshold == 50.0
        assert config.fail_mode == "fail-close"
        assert config.hitl_timeout_seconds == 300

    def test_load_from_tmp_path(self, defaults_data):
        """Geçici dosyadan yükleme de çalışır (CWD bağımsız)."""
        p = Path(str(Path.cwd())) / "tmp_test_config.yaml"
        try:
            p.write_text(yaml.dump(defaults_data))
            cfg = ConfigManager(str(p))
            config = cfg.load()
            assert config.tier1_threshold == 80.0
        finally:
            if p.exists():
                p.unlink()

    def test_missing_file_raises_filenotfound(self):
        """Varolmayan dosya → FileNotFoundError."""
        cfg = ConfigManager("/nonexistent/path/config.yaml")
        with pytest.raises(FileNotFoundError):
            cfg.load()

    def test_invalid_yaml_raises_exception(self, tmp_path):
        """Geçersiz YAML içeriği → exception."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{invalid: yaml: broken}")
        cfg = ConfigManager(str(bad_file))
        with pytest.raises(Exception):
            cfg.load()

    def test_invalid_config_raises_validation_error(self, tmp_path):
        """Geçersiz config değerleri → ValidationError."""
        bad_file = tmp_path / "invalid.yaml"
        bad_file.write_text(yaml.dump({"tier1_threshold": -5}))
        cfg = ConfigManager(str(bad_file))
        with pytest.raises(ValidationError):
            cfg.load()

    def test_get_dotted_key(self, loaded_cfg):
        """get() dotted key ile iç içe değerlere erişir."""
        assert loaded_cfg.get("tier1_threshold") == 80.0
        assert loaded_cfg.get("logging.level") == "INFO"
        assert loaded_cfg.get("logging.format") == "json"

    def test_get_default_fallback(self, loaded_cfg):
        """get() varolmayan key için default döner."""
        assert loaded_cfg.get("nonexistent", 42) == 42
        assert loaded_cfg.get("logging.nonexistent", "fallback") == "fallback"

    def test_set_runtime_value(self, loaded_cfg):
        """set() runtime'da değer değiştirir."""
        loaded_cfg.set("tier1_threshold", 95.0)
        assert loaded_cfg.get("tier1_threshold") == 95.0

    def test_set_dotted_key(self, loaded_cfg):
        """set() dotted key ile iç içe değer değiştirir."""
        loaded_cfg.set("logging.level", "DEBUG")
        assert loaded_cfg.get("logging.level") == "DEBUG"

    def test_reload_returns_fresh_config(self, tmp_path):
        """reload() disk'ten taze config okur."""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump({"tier1_threshold": 80.0}))
        cfg = ConfigManager(str(config_path))
        cfg.load()
        assert cfg.get("tier1_threshold") == 80.0

        # Dosyayı değiştir
        config_path.write_text(yaml.dump({"tier1_threshold": 90.0}))
        cfg.reload()
        assert cfg.get("tier1_threshold") == 90.0

    def test_get_before_load_raises_runtimeerror(self):
        """load() çağrılmadan get() → RuntimeError."""
        cfg = ConfigManager("stateguard/config/defaults.yaml")
        with pytest.raises(RuntimeError, match="not loaded"):
            cfg.get("tier1_threshold")

    def test_set_before_load_raises_runtimeerror(self):
        """load() çağrılmadan set() → RuntimeError."""
        cfg = ConfigManager("stateguard/config/defaults.yaml")
        with pytest.raises(RuntimeError, match="not loaded"):
            cfg.set("tier1_threshold", 99.0)

    def test_property_path(self):
        """path property doğru yolu döner."""
        cfg = ConfigManager("custom/path.yaml")
        assert str(cfg.path) == "custom/path.yaml"

    def test_property_config_none_before_load(self):
        """config property load öncesi None döner."""
        cfg = ConfigManager("stateguard/config/defaults.yaml")
        assert cfg.config is None

    def test_property_config_after_load(self):
        """config property load sonrası StateGuardConfig döner."""
        cfg = ConfigManager("stateguard/config/defaults.yaml")
        config = cfg.load()
        assert cfg.config is config
