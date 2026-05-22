"""Tests for example validator plugins — JSON Schema, Keyword, Length, and Regex validators."""

from __future__ import annotations

import re
from typing import Any

import pytest

from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.plugin.examples.json_schema import JsonSchemaValidator
from stateguard.plugin.examples.keyword import KeywordValidator
from stateguard.plugin.examples.length import LengthValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def json_validator() -> JsonSchemaValidator:
    return JsonSchemaValidator()


@pytest.fixture
def keyword_validator() -> KeywordValidator:
    return KeywordValidator()


@pytest.fixture
def length_validator() -> LengthValidator:
    return LengthValidator()


# ---------------------------------------------------------------------------
# JsonSchemaValidator Tests — AC 1
# ---------------------------------------------------------------------------

class TestJsonSchemaValidator:
    """JsonSchemaValidator.validate() — AC 1"""

    def test_valid_json_no_schema(self, json_validator: JsonSchemaValidator):
        """Geçerli JSON, schema yok → PASS, score=100"""
        result = json_validator.validate('{"key": "value"}')
        assert result.passed is True
        assert result.score == 100.0
        assert result.dimension == ValidationDimension.STRUCTURAL

    def test_valid_json_with_list(self, json_validator: JsonSchemaValidator):
        """Geçerli JSON array → PASS"""
        result = json_validator.validate('[1, 2, 3]')
        assert result.passed is True
        assert result.score == 100.0

    def test_valid_json_with_number(self, json_validator: JsonSchemaValidator):
        """Geçerli JSON sayı → PASS"""
        result = json_validator.validate('42')
        assert result.passed is True
        assert result.score == 100.0

    def test_invalid_json_syntax(self, json_validator: JsonSchemaValidator):
        """Geçersiz JSON → FAIL, score=0, detail'de error mesajı var"""
        result = json_validator.validate("{invalid}")
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    def test_invalid_json_empty(self, json_validator: JsonSchemaValidator):
        """Boş string → FAIL"""
        result = json_validator.validate("")
        assert result.passed is False
        assert result.score == 0.0

    def test_invalid_json_none(self, json_validator: JsonSchemaValidator):
        """None output → FAIL"""
        result = json_validator.validate(None)  # type: ignore[arg-type]
        assert result.passed is False
        assert result.score == 0.0

    def test_with_valid_schema(self, json_validator: JsonSchemaValidator):
        """Schema ile validasyon: geçerli veri → PASS"""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        result = json_validator.validate(
            '{"name": "test", "age": 30}',
            context={"schema": schema},
        )
        assert result.passed is True
        assert result.score == 100.0

    def test_with_schema_validation_error(self, json_validator: JsonSchemaValidator):
        """Schema ile validasyon: geçersiz veri → FAIL"""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = json_validator.validate(
            '{"age": 30}',
            context={"schema": schema},
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    def test_details_format(self, json_validator: JsonSchemaValidator):
        """Başarılı validasyonda details.format == 'valid'"""
        result = json_validator.validate('{"ok": true}')
        assert result.details.get("format") == "valid"


# ---------------------------------------------------------------------------
# KeywordValidator Tests — AC 2
# ---------------------------------------------------------------------------

class TestKeywordValidator:
    """KeywordValidator.validate() — AC 2"""

    @staticmethod
    def _call(
        validator: KeywordValidator,
        output: str,
        required: list[str] | None = None,
        forbidden: list[str] | None = None,
    ) -> Any:
        ctx: dict[str, Any] = {}
        if required is not None:
            ctx["required_keywords"] = required
        if forbidden is not None:
            ctx["forbidden_keywords"] = forbidden
        return validator.validate(output, context=ctx)

    def test_all_required_found(self, keyword_validator: KeywordValidator):
        """Tüm gerekli keyword'ler bulundu → PASS"""
        result = self._call(keyword_validator, "Bu bir test mesajıdır", required=["test", "mesaj"])
        assert result.passed is True
        assert result.score == 100.0

    def test_some_required_missing(self, keyword_validator: KeywordValidator):
        """Bazı required keyword'ler eksik → skor düşer"""
        result = self._call(keyword_validator, "merhaba dünya", required=["merhaba", "test", "foo"])
        assert result.score == pytest.approx(100.0 / 3, rel=0.01)  # 1/3 found
        assert result.passed is True  # forbidden yoksa passed=True

    def test_forbidden_keyword_present(self, keyword_validator: KeywordValidator):
        """Yasak keyword var → FAIL"""
        result = self._call(keyword_validator, "bu bir spam mesajıdır", forbidden=["spam"])
        assert result.passed is False
        assert result.score == 0.0

    def test_required_and_forbidden(self, keyword_validator: KeywordValidator):
        """Hem required hem forbidden — forbidden öncelikli → FAIL"""
        result = self._call(
            keyword_validator,
            "acil önemli spam mesaj",
            required=["acil", "önemli"],
            forbidden=["spam"],
        )
        assert result.passed is False
        assert result.score == 0.0

    def test_case_insensitive(self, keyword_validator: KeywordValidator):
        """Case-insensitive eşleştirme → PASS"""
        result = self._call(keyword_validator, "HELLO WORLD", required=["hello"])
        assert result.passed is True
        assert result.score == 100.0

    def test_empty_keyword_lists(self, keyword_validator: KeywordValidator):
        """Boş liste = hiçbir kısıtlama yok → PASS, score=100"""
        result = self._call(keyword_validator, "herhangi bir metin", required=[], forbidden=[])
        assert result.passed is True
        assert result.score == 100.0

    def test_no_context_at_all(self, keyword_validator: KeywordValidator):
        """Hiç context verilmezse → PASS, score=100"""
        result = keyword_validator.validate("test")
        assert result.passed is True
        assert result.score == 100.0

    def test_details_content(self, keyword_validator: KeywordValidator):
        """details doğru anahtarları içerir"""
        result = self._call(
            keyword_validator,
            "test önemli acil",
            required=["test", "acil", "olmayan"],
            forbidden=["spam"],
        )
        assert "required_found" in result.details
        assert "required_missing" in result.details
        assert "forbidden_found" in result.details
        assert "test" in result.details["required_found"]
        assert "olmayan" in result.details["required_missing"]

    def test_no_forbidden_found(self, keyword_validator: KeywordValidator):
        """Yasak keyword yoksa forbidden_found boş liste"""
        result = self._call(keyword_validator, "temiz metin", forbidden=["spam", "reklam"])
        assert result.details["forbidden_found"] == []


# ---------------------------------------------------------------------------
# LengthValidator Tests — AC 3
# ---------------------------------------------------------------------------

class TestLengthValidator:
    """LengthValidator.validate() — AC 3"""

    def test_within_bounds(self, length_validator: LengthValidator):
        """Uzunluk min/max arasında → PASS, score=100"""
        result = length_validator.validate("hello", context={"min_length": 1, "max_length": 10})
        assert result.passed is True
        assert result.score == 100.0
        assert result.details["length"] == 5
        assert result.details["min_ok"] is True
        assert result.details["max_ok"] is True

    def test_below_minimum(self, length_validator: LengthValidator):
        """Min'in altında → skor düşer"""
        result = length_validator.validate("ab", context={"min_length": 5})
        assert result.score == 50.0  # min fail, max yok = 50
        assert result.passed is True
        assert result.details["min_ok"] is False

    def test_above_maximum(self, length_validator: LengthValidator):
        """Max'in üstünde → skor düşer"""
        result = length_validator.validate("hello world", context={"max_length": 5})
        assert result.score == 50.0  # min yok, max fail = 50
        assert result.passed is True

    def test_both_sides_fail(self, length_validator: LengthValidator):
        """Hem min hem max başarısız → score=0"""
        result = length_validator.validate("xxxxx", context={"min_length": 10, "max_length": 3})
        assert result.score == 0.0
        assert result.passed is True  # score 0 ama forbidden yok → passed=True

    def test_only_min_specified(self, length_validator: LengthValidator):
        """Sadece min_length var → max kontrolü atlanır"""
        result = length_validator.validate("hello!!!", context={"min_length": 3})
        assert result.passed is True
        assert result.score == 100.0
        assert result.details["min_ok"] is True

    def test_only_max_specified(self, length_validator: LengthValidator):
        """Sadece max_length var → min kontrolü atlanır"""
        result = length_validator.validate("hello", context={"max_length": 3})
        assert result.score == 50.0
        assert result.details["max_ok"] is False

    def test_no_context(self, length_validator: LengthValidator):
        """Hiç context yok → PASS, score=100"""
        result = length_validator.validate("anything")
        assert result.passed is True
        assert result.score == 100.0

    def test_non_string_input(self, length_validator: LengthValidator):
        """String olmayan input str() ile çevrilir"""
        result = length_validator.validate([1, 2, 3], context={"min_length": 2, "max_length": 10})
        # repr of list is "[1, 2, 3]" = 9 chars
        assert result.passed is True
        assert result.score == 100.0

    def test_none_input(self, length_validator: LengthValidator):
        """None input → str(None) = 'None' = 4 chars"""
        result = length_validator.validate(None, context={"min_length": 1, "max_length": 10})  # type: ignore[arg-type]
        assert result.passed is True
        assert result.score == 100.0

    def test_details_structure(self, length_validator: LengthValidator):
        """details doğru alanları içerir"""
        result = length_validator.validate("test", context={"min_length": 1, "max_length": 10})
        assert "length" in result.details
        assert "min_length" in result.details
        assert "max_length" in result.details
        assert "min_ok" in result.details
        assert "max_ok" in result.details

    def test_dimension_quantitative(self, length_validator: LengthValidator):
        """LengthValidator dimension=QUANTITATIVE"""
        assert length_validator.dimension == ValidationDimension.QUANTITATIVE


# ---------------------------------------------------------------------------
# RegexValidator Tests — AC 4
# ---------------------------------------------------------------------------

class TestRegexValidator:
    """RegexValidator.validate()"""

    def test_match_pattern(self):
        """Output pattern'e uyuyor -> PASS."""
        from stateguard.plugin.examples.regex import RegexValidator

        v = RegexValidator()
        result = v.validate("hello123", context={"pattern": r"^[a-z]+\d+$"})
        assert result.passed is True
        assert result.score == 100.0

    def test_no_match_pattern(self):
        """Output pattern'e uymuyor -> FAIL."""
        from stateguard.plugin.examples.regex import RegexValidator

        v = RegexValidator()
        result = v.validate("HELLO", context={"pattern": r"^[a-z]+$"})
        assert result.passed is False
        assert result.score == 0.0

    def test_no_pattern_in_context(self):
        """Context'te pattern yoksa -> PASS (kisitlama yok)."""
        from stateguard.plugin.examples.regex import RegexValidator

        v = RegexValidator()
        result = v.validate("anything")
        assert result.passed is True
        assert result.score == 100.0

    def test_invalid_pattern_in_context(self):
        """Gecersiz regex pattern -> hata mesaji iceren FAIL."""
        from stateguard.plugin.examples.regex import RegexValidator

        v = RegexValidator()
        result = v.validate("test", context={"pattern": r"["})
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    def test_class_attributes(self):
        """RegexValidator class attribute'lari dogru."""
        from stateguard.plugin.examples.regex import RegexValidator

        assert RegexValidator.name == "regex"
        assert RegexValidator.dimension == ValidationDimension.STRUCTURAL
        assert RegexValidator.tier == ValidationTier.TIER_1
