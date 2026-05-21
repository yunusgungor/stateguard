"""Tests for StructuralValidator — JSON schema, regex, and type checks."""

from __future__ import annotations

from typing import Any

import pytest

from stateguard.dimensions.structural import StructuralValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def validator() -> StructuralValidator:
    return StructuralValidator()


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------

class TestStructuralValidatorAttributes:
    """AC 1 — Class-level attribute enforcement."""

    def test_name(self, validator: StructuralValidator) -> None:
        assert validator.name == "structural"

    def test_dimension(self, validator: StructuralValidator) -> None:
        assert validator.dimension == ValidationDimension.STRUCTURAL

    def test_tier(self, validator: StructuralValidator) -> None:
        assert validator.tier == ValidationTier.TIER_1


# ---------------------------------------------------------------------------
# JSON schema validaton — AC 2
# ---------------------------------------------------------------------------

class TestJsonSchemaValidation:
    """AC 2 — JSON schema conformance."""

    @pytest.fixture
    def simple_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }

    def test_valid_json_no_schema(self, validator: StructuralValidator) -> None:
        """Schema yok → all checks pass if nothing else configured."""
        result = validator.validate('{"key": "value"}')
        assert result.passed is True
        assert result.score == 100.0

    def test_valid_json_matches_schema(
        self, validator: StructuralValidator, simple_schema: dict[str, Any],
    ) -> None:
        """Schema ile uyumlu JSON → PASS."""
        result = validator.validate(
            '{"name": "test", "age": 30}',
            context={"schema": simple_schema},
        )
        assert result.passed is True
        assert result.score == 100.0
        assert result.details.get("schema") == "valid"

    def test_invalid_json_against_schema(
        self, validator: StructuralValidator, simple_schema: dict[str, Any],
    ) -> None:
        """Schema ile uyumsuz JSON → FAIL, score=0.0."""
        result = validator.validate(
            '{"age": 30}',
            context={"schema": simple_schema},
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    def test_invalid_json_syntax(
        self, validator: StructuralValidator, simple_schema: dict[str, Any],
    ) -> None:
        """Geçersiz JSON syntax → FAIL, score=0.0."""
        result = validator.validate(
            "{invalid}",
            context={"schema": simple_schema},
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    def test_null_output(
        self, validator: StructuralValidator, simple_schema: dict[str, Any],
    ) -> None:
        """None output → FAIL."""
        result = validator.validate(None, context={"schema": simple_schema})  # type: ignore[arg-type]
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_string(
        self, validator: StructuralValidator, simple_schema: dict[str, Any],
    ) -> None:
        """Boş string → FAIL."""
        result = validator.validate("", context={"schema": simple_schema})
        assert result.passed is False
        assert result.score == 0.0

    def test_jsonschema_not_installed_graceful(
        self, validator: StructuralValidator, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """jsonschema yoksa graceful degradation: FAIL + error mesajı."""
        import stateguard.dimensions.structural as structural_mod

        monkeypatch.setattr(structural_mod, "HAS_JSCHEMA", False)
        result = validator.validate(
            '{"name": "test"}',
            context={"schema": {"type": "object"}},
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "jsonschema library not installed" in result.details.get("error", "")

    def test_invalid_schema_definition(
        self, validator: StructuralValidator,
    ) -> None:
        """Geçersiz schema tanımı → FAIL."""
        result = validator.validate(
            '{"name": "test"}',
            context={"schema": {"type": 123}},  # invalid schema
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details


# ---------------------------------------------------------------------------
# Regex pattern matching — AC 3
# ---------------------------------------------------------------------------

class TestRegexPatternMatching:
    """AC 3 — Regex pattern matching with re.search()."""

    def test_all_patterns_match(self, validator: StructuralValidator) -> None:
        """Tüm pattern'ler eşleşir → PASS."""
        result = validator.validate(
            "Hello World",
            context={"regex_patterns": [r"^Hello", r"World$"]},
        )
        assert result.passed is True
        assert result.score == 100.0

    def test_some_patterns_missing(self, validator: StructuralValidator) -> None:
        """Bazı pattern'ler eşleşmez → FAIL."""
        result = validator.validate(
            "Hello World",
            context={"regex_patterns": [r"^Hello", r"Goodbye"]},
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    def test_empty_pattern_list(self, validator: StructuralValidator) -> None:
        """Boş pattern listesi → skip, PASS."""
        result = validator.validate("Hello", context={"regex_patterns": []})
        assert result.passed is True
        assert result.score == 100.0

    def test_case_sensitive_matching(self, validator: StructuralValidator) -> None:
        """Case-sensitive: 'hello' != 'Hello' (default re.search)."""
        result = validator.validate(
            "Hello World",
            context={"regex_patterns": ["hello"]},
        )
        assert result.passed is False
        assert result.score == 0.0

    def test_regex_details(
        self, validator: StructuralValidator,
    ) -> None:
        """Başarılı eşleşmede details.regex_matches doldurulur."""
        result = validator.validate(
            "Hello World",
            context={"regex_patterns": [r"Hello"]},
        )
        assert result.passed is True
        assert "regex_matches" in result.details
        assert result.details["regex_matches"].get(r"Hello") is True

    def test_invalid_regex_pattern(
        self, validator: StructuralValidator,
    ) -> None:
        """Geçersiz regex pattern → FAIL + error mesajı."""
        result = validator.validate(
            "Hello",
            context={"regex_patterns": [r"[invalid"]},
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details


# ---------------------------------------------------------------------------
# Type checks — AC 4
# ---------------------------------------------------------------------------

class TestTypeChecks:
    """AC 4 — Type checking on JSON fields."""

    def test_correct_types(self, validator: StructuralValidator) -> None:
        """Tüm alanlar doğru tipte → PASS."""
        result = validator.validate(
            '{"name": "test", "count": 42, "active": true}',
            context={
                "type_checks": {
                    "name": {"type": "string", "required": True},
                    "count": {"type": "integer", "required": True},
                    "active": {"type": "boolean", "required": False},
                },
            },
        )
        assert result.passed is True
        assert result.score == 100.0

    def test_wrong_type(self, validator: StructuralValidator) -> None:
        """Yanlış tip → FAIL."""
        result = validator.validate(
            '{"name": 123}',
            context={
                "type_checks": {
                    "name": {"type": "string", "required": True},
                },
            },
        )
        assert result.passed is False
        assert result.score == 0.0

    def test_required_field_missing(self, validator: StructuralValidator) -> None:
        """Required field eksik → FAIL."""
        result = validator.validate(
            '{"name": "test"}',
            context={
                "type_checks": {
                    "name": {"type": "string", "required": True},
                    "email": {"type": "string", "required": True},
                },
            },
        )
        assert result.passed is False
        assert result.score == 0.0

    def test_optional_field_missing(self, validator: StructuralValidator) -> None:
        """Optional field eksik → PASS (sorun değil)."""
        result = validator.validate(
            '{"name": "test"}',
            context={
                "type_checks": {
                    "name": {"type": "string", "required": True},
                    "email": {"type": "string", "required": False},
                },
            },
        )
        assert result.passed is True
        assert result.score == 100.0

    def test_number_type_accepts_int_and_float(
        self, validator: StructuralValidator,
    ) -> None:
        """'number' tipi hem int hem float kabul eder."""
        result = validator.validate(
            '{"rate": 42.5, "count": 10}',
            context={
                "type_checks": {
                    "rate": {"type": "number", "required": True},
                    "count": {"type": "number", "required": True},
                },
            },
        )
        assert result.passed is True
        assert result.score == 100.0

    def test_array_type(self, validator: StructuralValidator) -> None:
        """'array' tipi list kabul eder."""
        result = validator.validate(
            '{"items": [1, 2, 3]}',
            context={
                "type_checks": {
                    "items": {"type": "array", "required": True},
                },
            },
        )
        assert result.passed is True
        assert result.score == 100.0

    def test_null_type(self, validator: StructuralValidator) -> None:
        """'null' tipi None kabul eder."""
        result = validator.validate(
            '{"maybe": null}',
            context={
                "type_checks": {
                    "maybe": {"type": "null", "required": False},
                },
            },
        )
        assert result.passed is True
        assert result.score == 100.0

    def test_empty_type_checks_dict(
        self, validator: StructuralValidator,
    ) -> None:
        """Boş type_checks → skip, PASS."""
        result = validator.validate("test", context={"type_checks": {}})
        assert result.passed is True
        assert result.score == 100.0


# ---------------------------------------------------------------------------
# Combined validation — all three at once
# ---------------------------------------------------------------------------

class TestCombinedValidation:
    """Tüm kontroller aynı anda."""

    def test_all_three_pass(self, validator: StructuralValidator) -> None:
        """JSON schema + regex + type checks hepsi geçer → PASS."""
        schema = {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        }
        result = validator.validate(
            '{"email": "test@example.com"}',
            context={
                "schema": schema,
                "regex_patterns": [r".+@.+\..+"],
                "type_checks": {
                    "email": {"type": "string", "required": True},
                },
            },
        )
        assert result.passed is True
        assert result.score == 100.0

    def test_regex_fails_others_pass(
        self, validator: StructuralValidator,
    ) -> None:
        """Regex fail → FAIL, diğer kontroller geçse bile."""
        schema = {"type": "object", "properties": {"email": {"type": "string"}}}
        result = validator.validate(
            '{"email": "invalid"}',
            context={
                "schema": schema,
                "regex_patterns": [r".+@.+\..+"],
                "type_checks": {"email": {"type": "string", "required": True}},
            },
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case'ler: None, boş, büyük JSON, unicode."""

    def test_none_output_no_context(
        self, validator: StructuralValidator,
    ) -> None:
        """None output, hiç context yok → FAIL."""
        result = validator.validate(None)  # type: ignore[arg-type]
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_string_no_context(
        self, validator: StructuralValidator,
    ) -> None:
        """Boş string, hiç context yok → FAIL."""
        result = validator.validate("")
        assert result.passed is False
        assert result.score == 0.0

    def test_dict_input(
        self, validator: StructuralValidator,
    ) -> None:
        """Dict input → otomatik json.dumps ile string'e çevrilir."""
        result = validator.validate(
            {"key": "value"},
            context={
                "schema": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                },
            },
        )
        assert result.passed is True
        assert result.score == 100.0

    def test_list_input(
        self, validator: StructuralValidator,
    ) -> None:
        """List input → otomatik json.dumps."""
        result = validator.validate([1, 2, 3])
        assert result.passed is True
        assert result.score == 100.0

    def test_unicode_output(self, validator: StructuralValidator) -> None:
        """Unicode karakterlerle regex çalışır."""
        result = validator.validate(
            "Merhaba Dünya 🎉",
            context={"regex_patterns": [r"Merhaba", r"🎉"]},
        )
        assert result.passed is True
        assert result.score == 100.0

    def test_no_context_at_all(self, validator: StructuralValidator) -> None:
        """Hiç context verilmezse → PASS, score=100."""
        result = validator.validate("plain text")
        assert result.passed is True
        assert result.score == 100.0

    def test_dimension_preserved(
        self, validator: StructuralValidator,
    ) -> None:
        """Tüm sonuçlarda dimension=STRUCTURAL."""
        result = validator.validate("anything")
        assert result.dimension == ValidationDimension.STRUCTURAL
        result2 = validator.validate("")
        assert result2.dimension == ValidationDimension.STRUCTURAL


# ---------------------------------------------------------------------------
# Performance — AC 5
# ---------------------------------------------------------------------------

class TestPerformance:
    """AC 5 — 50ms altında tamamlanma."""

    def test_completes_under_50ms(
        self, validator: StructuralValidator,
    ) -> None:
        """Her bir validasyon çağrısı 50ms altında tamamlanır (AC 5)."""
        import time

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "value": {"type": "integer"}},
            "required": ["name"],
        }
        ctx = {
            "schema": schema,
            "regex_patterns": [r"^\{", r"\}$"],
            "type_checks": {
                "name": {"type": "string", "required": True},
                "value": {"type": "integer", "required": False},
            },
        }
        start = time.perf_counter()
        for _ in range(100):
            validator.validate('{"name": "test", "value": 42}', context=ctx)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 50, f"Average call took {avg_ms:.1f}ms (limit 50ms)"

    def test_large_json_performance(
        self, validator: StructuralValidator,
    ) -> None:
        """Büyük JSON (~1MB) — hızlı tamamlanır."""
        import time

        # Build a ~1.1MB JSON string with 7000 items, each with long data field
        data_chunk = "x" * 200
        items = [f'{{"id": {i}, "data": "{data_chunk}"}}' for i in range(7000)]
        large = '{"items": [' + ','.join(items) + ']}'
        assert len(large) >= 1_000_000, f"Test data too small: {len(large)} bytes"

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}, "data": {"type": "string"}},
                    },
                },
            },
        }
        start = time.perf_counter()
        result = validator.validate(large, context={"schema": schema})
        elapsed = time.perf_counter() - start
        elapsed_ms = elapsed * 1000

        assert result.passed is True
        assert elapsed_ms < 3000, f"Large JSON took {elapsed_ms:.1f}ms (limit 3000ms)"
