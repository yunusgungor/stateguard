"""Structural validation — JSON schema, regex, and type checks.

Provides :class:`StructuralValidator` which validates that an output
conforms to a declared JSON Schema, passes regular-expression
constraints, and satisfies basic type expectations.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..models.enums import ValidationDimension, ValidationTier
from ..models.result import ValidationResult
from ..plugin.base import BaseValidator

# Optional jsonschema dependency
try:
    import jsonschema

    HAS_JSCHEMA = True
except ImportError:
    HAS_JSCHEMA = False


_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


class StructuralValidator(BaseValidator):
    """Validator that enforces structural constraints on outputs.

    Supports:

        * JSON Schema conformance (via ``jsonschema`` or equivalent)
        * Regex pattern matching on string fields
        * Python type-annotation checks (isinstance-style)

    Configuration is passed through the *context* dict.
    """

    name: str = "structural"
    dimension: ValidationDimension = ValidationDimension.STRUCTURAL
    tier: ValidationTier = ValidationTier.TIER_1

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Validate *output* against structural rules.

        Args:
            output:  The data to validate (usually a JSON string or plain text).
            context: Must be a dict. May carry ``schema``, ``regex_patterns``, or
                     ``type_checks`` keys.

        Returns:
            A :class:`ValidationResult` summarising structural conformance.
        """
        # Patch 1: validate context is actually a dict (not str/int/etc.)
        ctx = context if isinstance(context, dict) else {}
        details: dict[str, Any] = {}

        # --- Handle None / empty -------------------------------------------------
        if output is None:
            output = ""

        if not isinstance(output, str):
            try:
                output = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
            except Exception:
                output = str(output)

        if not output.strip():
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"error": "Empty output"},
            )

        # --- Helper: parse JSON once (lazy) --------------------------------------
        _parsed: Any = None  # cached parsed JSON

        def _ensure_parsed() -> Any:
            nonlocal _parsed
            if _parsed is None:
                try:
                    _parsed = json.loads(output)
                except json.JSONDecodeError as e:
                    raise _ValidationHalt(e)
            return _parsed

        # --- 1. JSON Schema validation -------------------------------------------
        schema = ctx.get("schema")
        if schema is not None:
            try:
                parsed = _ensure_parsed()
            except _ValidationHalt as e:
                return _fail(self.dimension, str(e), accumulated=details)

            if not HAS_JSCHEMA:
                return ValidationResult(
                    score=0.0,
                    passed=False,
                    dimension=self.dimension,
                    details={"error": "jsonschema library not installed"},
                )
            try:
                jsonschema.validate(instance=parsed, schema=schema)
                details["schema"] = "valid"
            except jsonschema.ValidationError as e:
                return _fail(self.dimension, str(e), accumulated=details)
            except jsonschema.SchemaError as e:
                return _fail(self.dimension, f"Invalid schema: {e}", accumulated=details)
            # Patch 6: catch RefResolutionError (sibling of SchemaError)
            except jsonschema.exceptions.RefResolutionError as e:
                return _fail(self.dimension, f"Schema reference error: {e}", accumulated=details)

        # --- 2. Regex pattern matching -------------------------------------------
        # Patch 1: validate regex_patterns is actually a list
        raw_patterns = ctx.get("regex_patterns", [])
        regex_patterns = raw_patterns if isinstance(raw_patterns, list) else []

        if regex_patterns:
            matches: dict[str, bool] = {}
            for pattern in regex_patterns:
                # Patch 1: guard non-string pattern elements
                if not isinstance(pattern, str):
                    return _fail(
                        self.dimension,
                        f"Regex pattern must be a string, got {type(pattern).__name__}",
                        accumulated=details,
                    )
                try:
                    m = re.search(pattern, output)
                    matches[pattern] = m is not None
                except re.error as e:
                    return _fail(
                        self.dimension,
                        f"Invalid regex pattern {pattern!r}: {e}",
                        accumulated=details,
                    )

            failed_patterns = [p for p, ok in matches.items() if not ok]
            if failed_patterns:
                return _fail(
                    self.dimension,
                    f"Regex patterns not matched: {failed_patterns}",
                    extra={"regex_matches": matches},
                    accumulated=details,
                )
            details["regex_matches"] = matches

        # --- 3. Type checks ------------------------------------------------------
        raw_type_checks = ctx.get("type_checks", {})
        type_checks = raw_type_checks if isinstance(raw_type_checks, dict) else {}

        if type_checks:
            try:
                parsed = _ensure_parsed()
            except _ValidationHalt as e:
                return _fail(self.dimension, f"Cannot parse output for type checks: {e}", accumulated=details)

            # Patch 2: guard against non-dict JSON roots (None, bool, list, str)
            if not isinstance(parsed, dict):
                return _fail(
                    self.dimension,
                    f"Cannot perform type checks: output root is {type(parsed).__name__}, not an object",
                    accumulated=details,
                )

            for field_name, check in type_checks.items():
                # Patch 1: validate check value is a dict
                if not isinstance(check, dict):
                    return _fail(
                        self.dimension,
                        f"Type check for {field_name!r} must be a dict, got {type(check).__name__}",
                        accumulated=details,
                    )

                expected_type_name = check.get("type")
                # Patch 4: strict boolean check for "required" (not bool("false"))
                required_raw = check.get("required", False)
                required = required_raw if isinstance(required_raw, bool) else False

                if required and field_name not in parsed:
                    return _fail(
                        self.dimension,
                        f"Required field {field_name!r} is missing",
                        accumulated=details,
                    )

                if field_name in parsed:
                    # Patch 3: fail if "type" key is missing from check entry
                    if expected_type_name is None:
                        return _fail(
                            self.dimension,
                            f"Type check for {field_name!r} is missing the 'type' field",
                            accumulated=details,
                        )

                    expected = _TYPE_MAP.get(expected_type_name)
                    if expected is None:
                        return _fail(
                            self.dimension,
                            f"Unknown type {expected_type_name!r} for field {field_name!r}",
                            accumulated=details,
                        )

                    actual_value = parsed[field_name]
                    if not isinstance(actual_value, expected):
                        # Patch 5: float-as-integer guard (42.0 is a valid integer)
                        if expected_type_name == "integer" and isinstance(actual_value, float) and actual_value == int(actual_value):
                            pass  # Accept 42.0 as integer
                        else:
                            return _fail(
                                self.dimension,
                                f"Field {field_name!r} expected type {expected_type_name}, "
                                f"got {type(actual_value).__name__}",
                                accumulated=details,
                            )

        # --- All passed ----------------------------------------------------------
        return ValidationResult(
            score=100.0,
            passed=True,
            dimension=self.dimension,
            details=details or {"format": "valid"},
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ValidationHalt(Exception):
    """Raised internally to short-circuit on JSON parse failure."""


def _fail(
    dimension: ValidationDimension,
    error: str,
    extra: dict[str, Any] | None = None,
    accumulated: dict[str, Any] | None = None,
) -> ValidationResult:
    """Build a failing ValidationResult with the given *error*.

    Args:
        dimension: The validation dimension.
        error:     Human-readable error message.
        extra:     Additional detail keys to merge into details (overwrites
                   ``error`` if included — callers should avoid this).
        accumulated: Previously accumulated details from earlier validation
                     steps that passed (e.g. ``details`` dict with
                     ``schema='valid'``). These are merged *before* the
                     error key so the final result shows what passed and
                     what failed.
    """
    # Patch 7: preserve accumulated details from steps that already passed
    details: dict[str, Any] = {}
    if accumulated:
        details.update(accumulated)
    details["error"] = error
    if extra:
        details.update(extra)
    return ValidationResult(
        score=0.0,
        passed=False,
        dimension=dimension,
        details=details,
    )
