"""Structural validation — JSON schema, regex, and type checks.

Provides :class:`StructuralValidator` which validates that an output
conforms to a declared JSON Schema, passes regular-expression
constraints, and satisfies basic type expectations.
"""

from typing import Any

from ..plugin.base import BaseValidator
from ..models.result import ValidationResult


class StructuralValidator(BaseValidator):
    """Validator that enforces structural constraints on outputs.

    Supports:
        * JSON Schema conformance (via ``jsonschema`` or equivalent)
        * Regex pattern matching on string fields
        * Python type-annotation checks (isinstance-style)

    Configuration is passed through the *context* dict or set directly
    on the instance.
    """

    def __init__(self, schema: dict | None = None) -> None:
        """Initialise the structural validator.

        Args:
            schema: Optional JSON Schema dictionary to validate against.
        """
        super().__init__()
        self._schema = schema or {}

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Validate *output* against structural rules.

        Args:
            output:  The data to validate (usually a dict or JSON-serialisable value).
            context: May carry ``schema``, ``regex_patterns``, or ``type_checks`` keys.

        Returns:
            A :class:`ValidationResult` summarising structural conformance.
        """
        ...
