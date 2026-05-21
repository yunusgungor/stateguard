"""JSON Schema validator — validates output against a JSON schema.

The :class:`JsonSchemaValidator` checks whether the provided *output*
conforms to a configurable JSON Schema definition.
"""

import json
from typing import Any

from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.plugin.base import BaseValidator

# Optional jsonschema dependency
try:
    import jsonschema

    HAS_JSCHEMA = True
except ImportError:
    HAS_JSCHEMA = False


class JsonSchemaValidator(BaseValidator):
    """Validator that checks JSON schema compliance.

    Class Attributes:
        name:      ``"json_schema"``
        dimension: :attr:`ValidationDimension.STRUCTURAL`
        tier:      :attr:`ValidationTier.TIER_1`
    """

    name: str = "json_schema"
    dimension: ValidationDimension = ValidationDimension.STRUCTURAL
    tier: ValidationTier = ValidationTier.TIER_1

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Validate *output* against a JSON schema.

        Args:
            output:  The data to validate (expected to be a JSON-serialisable
                     object or a string containing JSON).
            context: Optional contextual information (may contain a ``schema``
                     key with the target JSON Schema definition).

        Returns:
            A :class:`ValidationResult` indicating schema compliance.
        """
        if output is None:
            output = ""

        if not isinstance(output, str):
            # Use json.dumps for dicts/lists (valid JSON), str() for other types
            try:
                output = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
            except Exception:
                output = str(output)

        if not output.strip():
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"format": "invalid", "error": "Empty output"},
            )

        # Attempt JSON parse
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as e:
            return ValidationResult(
                score=0.0,
                passed=False,
                dimension=self.dimension,
                details={"format": "invalid", "error": str(e)},
            )

        # If schema provided and jsonschema available, validate
        if context and isinstance(context, dict) and "schema" in context:
            schema = context["schema"]
            if not HAS_JSCHEMA:
                return ValidationResult(
                    score=0.0,
                    passed=False,
                    dimension=self.dimension,
                    details={"format": "unknown", "error": "jsonschema library not installed"},
                )
            if schema is not None:
                try:
                    jsonschema.validate(instance=parsed, schema=schema)
                except jsonschema.ValidationError as e:
                    return ValidationResult(
                        score=0.0,
                        passed=False,
                        dimension=self.dimension,
                        details={"format": "invalid", "error": str(e)},
                    )
                except jsonschema.SchemaError as e:
                    return ValidationResult(
                        score=0.0,
                        passed=False,
                        dimension=self.dimension,
                        details={"format": "invalid", "error": f"Invalid schema: {e}"},
                    )

        return ValidationResult(
            score=100.0,
            passed=True,
            dimension=self.dimension,
            details={"format": "valid"},
        )
