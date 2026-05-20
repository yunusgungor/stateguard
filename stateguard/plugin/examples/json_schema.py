"""JSON Schema validator — validates output against a JSON schema.

The :class:`JsonSchemaValidator` checks whether the provided *output*
conforms to a configurable JSON Schema definition.
"""

from typing import Any

from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult
from stateguard.plugin.base import BaseValidator


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
        ...
