"""Base validator abstract class for the StateGuard plugin system.

Every dimension-validator and tier-validator inherits from
:class:`BaseValidator` and implements a ``validate`` method that
returns a :class:`ValidationResult`.
"""

import abc
from typing import Any

from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult


class BaseValidator(abc.ABC):
    """Abstract base class for all validators in the StateGuard pipeline.

    Subclasses must implement :meth:`validate` and provide values for
    the class attributes *name*, *dimension*, and *tier*.

    Attributes:
        name:      Human-readable validator name.
        dimension: The :class:`ValidationDimension` this validator targets.
        tier:      The :class:`ValidationTier` this validator belongs to.
    """

    name: str = ""
    dimension: ValidationDimension = ValidationDimension.STRUCTURAL
    tier: ValidationTier = ValidationTier.TIER_1

    @abc.abstractmethod
    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Run validation on *output* given an optional *context*.

        Args:
            output:  The data to validate (type depends on the dimension).
            context: Optional contextual information (default: ``None``).

        Returns:
            A :class:`ValidationResult` summarising the outcome.
        """
        ...
