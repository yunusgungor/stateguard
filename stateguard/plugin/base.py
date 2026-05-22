"""Base validator abstract class — pluggable validation unit.

All validators in StateGuard inherit from :class:`BaseValidator`.
The abstract interface enforces that every subclass provides
``name``, ``dimension``, ``tier``, and a ``validate()`` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult


class BaseValidator(ABC):
    """Abstract base class for all StateGuard validators.

    Subclasses **must** define the following class attributes:

    * ``name`` — short identifier (e.g. ``"embedding-similarity"``)
    * ``dimension`` — a :class:`ValidationDimension` member
    * ``tier`` — a :class:`ValidationTier` member

    and **must** implement :meth:`validate`.

    Optional metadata (not enforced):

    * ``description`` — human-readable description (default ``""``)
    * ``version`` — SemVer string (default ``"0.1.0"``)

    Optional lifecycle hooks (no-op by default):

    * :meth:`setup` — called once when the validator is initialised
    * :meth:`teardown` — called when the validator is torn down

    Example::

        class MyValidator(BaseValidator):
            name = "my-validator"
            dimension = ValidationDimension.STRUCTURAL
            tier = ValidationTier.TIER_1

            def validate(self, output, context=None):
                ...  # return ValidationResult
    """

    # --- Required class attributes (enforced via __init_subclass__) ---
    name: str = "base-validator"
    dimension: ValidationDimension = ValidationDimension.STRUCTURAL
    tier: ValidationTier = ValidationTier.TIER_1

    # --- Optional metadata (not enforced) ---
    description: str = ""
    version: str = "0.1.0"

    # ------------------------------------------------------------------
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Enforce that subclasses define required class attributes.

        Subclasses that are themselves abstract (their metaclass is
        ``ABCMeta``, e.g. they inherit from ``ABC`` or have
        ``@abstractmethod``) are allowed to skip the enforcement so
        that intermediate abstract base classes can be created.

        Raises:
            TypeError: If ``name``, ``dimension``, or ``tier`` are not
                       overridden by the subclass (unless the subclass
                       is itself abstract).
        """
        super().__init_subclass__(**kwargs)

        # Allow abstract intermediate classes to skip enforcement.
        if ABC in cls.__bases__ or hasattr(cls, "__abstractmethods__"):
            return

        # Check that required class attrs are actually overridden.
        # Uses MRO traversal: walk the MRO from cls up to (but not including)
        # BaseValidator and check if ANY class in that chain defined the attr.
        # This allows deep inheritance (Mid → Child → GrandChild) as long as
        # at least one intermediate class defines the required attr.
        #
        # NOTE: Value comparison with `is` won't work because enum singletons
        # (e.g. ValidationTier.TIER_1) are the same object whether defined
        # on BaseValidator or a subclass — both resolve to the exact same
        # singleton. Hence the MRO-based __dict__ traversal.
        mro = cls.__mro__
        base_idx = mro.index(BaseValidator)
        chain_classes = mro[:base_idx]  # cls through the class just before BaseValidator

        missing = []
        for attr in ("name", "dimension", "tier"):
            overridden = any(attr in k.__dict__ for k in chain_classes)
            if not overridden:
                missing.append(attr)

        if missing:
            raise TypeError(
                f"{cls.__name__} must define the following class "
                f"attributes: {', '.join(missing)}"
            )

        # Validate types of dimension and tier at class-definition time.
        if not isinstance(cls.dimension, ValidationDimension):
            raise TypeError(
                f"{cls.__name__}.dimension must be a ValidationDimension "
                f"member, got {type(cls.dimension).__name__}"
            )
        if not isinstance(cls.tier, ValidationTier):
            raise TypeError(
                f"{cls.__name__}.tier must be a ValidationTier "
                f"member, got {type(cls.tier).__name__}"
            )

        # Validate that validate is callable (W2 fix).
        validate_attr = cls.__dict__.get("validate")
        if validate_attr is not None and not callable(validate_attr):
            raise TypeError(
                f"{cls.__name__}.validate must be a callable method, "
                f"got {type(validate_attr).__name__}"
            )

    def __init__(self) -> None:
        """Initialise the validator.

        The default implementation is a no-op.  Subclasses may override
        to perform custom initialisation.
        """

    # ------------------------------------------------------------------
    # --- Main validation interface -------------------------------------
    # ------------------------------------------------------------------
    @abstractmethod
    def validate(
        self,
        output: Any,
        context: dict | None = None,
    ) -> ValidationResult:
        """Run validation on *output*.

        Args:
            output:  The data to validate (typically an LLM response).
            context: Optional contextual information (dimension or
                     engine-level data).

        Returns:
            A :class:`ValidationResult` with the validation outcome.
        """

    # ------------------------------------------------------------------
    # --- Optional lifecycle hooks (no-op by default) ------------------
    # ------------------------------------------------------------------
    def setup(self) -> None:
        """Prepare the validator for use (model loading, resource init).

        Override to implement custom initialisation.  The default is a
        no-op.
        """

    def teardown(self) -> None:
        """Clean up resources held by the validator.

        Override to implement custom teardown.  The default is a no-op.
        """
