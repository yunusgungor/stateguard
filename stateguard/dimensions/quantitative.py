"""Quantitative validation — Isolation Forest & Z-Score outlier detection.

Provides :class:`QuantitativeValidator` that treats validation as an
anomaly-detection problem.  Supports both unsupervised (Isolation Forest)
and statistical (Z-Score) methods to flag outputs that deviate from
expected distributions.
"""

from typing import Any

from ..plugin.base import BaseValidator
from ..models.result import ValidationResult


class QuantitativeValidator(BaseValidator):
    """Validator that flags outliers using statistical or ML methods.

    Two detection modes are available:

        * ``"zscore"`` — Classic Z-Score thresholding.  Values with
          |Z| > *z_threshold* are flagged.
        * ``"isolation_forest"`` — Unsupervised anomaly detection via
          :class:`sklearn.ensemble.IsolationForest`.

    The validator can be fitted on a reference dataset at initialisation
    time or incrementally over time.
    """

    def __init__(
        self,
        method: str = "zscore",
        z_threshold: float = 3.0,
        contamination: float = 0.1,
    ) -> None:
        """Initialise the quantitative validator.

        Args:
            method:        Detection mode — ``"zscore"`` or ``"isolation_forest"``.
            z_threshold:   Z-Score cutoff (only used when *method* is ``"zscore"``).
            contamination: Expected proportion of outliers in the data
                           (only used when *method* is ``"isolation_forest"``).
        """
        super().__init__()
        self._method = method
        self._z_threshold = z_threshold
        self._contamination = contamination

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Validate *output* by checking whether it is an outlier.

        Args:
            output:  Numeric value or vector to evaluate.
            context: May contain ``reference_data``, ``method`` override,
                     or per-call parameters.

        Returns:
            A :class:`ValidationResult` where low scores indicate anomalies.
        """
        ...
