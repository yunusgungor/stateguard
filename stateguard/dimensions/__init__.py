"""Dimensions module — Multi-dimensional validation plugins.

Modules:
    structural:   StructuralValidator — JSON schema, regex, type checks
    semantic:     SemanticValidator — embedding cross-validation
    quantitative: QuantitativeValidator — outlier detection (Isolation Forest, Z-Score)
    behavioral:   BehavioralValidator — state machine & snapshot comparison
    security:     SecurityValidator — prompt injection detection
"""

from stateguard.dimensions.quantitative import QuantitativeValidator
from stateguard.dimensions.semantic import SemanticValidator
from stateguard.dimensions.structural import StructuralValidator

__all__ = [
    "QuantitativeValidator",
    "SemanticValidator",
    "StructuralValidator",
]
