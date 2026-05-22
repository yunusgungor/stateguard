# StateGuard — AI Agent Validation Engine

Multi-dimensional, deterministic output validator for AI agent outputs.

## Installation

```bash
pip install stateguard
```

## Quick Start

```python
from stateguard import ValidationEngine

engine = ValidationEngine()
result = engine.validate("LLM output text", context={})
print(f"Score: {result.overall_score}, Passed: {result.passed}")
```

## Features

- **Cascade Validation Pipeline** — Tier 1 (Embedding) → Tier 2 (ML Ensemble) → Tier 3 (LLM)
- **5 Validation Dimensions** — Structural, Semantic, Quantitative, Behavioral, Security
- **Plugin System** — Extend with custom validators via Python classes
- **State Machine + Snapshot** — Pipeline safety and error cascade detection
- **Auto-Retry → HITL → Rollback** — Graduated error handling

## Plugin Development Quickstart

Create your own validator in minutes:

```python
from stateguard.plugin.base import BaseValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult

class MyValidator(BaseValidator):
    name: str = "my-validator"
    dimension: ValidationDimension = ValidationDimension.STRUCTURAL
    tier: ValidationTier = ValidationTier.TIER_1

    def validate(self, output, context=None):
        # Your validation logic here
        return ValidationResult(
            score=100.0,
            passed=True,
            dimension=self.dimension,
        )
```

See [Plugin Development Guide](docs/plugin-dev-guide.md) for full documentation,
inheritance rules, lifecycle hooks, and built-in examples (JsonSchema,
Keyword, Length, Regex validators).

## License

MIT
