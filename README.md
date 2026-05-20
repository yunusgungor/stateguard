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

## License

MIT
