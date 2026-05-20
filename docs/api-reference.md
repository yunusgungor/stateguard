# API Referansı

Bu sayfa StateGuard'ın public API'sini belgelemektedir.

## ValidationEngine

```python
class ValidationEngine:
    """Ana validasyon motoru."""
    
    def __init__(self, config: ValidationConfig | None = None):
        ...
    
    def validate(
        self,
        output: str,
        context: dict | None = None,
        expected_format: str | None = None
    ) -> EngineResult:
        ...
```

### Parametreler

| Parametre | Tip | Varsayılan | Açıklama |
|:----------|:----|:-----------|:---------|
| `output` | `str` | — | Doğrulanacak LLM çıktısı |
| `context` | `dict` | `None` | Validasyon için bağlam bilgisi |
| `expected_format` | `str` | `None` | Beklenen çıktı formatı (json, xml, text) |

### Dönüş Değeri

`EngineResult` — validasyon sonucu (score, passed, dimension_scores, errors, metadata)

---

## ValidationConfig

```python
class ValidationConfig(BaseModel):
    """Validasyon motoru konfigürasyonu."""
    
    tier1_threshold: float = 80.0
    tier2_threshold: float = 70.0
    adaptive: bool = True
    max_retries: int = 2
    hitl_enabled: bool = True
    snapshot_enabled: bool = True
```

---

## PluginRegistry

```python
class PluginRegistry:
    """Plugin kayıt ve keşif sistemi."""
    
    @classmethod
    def register(cls, name: str, validator_class: type[BaseValidator]) -> None:
        ...
    
    def get(self, name: str) -> BaseValidator:
        ...
    
    def list(self) -> dict[str, ValidatorInfo]:
        ...
```

---

## StateMachine

```python
class StateMachine:
    """Pipeline FSM yönetimi."""
    
    def transition(self, target: PipelineState) -> None:
        ...
    
    def snapshot(self) -> StateSnapshot:
        ...
```

---

## DimensionResult

```python
class DimensionResult(BaseModel):
    """Tek bir validasyon boyutunun sonucu."""
    
    name: str
    score: float
    passed: bool
    details: str
    errors: list[str]
```

---

## EngineResult

```python
class EngineResult(BaseModel):
    """Validasyon motoru nihai sonucu."""
    
    score: float
    passed: bool
    dimension_scores: dict[str, DimensionResult]
    tier_used: int
    total_time_ms: float
    errors: list[str]
    metadata: dict
```

---

## DecisionEntry

```python
class DecisionEntry(BaseModel):
    """Decision log kaydı."""
    
    timestamp: datetime
    input_hash: str
    output: str
    score: float
    passed: bool
    tier_used: int
    dimension_results: dict[str, DimensionResult]
    feedback: str | None
```

---

> **Not:** Bu API referansı henüz tamamlanmamıştır. Detaylı dokümantasyon için `stateguard/` kaynak koduna ve `pydoc` çıktısına bakabilirsiniz.
