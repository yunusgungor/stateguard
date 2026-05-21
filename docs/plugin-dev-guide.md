# Plugin Geliştirme Kılavuzu

StateGuard, Python class tabanlı bir plugin sistemi sunar. Kendi validator'larınızı yazarak validasyon motorunu her tür çıktı için genişletebilirsiniz.

---

## Getting Started

### Kurulum

```bash
pip install stateguard
# veya Poetry ile
poetry add stateguard
```

### Temel Kullanım

```python
from stateguard.config.settings import ConfigManager
from stateguard.core.engine import ValidationEngine

# Varsayılan config'i yükle (engine otomatik kullanır)
ConfigManager().load()

# Engine oluştur
engine = ValidationEngine(agent_id="my-agent")

# Bir LLM çıktısını doğrula
result = engine.validate("Merhaba, bugün nasılsınız?")
print(f"Skor: {result.overall_score}")
print(f"Geçti: {result.passed}")
print(f"İzlenen yol: Tier {result.tier_path}")
```

### Config ile Kullanım

```python
from stateguard.config.settings import ConfigManager

cfg = ConfigManager()
config = cfg.load()

print(config.tier1_threshold)  # 80.0
print(config.fail_mode)        # "fail-close"
```

---

## Temel Validator Arayüzü

Tüm validator'lar `BaseValidator` abstract class'ından türetilmelidir.

```python
from stateguard.plugin.base import BaseValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult


class MyValidator(BaseValidator):
    """Örnek validator."""

    name: str = "my-validator"
    dimension: ValidationDimension = ValidationDimension.STRUCTURAL
    tier: ValidationTier = ValidationTier.TIER_1

    def validate(
        self, output: str, context: dict | None = None
    ) -> ValidationResult:
        # Validasyon mantığını buraya yazın
        return ValidationResult(
            score=100.0,
            passed=True,
            dimension=self.dimension,
            details={"message": "Validasyon başarılı"},
        )
```

### Zorunlu Alanlar

| Alan | Tip | Açıklama |
|:-----|:----|:---------|
| `name` | `str` | Benzersiz validator adı |
| `dimension` | `ValidationDimension` | Validasyon boyutu (STRUCTURAL, SEMANTIC, etc.) |
| `tier` | `ValidationTier` | Tier seviyesi (TIER_1, TIER_2, TIER_3) |

### Lifecycle Hook'ları

```python
class MyValidator(BaseValidator):
    name = "my-validator"
    dimension = ValidationDimension.STRUCTURAL
    tier = ValidationTier.TIER_1

    def setup(self) -> None:
        """Kayıt anında çağrılır (opsiyonel)."""
        self._model = self._load_model()

    def teardown(self) -> None:
        """Kaldırma anında çağrılır (opsiyonel)."""
        self._cleanup()

    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        ...
```

---

## Creating Your First Validator

### Adım 1: Sınıfı oluşturun

```python
# my_validator.py
from stateguard.plugin.base import BaseValidator
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult


class LengthCheckValidator(BaseValidator):
    name = "length-check"
    dimension = ValidationDimension.QUANTITATIVE
    tier = ValidationTier.TIER_1

    def validate(self, output, context=None):
        length = len(str(output))
        min_len = context.get("min_length", 1) if context else 1
        max_len = context.get("max_length", 1000) if context else 1000

        passed = min_len <= length <= max_len
        score = 100.0 if passed else 0.0

        return ValidationResult(
            score=score,
            passed=passed,
            dimension=self.dimension,
            details={"length": length, "min_length": min_len, "max_length": max_len},
        )
```

### Adım 2: Kaydedin

```python
from stateguard.plugin.registry import PluginRegistry

registry = PluginRegistry()
validator = LengthCheckValidator()
registry.register(validator)

# Doğrulama
assert "length-check" in [v["name"] for v in registry.list_validators()]
```

### Adım 3: Test yazın

```python
def test_length_check_validator():
    from my_validator import LengthCheckValidator

    v = LengthCheckValidator()
    result = v.validate("Hello", context={"min_length": 1, "max_length": 10})
    assert result.passed
    assert result.score == 100.0

    result = v.validate("A", context={"min_length": 5})
    assert result.score == 0.0
```

---

## API Referansı

### `BaseValidator`

```python
class BaseValidator(ABC):
    name: str                        # Validator adı (zorunlu)
    dimension: ValidationDimension   # Boyut (zorunlu)
    tier: ValidationTier             # Tier (zorunlu)
    description: str                 # Açıklama (opsiyonel, default "")
    version: str                     # Versiyon (opsiyonel, default "0.1.0")

    def setup(self) -> None: ...     # Kayıt anında çağrılır
    def teardown(self) -> None: ...  # Kaldırma anında çağrılır

    @abstractmethod
    def validate(self, output: Any, context: dict | None = None) -> ValidationResult: ...
```

### `PluginRegistry`

```python
class PluginRegistry:
    def register(self, validator: BaseValidator) -> None: ...
    def unregister(self, name: str) -> None: ...
    def list_validators(
        self, dimension: ValidationDimension | None = None
    ) -> list[dict[str, Any]]: ...
    def discover_plugins(
        self, path: str | list[str] | None = None
    ) -> list[str]: ...
```

### `ValidationResult`

```python
class ValidationResult(BaseModel):
    score: float = Field(default=0.0, ge=0.0, le=100.0)  # 0.0 - 100.0
    passed: bool = False                # Geçti/Kaldı
    dimension: ValidationDimension      # Validasyon boyutu
    details: dict[str, Any] = Field(default_factory=dict)  # Detaylı bilgi
    error: str | None = None            # Hata mesajı (varsa)
```

### `EngineResult`

```python
class EngineResult(BaseModel):
    overall_score: float
    passed: bool
    tier_path: list[int]            # Hangi tier'lar çalıştı
    dimension_scores: dict[str, float]      # Tier bazında skorlar (örn. {\"tier_1\": 85.0})
    details: dict[str, Any]         # decision_log, tier_results, config
```

### `DecisionEntry` / `DecisionLogger`

```python
class DecisionEntry(BaseModel):
    timestamp: datetime             # UTC zaman damgası
    agent_id: str                   # Agent kimliği
    step_id: str                    # Pipeline adımı
    dimension: ValidationDimension  # Validasyon boyutu
    score: float                    # 0.0 - 100.0
    decision: str                   # "pass", "fail", "retry", "escalate"
    details: dict[str, Any]         # Metadata

class DecisionLogger:
    def log(self, entry: DecisionEntry) -> None: ...
    def query(
        self,
        agent_id: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        result: str | None = None,
    ) -> list[DecisionEntry]: ...
```

---

## Konfigürasyon Referansı

Varsayılan konfigürasyon `stateguard/config/defaults.yaml` dosyasında tanımlıdır:

```yaml
# --- Tier Thresholds ---
tier1_threshold: 80.0     # score >= 80 → PASS, score < 50 → FAIL
tier2_threshold: 50.0     # Ensemble geçer/kal eşiği

# --- Tier 3 ---
tier3_enabled: true
tier3:
  endpoint: "http://localhost:8000"
  model: "llama-3.2-1b"
  timeout_seconds: 5.0
  usage_limit: 10            # maksimum Tier 3 cagrisi (0 = sinirsiz)

# --- Embedding ---
default_embedding_model: "all-MiniLM-L6-v2"
embedding_device: "cpu"

# --- HITL ---
hitl_timeout_seconds: 300

# --- Fail Mode ---
fail_mode: "fail-close"   # veya "fail-open"

# --- Logging ---
logging:
  level: "INFO"
  format: "json"

# --- Global Threshold ---
default_threshold: 0.7      # 0.0-1.0 arası global geçer/kal eşiği

# --- Scoring ---
scoring:
  weights:
    structural: 0.25
    semantic: 0.25
    quantitative: 0.15
    behavioral: 0.20
    security: 0.15
  thresholds:
    pass: 75.0         # >= 75 → PASS
    borderline: 50.0   # 50-74 → borderline, < 50 → FAIL

# --- Plugin Registry ---
plugins:
  enabled: []               # boş = tümü aktif; doluysa sadece listedekiler
```

### Plugin Konfigürasyonu

Validator'lar context parametresi üzerinden yapılandırılır:

```python
result = validator.validate(
    output,
    context={
        "schema": {"type": "object", "properties": {"name": {"type": "string"}}},  # JsonSchemaValidator
        "required_keywords": ["evet", "tamam"],          # KeywordValidator
        "forbidden_keywords": ["spam", "reklam"],        # KeywordValidator
        "min_length": 10,                                 # LengthValidator
        "max_length": 500,                                # LengthValidator
    },
)
```

---

## Örnek Validator'lar

StateGuard 3 adet referans validator ile birlikte gelir:

### JsonSchemaValidator

```python
from stateguard.plugin.examples.json_schema import JsonSchemaValidator

v = JsonSchemaValidator()

# JSON format kontrolü
result = v.validate('{"name": "test"}')
assert result.passed

# Schema validasyonu (jsonschema opsiyonel)
schema = {"type": "object", "properties": {"name": {"type": "string"}}}
result = v.validate('{"name": "test"}', context={"schema": schema})
```

### KeywordValidator

```python
from stateguard.plugin.examples.keyword import KeywordValidator

v = KeywordValidator()

result = v.validate("Bu bir test mesajıdır", context={
    "required_keywords": ["test"],
    "forbidden_keywords": ["spam"],
})
```

### LengthValidator

```python
from stateguard.plugin.examples.length import LengthValidator

v = LengthValidator()

result = v.validate("Merhaba", context={
    "min_length": 1,
    "max_length": 100,
})
```

---

## İleri Düzey İpuçları

1. **Thread-safe**: Validator'lar thread-safe olmalıdır — StateGuard aynı anda birden çok validate() çağırabilir
2. **State Tutma**: setup()/teardown() ile kaynak yönetimi yapın
3. **Hata Yönetimi**: Beklenmeyen hatalarda her zaman `ValidationResult` döndürün, exception fırlatmayın
4. **Loglama**: `StructLogAdapter` ile yapılandırılmış loglama yapabilirsiniz
5. **Test**: Her validator için unit test yazın, `tmp_path` fixture'ı ile dosya keşfi test edin
6. **NaN/Inf Koruması**: Skor hesaplamalarında `math.isfinite()` ile NaN/Inf kontrolü yapın
7. **f-string Kullanın**: `.format()` kullanmayın — LLM içeriklerinde `{}` crash'e yol açar
