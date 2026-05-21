# API Referansı

Bu sayfa StateGuard'ın public API'sini belgelemektedir.

---

## `ValidationEngine`

```python
class ValidationEngine:
    """Ana validasyon motoru. Cascade pipeline: Tier 1 → Tier 2 → Tier 3."""

    def __init__(
        self,
        embedding_validator: EmbeddingValidator | None = None,
        ensemble_validator: EnsembleValidator | None = None,
        llm_validator: LLMValidator | None = None,
        agent_id: str | None = None,
        tier3_enabled: bool | None = None,
        fail_mode: str | None = None,
        decision_logger: DecisionLogger | None = None,
    ) -> None: ...

    def validate(
        self,
        output: Any,
        context: dict | None = None,
    ) -> EngineResult: ...
```

### Parametreler (`__init__`)

| Parametre | Tip | Varsayılan | Açıklama |
|:----------|:----|:-----------|:---------|
| `embedding_validator` | `EmbeddingValidator \| None` | `None` | Tier 1 (otomatik oluşturulur) |
| `ensemble_validator` | `EnsembleValidator \| None` | `None` | Tier 2 (otomatik oluşturulur) |
| `llm_validator` | `LLMValidator \| None` | `None` | Tier 3 (config'den oluşturulur) |
| `agent_id` | `str \| None` | `None` | Decision log için agent kimliği |
| `tier3_enabled` | `bool \| None` | `None` | Config'deki değeri override eder |
| `fail_mode` | `str \| None` | `None` | `"fail-close"` veya `"fail-open"` |
| `decision_logger` | `DecisionLogger \| None` | `None` | Karar loglayıcı (otomatik oluşturulur) |

### Dönüş Değeri

`EngineResult` — validasyon sonucu:

| Alan | Tip | Açıklama |
|:-----|:----|:---------|
| `overall_score` | `float` | 0.0-100.0 arası toplam skor |
| `passed` | `bool` | Geçti/Kaldı |
| `tier_path` | `list[int]` | Hangi tier'lar çalıştı (`[1]`, `[1,2]`, `[1,2,3]`) |
| `dimension_scores` | `dict` | Boyut bazında skorlar |
| `details` | `dict` | `decision_log`, `tier_results`, `config` |

---

## `ValidationResult`

```python
class ValidationResult(BaseModel):
    score: float                          # 0.0 - 100.0 (Field(ge=0.0, le=100.0))
    passed: bool                          # Geçti/Kaldı
    dimension: ValidationDimension        # Validasyon boyutu
    details: dict[str, Any]               # Detaylı bilgi (default_factory=dict)
    error: str | None = None              # Hata mesajı (opsiyonel)
```

---

## `BaseValidator`

```python
class BaseValidator(ABC):
    name: str                            # Zorunlu — validator adı
    dimension: ValidationDimension       # Zorunlu — boyut
    tier: ValidationTier                 # Zorunlu — tier
    description: str = ""                # Opsiyonel — açıklama
    version: str = "0.1.0"               # Opsiyonel — versiyon

    def setup(self) -> None: ...         # Kayıt anında çağrılır
    def teardown(self) -> None: ...      # Kaldırma anında çağrılır

    @abstractmethod
    def validate(
        self, output: Any, context: dict | None = None
    ) -> ValidationResult: ...
```

### Enum: `ValidationDimension`

```python
class ValidationDimension(str, Enum):
    STRUCTURAL = "structural"        # Yapısal validasyon
    SEMANTIC = "semantic"            # Anlamsal validasyon
    QUANTITATIVE = "quantitative"    # Sayısal validasyon
    BEHAVIORAL = "behavioral"        # Davranışsal validasyon
    SECURITY = "security"            # Güvenlik validasyonu
```

### Enum: `ValidationTier`

```python
class ValidationTier(str, Enum):
    TIER_1 = "tier_1"   # Hızlı geçiş (Embedding, Kural bazlı)
    TIER_2 = "tier_2"   # Orta (Ensemble ML)
    TIER_3 = "tier_3"   # Derin (Küçük LLM)
```

---

## `PluginRegistry`

```python
class PluginRegistry:
    def __init__(self) -> None: ...

    def register(self, validator: BaseValidator) -> None:
        """Validator kaydeder.
        Raises: TypeError (BaseValidator değilse), ValueError (duplicate name)
        """

    def unregister(self, name: str) -> None:
        """Validator kaldırır.
        Raises: KeyError (name bulunamazsa)
        """

    def list_validators(
        self,
        dimension: ValidationDimension | None = None,
    ) -> list[dict[str, Any]]:
        """Tüm validatörleri metadata ile listeler."""

    def discover_plugins(
        self,
        path: str | list[str] | None = None,
    ) -> list[str]:
        """Dosya sisteminden plugin keşfeder ve kaydeder."""
```

### Return Format (`list_validators`)

Her validator için dönen dict anahtarları:

| Anahtar | Tip | Açıklama |
|:--------|:----|:---------|
| `name` | `str` | Validator adı |
| `dimension` | `ValidationDimension` | Boyut |
| `tier` | `ValidationTier` | Tier |
| `type` | `str` | Sınıf adı |
| `description` | `str` | Açıklama |
| `version` | `str` | Versiyon |

---

## `DecisionEntry` / `DecisionLogger`

```python
class DecisionEntry(BaseModel):
    timestamp: datetime             # UTC (default_factory=datetime.now(timezone.utc))
    agent_id: str                   # Agent kimliği
    step_id: str                    # Pipeline adımı
    dimension: ValidationDimension  # Boyut
    score: float                    # 0.0 - 100.0 (Field(ge=0.0, le=100.0))
    decision: str                   # "pass" | "fail" | "retry" | "escalate"
    details: dict[str, Any]         # Metadata

class DecisionLogger:
    """Thread-safe in-memory decision log."""

    def log(self, entry: DecisionEntry) -> None: ...

    def query(
        self,
        agent_id: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        result: str | None = None,
    ) -> list[DecisionEntry]:
        """Filtrelenmiş sorgu. Tüm filtreler AND ile birleşir.
        - agent_id: Tam eşleşme
        - time_range: (start, end) inclusive
        - result: Case-insensitive decision eşleşmesi
        """
```

---

## `ConfigManager` / `StateGuardConfig`

```python
class ConfigManager:
    def __init__(self, config_path: str | None = None) -> None: ...

    def load(self) -> StateGuardConfig:
        """defaults.yaml dosyasını yükler ve Pydantic ile doğrular."""

class StateGuardConfig(BaseModel):
    tier1_threshold: float = 80.0
    tier2_threshold: float = 50.0
    tier3_enabled: bool = True
    tier3: Tier3Config
    default_embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    hitl_timeout_seconds: int = 300
    fail_mode: str = "fail-close"
    logging: dict[str, Any]
    default_threshold: float = 0.7
    scoring: dict[str, Any]
    plugins: dict[str, Any]
```

---

## `StructLogAdapter`

```python
class StructLogAdapter:
    """Yapılandırılmış loglama adapter'ı. structlog tabanlıdır.
    structlog yoksa logging fallback kullanır.
    """

    def __init__(self, name: str | None = None) -> None: ...

    @classmethod
    def configure(
        cls,
        level: str = "INFO",
        json_format: bool = True,
        **kwargs: Any,
    ) -> None:
        """Global structlog yapılandırması. Uygulama başlangıcında bir kez çağrılır."""

    def bind(self, **kwargs: Any) -> StructLogAdapter:
        """Context bağlar (structlog gerektirir)."""

    def debug(self, event: str, **kwargs: Any) -> None: ...
    def info(self, event: str, **kwargs: Any) -> None: ...
    def warning(self, event: str, **kwargs: Any) -> None: ...
    def error(self, event: str, **kwargs: Any) -> None: ...
    def critical(self, event: str, **kwargs: Any) -> None: ...
```

---

## Tüm Public API Sınıfları

| Sınıf | Modül | Açıklama |
|:------|:------|:---------|
| `ValidationEngine` | `stateguard.core.engine` | Cascade pipeline motoru |
| `ValidationResult` | `stateguard.models.result` | Validasyon sonucu |
| `EngineResult` | `stateguard.models.result` | Engine nihai sonucu |
| `BaseValidator` | `stateguard.plugin.base` | Validator abstract class |
| `PluginRegistry` | `stateguard.plugin.registry` | Plugin kayıt/keşif |
| `DecisionEntry` | `stateguard.models.log` | Karar log kaydı |
| `DecisionLogger` | `stateguard.models.log` | Karar log yöneticisi |
| `ConfigManager` | `stateguard.config.settings` | Config yükleyici |
| `StateGuardConfig` | `stateguard.config.schema` | Pydantic config modeli |
| `ValidationDimension` | `stateguard.models.enums` | Boyut enum'u |
| `ValidationTier` | `stateguard.models.enums` | Tier enum'u |
| `StructLogAdapter` | `stateguard.utils.logging` | Yapılandırılmış loglama |
| `JsonSchemaValidator` | `stateguard.plugin.examples.json_schema` | JSON Schema validator |
| `KeywordValidator` | `stateguard.plugin.examples.keyword` | Keyword validator |
| `LengthValidator` | `stateguard.plugin.examples.length` | Uzunluk validator |
