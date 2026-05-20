# Plugin Geliştirme Kılavuzu

StateGuard, Python class tabanlı bir plugin sistemi sunar. Kendi validator'larınızı yazarak validasyon motorunu her tür çıktı için genişletebilirsiniz.

## Temel Validator Arayüzü

Tüm validator'lar `BaseValidator` abstract class'ından türetilmelidir:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    """Tek bir validator'ın çıktısı."""
    passed: bool
    score: float          # 0.0 - 1.0 arası
    confidence: float     # 0.0 - 1.0 arası
    details: str          # Açıklama
    errors: list[str]     # Hata listesi


class BaseValidator(ABC):
    """Tüm validator'ların temel sınıfı."""

    @abstractmethod
    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        """Çıktıyı doğrula.
        
        Args:
            output: Doğrulanacak LLM çıktısı
            context: Opsiyonel bağlam bilgisi
            
        Returns:
            ValidationResult: Validasyon sonucu
        """
        ...

    def configure(self, config: dict) -> None:
        """Validator konfigürasyonu.
        
        Args:
            config: Validator'a özel konfigürasyon parametreleri
        """
        pass
```

## Örnek Validator'lar

### JSON Şema Validator

```python
import json
import jsonschema
from stateguard.plugin import BaseValidator, ValidationResult


class JsonSchemaValidator(BaseValidator):
    """Çıktının JSON şemasına uygunluğunu doğrular."""

    def __init__(self):
        self.schema = {}

    def configure(self, config: dict) -> None:
        self.schema = config.get("schema", {})

    def validate(self, output: str, context: dict | None = None) -> ValidationResult:
        try:
            data = json.loads(output)
            jsonschema.validate(data, self.schema)
            return ValidationResult(
                passed=True,
                score=1.0,
                confidence=1.0,
                details="JSON şema validasyonu geçti",
                errors=[]
            )
        except json.JSONDecodeError as e:
            return ValidationResult(
                passed=False,
                score=0.0,
                confidence=1.0,
                details=f"Geçersiz JSON: {e}",
                errors=[str(e)]
            )
        except jsonschema.ValidationError as e:
            return ValidationResult(
                passed=False,
                score=0.3,
                confidence=0.9,
                details=f"JSON şema uyuşmazlığı: {e.message}",
                errors=[e.message]
            )
```

### Anahtar Kelime Validator

```python
from stateguard.plugin import BaseValidator, ValidationResult


class KeywordValidator(BaseValidator):
    """Çıktıda olması/olmaması gereken anahtar kelimeleri kontrol eder."""

    def __init__(self):
        self.required_keywords = []
        self.forbidden_keywords = []

    def configure(self, config: dict) -> None:
        self.required_keywords = config.get("required", [])
        self.forbidden_keywords = config.get("forbidden", [])

    def validate(self, output: str, context: dict | None = None) -> ValidationResult:
        errors = []
        output_lower = output.lower()

        # Zorunlu kelimeler
        for kw in self.required_keywords:
            if kw.lower() not in output_lower:
                errors.append(f"Zorunlu kelime eksik: '{kw}'")

        # Yasaklı kelimeler
        for kw in self.forbidden_keywords:
            if kw.lower() in output_lower:
                errors.append(f"Yasaklı kelime bulundu: '{kw}'")

        passed = len(errors) == 0
        score = 1.0 - (len(errors) / max(len(self.required_keywords) + len(self.forbidden_keywords), 1))
        
        return ValidationResult(
            passed=passed,
            score=max(0.0, score),
            confidence=0.95,
            details=f"{len(errors)} kural ihlali bulundu" if errors else "Tüm kelime kontrolleri geçti",
            errors=errors
        )
```

## Plugin Kaydı

Validator'ları StateGuard'a kaydetmek için `PluginRegistry` kullanılır:

```python
from stateguard.plugin import PluginRegistry

# Validator'ları kaydet
PluginRegistry.register("json-schema", JsonSchemaValidator)
PluginRegistry.register("keyword", KeywordValidator)

# Validator'ları kullan
registry = PluginRegistry()
validator = registry.get("json-schema")
validator.configure({"schema": my_schema})
result = validator.validate(output_data)
```

## Konfigürasyon

Validator'lar YAML dosyası ile de yapılandırılabilir:

```yaml
# validators.yaml
validators:
  - name: json-schema
    enabled: true
    config:
      schema:
        type: object
        properties:
          action:
            type: string
          parameters:
            type: object

  - name: keyword
    enabled: true
    config:
      required: ["teşekkür", "yardım"]
      forbidden: ["şifre", "kredi kartı"]
```

## İleri Düzey: Custom Validator İpuçları

1. **Performans**: Validator'lar thread-safe olmalıdır — StateGuard `ThreadPoolExecutor` ile paralel çalıştırabilir
2. **State Tutma**: Validator'lar stateful olabilir (`configure()` ile başlatılır)
3. **Hata Yönetimi**: Beklenmeyen hatalarda `ValidationResult` döndürün, exception fırlatmayın
4. **Loglama**: structlog ile validator içinden loglama yapabilirsiniz
5. **Test**: Her validator için unit test yazın, StateGuard'ın test fixture'larını kullanın

```python
# Test örneği
def test_json_schema_validator():
    validator = JsonSchemaValidator()
    validator.configure({
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
    })
    
    result = validator.validate('{"name": "StateGuard"}')
    assert result.passed
    assert result.score == 1.0
```
