# Edge Case & Boundary Condition Analysis — StateGuard Plugin System

**Review date:** 2026-05-21  
**Scope:** `stateguard/plugin/base.py`, `tests/test_plugin/test_base.py`, `stateguard/core/tier{1,2,3}.py`, `stateguard/plugin/examples/*.py`  
**Test run:** 13/13 passed ✅

---

## 🔴 CRITICAL (0 findings)

Hiçbir kritik sorun tespit edilmedi. Mevcut validatörler yeni `BaseValidator` ile tam uyumlu. Tüm testler geçiyor.

---

## 🟡 WARNING (6 findings)

### W1. Deep inheritance chain broken by `__init_subclass__`

| Aspect | Detail |
|--------|--------|
| **File** | `stateguard/plugin/base.py:72` |
| **Code** | `if attr not in cls.__dict__:` |
| **Risk** | Medium |

`__init_subclass__` checki `cls.__dict__` kullandığı için **yalnızca** doğrudan subclass'ın kendi namespace'inde tanımlanmış attributeları görür. Miras alınan attribute'lar (MRO'dan gelenler) `cls.__dict__`'te bulunmaz.

```python
class Mid(BaseValidator):
    name = "mid"         # Mid.__dict__'te var ✓
    dimension = ...
    tier = ...
    def validate(self, ...): ...

class Child(Mid):
    pass                  # Child.__dict__'te name/dimension/tier YOK → TypeError ✗
```

Bu, **2+ seviyeli inheritance chain'i imkansız kılar**. Kullanıcılar her alt sınıfta metadata'yı tekrar bildirmek zorunda. Özellikle abstract intermediate class'lar veya plugin versionlama (ör. `MyPluginV1` → `MyPluginV2`) desenlerini engeller.

**Öneri:** `__init_subclass__`'te `cls.__dict__` yerine, required attr'ların `cls` üzerinde `getattr` ile erişilebilir olup olmadığını ve `cls`'in kendisinin `BaseValidator` olmadığını kontrol eden bir mekanizma düşünülebilir. Veya mevcut davranış bilinçli bir tercih olarak dokümante edilmeli.

---

### W2. `validate` attribute'u callable olarak enforce edilmiyor

| Aspect | Detail |
|--------|--------|
| **File** | `stateguard/plugin/base.py` (tüm dosya) |
| **Risk** | Medium |

İki mekanizma da `validate`'in callable olduğunu garanti etmez:

1. **`__init_subclass__`**: Sadece `name`, `dimension`, `tier` varlığını kontrol eder.
2. **ABC `@abstractmethod`**: Sadece `validate`'in `__isabstractmethod__ = True` olmadığını kontrol eder.

```python
class BrokenValidator(BaseValidator):
    name = "broken"
    dimension = ValidationDimension.STRUCTURAL
    tier = ValidationTier.TIER_1
    validate = 42        # cls.__dict__'te var ✓, ABC için abstract değil ✓

v = BrokenValidator()    # Instantiate OLUR
v.validate("test")       # TypeError: 'int' object is not callable ✗
```

**Risk:** Hata sadece runtime'da ortaya çıkar. Kasti veya dikkatsiz kullanımda sessiz kalır.

**Öneri:** `__init_subclass__` içinde `validate`'in callable olup olmadığı da kontrol edilebilir:
```python
if callable(getattr(cls, 'validate', None)) is False:
    raise TypeError(...)
```

---

### W3. `LLMValidator`, `HTTPLLMClient._build_prompt` private method'una sıkı bağımlı

| Aspect | Detail |
|--------|--------|
| **File** | `stateguard/core/tier3.py:71` |
| **Code** | `prompt = HTTPLLMClient._build_prompt(output)` |
| **Risk** | Medium |

`LLMClient` bir **Protocol** olarak tanımlanmış (sadece `ask` zorunlu), ancak `LLMValidator` doğrudan `HTTPLLMClient._build_prompt`'u çağırıyor. Bu şu anlama gelir:

- Özel bir `LLMClient` implementasyonu (ör. `GeminiClient`, `ClaudeClient`) `HTTPLLMClient`'ın sabit prompt template'ini kullanır.
- Alternatif client'lar kendi prompt formatlarını kullanamaz.
- `_build_prompt` private (`_` prefix) olduğu için API sözleşmesinin parçası değildir.

**Öneri:** Prompt template'ini `BaseValidator` seviyesine taşımak veya `LLMClient` protocol'üne `build_prompt` metodunu eklemek.

---

### W4. `EmbeddingValidator.__init__` conditional logic misleading

| Aspect | Detail |
|--------|--------|
| **File** | `stateguard/core/tier1.py:58-67` |
| **Code** | `if model_name is not None and device is not None and threshold is not None:` |
| **Risk** | Low |

Koşul **ALL-OR-NOTHING** gibi görünür, ancak `else` branch'i partial override'ları doğru işler:

```python
# ALL three required gibi görünüyor...
if model_name is not None and device is not None and threshold is not None:
    ...
else:
    # ...ama aslında partial override çalışır
    self._model_name = model_name or config.default_embedding_model  # model_name kullanılır
```

`model_name="custom"` tek başına verilse bile çalışır (doğru davranış), ancak kod okuyucusunda "hepsi veya hiçbiri" izlenimi yaratır.

**Öneri:** Koşulu kaldırıp her assignment'ı bağımsız yapmak daha net olur:
```python
cfg = ConfigManager()
config = cfg.load()
self._model_name = model_name or config.default_embedding_model
self._device = device or config.embedding_device
self._threshold = threshold or config.tier1_threshold
```

---

### W5. `EnsembleValidator` fits anomaly detectors on validation data (not training data)

| Aspect | Detail |
|--------|--------|
| **File** | `stateguard/core/tier2.py:316` |
| **Code** | `analyzer.fit(features)` (her `validate()` çağrısında) |
| **Risk** | Medium |

Anomaly detection modelleri her `validate()` çağrısında **validation data'sına fit edilir**. Bu birkaç sorun yaratır:

1. **Data leakage:** Model, değerlendirdiği veri üzerinde eğitilir → anomalileri tespit etme yeteneği kaybolur.
2. **Stateless:** Her çağrıda sıfırdan eğitilir → önceki çağrılardan öğrenme yok.
3. **Yanıltıcı sonuçlar:** `IsolationForest` contamination oranına göre anomaly bulur. Aynı veriye fit edilip aynı veriden tahmin yapıldığında, model "en uzaktaki" noktaları anomaly olarak etiketler — gerçek anomaly değil.
4. **Z-Score Analyzer:** `std_`'nin sıfır olduğu durumlarda `self.std_[self.std_ == 0] = 1.0` ile guard var (line 49) — iyi, ancak constant-feature durumunda anlamlı sonuç üretmez.

**Öneri:** Modellerin `setup()` hook'unda referans/normal veri ile eğitilmesi, `validate()`'te sadece predict yapılması gerekir. Bu, mevcut dizaynda bir feature eksikliğidir.

---

### W6. `EmbeddingValidator.validate()` LSP violation (type narrowing)

| Aspect | Detail |
|--------|--------|
| **File** | `stateguard/core/tier1.py:81` |
| **Code** | `output: str` vs base'de `output: Any` |
| **Risk** | Low |

Base class `output: Any` tanımlarken, `EmbeddingValidator` `output: str` ile daraltır. Python type checker'ları bunu LSP ihlali olarak işaretler (parametre contravariant olmalı, narrowing covariance'dır).

Çalışma zamanında sorun yoktur çünkü Python dynamic typing kullanır. Ancak mypy/pyright gibi araçlar uyarı verebilir.

---

## 🔵 INFO (7 findings)

### I1. Test izolasyonu yeterli — ancak fixture kullanımı yok

`tests/test_plugin/test_base.py`'de her test method'u `GoodValidator()` instance'ını fresh oluşturur. Shared state yok. Ancak `conftest.py`'deki `mock_embedding_model` gibi fixtur'lar kullanılmamış.

### I2. Örnek plugin'ler skeleton (ellipsis implementasyon)

`json_schema.py`, `keyword.py`, `length.py` validatörleri `validate()` metodunda `...` (ellipsis) içerir. `__init_subclass__` ve ABC kontrollerini geçer, ancak çağrıldığında `NotImplementedError` fırlatır. Bu, testlerin yalnızca instantiation'ı kontrol ettiği anlamına gelir — validation logic test edilmez.

### I3. `__init_subclass__` kwargs forwarding test edilmemiş

`__init_subclass__(cls, **kwargs: Any)` — `**kwargs` `super().__init_subclass__(**kwargs)` ile forward edilir. Ancak hiçbir test `class X(BaseValidator, metaclass=...):` gibi kwargs gerektiren kalıtım senaryolarını test etmez.

### I4. Multiple inheritance / diamond inheritance test edilmemiş

`DiamondValidator(BaseValidator, SomeMixin)` şeklinde bir senaryo testlerde yok. Çalıştığı doğrulandı (manuel testte), ancak test coverage'ı yok.

### I5. `output` için edge case testleri eksik

BaseValidator seviyesinde `output: Any` için şu durumlar test edilmemiş:
- `output=None`
- `output=bytes`
- `output=complex_object` (Pydantic model, dataclass)
- `output=empty_string`

Her validatör bunları kendi içinde handle eder, ancak base seviyesinde bir contract testi yok.

### I6. `EnsembleValidator` context parametresini kullanmıyor

`EnsembleValidator.validate()`'un `context` parametresi almasına rağmen kullanılmaz. Aynı durum `LLMValidator` için de geçerli. Bu, base interface'in bir parçası olarak eklenmiş ancak implementasyonlarda boş geçilmiştir.

### I7. `ValidationResult.error` alanı bazı validatörlerde `None` bazılarında string

| Validator | Hata durumunda `error` | Başarı durumunda `error` |
|-----------|----------------------|------------------------|
| EmbeddingValidator | `str` (açıklayıcı) | `None` |
| EnsembleValidator | `str` (açıklayıcı) | `None` |
| LLMValidator | `str` (açıklayıcı) | `None` |

Tutarlı görünüyor. ✅

---

## ✅ Summary

| Severity | Count |
|----------|-------|
| 🔴 CRITICAL | 0 |
| 🟡 WARNING | 6 |
| 🔵 INFO | 7 |
| **Total** | **13** |

### Key Findings

1. **En büyük risk:** Deep inheritance chain'in `__init_subclass__` tarafından engellenmesi (W1). Kod genişledikçe bu kısıtlama problem yaratabilir.
2. **En sessiz hata:** `validate`'in callable olarak enforce edilmemesi (W2). Kullanıcı hatası durumunda runtime crash kaçınılmaz.
3. **En büyük tasarım sorunu:** `EnsembleValidator`'ın validation data'sına fit etmesi (W5). Anomaly detection'un temel prensibine aykırı.
4. **Küçük ama rahatsız edici:** `EmbeddingValidator.__init__`'teki misleading conditional (W4) ve `LLMValidator`'ın private method'a bağımlılığı (W3).
5. **Mevcut validatörler:** Tümü BaseValidator ile çalışır, hiçbir regression yok ✅.
