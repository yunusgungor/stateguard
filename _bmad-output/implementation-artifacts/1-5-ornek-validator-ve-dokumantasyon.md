# Story 1-5: Örnek Validator ve Dokümantasyon Tamamlama

**Epic:** 1 — StateGuard Productization & Hardening
**Story ID:** 1-5
**Story Key:** 1-5-ornek-validator-ve-dokumantasyon
**Status:** ready-for-dev
**Başlangıç:** 2026-05-22
**Tahmini Süre:** 1 gün
**Öncelik:** ORTA

---

## Kullanıcı Hikayesi

**Bir** StateGuard'a yeni başlayan geliştirici olarak, **istiyorum ki** çalışan, gerçekçi örnek plugin validator'ları görebileyim ve dokümantasyonda plugin geliştirme adımlarını takip edebileyim, **böylece** kendi validator'ımı hızlıca geliştirip projeme entegre edebileyim.

---

## Mevcut Durum (I2)

`EDGE_CASE_ANALYSIS.md` I2 bulgusu:

`stateguard/plugin/examples/` içindeki validator'ların `validate()` metodu `...` (ellipsis) içerir:

```python
# json_schema.py, keyword.py, length.py
class SomeValidator(BaseValidator):
    ...
    def validate(self, output: Any, context: dict | None = None) -> ValidationResult:
        ...  # ← NotImplementedError fırlatır
```

Bu:
- `__init_subclass__` ve ABC kontrollerini geçer
- Ancak çağrıldığında `NotImplementedError` fırlatır
- Testler sadece instantiation'ı kontrol eder, validation logic test edilmez
- Yeni başlayanlar için "boş şablon" izlenimi verir

**Not:** `keyword.py` ve `length.py` zaten dolu implementasyon içeriyor — sadece `json_schema.py`'deki `validate` ellipsis kalmış olabilir. Kod incelemesi yapılacak.

---

## Kabul Kriterleri

### AC1: Örnek Validator'lar Doldurulacak
- [ ] `stateguard/plugin/examples/length.py` — `validate()` dolu ✅ (mevcut)
- [ ] `stateguard/plugin/examples/keyword.py` — `validate()` dolu ✅ (mevcut)
- [ ] `stateguard/plugin/examples/json_schema.py` — `validate()` dolu ✅ (mevcut)
- [ ] Eğer herhangi birinde `...` kalmışsa, **gerçekçi bir implementasyon** ile doldurulacak
- [ ] Yeni bir örnek eklenecek: `regex.py` — regex pattern matching validator
  - `name = "regex"`, `dimension = STRUCTURAL`, `tier = TIER_1`
  - Context'ten `pattern` alır, output'un pattern'e uyup uymadığını kontrol eder
  - Test: `tests/test_plugin/test_examples.py`'de test eklenecek

### AC2: Örnek Validator Test'leri Eklenecek
- [ ] Her örnek validator için en az 1 validation testi
- [ ] `test_length_validator_checks_length()` — length.doğrulama
- [ ] `test_keyword_validator_checks_keywords()` — keyword.doğrulama
- [ ] `test_json_schema_validator_checks_schema()` — json_schema.doğrulama (jsonschema kütüphanesi yoksa skip)
- [ ] `test_regex_validator_checks_pattern()` — regex.doğrulama
- [ ] Bu testler `tests/test_plugin/test_examples.py`'ye eklenecek

### AC3: Plugin Geliştirme Dokümantasyonu Güncellenecek
- [ ] `docs/plugin-dev-guide.md` — adım adım plugin geliştirme rehberi
  - BaseValidator'dan kalıtım
  - Gerekli attribute'lar
  - `validate()` implementasyonu
  - `setup()` ve `teardown()` lifecycle hook'ları
  - Plugin registry'e kayıt
  - Test etme
- [ ] `docs/architecture.md` — plugin sistemi mimarisi güncellenecek
  - Validator tipleri (Built-in vs Plugin)
  - Tier'ler ve Cascade mantığı
  - Inheritance kuralları
- [ ] Mevcut örnek validator'lara dokümantasyon içinde referans verilecek

### AC4: README Güncellenecek
- [ ] Plugin geliştirme bölümü eklenecek (kısa, "hızlı başlangıç")
- [ ] Örnek kod bloğu ile plugin oluşturma adımları

---

## Değiştirilecek Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `stateguard/plugin/examples/json_schema.py` | Varsa `...` kontrol et ve doldur |
| `stateguard/plugin/examples/regex.py` | **YENİ** — regex validator örneği |
| `tests/test_plugin/test_examples.py` | Yeni validation testleri |
| `docs/plugin-dev-guide.md` | Plugin geliştirme rehberi güncellemesi |
| `docs/architecture.md` | Plugin sistemi mimari güncellemesi |
| `README.md` | Plugin geliştirme hızlı başlangıç bölümü |

---

## Dikkat Edilmesi Gerekenler

1. **Mevcut örnekler:** `length.py`, `keyword.py` ve `json_schema.py` **zaten dolu implementasyon içeriyor** — kod incelemesi yapıldı ve doğrulandı. Sadece `json_schema.py`'nin ellipsis ihtimali kontrol edilecek.
2. **RegexValidator:** Yeni eklenecek. `re` modülü standart kütüphanede olduğu için ek bağımlılık gerekmez.
3. **Testlerde `jsonschema` import'u:** `test_json_schema_validator_checks_schema()`'da `jsonschema` import'u try/except ile korunmalı, yoksa `pytest.skip()` yapılmalı.
4. **Dokümantasyon dili:** Proje standardına uygun olarak İngilizce.
