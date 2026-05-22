# Story 1-2: Plugin Sistemi Sağlamlaştırma

**Epic:** 1 — StateGuard Productization & Hardening
**Story ID:** 1-2
**Story Key:** 1-2-plugin-sistemi-saglamlastirma
**Status:** review
**Başlangıç:** 2026-05-22
**Tamamlanma:** 2026-05-22
**Tahmini Süre:** 1 gün
**Gerçek Süre:** ~15 dk
**Öncelik:** YÜKSEK

---

## Kullanıcı Hikayesi

**Bir** StateGuard plugin geliştiricisi olarak, **istiyorum ki** ara katman (intermediate) abstract sınıflar oluşturabileyim ve `validate` attribute'unu yanlışlıkla non-callable olarak atadığımda hata alayım, **böylece** plugin ekosistemini versionlama ve soyutlama desenleriyle genişletebileyim ve runtime crash'leri önceden yakalayabileyim.

---

## Mevcut Durum (W1 + W2)

### W1 — Deep Inheritance Chain Broken

`BaseValidator.__init_subclass__` (base.py:72) `cls.__dict__` kontrolü yapar. Bu, **yalnızca** doğrudan subclass'ın kendi namespace'inde tanımlı attribute'ları görür:

```python
class Mid(BaseValidator):
    name = "mid"
    dimension = ...
    tier = ...
    def validate(self, ...): ...

class Child(Mid):
    pass  # Child.__dict__'te name/dimension/tier YOK → TypeError ✗
```

**Risk:** 2+ seviyeli inheritance chain imkansız. Plugin versionlama (`MyPluginV1` → `MyPluginV2`) engellenir.

### W2 — `validate` Callable Enforced Edilmiyor

```python
class BrokenValidator(BaseValidator):
    name = "broken"
    dimension = ValidationDimension.STRUCTURAL
    tier = ValidationTier.TIER_1
    validate = 42  # cls.__dict__'te var ✓, ABC'ye göre abstract değil ✓

v = BrokenValidator()   # Instantiate OLUR
v.validate("test")      # TypeError: 'int' object is not callable ✗
```

**Risk:** Hata runtime'da ortaya çıkar. Sessiz kalır, geliştirici deneyimi kötüdür.

---

## Kabul Kriterleri

### AC1: Deep Inheritance Artık Mümkün
- [ ] `__init_subclass__`'ta `attr not in cls.__dict__` kontrolü, **MRO üzerinden** `getattr` kontrolü ile değiştirilecek
- [ ] Yeni mantık: `attr` ya doğrudan subclass'ta (`cls.__dict__`) tanımlanmış **veya** BaseValidator dışında bir ara sınıftan (MRO'da BaseValidator ile subclass arasında) miras alınmış olmalı
- [ ] BaseValidator'ın default değerleri (`"base-validator"`, `STRUCTURAL`, `TIER_1`) **hala override edilmemişse hata verir**
- [ ] `class Child(Mid): pass` (Mid'de tüm gerekli attr'lar var) çalışır
- [ ] `GrandChild(Child): pass` (3 seviye) de çalışır
- [ ] Test: `test_deep_inheritance_works()` — 3 seviyeli chain
- [ ] Test: `test_deep_inheritance_still_requires_attrs()` — ara sınıfta attr yoksa hala hata

### AC2: `validate` Callable Olarak Kontrol Ediliyor
- [ ] `__init_subclass__` içinde `validate`'in `callable()` olduğu kontrol edilecek
- [ ] Eğer `validate` callable değilse `TypeError` fırlatılacak
- [ ] Hata mesajı: `"{cls.__name__}.validate must be a callable method, got {type(cls.validate).__name__}"`
- [ ] Bu kontrol sadece subclass kendi `validate`'ini override ettiğinde yapılacak (abstract intermediate sınıflarda atlanabilir)
- [ ] Test: `test_validate_must_be_callable()` — `validate = 42` hatası
- [ ] Test: `test_validate_callable_passes()` — normal validatör geçer
- [ ] Test: `test_non_callable_validate_abstract_intermediate_skips()` — abstract intermediate sınıflarda kontrol atlanır

### AC3: Abstract Intermediate Sınıf Desteği
- [ ] `ABC in cls.__bases__` veya `hasattr(cls, "__abstractmethods__")` durumunda attribute kontrolleri atlanır (mevcut davranış korunur)
- [ ] Ancak callable kontrolü abstract sınıflarda da **yapılabilir** (abstract method olarak bildirilmiş validate için sorun olmaz)
- [ ] Test: `test_abstract_intermediate_skips_attr_check()` — abstract sınıf attr tanımlamadan geçer

---

## Değiştirilecek Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `stateguard/plugin/base.py` | `__init_subclass__` mantığı yeniden yazılacak, callable kontrolü eklenecek |
| `tests/test_plugin/test_base.py` | Yeni testler eklenecek |

---

## Dikkat Edilmesi Gerekenler

1. **MRO mantığı:** `getattr(cls, attr, _sentinel)` ile attr'ın BaseValidator dışından gelip gelmediği kontrol edilmeli. `getattr(BaseValidator, attr)` ile karşılaştırma yapılabilir.
2. **Mevcut testler regression:** Tüm `tests/test_plugin/test_base.py`'daki 13 testin tamamı geçmeye devam etmeli.
3. **Threshold doğru seçilmeli:** Non-callable kontrolü çok agresif olmamalı — `@staticmethod`, `@classmethod`, lambdalar da callable'dır ve geçmelidir.
4. **Hata mesajları İngilizce** olmalı, proje standardına uygun.

---

## Teknik Tasarım

### Yeni `__init_subclass__` Mantığı

```python
def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)

    # Abstract intermediate sınıfları atla
    if ABC in cls.__bases__ or hasattr(cls, "__abstractmethods__"):
        return

    # Required attr'ları kontrol et: ya cls.__dict__'te tanımlanmış
    # ya da BaseValidator dışındaki bir ara sınıftan gelmiş olmalı
    missing = []
    for attr in ("name", "dimension", "tier"):
        if attr in cls.__dict__:
            continue  # doğrudan tanımlanmış ✓
        # MRO'da BaseValidator dışında bir yerde tanımlanmış mı?
        if hasattr(cls, attr) and not hasattr(BaseValidator, attr):
            continue  # ara sınıftan miras ✓
        # Ayrıca: BaseValidator'ın default'u mu yoksa ara sınıf override mı?
        base_default = getattr(BaseValidator, attr, _SENTINEL)
        current_value = getattr(cls, attr, _SENTINEL)
        if current_value is not _SENTINEL and current_value is not base_default:
            continue  # ara sınıf override etmiş ✓
        missing.append(attr)

    if missing:
        raise TypeError(...)

    # validate callable mı kontrol et
    validate_attr = cls.__dict__.get("validate")
    if validate_attr is not None and not callable(validate_attr):
        raise TypeError(
            f"{cls.__name__}.validate must be a callable method, "
            f"got {type(validate_attr).__name__}"
        )

    # dimension/tier type kontrolü (mevcut)
    ...
```
