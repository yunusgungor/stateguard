# Story 1-1: Tier-2 ML Ensemble Eğitim/Kalibrasyon Altyapısı

**Epic:** 1 — StateGuard Productization & Hardening
**Story ID:** 1-1
**Story Key:** 1-1-tier2-ml-ensemble-egitim-kalibrasyon
**Status:** review
**Başlangıç:** 2026-05-22
**Tamamlanma:** 2026-05-22
**Tahmini Süre:** 2-3 gün
**Gerçek Süre:** ~30 dk
**Öncelik:** YÜKSEK

---

## Kullanıcı Hikayesi

**Bir** StateGuard geliştiricisi olarak, **istiyorum ki** EnsembleValidator'ı referans/normal veri ile önceden eğitebileyim ve eğitilmiş modeli persist edip tekrar yükleyebileyim, **böylece** her `validate()` çağrısında validation data'sına fit edilmesin ve data leakage olmadan gerçek anomali tespiti yapabileyim.

---

## Mevcut Durum (W5)

`EDGE_CASE_ANALYSIS.md` W5 bulgusu:

`EnsembleValidator.validate()` (tier2.py:316) her çağrıda `analyzer.fit(features)` yapıyor. Bu:

1. **Data leakage:** Model değerlendirdiği veri üzerinde eğitiliyor
2. **Stateless:** Her çağrıda sıfırdan eğitim → önceki çağrılardan öğrenme yok
3. **Yanıltıcı sonuçlar:** IsolationForest contamination oranına göre anomaly bulur
4. **Z-Score Analyzer:** `std_` sıfır olduğunda guard var ama anlamlı sonuç üretmez

---

## Kabul Kriterleri

### AC1: EnsembleValidator `fit()` metodu
- [ ] `fit(reference_data: np.ndarray | list[list[float]] | dict)` metodu eklenecek
- [ ] Tüm alt analyzeler (IsolationForest, SVM, Z-Score) referans veri ile eğitilecek
- [ ] `fit()` birden çok çağrılabilir — sonraki çağrılar önceki eğitimi **günceller** (incremental fit)
- [ ] `fit()` öncesinde `_ensure_analyzers()` lazily initialize edilecek
- [ ] Eğitim durumu `_is_fitted: bool` flag'i ile izlenecek

### AC2: EnsembleValidator `validate()` artık fit ETMEYECEK
- [ ] `validate()` içindeki `analyzer.fit(features)` çağrısı kaldırılacak
- [ ] Eğer `_is_fitted == False` ise `validate()` öncesinde otomatik fit yapılacak (geriye uyumluluk)
- [ ] Ancak bu otomatik fit durumunda bir `warnings.warn()` ile uyarı basılacak
- [ ] Otomatik fit durumu `details` içinde `"auto_fitted": True` olarak raporlanacak

### AC3: Model Persistence (save/load)
- [ ] `save(path: str | Path)` metodu eklenecek
- [ ] `load(path: str | Path, ...)` classmethod'u eklenecek
- [ ] Persistence formatı: `pickle` veya `joblib` (scikit-learn uyumluluğu için)
- [ ] Kaydedilen dosya: `(analyzers state, config params, fitted flag)`
- [ ] `save()`/`load()` thread-safe olacak (temp file + atomic rename)
- [ ] Yüklenen modelin versiyonu mevcut kod versiyonuyla uyumsuzsa `ValueError`

### AC4: BaseValidator `setup()` hook'u EnsembleValidator'da implemente edilecek
- [ ] `setup()` içinde referans veri yolu config'den okunacak
- [ ] Config'de `tier2_reference_data_path` veya `tier2_model_path` varsa otomatik yüklenecek
- [ ] `setup()` `ConfigManager` üzerinden yapılandırılabilir olacak

### AC5: Test Coverage
- [ ] `test_fit_with_reference_data()` — normal veri ile eğitim doğrulama
- [ ] `test_validate_after_fit_predicts_only()` — fit sonrası validate'de fit çağrılmadığını doğrulama (mock ile)
- [ ] `test_validate_without_fit_warns()` — fit edilmemiş modelin auto-fit yapması ve uyarı basması
- [ ] `test_save_and_load_model()` — persistence çalışması
- [ ] `test_save_and_load_restores_state()` — yüklenen modelin önceki fit durumunu koruması
- [ ] `test_load_incompatible_version_raises()` — uyumsuz versiyon hatası
- [ ] `test_incremental_fit_updates_model()` — çoklu fit çağrısının güncelleme yapması
- [ ] Tüm yeni testler `tests/test_core/test_tier2.py` içine eklenecek

---

## Değiştirilecek Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `stateguard/core/tier2.py` | Fit/predict ayrıştırması, save/load, `setup()` implementasyonu |
| `stateguard/config/settings.py` | Yeni config alanları: `tier2_reference_data_path`, `tier2_model_path` |
| `stateguard/config/schema.py` | Config schema güncellemesi (opsiyonel alanlar) |
| `tests/test_core/test_tier2.py` | Yeni testler |
| `tests/test_config/conftest.py` | Yeni config alanları için fixture (opsiyonel) |

---

## Dikkat Edilmesi Gerekenler

1. **Geriye uyumluluk:** Mevcut kullanıcılar hiçbir değişiklik yapmadan EnsembleValidator'ı kullanmaya devam edebilmeli. Auto-fit + warning ile eski davranış korunmalı.
2. **Pickle güvenliği:** `load()` için `yaml.Loader` benzeri güvenlik uyarısı eklenmeli.
3. **Joblib bağımlılığı:** sklearn ile gelen `joblib` kullanılmalı, ek bağımlılık eklenmemeli.
4. **Z-Score Analyzer:** `fit()` zaten doğru çalışıyor — sadece çağrılma yeri değişecek.
5. **Hata mesajları:** Türkçe değil, İngilizce olmalı (proje standardı).
6. **Mevcut testlerin tamamı geçmeye devam etmeli** — hiçbirinde regression olmamalı.

---

## Teknik Tasarım

### Yeni Akış

```
┌──────────────────────────────────────────┐
│  KULLANICI KODU                          │
│                                          │
│  v = EnsembleValidator()                 │
│  v.fit(reference_features)               │  ← YENİ: referans veri ile eğitim
│  v.save("/path/to/model.joblib")         │  ← YENİ: persist
│                                          │
│  # Başka bir yerde:                      │
│  v2 = EnsembleValidator.load("/path/...")│  ← YENİ: load
│  result = v2.validate(new_output)        │  ← validate'de fit YOK
└──────────────────────────────────────────┘

Geriye uyumluluk akışı (değişmeyen):
  v = EnsembleValidator()
  result = v.validate(output)               # Auto-fit + warning
```

### EnsembleValidator Yapısı

```python
class EnsembleValidator(BaseValidator):
    _is_fitted: bool = False
    _fitted_at: str | None = None        # ISO timestamp
    _fit_version: str | None = None      # stateguard.__version__

    def fit(self, reference_data) -> None:
        """Fit all analyzers on reference/normal data."""
        ...

    def save(self, path: str | Path) -> None:
        """Persist trained model to disk."""
        ...

    @classmethod
    def load(cls, path: str | Path, ...) -> EnsembleValidator:
        """Load a previously trained model."""
        ...

    def setup(self) -> None:
        """Lifecycle hook: load reference data or pre-trained model from config."""
        ...

    def validate(self, output, context=None) -> ValidationResult:
        """Predict only — no fitting."""
        ...
```
