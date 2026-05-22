# Story 1-4: Snapshot/State Robustness Fix

**Epic:** 1 — StateGuard Productization & Hardening
**Story ID:** 1-4
**Story Key:** 1-4-snapshot-state-robustness
**Status:** review
**Başlangıç:** 2026-05-22
**Tamamlanma:** 2026-05-22
**Tahmini Süre:** 0.5 gün
**Gerçek Süre:** ~5 dk
**Öncelik:** ORTA

---

## Kullanıcı Hikayesi

**Bir** StateGuard kullanıcısı olarak, **istiyorum ki** snapshot alırken JSON-serializable olmayan dict key'leri dostane bir hata mesajıyla karşılansın, **böylece** beklenmedik `TypeError` crash'leri yerine anlamlı hatalar alayım.

---

## Mevcut Durum (W1)

`EDGE_CASE_ANALYSIS_SNAPSHOT.md` W1 bulgusu:

`snapshot.py:53-58` — `take_snapshot()` sadece `(ValueError, RecursionError)` yakalar:

```python
try:
    raw_size = len(json.dumps(data))
except (ValueError, RecursionError) as exc:
    raise ValueError(
        f"state_data cannot be serialized: {exc}"
    ) from exc
```

`json.dumps(data, default=str)` — `default=str` non-serializable **values** için çalışır, ama non-serializable **keys** için `TypeError` fırlatır (tuple, frozenset, custom object, complex number).

```python
m.take_snapshot({(1,2): "value"})  # → unhandled TypeError ✗
```

Bu hata, line 38-41'deki "not a dict" input validation hatasıyla **aynı exception tipi** olduğu için ayırt edilemez.

Ayrıca W2 ve W3 bulguları da bu story kapsamında ele alınacak:
- **W2:** TestDuplicate — `TestTakeSnapshot` ve `TestEdgeCases` arasında 4 testin tekrarı
- **W3:** Unused fixture `manager_with_snapshot`

---

## Kabul Kriterleri

### AC1: `TypeError` Yakalanacak
- [ ] `except (ValueError, RecursionError) as exc:` → `except (ValueError, RecursionError, TypeError) as exc:`
- [ ] Hata mesajı: `"state_data cannot be serialized: {exc}"` (mevcut formatta kalır)
- [ ] Test: `test_snapshot_tuple_key_raises_value_error()` — tuple key ile TypeError → ValueError dönüşümü
- [ ] Test: `test_snapshot_frozenset_key_raises_value_error()` — frozenset key ile
- [ ] Test: `test_snapshot_custom_object_key_raises_value_error()` — custom object key ile

### AC2: Test Duplication Temizlenecek (W2)
- [ ] `TestEdgeCases`'den şu 4 test kaldırılacak (zaten `TestTakeSnapshot`'te var):
  - `test_empty_state_data` (line 314)
  - `test_none_state_data` (line 319)
  - `test_string_state_data` (line 323)
  - `test_list_state_data` (line 327)
- [ ] Alternatif: eğer bu testler farklı bir fixture/scenario test ediyorsa, her birine unique assertion eklenip yorum satırı ile çapraz referans verilecek
- [ ] Test: duplicate temizliği sonrası test sayısı 45'ten 41'e düşecek (W1 eklemeleriyle tekrar artabilir)

### AC3: Unused Fixture Kaldırılacak (W3)
- [ ] `manager_with_snapshot` fixture (test_snapshot.py:37-41) kaldırılacak
- [ ] Eğer fixture başka testlerde referans ediliyorsa (import edilmişse) kontrol edilecek
- [ ] Test: fixture kaldırıldıktan sonra tüm testler geçiyor

### AC4: Mevcut Testler Regression
- [ ] Tüm `tests/test_state/test_snapshot.py` testleri (geçerli 45) geçmeye devam etmeli
- [ ] W1 eklemeleri sonrası test sayısı en az 3 artmalı

---

## Değiştirilecek Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `stateguard/state/snapshot.py` | `except` clause'a `TypeError` ekle |
| `tests/test_state/test_snapshot.py` | Yeni testler + duplication cleanup + unused fixture removal |

---

## Dikkat Edilmesi Gerekenler

1. **Hata ayırt edilebilirliği:** `TypeError`'dan dönüştürülen `ValueError`'un mesajı, "not a dict" TypeError'ından farklı olmalı. JSON serialize hatası için: `"state_data cannot be serialized: ..."` (mevcut). "not a dict" hatası için: `"state_data must be a dict, got ..."` (mevcut).
2. **`default=str` parametresi** korunmalı — non-JSON-serializable **values** için hala çalışır.
3. **W2 temizliği** dikkatli yapılmalı — silinen testlerin başka testler tarafından referans edilmediğinden emin olunmalı.
