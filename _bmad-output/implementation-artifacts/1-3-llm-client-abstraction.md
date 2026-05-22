# Story 1-3: LLM Client Abstraction İyileştirme

**Epic:** 1 — StateGuard Productization & Hardening
**Story ID:** 1-3
**Story Key:** 1-3-llm-client-abstraction
**Status:** review
**Başlangıç:** 2026-05-22
**Tamamlanma:** 2026-05-22
**Tahmini Süre:** 0.5 gün
**Gerçek Süre:** ~10 dk
**Öncelik:** ORTA

---

## Kullanıcı Hikayesi

**Bir** StateGuard geliştiricisi olarak, **istiyorum ki** farklı LLM sağlayıcıları (OpenAI, Claude, Gemini) kendi prompt formatlarını kullanabilsin, **böylece** `LLMValidator` herhangi bir `LLMClient` implementasyonuyla çalışabilsin ve `HTTPLLMClient._build_prompt` private metoduna sıkı bağımlılık ortadan kalksın.

---

## Mevcut Durum (W3)

`EDGE_CASE_ANALYSIS.md` W3 bulgusu:

`LLMValidator` (tier3.py:71) doğrudan `HTTPLLMClient._build_prompt(output)` çağırıyor:

```python
# tier3.py:71
prompt = HTTPLLMClient._build_prompt(output)
```

Bu:
- `LLMClient` bir **Protocol** olarak tanımlanmış (sadece `ask` zorunlu)
- Ancak `LLMValidator` private (`_` prefix) bir metoda bağımlı
- Özel client implementasyonları `HTTPLLMClient`'ın sabit prompt template'ini kullanmak **zorunda**
- Prompt template'inin özelleştirilmesi için temiz bir API yok

---

## Kabul Kriterleri

### AC1: `LLMClient` Protocol'üne `build_prompt` Eklenecek
- [ ] `build_prompt(output: str) -> str` metodu `LLMClient` protocol'üne eklenecek
- [ ] Bu metod **opsiyonel** olacak (`Protocol`'de varsayılan implementasyon ile)
- [ ] Varsayılan implementasyon: `return f"Evaluate the following output:\n{output}"` (İngilizce, jenerik)
- [ ] Dokümantasyon: `build_prompt` override edilerek özel prompt formatı kullanılabilir

### AC2: `HTTPLLMClient` Kendi `build_prompt`'unu Sağlayacak
- [ ] `HTTPLLMClient.build_prompt(output)` — `_build_prompt`'u çağıran public wrapper
- [ ] Mevcut `_build_prompt` classmethod'u **korunacak** (private kalabilir veya public yapılabilir)
- [ ] `HTTPLLMClient.JUDGE_PROMPT_TEMPLATE` değişkeni değiştirilerek farklı prompt şablonu kullanılabilir

### AC3: `LLMValidator` Artık `HTTPLLMClient._build_prompt` Çağırmayacak
- [ ] `tier3.py:71` — `HTTPLLMClient._build_prompt(output)` → `self._llm_client.build_prompt(output)`
- [ ] Bu değişiklikle herhangi bir `LLMClient` implementasyonu kendi prompt'unu üretebilir
- [ ] Geriye uyumluluk: Mevcut `HTTPLLMClient` kullanıcıları etkilenmez

### AC4: Test Coverage
- [ ] `test_llm_validator_uses_client_build_prompt()` — client'ın `build_prompt`'u çağrıldığını doğrulama
- [ ] `test_custom_client_with_custom_prompt()` — özel client'ın farklı prompt kullanması
- [ ] `test_http_llm_client_public_build_prompt()` — `HTTPLLMClient.build_prompt()` çalışıyor

---

## Değiştirilecek Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `stateguard/core/llm_client.py` | Protocol'e `build_prompt` ekle, `HTTPLLMClient`'a public wrapper |
| `stateguard/core/tier3.py` | `HTTPLLMClient._build_prompt` → `self._llm_client.build_prompt` |
| `tests/test_core/test_tier3.py` | Yeni testler |
| `tests/test_core/test_llm_client.py` | `build_prompt` testleri |

---

## Dikkat Edilmesi Gerekenler

1. **Protocol değişiklikleri:** `Protocol`'e metod eklemek mevcut implementasyonları bozmaz çünkü Python protocol'leri structural subtyping kullanır — bir sınıfın `build_prompt` metodu varsa Protocol'e uyar.
2. **Varsayılan implementasyon:** Protocol'de varsayılan implementasyon için `@staticmethod` veya class-level fonksiyon tanımı kullanılabilir. Ancak Protocol'lerde varsayılan implementasyon için en temiz yol:
   ```python
   class LLMClient(Protocol):
       model: str
       
       def ask(self, prompt: str) -> str: ...
       
       def build_prompt(self, output: str) -> str:
           return f"Evaluate:\n{output}"
   ```
3. **Mevcut testler:** Tüm testlerin geçmeye devam etmesi zorunlu.
4. **`_build_prompt`'u silme:** Mevcut private metodu silme, sadece public wrapper ekle. Eski kullanımı kırmamak için `_build_prompt`'u `build_prompt`'a yönlendir.
