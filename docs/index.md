# StateGuard

**AI Agent Validation Engine** — Multi-dimensional deterministic output validator.

StateGuard, AI agent'larının non-deterministik doğasından kaynaklanan güvenilirlik sorununu çözen, **genel amaçlı, çok boyutlu ve genişletilebilir bir validasyon motorudur**.

## Neden StateGuard?

LLM'ler aynı girdiye her defasında farklı çıktılar üretebilir. Bu, üretim ortamında agent kullanan her sistem için ciddi bir güven sorunudur. StateGuard, LLM'in kendisini değiştirmeye çalışmak yerine, çıktısını **beş boyutta** deterministik olarak doğrular ve sapma tespitinde düzelten bir **doğrulama döngüsü (verification loop)** katmanıdır.

## Özellikler

- **🔬 Kademeli Validasyon** — 3 aşamalı cascade pipeline: Embedding → Kural+Ensemble → Küçük LLM
- **📐 Çok Boyutlu Analiz** — 5 validasyon boyutu: Structural, Semantic, Quantitative, Behavioral, Security
- **🎯 Skor Kartı** — Ağırlıklı çok boyutlu skorlama sistemi (0-100)
- **🔌 Plugin Mimarisi** — Python class tabanlı genişletilebilir validasyon
- **⚙️ State Machine** — Pipeline adım takibi, snapshot ve drift tespiti
- **🔄 Verification Loop** — Otomatik retry, HITL, pipeline rollback
- **📝 Decision Log** — Tüm validasyon adımlarının kayıt altına alınması

## Hızlı Başlangıç

```python
from stateguard import ValidationEngine

# Validation engine'i başlat
engine = ValidationEngine()

# Bir agent çıktısını doğrula
result = engine.validate(
    output="Kullanıcının talebi doğrultusunda indirim kodu oluşturuldu: SAVE10",
    expected_format="json"
)

print(f"Skor: {result.score}/100")
print(f"Geçti: {result.passed}")
print(f"Detay: {result.details}")
```

## Kullanım Senaryoları

| Senaryo | Açıklama |
|:---------|:----------|
| **Code Review Agent** | Kod inceleme çıktılarının format ve mantık tutarlılığı |
| **Content Pipeline** | Çok adımlı içerik üretiminde adım adım validasyon |
| **Data Analysis Agent** | Veri analizi raporlarının yapısal ve nicel doğrulaması |
| **Chatbot** | Kullanıcıya giden yanıtların güvenlik ve tutarlılık kontrolü |

## Mimari Bakış

```
LLM Çıktısı → Tier 1 (Embedding) → Geçti ✅
                                 → Sınırda → Tier 2 (Kural + Ensemble) → Geçti ✅
                                                                       → Sınırda → Tier 3 (Küçük LLM) → Sonuç
```

## Lisans

MIT License — detaylar için `LICENSE` dosyasına bakın.
