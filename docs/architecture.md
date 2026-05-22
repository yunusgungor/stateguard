# Mimari Doküman

StateGuard, LLM çıktılarını **üç aşamalı kademeli (cascade)** bir pipeline ile doğrular. Her aşama bir sonrakinden daha derin ve daha maliyetlidir.

## Cascade Pipeline

```
LLM Çıktısı
     │
     ▼
┌─────────────────────────────┐
│  Tier 1: Embedding Similarity │  Hızlı (~10ms)
│  Skor ≥ 80 → ✅ PASS         │  Çıktıların ~%70'i
│  Skor < 50 → ❌ FAIL         │
│  50-79 → Tier 2'ye yönlendir │
└─────────────────────────────┘
              │ 50-79 arası
              ▼
┌─────────────────────────────┐
│  Tier 2: Kural + Ensemble    │  Orta (~100ms)
│  • Kural bazlı kontroller    │
│  • Isolation Forest          │
│  • One-Class SVM             │
│  • Z-Score analizi           │
│  Geçmezse → Tier 3           │
└─────────────────────────────┘
              │ Sınırda vakalar
              ▼
┌─────────────────────────────┐
│  Tier 3: Küçük LLM           │  Derin (~1-5s)
│  • Semantik doğrulama        │
│  • Bağlam tutarlılığı        │
│  • Nihai karar               │
└─────────────────────────────┘
              │
              ▼
       Nihai Sonuç
```

## Validasyon Boyutları

Her boyut bağımsız bir validasyon perspektifi sunar ve **Score Card** ile ağırlıklı olarak birleştirilir:

| Boyut | Ağırlık | Amaç |
|:------|:-------:|:-----|
| **Structural** | %25 | JSON/XML şema uyumu, tip kontrolü |
| **Semantic** | %25 | Anlamsal tutarlılık, embedding benzerlik |
| **Quantitative** | %15 | Sayısal doğruluk, istatistiksel tutarlılık |
| **Behavioral** | %20 | Davranışsal beklentilere uygunluk |
| **Security** | %15 | Zararlı içerik, prompt injection tespiti |

**Toplam Skor** = (Structural × 0.25) + (Semantic × 0.25) + (Quantitative × 0.15) + (Behavioral × 0.20) + (Security × 0.15)

## State Machine

Pipeline'ın her adımı **5 durumlu bir Finite State Machine (FSM)** ile takip edilir:

```
    ┌──────────┐
    │   IDLE   │
    └────┬─────┘
         │ start
         ▼
    ┌────────────┐
    │ VALIDATING │
    └─┬────┬────┬┘
      │  │  │
      ▼  │  ▼
  ┌────┐ │ ┌──────┐
  │PASS│ │ │FAILED│
  └────┘ │ └──┬───┘
         │    │ retry
         │    ▼
         │ ┌──────┐
         │ │RETRY │
         │ └──┬───┘
         │    │ 2. başarısızlık
         │    ▼
         │ ┌───────────┐
         │ │HITL_WAITING│
         │ └───────────┘
         │
         ▼
    ┌───────────┐
    │ COMPLETED │
    └───────────┘
```

**State Snapshot** her adımda system state'in serialize edilmiş görüntüsünü alır. Bu snapshot'lar:
- Kümülatif sapma tespiti için karşılaştırma bazı sağlar
- Pipeline rollback durumunda geri yükleme noktası oluşturur
- Decision log için denetim izi bırakır

## Plugin Mimarisi

```
Plugin Sistemi
├── BaseValidator (abstract class)
│   ├── validate(output, context) → ValidationResult
│   ├── setup() → None             # Kaynak yükleme (opsiyonel)
│   └── teardown() → None          # Kaynak temizleme (opsiyonel)
├── PluginRegistry
│   ├── register(name, validator_class)
│   ├── get(name) → BaseValidator
│   └── list() → dict[str, ValidatorInfo]
└── Examples
    ├── JsonSchemaValidator (STRUCTURAL, TIER_1)
    ├── KeywordValidator (SEMANTIC, TIER_1)
    ├── LengthValidator (QUANTITATIVE, TIER_1)
    └── RegexValidator (STRUCTURAL, TIER_1)
```

Plugin'ler Python class inheritance ile tanımlanır ve YAML konfigürasyonla yüklenir.

## Geri Bildirim Motoru

3 aşamalı hata yönetimi:

1. **Auto Retry** — İlk başarısızlıkta LLM'e otomatik yeniden gönder
2. **HITL (Human In The Loop)** — 2. başarısızlıkta insan onayı bekle
3. **Pipeline Rollback** — Kritik hatalarda pipeline'ı önceki sağlıklı state'e döndür

## Proje Yapısı

```
stateguard/
├── pyproject.toml
├── stateguard/
│   ├── core/          ← engine, tier1, tier2, tier3, scoring
│   ├── dimensions/    ← structural, semantic, quantitative, behavioral, security
│   ├── state/         ← machine, snapshot, drift
│   ├── feedback/      ← retry, hitl, rollback
│   ├── plugin/        ← base, registry, examples
│   ├── config/        ← schema, settings, defaults
│   ├── models/        ← result, enums, log
│   └── utils/         ← embedding, logging
├── tests/
│   ├── test_core/
│   ├── test_dimensions/
│   ├── test_state/
│   ├── test_feedback/
│   ├── test_plugin/
│   └── integration/
└── docs/              ← Bu dokümanlar
```

## Teknoloji Yığını

| Bileşen | Teknoloji |
|:--------|:----------|
| Dil | Python >= 3.11 |
| Paket Yönetimi | Poetry |
| Embedding | sentence-transformers >= 3.0 |
| ML | scikit-learn >= 1.5 (Isolation Forest, OCSVM) |
| Veri Modelleri | Pydantic v2 |
| Test | pytest, pytest-cov |
| Loglama | structlog |
| Dokümantasyon | MkDocs + Material Theme |
