# Kurulum Kılavuzu

## Gereksinimler

- Python >= 3.11
- Poetry >= 1.8 (öncelikli) veya pip
- En az 4GB RAM (embedding model için)

## Kurulum

### Poetry ile

```bash
# Projeyi klonla
git clone https://github.com/yunusgungor/stateguard.git
cd stateguard

# Bağımlılıkları yükle
poetry install

# Sanal ortamı aktifleştir
poetry shell
```

### pip ile

```bash
pip install stateguard
```

## İlk Validasyon

```python
from stateguard import ValidationEngine
from stateguard.models import ValidationConfig

# Konfigürasyon
config = ValidationConfig(
    tier1_threshold=80,    # Embedding similarity eşiği
    tier2_threshold=70,    # Ensemble eşiği
    adaptive=True          # Adaptif eşik güncelleme
)

# Engine'i başlat
engine = ValidationEngine(config=config)

# Çıktıyı doğrula
result = engine.validate(
    output="Merhaba, size nasıl yardımcı olabilirim?",
    context="customer_service",
    expected_format="text"
)

if result.passed:
    print(f"✅ Validasyon geçti (skor: {result.score})")
else:
    print(f"❌ Validasyon başarısız (skor: {result.score})")
    print(f"Hatalar: {result.errors}")
```

## Önemli Bağımlılıklar

| Paket | Sürüm | Amaç |
|:------|:-----|:-----|
| sentence-transformers | >= 3.0 | Embedding tabanlı semantik benzerlik |
| scikit-learn | >= 1.5 | Ensemble anomali tespiti (Isolation Forest, OCSVM) |
| pydantic | >= 2.0 | Tip güvenli veri modelleri |
| structlog | >= 24.0 | Yapılandırılmış loglama |

## Sonraki Adımlar

1. [Mimari dokümanı](architecture.md) inceleyerek validasyon akışını anlayın
2. [Plugin geliştirme kılavuzu](plugin-dev-guide.md) ile özel validator'lar yazın
3. Kapsamlı örnekler için `examples/` dizinine göz atın
