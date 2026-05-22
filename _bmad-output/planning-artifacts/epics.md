# StateGuard Epic 1: Ürünleştirme ve Sağlamlaştırma

**Epic ID:** 1
**Epic Adı:** StateGuard Productization & Hardening
**Başlangıç:** 2026-05-22
**Durum:** planning

## Epic Amacı

StateGuard'ın mevcut edge case analizlerinde tespit edilen 5 kritik tasarım ve robust-ness eksikliğini kapatarak projeyi üretime hazır hale getirmek. Bu epic, projenin "çalışan demo" seviyesinden "güvenilir kütüphane" seviyesine çıkmasını sağlar.

## Kapsanan Bulgular

| # | Bulgu | Kaynak | Severity |
|---|-------|--------|----------|
| W5 | EnsembleValidator validation data'sına fit ediyor (data leakage) | EDGE_CASE_ANALYSIS.md | 🟡 WARNING |
| W1 | Deep inheritance chain `__init_subclass__` tarafından engelleniyor | EDGE_CASE_ANALYSIS.md | 🟡 WARNING |
| W2 | `validate` callable olarak enforce edilmiyor | EDGE_CASE_ANALYSIS.md | 🟡 WARNING |
| W3 | LLMValidator, `HTTPLLMClient._build_prompt` private method'una bağımlı | EDGE_CASE_ANALYSIS.md | 🟡 WARNING |
| W1 | Snapshot `TypeError` yakalamıyor (non-JSON dict key) | EDGE_CASE_ANALYSIS_SNAPSHOT.md | 🟡 WARNING |
| I2 | Örnek plugin'ler skeleton/ellipsis implementasyon | EDGE_CASE_ANALYSIS.md | 🔵 INFO |

## Çıktılar

- EnsembleValidator'a train/persist/load akışı
- BaseValidator'da inheritance fix + validate callable guard
- LLMClient protocol'ünde build_prompt desteği
- SnapshotManager'da TypeError handling
- Çalışan örnek plugin validator'lar + dökümantasyon

## Story'ler

| # | Story | Tahmini Süre | Öncelik |
|---|-------|-------------|---------|
| 1-1 | Tier-2 ML Ensemble Eğitim/Kalibrasyon Altyapısı | 2-3 gün | YÜKSEK |
| 1-2 | Plugin Sistemi Sağlamlaştırma | 1 gün | YÜKSEK |
| 1-3 | LLM Client Abstraction İyileştirme | 0.5 gün | ORTA |
| 1-4 | Snapshot/State Robustness Fix | 0.5 gün | ORTA |
| 1-5 | Örnek Validator ve Dokümantasyon Tamamlama | 1 gün | ORTA |
