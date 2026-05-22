"""Tests for EnsembleValidator — Tier 2 ensemble anomaly detection."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from stateguard.core.tier2 import (
    EnsembleValidator,
    IsolationForestAnalyzer,
    SVMAnomalyAnalyzer,
    ZScoreAnalyzer,
)
from stateguard.models.enums import ValidationDimension, ValidationTier
from stateguard.models.result import ValidationResult


# ── ZScoreAnalyzer Tests ─────────────────────────────────────────────


class TestZScoreAnalyzer:
    """ZScoreAnalyzer birim testleri."""

    def test_fit_computes_mean_and_std(self):
        """fit() sonrası mean ve std doğru hesaplanır."""
        data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)
        analyzer = ZScoreAnalyzer(threshold=3.0)
        analyzer.fit(data)
        assert analyzer.mean_ is not None
        assert analyzer.std_ is not None
        np.testing.assert_array_almost_equal(analyzer.mean_, [2.0, 3.0])
        np.testing.assert_array_almost_equal(analyzer.std_, [0.8165, 0.8165], decimal=4)

    def test_predict_normal(self):
        """Normal değerler → 1 (normal)."""
        data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)
        analyzer = ZScoreAnalyzer(threshold=3.0)
        analyzer.fit(data)
        # Ortalama değer
        result = analyzer.predict(np.array([[2.0, 3.0]]))
        np.testing.assert_array_equal(result, np.array([1]))

    def test_predict_anomaly(self):
        """Aykırı değerler → -1 (anomaly)."""
        data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)
        analyzer = ZScoreAnalyzer(threshold=2.0)
        analyzer.fit(data)
        # Mean'den çok uzak değer
        result = analyzer.predict(np.array([[100.0, 100.0]]))
        np.testing.assert_array_equal(result, np.array([-1]))

    def test_score_samples(self):
        """score_samples anomali skoru döndürür."""
        data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)
        analyzer = ZScoreAnalyzer(threshold=3.0)
        analyzer.fit(data)
        scores = analyzer.score_samples(np.array([[2.0, 3.0]]))
        # 1.0 = normal (z-skor 0)
        assert scores[0] > 0

    def test_zero_std_does_not_crash(self):
        """std=0 durumunda NaN oluşmaz."""
        data = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        analyzer = ZScoreAnalyzer(threshold=3.0)
        analyzer.fit(data)
        result = analyzer.predict(np.array([[1.0, 1.0]]))
        np.testing.assert_array_equal(result, np.array([1]))

    def test_default_threshold(self):
        """Varsayılan threshold 3.0."""
        analyzer = ZScoreAnalyzer()
        assert analyzer.threshold == 3.0

    def test_custom_threshold(self):
        """Custom threshold constructor'dan geçilebilir."""
        analyzer = ZScoreAnalyzer(threshold=2.5)
        assert analyzer.threshold == 2.5


class TestIsolationForestAnalyzer:
    """IsolationForestAnalyzer birim testleri."""

    def test_fit_and_predict(self):
        """fit + predict döngüsü çalışır."""
        with patch("sklearn.ensemble.IsolationForest") as mock_if:
            instance = mock_if.return_value
            instance.fit.return_value = None
            instance.predict.return_value = np.array([1, 1, -1])

            analyzer = IsolationForestAnalyzer(contamination=0.1, random_state=42)
            data = np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 20.0]], dtype=np.float64)
            analyzer.fit(data)
            result = analyzer.predict(data)
            np.testing.assert_array_equal(result, np.array([1, 1, -1]))
            instance.fit.assert_called_once()

    def test_score_samples(self):
        """score_samples decision_function çağırır."""
        with patch("sklearn.ensemble.IsolationForest") as mock_if:
            instance = mock_if.return_value
            instance.fit.return_value = None
            instance.decision_function.return_value = np.array([0.5, 0.3, -0.1])

            analyzer = IsolationForestAnalyzer()
            data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)
            analyzer.fit(data)
            scores = analyzer.score_samples(data)
            np.testing.assert_array_almost_equal(scores, np.array([0.5, 0.3, -0.1]))


class TestSVMAnomalyAnalyzer:
    """SVMAnomalyAnalyzer birim testleri."""

    def test_fit_and_predict(self):
        """fit + predict döngüsü çalışır."""
        with patch("sklearn.svm.OneClassSVM") as mock_svm:
            instance = mock_svm.return_value
            instance.fit.return_value = None
            instance.predict.return_value = np.array([1, 1, -1])

            analyzer = SVMAnomalyAnalyzer(nu=0.1, kernel="rbf", gamma="auto")
            data = np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 20.0]], dtype=np.float64)
            analyzer.fit(data)
            result = analyzer.predict(data)
            np.testing.assert_array_equal(result, np.array([1, 1, -1]))

    def test_score_samples(self):
        """score_samples decision_function çağırır."""
        with patch("sklearn.svm.OneClassSVM") as mock_svm:
            instance = mock_svm.return_value
            instance.fit.return_value = None
            instance.decision_function.return_value = np.array([0.8, 0.6, -0.2])

            analyzer = SVMAnomalyAnalyzer()
            data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)
            analyzer.fit(data)
            scores = analyzer.score_samples(data)
            np.testing.assert_array_almost_equal(scores, np.array([0.8, 0.6, -0.2]))


# ── EnsembleValidator Tests ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def patch_sklearn_analyzers():
    """Mock sklearn modelleri — gerçek model indirme/eğitim yok."""
    with (
        patch("sklearn.ensemble.IsolationForest") as mock_if,
        patch("sklearn.svm.OneClassSVM") as mock_svm,
    ):
        # IsolationForest mock
        if_instance = mock_if.return_value
        if_instance.fit.return_value = None
        if_instance.predict.return_value = np.array([1, 1, 1])
        if_instance.decision_function.return_value = np.array([0.5, 0.5, 0.5])

        # OneClassSVM mock
        svm_instance = mock_svm.return_value
        svm_instance.fit.return_value = None
        svm_instance.predict.return_value = np.array([1, 1, 1])
        svm_instance.decision_function.return_value = np.array([0.8, 0.8, 0.8])

        yield


class TestEnsembleValidator:
    """EnsembleValidator birim testleri."""

    def test_class_attributes(self):
        """Sınıf attribute'ları doğru set edilmiş."""
        v = EnsembleValidator()
        assert v.name == "ensemble-anomaly"
        assert v.dimension == ValidationDimension.SEMANTIC
        assert v.tier == ValidationTier.TIER_2

    def test_lazy_initialization(self):
        """Analyzer'lar constructor'da değil, ilk validate'de initialize edilir."""
        v = EnsembleValidator()
        # Constructor sonrası analyzer yok
        assert not hasattr(v, "_analyzers") or v._analyzers is None

    def test_all_methods_normal_passes(self):
        """3/3 yöntem normal derse → passed=True, score=100."""
        v = EnsembleValidator()
        data = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float64)
        result = v.validate({"features": data.tolist()})
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.score == 100.0
        assert result.dimension == ValidationDimension.SEMANTIC

    def test_majority_passes(self):
        """2/3 yöntem normal derse → passed=True (2/3 çoğunluk)."""
        v = EnsembleValidator()
        # Predict'i override et: 2 normal, 1 anomali
        analyzers = v._ensure_analyzers()
        for name, analyzer in analyzers:
            if name == "z_score":  # ZScoreAnalyzer → anomali desin
                analyzer.predict = lambda x: np.array([-1])
            else:
                analyzer.predict = lambda x: np.array([1])

        data = {"features": [1.0, 2.0, 3.0, 4.0]}
        result = v.validate(data)
        assert result.passed is True
        assert 66.0 <= result.score <= 67.0  # 2/3 ≈ 66.67

    def test_majority_fails(self):
        """2/3 yöntem anomali derse → passed=False."""
        v = EnsembleValidator()
        analyzers = v._ensure_analyzers()
        for name, analyzer in analyzers:
            if name == "isolation_forest":  # Sadece IsolationForest normal
                analyzer.predict = lambda x: np.array([1])
            else:
                analyzer.predict = lambda x: np.array([-1])

        data = {"features": [10.0, 20.0, 30.0, 40.0]}
        result = v.validate(data)
        assert result.passed is False
        assert 33.0 <= result.score <= 34.0  # 1/3 ≈ 33.33

    def test_all_anomaly_fails_zero_score(self):
        """3/3 yöntem anomali derse → passed=False, score=0."""
        v = EnsembleValidator()
        analyzers = v._ensure_analyzers()
        for name, analyzer in analyzers:
            analyzer.predict = lambda x: np.array([-1])

        data = {"features": [100.0, 200.0]}
        result = v.validate(data)
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_input_returns_error(self):
        """Boş/geçersiz girdi → error döner."""
        v = EnsembleValidator()
        result = v.validate({})
        assert result.passed is False
        assert result.score == 0.0
        assert "features" in (result.error or "")

    def test_none_input_returns_error(self):
        """None girdi → error döner."""
        v = EnsembleValidator()
        result = v.validate(None)
        assert result.passed is False
        assert result.score == 0.0

    def test_string_input_returns_error(self):
        """String girdi → error döner (Tier 2 sayısal veri bekler)."""
        v = EnsembleValidator()
        result = v.validate("string input")
        assert result.passed is False

    def test_details_contains_three_methods(self):
        """Details alanı 3 yöntemin de sonucunu içerir."""
        v = EnsembleValidator()
        data = {"features": [1.0, 2.0, 3.0]}
        result = v.validate(data)
        assert "isolation_forest" in result.details
        assert "one_class_svm" in result.details
        assert "z_score" in result.details
        assert "ensemble" in result.details
        assert result.details["ensemble"]["total_methods"] == 3
        assert "vote" in result.details["ensemble"]

    def test_validation_result_format(self):
        """ValidationResult formatı doğru (dimension=SEMANTIC)."""
        v = EnsembleValidator()
        data = {"features": [1.0, 2.0, 3.0]}
        result = v.validate(data)
        assert isinstance(result, ValidationResult)
        assert result.dimension == ValidationDimension.SEMANTIC
        assert 0.0 <= result.score <= 100.0
        assert isinstance(result.passed, bool)

    def test_feature_extraction_from_dict(self):
        """features list/dict'ten dogru cikarilir."""
        v = EnsembleValidator()
        data_list = {"features": [0.5, 0.3, 0.8, 0.1]}
        result = v.validate(data_list)
        assert isinstance(result, ValidationResult)

        # Use a fresh validator for different feature dimensions.
        v2 = EnsembleValidator()
        data_array = {"features": [[0.5, 0.3], [0.8, 0.1]]}
        result = v2.validate(data_array)
        assert isinstance(result, ValidationResult)


class TestEnsembleValidatorFit:
    """fit() metodu — AC1."""

    def test_fit_method_exists(self):
        """fit() metodu mevcut ve cagrilabilir."""
        v = EnsembleValidator()
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        # Henuz fit yok — AttributeError bekleniyor (RED phase)
        v.fit(data)
        # fit sonrasi _is_fitted True olmali
        assert v._is_fitted is True

    def test_fit_with_dict_input(self):
        """fit() dict ile cagrilabilir (features anahtari ile)."""
        v = EnsembleValidator()
        result = v.fit({"features": [1.0, 2.0, 3.0, 4.0]})
        assert v._is_fitted is True
        assert result is None  # fit void dondurur

    def test_fit_with_list_input(self):
        """fit() list of lists ile cagrilabilir."""
        v = EnsembleValidator()
        v.fit([[1.0, 2.0], [3.0, 4.0]])
        assert v._is_fitted is True

    def test_fit_updates_is_fitted_and_timestamp(self):
        """fit() sonrasi _is_fitted=True ve _fitted_at timestamp alir."""
        v = EnsembleValidator()
        assert v._is_fitted is False
        v.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert v._is_fitted is True
        assert v._fitted_at is not None


class TestEnsembleValidatorValidateAfterFit:
    """validate() fit sonrasi sadece predict yapar — AC2."""

    def test_validate_after_fit_predicts_only(self):
        """fit sonrasi validate'de analyzer.fit cagrilmaz (mock ile)."""
        from unittest.mock import MagicMock

        v = EnsembleValidator()
        # Analyzer'lari mockla
        v._analyzers = [
            ("mock", MagicMock()),
        ]
        v._analyzers[0][1].predict.return_value = np.array([1])
        v._analyzers[0][1].score_samples.return_value = np.array([0.5])
        v._is_fitted = True

        result = v.validate({"features": [1.0, 2.0]})
        v._analyzers[0][1].fit.assert_not_called()
        v._analyzers[0][1].predict.assert_called_once()

    def test_validate_without_fit_auto_fits_and_warns(self):
        """fit edilmemis validate auto-fit yapar ve warning basar."""
        import warnings

        v = EnsembleValidator()
        data = {"features": [1.0, 2.0, 3.0]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = v.validate(data)

        assert result.passed is True
        assert len(w) >= 1
        assert any("not fitted" in str(msg.message).lower() for msg in w)

    def test_validate_without_fit_reports_auto_fitted(self):
        """Auto-fit detaylarda auto_fitted=True olarak raporlanir."""
        v = EnsembleValidator()
        result = v.validate({"features": [1.0, 2.0]})
        assert result.details.get("auto_fitted") is True


class TestEnsembleValidatorPersistence:
    """save/load — AC3."""

    def test_save_and_load_model(self, tmp_path):
        """save() sonrasi load() ile ayni state geri yuklenir."""
        import joblib

        v = EnsembleValidator()
        v._is_fitted = True
        v._fitted_at = "2026-05-22T12:00:00+00:00"
        v._fit_version = "1.0.0"

        # Create simple fitted analyzers (no sklearn mock involved).
        z = ZScoreAnalyzer(threshold=3.0)
        z.fit(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        v._analyzers = [("z_score", z)]

        model_path = str(tmp_path / "ensemble_model.joblib")
        v.save(model_path)

        v2 = EnsembleValidator.load(model_path)
        assert v2._is_fitted is True
        assert v2._fitted_at == v._fitted_at
        assert v2._contamination == v._contamination
        assert v2._z_score_threshold == v._z_score_threshold

    def test_load_unfitted_saves_and_loads(self, tmp_path):
        """fit edilmemis model de save/load yapilabilir."""
        v = EnsembleValidator()
        v._analyzers = []  # No analyzers to avoid mock pickling
        model_path = str(tmp_path / "unfitted.joblib")
        v.save(model_path)
        v2 = EnsembleValidator.load(model_path)
        assert v2._is_fitted is False

    def test_load_invalid_path_raises(self):
        """Gecersiz path load'da FileNotFoundError firlatir."""
        with pytest.raises((FileNotFoundError, ValueError)):
            EnsembleValidator.load("/nonexistent/path/model.joblib")

    def test_save_creates_file(self, tmp_path):
        """save() dosya olusturur."""
        v = EnsembleValidator()
        v._analyzers = []
        model_path = str(tmp_path / "test_model.joblib")
        v.save(model_path)
        assert tmp_path.joinpath("test_model.joblib").exists()

    def test_load_incompatible_version_raises(self, tmp_path):
        """Uyumsuz model versiyonu ValueError firlatir."""
        v = EnsembleValidator()
        v._analyzers = []
        model_path = str(tmp_path / "bad_version.joblib")
        # Save then corrupt the version
        v.save(model_path)

        # Manually overwrite the version field
        import joblib as _joblib
        state = _joblib.load(str(model_path))
        state["_model_version"] = "0.0.0"
        _joblib.dump(state, str(model_path))

        with pytest.raises(ValueError, match="version"):
            EnsembleValidator.load(str(model_path))

    def test_incremental_fit_updates_model(self):
        """Coklu fit cagrisi modeli gunceller (ikinci fit override eder)."""
        v = EnsembleValidator()
        fit1_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        fit2_data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

        v.fit(fit1_data)
        assert v._is_fitted is True
        ts1 = v._fitted_at

        v.fit(fit2_data)  # re-fit
        assert v._is_fitted is True
        ts2 = v._fitted_at
        # Timestamp guncellenmeli
        assert ts2 != ts1


class TestEnsembleValidatorSetup:
    """setup() hook — AC4."""

    def test_setup_does_not_crash(self):
        """setup() varsayilan halde no-op'tur (config yoksa)."""
        v = EnsembleValidator()
        v.setup()  # should not raise
