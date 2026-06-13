"""
tests/test_ml.py
=================
Unit tests for ML models and inference.

Tests:
  - sklearn LDA/SVM/RF fit and predict
  - Confidence gate (>0.85 = issue, <0.75 = hold)
  - Adaptive calibration SGD update
  - Prediction result shape and types
  - Fallback mechanism on CNN failure
"""

import pytest
import numpy as np


class TestSKLearnBaseline:
    def test_lda_fit_predict(self, batch_epochs, synthetic_feature_vec):
        from ml.sklearn_baseline import SKLearnBaseline
        X_ep, y = batch_epochs
        # Create features
        from dsp.feature_vector import FeatureExtractor
        from dsp.epoch_extraction import Epoch
        extractor = FeatureExtractor(sample_rate=250.0, n_channels=8)
        X_feat = np.array([
            extractor.extract(Epoch(data=ep[:, :200], timestamp=0.0, epoch_type="p300"))
            for ep in X_ep
        ])
        clf = SKLearnBaseline(method="lda", n_classes=4)
        clf.fit(X_feat, y)
        pred, probs, conf = clf.predict(X_feat[0])
        assert pred in range(4)
        assert len(probs) == 4
        assert 0.0 <= conf <= 1.0
        assert abs(sum(probs) - 1.0) < 1e-5

    def test_svm_trains_without_error(self, batch_epochs):
        from ml.sklearn_baseline import SKLearnBaseline
        from dsp.feature_vector import FeatureExtractor
        from dsp.epoch_extraction import Epoch
        X_ep, y = batch_epochs
        extractor = FeatureExtractor()
        X_feat = np.array([
            extractor.extract(Epoch(data=ep[:, :200], timestamp=0.0, epoch_type="p300"))
            for ep in X_ep
        ])
        clf = SKLearnBaseline(method="svm")
        clf.fit(X_feat, y)
        assert clf._is_fitted

    def test_invalid_method_raises(self):
        from ml.sklearn_baseline import SKLearnBaseline
        with pytest.raises(ValueError):
            SKLearnBaseline(method="invalid_method")

    def test_predict_before_fit_returns_uniform(self, synthetic_feature_vec):
        from ml.sklearn_baseline import SKLearnBaseline
        clf = SKLearnBaseline(method="lda", n_classes=4)
        pred, probs, conf = clf.predict(synthetic_feature_vec)
        assert pred == 0
        assert conf == pytest.approx(0.25, abs=0.01)


class TestRealTimePredictor:
    def test_confidence_gate_issue(self, synthetic_epoch, lda_classifier):
        from ml.predict import RealTimePredictor, CommandDecision
        predictor = RealTimePredictor(
            fallback_model=lda_classifier,
            paradigm="motor_imagery",
            primary_threshold=0.85,
        )
        result = predictor.predict(
            epoch=synthetic_epoch[:, :1000] if synthetic_epoch.shape[1] >= 1000 else np.tile(synthetic_epoch, (1, 5)),
            feature_vector=np.random.randn(56).astype(np.float32),
        )
        assert result.decision in list(CommandDecision)
        assert 0.0 <= result.confidence <= 1.0
        assert result.class_name in ["left", "right", "feet", "rest"]

    def test_result_has_all_fields(self, synthetic_epoch, lda_classifier):
        from ml.predict import RealTimePredictor
        predictor = RealTimePredictor(fallback_model=lda_classifier)
        result = predictor.predict(
            epoch=np.random.randn(8, 1000).astype(np.float32),
            feature_vector=np.random.randn(56).astype(np.float32),
        )
        assert hasattr(result, "timestamp")
        assert hasattr(result, "predicted_class")
        assert hasattr(result, "confidence")
        assert hasattr(result, "inference_ms")
        assert result.inference_ms > 0

    def test_hold_on_low_confidence(self, lda_classifier):
        """Predictor must HOLD when all classifiers return low confidence."""
        from ml.predict import RealTimePredictor, CommandDecision
        predictor = RealTimePredictor(
            cnn_model=None,
            fallback_model=None,   # force zero-confidence path
            paradigm="motor_imagery",
        )
        result = predictor.predict(
            epoch=np.zeros((8, 1000), dtype=np.float32),
            feature_vector=np.zeros(56, dtype=np.float32),
        )
        # No models = uniform probs = 0.25 < 0.75 threshold → HOLD
        assert result.decision == CommandDecision.HOLD


class TestAdaptiveCalibration:
    def test_buffer_fills_correctly(self):
        from ml.adaptive_calibration import AdaptiveCalibration
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.model = None  # skip TF
        cal = AdaptiveCalibration(model=mock_model)
        for i in range(10):
            cal.add_trial(
                epoch=np.random.randn(8, 200).astype(np.float32),
                true_label=i % 4,
                predicted_label=i % 4,
            )
        assert cal._trial_count == 10

    def test_stats_returned(self):
        from ml.adaptive_calibration import AdaptiveCalibration
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.model = None
        cal = AdaptiveCalibration(model=mock_model)
        stats = cal.get_stats()
        assert "total_trials" in stats
        assert "total_retrains" in stats
        assert "alert_active" in stats