"""
Unit tests for the analysis worker components:
  - analysis_worker.preprocessor  (ClaimPreprocessor, _to_float, _to_int)
  - analysis_worker.fraud_detector (FraudDetector)

Run with:
    pytest tests/test_worker.py -v
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis_worker.preprocessor import ClaimPreprocessor, _to_float, _to_int
from analysis_worker.fraud_detector import FraudDetector


# ══════════════════════════════════════════════════════════════════════════════
# Helpers / Fixtures
# ══════════════════════════════════════════════════════════════════════════════

N_FEATURES = 25  # number of features built by ClaimPreprocessor._get_feature_names()


def _make_scaler(n_features: int = N_FEATURES):
    """Return a mock StandardScaler that matches the given feature count."""
    scaler = MagicMock()
    scaler.n_features_in_ = n_features
    # transform returns a float array with the same shape
    scaler.transform.side_effect = lambda x: x.astype(np.float64)
    return scaler


def _make_clinical_norms():
    return {}


def _make_preprocessor(scaler=None):
    if scaler is None:
        scaler = _make_scaler()
    return ClaimPreprocessor(scaler=scaler, clinical_norms=_make_clinical_norms())


def _minimal_claim(**overrides):
    """Return a minimal valid claim dict."""
    base = {
        "claim_id":              "CLM-001",
        "beneficiary_id":        "BENE-001",
        "provider_id":           "PROV-001",
        "claim_amount":          1500.0,
        "deductible_amt_paid":   50.0,
        "primary_diagnosis_code":"Z00.00",
        "age":                   45,
        "gender":                1,
        "race":                  1,
        "claim_start_date":      "2024-01-01",
        "claim_end_date":        "2024-01-05",
    }
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# 1. Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

class TestToFloat:
    def test_float_value(self):
        assert _to_float(3.14) == pytest.approx(3.14)

    def test_int_value(self):
        assert _to_float(5) == pytest.approx(5.0)

    def test_string_number(self):
        assert _to_float("2.5") == pytest.approx(2.5)

    def test_none_returns_default(self):
        assert _to_float(None, default=99.9) == pytest.approx(99.9)

    def test_invalid_string_returns_default(self):
        assert _to_float("abc", default=0.0) == pytest.approx(0.0)

    def test_zero_is_valid(self):
        assert _to_float(0) == pytest.approx(0.0)


class TestToInt:
    def test_integer_value(self):
        assert _to_int(7) == 7

    def test_float_truncated(self):
        assert _to_int(3.9) == 3

    def test_string_number(self):
        assert _to_int("42") == 42

    def test_none_returns_default(self):
        assert _to_int(None, default=5) == 5

    def test_invalid_string_returns_default(self):
        assert _to_int("xyz", default=0) == 0

    def test_negative_value(self):
        assert _to_int(-3) == -3


# ══════════════════════════════════════════════════════════════════════════════
# 2. ClaimPreprocessor Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestClaimPreprocessorInit:
    def test_feature_names_count(self):
        pp = _make_preprocessor()
        assert len(pp.feature_names) == N_FEATURES

    def test_known_feature_names_present(self):
        pp = _make_preprocessor()
        for expected in ('claim_amount', 'age', 'num_diagnosis_codes',
                         'total_chronic_conditions', 'claim_duration_days'):
            assert expected in pp.feature_names

    def test_scaler_features_detected(self):
        pp = _make_preprocessor(_make_scaler(n_features=N_FEATURES))
        assert pp._scaler_features == N_FEATURES

    def test_scaler_without_n_features_in_attr(self):
        """Fallback to len(scaler.mean_) when n_features_in_ is absent."""
        scaler = MagicMock(spec=[])   # no attributes
        scaler.mean_ = [0.0] * 10
        pp = ClaimPreprocessor(scaler=scaler, clinical_norms={})
        assert pp._scaler_features == 10


class TestClaimPreprocessorPreprocess:
    """End-to-end preprocessing tests."""

    def test_output_is_2d_numpy_array(self):
        pp = _make_preprocessor()
        result = pp.preprocess(_minimal_claim())
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_output_shape_matches_feature_count(self):
        pp = _make_preprocessor()
        result = pp.preprocess(_minimal_claim())
        assert result.shape == (1, N_FEATURES)

    def test_claim_amount_in_features(self):
        """Claim amount should appear at feature index 0 (before scaling)."""
        # Use identity scaler to inspect raw values
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        result = pp.preprocess(_minimal_claim(claim_amount=1234.0))
        # scaler.transform is identity mock, so index 0 = claim_amount
        assert result[0, 0] == pytest.approx(1234.0)

    def test_age_maps_correctly(self):
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        result = pp.preprocess(_minimal_claim(age=55))
        age_idx = pp.feature_names.index('age')
        assert result[0, age_idx] == pytest.approx(55.0)

    def test_missing_optional_fields_use_defaults(self):
        """Preprocessing must not crash when optional fields are absent."""
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        claim = {
            "claim_id":              "CLM-MIN",
            "claim_amount":          500.0,
            "primary_diagnosis_code":"Z00.00",
            "age":                   30,
        }
        result = pp.preprocess(claim)
        assert result.shape == (1, N_FEATURES)

    def test_gender_defaults_to_zero_when_missing(self):
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        result = pp.preprocess(_minimal_claim(gender=None))
        idx = pp.feature_names.index('gender')
        assert result[0, idx] == 0.0

    def test_coverage_defaults_to_12_months(self):
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        result = pp.preprocess(_minimal_claim())  # no coverage fields
        idx_a = pp.feature_names.index('no_of_months_part_a_cov')
        idx_b = pp.feature_names.index('no_of_months_part_b_cov')
        assert result[0, idx_a] == pytest.approx(12.0)
        assert result[0, idx_b] == pytest.approx(12.0)

    def test_chronic_conditions_encoded_as_binary(self):
        """Raw value 1 (Yes) → 1; value 2 (No) → 0."""
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        claim = _minimal_claim(chronic_cond_diabetes=1, chronic_cond_cancer=2)
        result = pp.preprocess(claim)
        diab_idx   = pp.feature_names.index('chronic_cond_diabetes')
        cancer_idx = pp.feature_names.index('chronic_cond_cancer')
        assert result[0, diab_idx]   == 1.0
        assert result[0, cancer_idx] == 0.0

    def test_total_chronic_conditions_count(self):
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        # Set 3 conditions to 1 (Yes)
        claim = _minimal_claim(
            chronic_cond_diabetes=1,
            chronic_cond_cancer=1,
            chronic_cond_stroke=1,
        )
        result = pp.preprocess(claim)
        idx = pp.feature_names.index('total_chronic_conditions')
        assert result[0, idx] == 3.0

    def test_renal_disease_indicator_y_maps_to_1(self):
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        result = pp.preprocess(_minimal_claim(renal_disease_indicator='Y'))
        idx = pp.feature_names.index('renal_disease_indicator_num')
        assert result[0, idx] == 1.0

    def test_renal_disease_indicator_default_maps_to_0(self):
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        result = pp.preprocess(_minimal_claim())
        idx = pp.feature_names.index('renal_disease_indicator_num')
        assert result[0, idx] == 0.0

    def test_num_diagnosis_codes_counts_non_empty(self):
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        claim = _minimal_claim(
            primary_diagnosis_code='Z00.00',
            diagnosis_code_2='E11.9',
            diagnosis_code_3='I10',
        )
        result = pp.preprocess(claim)
        idx = pp.feature_names.index('num_diagnosis_codes')
        assert result[0, idx] == 3.0

    def test_num_procedure_codes_counts_non_empty(self):
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        claim = _minimal_claim(procedure_code_1='99213', procedure_code_2='99214')
        result = pp.preprocess(claim)
        idx = pp.feature_names.index('num_procedure_codes')
        assert result[0, idx] == 2.0

    def test_has_multiple_physicians_flag(self):
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        # Single physician → 0
        r1 = pp.preprocess(_minimal_claim(attending_physician='P1'))
        # Two physicians → 1
        r2 = pp.preprocess(_minimal_claim(attending_physician='P1', operating_physician='P2'))
        idx = pp.feature_names.index('has_multiple_physicians')
        assert r1[0, idx] == 0.0
        assert r2[0, idx] == 1.0

    def test_scaler_called_when_dimensions_match(self):
        scaler = _make_scaler(N_FEATURES)
        pp = _make_preprocessor(scaler)
        pp.preprocess(_minimal_claim())
        scaler.transform.assert_called_once()

    def test_manual_normalize_used_when_scaler_mismatch(self):
        """When scaler feature count != built features, manual normalise is used."""
        scaler = _make_scaler(n_features=5)   # intentional mismatch
        pp = _make_preprocessor(scaler)
        result = pp.preprocess(_minimal_claim())
        # manual normalise clips to [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        # scaler.transform must NOT have been called
        scaler.transform.assert_not_called()


class TestCalculateDuration:
    """Tests for ClaimPreprocessor._calculate_duration"""

    def test_valid_iso_dates(self):
        pp = _make_preprocessor()
        assert pp._calculate_duration("2024-01-01", "2024-01-06") == 5

    def test_slash_date_format(self):
        pp = _make_preprocessor()
        assert pp._calculate_duration("01/01/2024", "01/11/2024") == 10

    def test_compact_date_format(self):
        pp = _make_preprocessor()
        assert pp._calculate_duration("20240101", "20240110") == 9

    def test_same_day_returns_1(self):
        pp = _make_preprocessor()
        assert pp._calculate_duration("2024-01-01", "2024-01-01") == 1

    def test_missing_start_date_returns_1(self):
        pp = _make_preprocessor()
        assert pp._calculate_duration(None, "2024-01-10") == 1

    def test_missing_end_date_returns_1(self):
        pp = _make_preprocessor()
        assert pp._calculate_duration("2024-01-01", None) == 1

    def test_invalid_format_returns_1(self):
        pp = _make_preprocessor()
        assert pp._calculate_duration("not-a-date", "also-not") == 1

    def test_inverted_dates_return_1(self):
        """End before start → max(negative, 1) == 1."""
        pp = _make_preprocessor()
        assert pp._calculate_duration("2024-01-10", "2024-01-01") == 1


class TestCountCodes:
    def test_counts_non_empty_codes(self):
        pp = _make_preprocessor()
        claim = {"code_1": "ABC", "code_2": "DEF", "code_3": None}
        assert pp._count_codes(claim, ["code_1", "code_2", "code_3"]) == 2

    def test_ignores_nan_string(self):
        pp = _make_preprocessor()
        claim = {"code_1": "NaN", "code_2": "Z00"}
        assert pp._count_codes(claim, ["code_1", "code_2"]) == 1

    def test_empty_string_not_counted(self):
        pp = _make_preprocessor()
        claim = {"code_1": "   ", "code_2": "Z00"}
        assert pp._count_codes(claim, ["code_1", "code_2"]) == 1

    def test_all_missing_returns_zero(self):
        pp = _make_preprocessor()
        assert pp._count_codes({}, ["code_1", "code_2"]) == 0


class TestManualNormalize:
    def test_output_in_0_1_range(self):
        pp = _make_preprocessor()
        vec = np.array([50_000, 5_000, 1_000, 60, 1, 2, 12, 12,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        3, 0, 5, 3, 1, 30], dtype=np.float64)
        # Pad to N_FEATURES (27)
        if len(vec) < N_FEATURES:
            vec = np.pad(vec, (0, N_FEATURES - len(vec)))
        out = pp._manual_normalize(vec)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_zero_vector_outputs_zeros(self):
        pp = _make_preprocessor()
        vec = np.zeros(N_FEATURES)
        out = pp._manual_normalize(vec)
        assert np.all(out == 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FraudDetector Tests
# ══════════════════════════════════════════════════════════════════════════════

def _make_detector(iforest_score: float = 0.0, ae_mse: float = 0.0):
    """
    Build a FraudDetector with mocked models and preprocessor.
    iforest_score  – raw decision_function value (more negative = more anomalous)
    ae_mse         – reconstruction MSE returned by autoencoder
    """
    preprocessor = MagicMock()
    features      = np.zeros((1, N_FEATURES))
    preprocessor.preprocess.return_value = features

    iforest = MagicMock()
    iforest.decision_function.return_value = np.array([iforest_score])
    # No detector_ attribute (avoid attribute cascade)
    del iforest.detector_
    iforest.n_features_in_ = N_FEATURES

    autoencoder = MagicMock()
    autoencoder.predict.return_value = features  # perfect reconstruction by default
    del autoencoder.detector_
    autoencoder.n_features_in_ = N_FEATURES

    # Patch torch import used in _detect_features
    with patch.dict('sys.modules', {'torch': MagicMock()}):
        det = FraudDetector(autoencoder=autoencoder, iforest=iforest,
                            preprocessor=preprocessor)
    return det, preprocessor, iforest, autoencoder


class TestFraudDetectorClassifyRisk:
    def _det(self):
        d, *_ = _make_detector()
        return d

    def test_score_below_medium_is_low(self):
        assert self._det()._classify_risk(0.49) == 'LOW'

    def test_score_at_medium_threshold(self):
        assert self._det()._classify_risk(0.50) == 'MEDIUM'

    def test_score_at_high_threshold(self):
        assert self._det()._classify_risk(0.70) == 'HIGH'

    def test_score_at_critical_threshold(self):
        assert self._det()._classify_risk(0.85) == 'CRITICAL'

    def test_score_1_is_critical(self):
        assert self._det()._classify_risk(1.0) == 'CRITICAL'

    def test_score_0_is_low(self):
        assert self._det()._classify_risk(0.0) == 'LOW'


class TestFraudDetectorSlice:
    def _det(self):
        d, *_ = _make_detector()
        return d

    def test_no_slice_when_sizes_match(self):
        d = self._det()
        arr = np.ones((1, 10))
        result = d._slice(arr, 10)
        assert result.shape == (1, 10)
        np.testing.assert_array_equal(result, arr)

    def test_slices_to_smaller_size(self):
        d = self._det()
        arr = np.arange(20).reshape(1, 20)
        result = d._slice(arr, 5)
        assert result.shape == (1, 5)
        np.testing.assert_array_equal(result[0], arr[0, :5])

    def test_pads_to_larger_size(self):
        d = self._det()
        arr = np.ones((1, 3))
        result = d._slice(arr, 6)
        assert result.shape == (1, 6)
        # Padded columns should be zero
        np.testing.assert_array_equal(result[0, 3:], [0, 0, 0])

    def test_returns_as_is_when_expected_n_is_none(self):
        d = self._det()
        arr = np.ones((1, 15))
        result = d._slice(arr, None)
        assert result.shape == arr.shape


class TestFraudDetectorDetect:
    def test_returns_required_keys(self):
        det, *_ = _make_detector()
        result = det.detect(_minimal_claim())
        for key in ('claim_id', 'is_anomaly', 'anomaly_score', 'iforest_score',
                    'autoencoder_score', 'risk_level', 'analyzed_at', 'details'):
            assert key in result

    def test_claim_id_propagated(self):
        det, *_ = _make_detector()
        result = det.detect(_minimal_claim(claim_id='CLM-XYZ'))
        assert result['claim_id'] == 'CLM-XYZ'

    def test_low_score_not_anomaly(self):
        """Very positive iforest score + perfect reconstruction → low combined score."""
        det, *_ = _make_detector(iforest_score=10.0)  # sigmoid(+10) ≈ 0.00005
        result = det.detect(_minimal_claim())
        assert result['is_anomaly'] is False
        assert result['risk_level'] == 'LOW'

    def test_high_score_is_anomaly(self):
        """Very negative iforest score = high anomaly probability."""
        det, _, iforest, ae = _make_detector(iforest_score=-10.0)
        # Make the autoencoder also return a moderate MSE to avoid overflow
        ae.predict.return_value = np.ones((1, N_FEATURES)) * 5
        result = det.detect(_minimal_claim())
        assert result['is_anomaly'] is True

    def test_combined_score_weighted(self):
        """combined = 0.6 * iforest + 0.4 * ae"""
        det, *_ = _make_detector()
        # Manually pin the sub-scores by reading the result
        result = det.detect(_minimal_claim())
        expected = 0.6 * result['iforest_score'] + 0.4 * result['autoencoder_score']
        assert result['anomaly_score'] == pytest.approx(expected)

    def test_anomaly_score_is_float(self):
        det, *_ = _make_detector()
        result = det.detect(_minimal_claim())
        assert isinstance(result['anomaly_score'], float)

    def test_is_anomaly_is_bool(self):
        det, *_ = _make_detector()
        result = det.detect(_minimal_claim())
        assert isinstance(result['is_anomaly'], bool)

    def test_analyzed_at_is_datetime(self):
        from datetime import datetime
        det, *_ = _make_detector()
        result = det.detect(_minimal_claim())
        assert isinstance(result['analyzed_at'], datetime)

    def test_details_contain_essential_fields(self):
        det, *_ = _make_detector()
        result = det.detect(_minimal_claim())
        details = result['details']
        assert 'claim_amount' in details
        assert 'num_features' in details
        assert 'model_version' in details

    def test_preprocessor_called_with_claim_data(self):
        det, pp, *_ = _make_detector()
        claim = _minimal_claim()
        det.detect(claim)
        pp.preprocess.assert_called_once_with(claim)

    def test_iforest_decision_function_called(self):
        det, _, iforest, _ = _make_detector()
        det.detect(_minimal_claim())
        iforest.decision_function.assert_called_once()

    def test_iforest_fallback_score_on_error(self):
        """If decision_function raises, iforest score should default to 0.5."""
        det, _, iforest, _ = _make_detector()
        iforest.decision_function.side_effect = RuntimeError("model failure")
        result = det.detect(_minimal_claim())
        assert result['iforest_score'] == pytest.approx(0.5)

    def test_autoencoder_fallback_to_decision_function(self):
        """If predict fails, autoencoder falls back to decision_function."""
        det, _, _, ae = _make_detector()
        ae.predict.side_effect = RuntimeError("predict failed")
        ae.decision_function.return_value = np.array([0.0])
        result = det.detect(_minimal_claim())
        # Should not raise; autoencoder_score should be a float
        assert isinstance(result['autoencoder_score'], float)

    def test_autoencoder_fallback_score_on_total_failure(self):
        """Both predict and decision_function fail → default 0.5."""
        det, _, _, ae = _make_detector()
        ae.predict.side_effect = RuntimeError("predict failed")
        ae.decision_function.side_effect = RuntimeError("df failed")
        result = det.detect(_minimal_claim())
        assert result['autoencoder_score'] == pytest.approx(0.5)


class TestFraudDetectorIForestScore:
    """Unit tests for _get_iforest_score normalisation."""

    def _get_score(self, raw: float) -> float:
        det, *_ = _make_detector(iforest_score=raw)
        return det._get_iforest_score(np.zeros((1, N_FEATURES)))

    def test_very_negative_score_near_1(self):
        """Very negative raw score → high anomaly probability ≈ 1."""
        assert self._get_score(-100.0) > 0.99

    def test_zero_score_is_half(self):
        """Raw score 0 → sigmoid(0) = 0.5."""
        assert self._get_score(0.0) == pytest.approx(0.5)

    def test_very_positive_score_near_0(self):
        """Very positive raw score → low anomaly probability ≈ 0."""
        assert self._get_score(100.0) < 0.01

    def test_output_in_0_1(self):
        for raw in (-5.0, 0.0, 5.0, 10.0):
            score = self._get_score(raw)
            assert 0.0 <= score <= 1.0
