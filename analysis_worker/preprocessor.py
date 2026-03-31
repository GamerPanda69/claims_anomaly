"""
Preprocessor - Transform claim data into features for ML models
"""
import numpy as np
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ClaimPreprocessor:
    """Transforms raw claim data into ML model input features"""

    def __init__(self, scaler, clinical_norms):
        self.scaler = scaler
        self.clinical_norms = clinical_norms
        self.feature_names = self._get_feature_names()

        # Detect how many features the scaler was actually trained on
        try:
            self._scaler_features = int(self.scaler.n_features_in_)
        except AttributeError:
            try:
                self._scaler_features = len(self.scaler.mean_)
            except Exception:
                self._scaler_features = None

        logger.info(
            f"Preprocessor ready: {len(self.feature_names)} features built, "
            f"scaler expects {self._scaler_features}"
        )

    def _get_feature_names(self) -> List[str]:
        """Define the expected feature columns for the model"""
        return [
            # Financial features
            'claim_amount',
            'deductible_amt_paid',
            'amount_per_day',

            # Age and demographics
            'age',
            'gender',
            'race',

            # Coverage duration
            'no_of_months_part_a_cov',
            'no_of_months_part_b_cov',

            # Chronic conditions (11 conditions)
            'chronic_cond_alzheimer',
            'chronic_cond_heartfailure',
            'chronic_cond_kidneydisease',
            'chronic_cond_cancer',
            'chronic_cond_obstrpulmonary',
            'chronic_cond_depression',
            'chronic_cond_diabetes',
            'chronic_cond_ischemicheart',
            'chronic_cond_osteoporasis',
            'chronic_cond_rheumatoidarthritis',
            'chronic_cond_stroke',

            # Derived features
            'total_chronic_conditions',
            'renal_disease_indicator_num',
            'num_diagnosis_codes',
            'num_procedure_codes',
            'has_multiple_physicians',
            'claim_duration_days',
        ]

    def preprocess(self, claim_data: Dict[str, Any]) -> np.ndarray:
        """
        Transform claim dictionary into feature vector.
        Safe defaults are applied for any missing / None fields so the
        preprocessor never crashes on optional claim fields.
        """
        try:
            features = {}

            # ── Financial ────────────────────────────────────────────────────
            features['claim_amount']        = _to_float(claim_data.get('claim_amount'), 0.0)
            features['deductible_amt_paid'] = _to_float(claim_data.get('deductible_amt_paid'), 0.0)

            claim_duration = self._calculate_duration(
                claim_data.get('claim_start_date'),
                claim_data.get('claim_end_date'),
            )
            features['claim_duration_days'] = claim_duration
            features['amount_per_day']      = features['claim_amount'] / max(claim_duration, 1)

            # ── Demographics ─────────────────────────────────────────────────
            features['age']    = _to_int(claim_data.get('age'),    0)
            features['gender'] = _to_int(claim_data.get('gender'), 0)
            features['race']   = _to_int(claim_data.get('race'),   0)

            # ── Coverage ─────────────────────────────────────────────────────
            features['no_of_months_part_a_cov'] = _to_int(claim_data.get('no_of_months_part_a_cov'), 12)
            features['no_of_months_part_b_cov'] = _to_int(claim_data.get('no_of_months_part_b_cov'), 12)

            # ── Chronic conditions (1=Yes/2=No → 1/0) ────────────────────────
            chronic_conditions = [
                'chronic_cond_alzheimer',
                'chronic_cond_heartfailure',
                'chronic_cond_kidneydisease',
                'chronic_cond_cancer',
                'chronic_cond_obstrpulmonary',
                'chronic_cond_depression',
                'chronic_cond_diabetes',
                'chronic_cond_ischemicheart',
                'chronic_cond_osteoporasis',
                'chronic_cond_rheumatoidarthritis',
                'chronic_cond_stroke',
            ]
            for cond in chronic_conditions:
                raw = _to_int(claim_data.get(cond), 2)  # default 2 = No
                features[cond] = 1 if raw == 1 else 0

            features['total_chronic_conditions'] = sum(features[c] for c in chronic_conditions)

            # ── Renal ─────────────────────────────────────────────────────────
            renal = claim_data.get('renal_disease_indicator', '0')
            features['renal_disease_indicator_num'] = 1 if str(renal) in ('1', 'Y') else 0

            # ── Code counts ──────────────────────────────────────────────────
            features['num_diagnosis_codes'] = self._count_codes(
                claim_data,
                ['primary_diagnosis_code'] + [f'diagnosis_code_{i}' for i in range(2, 11)],
            )
            features['num_procedure_codes'] = self._count_codes(
                claim_data,
                [f'procedure_code_{i}' for i in range(1, 7)],
            )

            # ── Physicians ───────────────────────────────────────────────────
            physicians     = [claim_data.get(k) for k in
                              ('attending_physician', 'operating_physician', 'other_physician')]
            unique_phys    = len([p for p in physicians if p and str(p).strip()])
            features['has_multiple_physicians'] = 1 if unique_phys > 1 else 0

            # ── Assemble feature vector ───────────────────────────────────────
            feature_vector = np.array(
                [features[name] for name in self.feature_names],
                dtype=np.float64,
            )

            # ── Scale only if the scaler dimensions match ─────────────────────
            if self._scaler_features is not None and self._scaler_features == len(feature_vector):
                feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
                logger.debug("Applied StandardScaler")
            else:
                # Scaler was trained on a different feature set — normalise manually
                # so models still get reasonably-scaled input.
                logger.debug(
                    f"Scaler feature mismatch ({self._scaler_features} vs "
                    f"{len(feature_vector)}); using manual min-max normalisation"
                )
                feature_vector_scaled = self._manual_normalize(feature_vector).reshape(1, -1)

            logger.debug(
                f"Preprocessed claim {claim_data.get('claim_id')} "
                f"→ {feature_vector_scaled.shape[1]} features"
            )
            return feature_vector_scaled

        except Exception as e:
            logger.error(f"Error preprocessing claim: {e}", exc_info=True)
            raise

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _manual_normalize(self, vec: np.ndarray) -> np.ndarray:
        """
        Lightweight per-feature min-max normalisation using known typical ranges.
        Used as a fallback when the saved scaler dimensions don't match.
        """
        # (min, max) per feature in self.feature_names order
        ranges = [
            (0, 100_000),   # claim_amount
            (0, 50_000),    # deductible_amt_paid
            (0, 10_000),    # amount_per_day
            (0, 120),       # age
            (0, 2),         # gender
            (0, 5),         # race
            (0, 12),        # no_of_months_part_a_cov
            (0, 12),        # no_of_months_part_b_cov
            # 11 chronic conditions (binary)
            *[(0, 1)] * 11,
            (0, 11),        # total_chronic_conditions
            (0, 1),         # renal_disease_indicator_num
            (0, 10),        # num_diagnosis_codes
            (0, 6),         # num_procedure_codes
            (0, 1),         # has_multiple_physicians
            (0, 365),       # claim_duration_days
        ]
        out = np.empty_like(vec, dtype=np.float64)
        for i, (lo, hi) in enumerate(ranges):
            span = hi - lo
            out[i] = (vec[i] - lo) / span if span > 0 else 0.0
        return np.clip(out, 0, 1)

    def _calculate_duration(self, start_date_str, end_date_str) -> int:
        """Calculate duration between two dates in days"""
        if not start_date_str or not end_date_str:
            return 1
        try:
            for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y%m%d'):
                try:
                    start = datetime.strptime(str(start_date_str), fmt)
                    end   = datetime.strptime(str(end_date_str),   fmt)
                    return max((end - start).days, 1)
                except ValueError:
                    continue
        except Exception:
            pass
        return 1

    def _count_codes(self, claim_data: Dict, field_names: List[str]) -> int:
        """Count non-empty codes"""
        return sum(
            1 for f in field_names
            if claim_data.get(f) and str(claim_data[f]).strip() and str(claim_data[f]).strip().upper() != 'NAN'
        )


# ── Module-level safe-cast helpers ────────────────────────────────────────────
def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value)) if value is not None else default
    except (ValueError, TypeError):
        return default
