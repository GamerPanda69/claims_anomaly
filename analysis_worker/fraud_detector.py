"""
Fraud Detector - Core anomaly detection using ensemble of models
"""
import numpy as np
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class FraudDetector:
    """Ensemble fraud detection using AutoEncoder and Isolation Forest"""

    def __init__(self, autoencoder, iforest, preprocessor):
        self.autoencoder   = autoencoder
        self.iforest       = iforest
        self.preprocessor  = preprocessor

        # Detect how many features each saved model actually expects so we can
        # slice the feature vector before passing it in.
        self._iforest_n   = self._detect_features(iforest,     'iforest')
        self._ae_n        = self._detect_features(autoencoder, 'autoencoder')

        logger.info(f"FraudDetector: iforest expects {self._iforest_n} features, "
                    f"autoencoder expects {self._ae_n} features")

        self.thresholds = {
            'critical': 0.85,
            'high':     0.70,
            'medium':   0.50,
        }

    # ── Feature detection ──────────────────────────────────────────────────────

    def _detect_features(self, model, name: str):
        """Return the number of input features the model expects, or None."""
        # PyOD wrappers expose inner sklearn model via .detector_
        inner = getattr(model, 'detector_', model)
        for attr in ('n_features_in_', 'n_features_'):
            val = getattr(inner, attr, None)
            if val is not None:
                logger.info(f"{name}: detected {val} expected features (via {attr})")
                return int(val)
        # AutoEncoder: check input layer size via torch model
        try:
            net = getattr(model, 'model_', None)
            if net is not None:
                import torch
                dummy = torch.zeros(1, 1)
                # Find first Linear layer's in_features
                for m in net.modules():
                    if hasattr(m, 'in_features'):
                        logger.info(f"{name}: detected {m.in_features} expected features (via Linear layer)")
                        return m.in_features
        except Exception:
            pass
        logger.warning(f"{name}: could not detect expected feature count")
        return None

    def _slice(self, features: np.ndarray, expected_n) -> np.ndarray:
        """
        Slice the feature vector to match the saved model's expected input width.
        If no slice is needed (or unknown), return as-is.
        """
        if expected_n is None:
            return features
        n = features.shape[1]
        if n == expected_n:
            return features
        if n > expected_n:
            logger.debug(f"Slicing feature vector from {n} → {expected_n}")
            return features[:, :expected_n]
        # Pad with zeros if somehow fewer features than expected
        logger.debug(f"Padding feature vector from {n} → {expected_n}")
        pad = np.zeros((features.shape[0], expected_n - n))
        return np.hstack([features, pad])

    # ── Main detection ─────────────────────────────────────────────────────────

    def detect(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        claim_id = claim_data.get('claim_id', 'UNKNOWN')
        logger.info(f"Analyzing claim {claim_id}")

        try:
            features = self.preprocessor.preprocess(claim_data)

            iforest_score     = self._get_iforest_score(features)
            autoencoder_score = self._get_autoencoder_score(features)

            # 60 % IForest + 40 % AutoEncoder
            combined_score = (0.6 * iforest_score) + (0.4 * autoencoder_score)
            is_anomaly     = combined_score >= self.thresholds['medium']
            risk_level     = self._classify_risk(combined_score)

            result = {
                'claim_id':         claim_id,
                'is_anomaly':       bool(is_anomaly),
                'anomaly_score':    float(combined_score),
                'iforest_score':    float(iforest_score),
                'autoencoder_score':float(autoencoder_score),
                'risk_level':       risk_level,
                'analyzed_at':      datetime.now(),
                'details': {
                    'claim_amount':  claim_data.get('claim_amount'),
                    'beneficiary_id':claim_data.get('beneficiary_id'),
                    'provider_id':   claim_data.get('provider_id'),
                    'num_features':  features.shape[1],
                    'model_version': '1.0',
                },
            }

            logger.info(f"Claim {claim_id}: risk={risk_level}, "
                        f"score={combined_score:.4f}, anomaly={is_anomaly}")
            return result

        except Exception as e:
            logger.error(f"Error detecting fraud for claim {claim_id}: {e}", exc_info=True)
            raise

    # ── Model scorers ──────────────────────────────────────────────────────────

    def _get_iforest_score(self, features: np.ndarray) -> float:
        try:
            f = self._slice(features, self._iforest_n)
            decision_score   = self.iforest.decision_function(f)
            # decision_function returns per-sample scores: more negative = more anomalous
            score = float(np.asarray(decision_score).ravel()[0])
            # Map to [0, 1]: sigmoid such that very negative → 1 (anomalous)
            normalized = 1.0 / (1.0 + np.exp(score))
            return float(normalized)
        except Exception as e:
            logger.error(f"IForest scoring error: {e}")
            return 0.5

    def _get_autoencoder_score(self, features: np.ndarray) -> float:
        f = self._slice(features, self._ae_n)
        # Try predict (reconstruct) → MSE
        try:
            reconstruction = self.autoencoder.predict(f)
            mse            = float(np.mean(np.square(f - reconstruction)))
            normalized     = 1.0 - 1.0 / (1.0 + np.exp(mse - 0.5))
            return float(np.clip(normalized, 0.0, 1.0))
        except Exception:
            pass
        # Fallback: decision_function (per-sample anomaly score from pyod)
        try:
            score      = float(np.asarray(self.autoencoder.decision_function(f)).ravel()[0])
            normalized = 1.0 / (1.0 + np.exp(-score))
            return float(np.clip(normalized, 0.0, 1.0))
        except Exception as e:
            logger.error(f"AutoEncoder scoring error: {e}")
            return 0.5


    def _classify_risk(self, score: float) -> str:
        if score >= self.thresholds['critical']:
            return 'CRITICAL'
        if score >= self.thresholds['high']:
            return 'HIGH'
        if score >= self.thresholds['medium']:
            return 'MEDIUM'
        return 'LOW'
