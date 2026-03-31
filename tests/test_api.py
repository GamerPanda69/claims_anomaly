"""
Unit tests for the Healthcare Claims Ingestion API (ingestion_api/main.py)
and shared Pydantic schemas (shared/schemas.py).

Run with:
    pytest tests/test_api.py -v
"""

import json
import sys
import os
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

# ── Path setup ─────────────────────────────────────────────────────────────────
# Ensure the project root is on PYTHONPATH so imports work without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_redis():
    """A fake Redis client wired into the API module."""
    r = MagicMock()
    r.ping.return_value = True
    r.llen.return_value = 0
    r.lpush.return_value = 1
    return r


@pytest.fixture(scope="module")
def api_client(mock_redis):
    """TestClient with Redis dependency patched out."""
    # Patch redis.from_url BEFORE importing main so the module-level redis_client
    # is already a mock when the app starts.
    with patch("redis.from_url", return_value=mock_redis):
        from ingestion_api.main import app
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


@pytest.fixture
def valid_claim_payload():
    """Minimal valid claim payload (uses field aliases for convenience)."""
    return {
        "claim_id":   "CLM-TEST-001",
        "user_id":    "BENE-001",        # alias → beneficiary_id
        "provider_id":"PROV-001",
        "amount":     1500.00,           # alias → claim_amount
        "icd_code":   "Z00.00",          # alias → primary_diagnosis_code
        "age":        45,
        "claim_start_date": "2024-01-01",
        "claim_end_date":   "2024-01-05",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. Pydantic Schema Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestClaimRequestSchema:
    """Tests for shared.schemas.ClaimRequest"""

    from shared.schemas import ClaimRequest  # module-level import for class body

    def _make(self, **overrides):
        from shared.schemas import ClaimRequest
        base = {
            "claim_id":   "CLM-001",
            "user_id":    "BENE-001",
            "provider_id":"PROV-001",
            "amount":     500.0,
            "icd_code":   "Z00.00",
            "age":        30,
        }
        base.update(overrides)
        return ClaimRequest(**base)

    # ── Happy-path ────────────────────────────────────────────────────────────

    def test_valid_minimal_claim(self):
        claim = self._make()
        assert claim.claim_id == "CLM-001"
        assert claim.beneficiary_id == "BENE-001"
        assert claim.claim_amount == 500.0
        assert claim.primary_diagnosis_code == "Z00.00"

    def test_field_aliases_mapped_correctly(self):
        """Aliases (user_id, amount, icd_code) must map to canonical field names."""
        claim = self._make(amount=999.99, icd_code="E11.9")
        assert claim.claim_amount == 999.99
        assert claim.primary_diagnosis_code == "E11.9"

    def test_optional_fields_default_values(self):
        claim = self._make()
        # Chronic conditions default to 2 (No)
        assert claim.chronic_cond_diabetes == 2
        assert claim.chronic_cond_cancer == 2
        # Coverage defaults
        assert claim.no_of_months_part_a_cov == 12
        assert claim.no_of_months_part_b_cov == 12
        # Claim type default
        assert claim.claim_type == "OUTPATIENT"

    def test_all_diagnosis_codes_optional(self):
        claim = self._make()
        assert claim.diagnosis_code_2 is None
        assert claim.diagnosis_code_10 is None

    def test_all_procedure_codes_optional(self):
        claim = self._make()
        assert claim.procedure_code_1 is None
        assert claim.procedure_code_6 is None

    def test_physicians_optional(self):
        claim = self._make(attending_physician="P-123", operating_physician="P-456")
        assert claim.attending_physician == "P-123"
        assert claim.other_physician is None

    # ── Validation errors ─────────────────────────────────────────────────────

    def test_zero_amount_rejected(self):
        with pytest.raises(ValidationError):
            self._make(amount=0)

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError):
            self._make(amount=-100)

    def test_amount_exceeding_limit_rejected(self):
        with pytest.raises(ValidationError):
            self._make(amount=1_500_000)

    def test_negative_age_rejected(self):
        with pytest.raises(ValidationError):
            self._make(age=-1)

    def test_age_above_120_rejected(self):
        with pytest.raises(ValidationError):
            self._make(age=121)

    def test_extreme_boundary_age_valid(self):
        claim_zero = self._make(age=0)
        assert claim_zero.age == 0
        claim_max  = self._make(age=120)
        assert claim_max.age == 120

    def test_gender_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            self._make(gender=3)

    def test_chronic_cond_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            self._make(chronic_cond_diabetes=3)  # must be 1 or 2

    def test_missing_required_fields_rejected(self):
        with pytest.raises(ValidationError):
            from shared.schemas import ClaimRequest
            ClaimRequest(claim_id="X", age=30)   # many required fields missing

    def test_empty_claim_id_rejected(self):
        with pytest.raises(ValidationError):
            self._make(claim_id="")

    def test_renal_disease_indicator_default(self):
        claim = self._make()
        assert claim.renal_disease_indicator == "0"


# ══════════════════════════════════════════════════════════════════════════════
# 2. FastAPI Endpoint Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_returns_200(self, api_client, mock_redis):
        mock_redis.llen.return_value = 2
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, api_client, mock_redis):
        mock_redis.llen.return_value = 5
        data = api_client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["service"] == "ingestion_api"
        assert data["redis"] == "connected"
        assert "queue_size" in data
        assert "timestamp" in data

    def test_health_reports_correct_queue_size(self, api_client, mock_redis):
        mock_redis.llen.return_value = 99
        data = api_client.get("/health").json()
        assert data["queue_size"] == 99

    def test_health_unhealthy_when_redis_fails(self, api_client, mock_redis):
        """When Redis raises, /health should return 503."""
        import redis as redis_lib
        mock_redis.ping.side_effect = redis_lib.ConnectionError("down")
        response = api_client.get("/health")
        assert response.status_code == 503
        # Restore for subsequent tests
        mock_redis.ping.side_effect = None
        mock_redis.ping.return_value = True


class TestRootEndpoint:
    """Tests for GET /"""

    def test_root_returns_200(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200

    def test_root_contains_service_info(self, api_client):
        data = api_client.get("/").json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data


class TestIngestEndpoint:
    """Tests for POST /ingest"""

    def test_successful_ingest(self, api_client, mock_redis, valid_claim_payload):
        mock_redis.llen.return_value = 1
        response = api_client.post("/ingest", json=valid_claim_payload)
        assert response.status_code == 200

    def test_ingest_response_body(self, api_client, mock_redis, valid_claim_payload):
        mock_redis.llen.return_value = 1
        data = api_client.post("/ingest", json=valid_claim_payload).json()
        assert data["status"] == "success"
        assert data["claim_id"] == valid_claim_payload["claim_id"]
        assert "queued_at" in data
        assert "message" in data

    def test_ingest_pushes_to_redis(self, api_client, mock_redis, valid_claim_payload):
        mock_redis.reset_mock()
        api_client.post("/ingest", json=valid_claim_payload)
        mock_redis.lpush.assert_called_once()
        queue_name = mock_redis.lpush.call_args[0][0]
        assert queue_name == "claims_queue"

    def test_ingest_queued_data_contains_claim_id(
        self, api_client, mock_redis, valid_claim_payload
    ):
        mock_redis.reset_mock()
        api_client.post("/ingest", json=valid_claim_payload)
        _, pushed_json = mock_redis.lpush.call_args[0]
        pushed = json.loads(pushed_json)
        assert pushed["claim_id"] == valid_claim_payload["claim_id"]

    def test_ingest_invalid_payload_returns_422(self, api_client):
        """Missing required fields → validation error from FastAPI."""
        response = api_client.post("/ingest", json={"claim_id": "X"})
        assert response.status_code == 422

    def test_ingest_zero_amount_returns_422(self, api_client, valid_claim_payload):
        bad = {**valid_claim_payload, "amount": 0}
        response = api_client.post("/ingest", json=bad)
        assert response.status_code == 422

    def test_ingest_negative_age_returns_422(self, api_client, valid_claim_payload):
        bad = {**valid_claim_payload, "age": -5}
        response = api_client.post("/ingest", json=bad)
        assert response.status_code == 422

    def test_ingest_redis_connection_error_returns_503(
        self, api_client, mock_redis, valid_claim_payload
    ):
        import redis as redis_lib
        mock_redis.lpush.side_effect = redis_lib.ConnectionError("down")
        response = api_client.post("/ingest", json=valid_claim_payload)
        assert response.status_code == 503
        mock_redis.lpush.side_effect = None   # restore

    def test_ingest_multiple_claims_increment_queue(
        self, api_client, mock_redis, valid_claim_payload
    ):
        mock_redis.llen.side_effect = [1, 2, 3]
        for i in range(3):
            payload = {**valid_claim_payload, "claim_id": f"CLM-MULTI-{i}"}
            resp = api_client.post("/ingest", json=payload)
            assert resp.status_code == 200
        mock_redis.llen.side_effect = None
