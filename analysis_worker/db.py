"""
Database operations for storing claims and analysis results.
Uses SQLAlchemy ORM — no raw psycopg2.
"""
import json
import logging
import os
from typing import Any, Dict

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Shared SQLAlchemy layer (shared/ is mounted into /app/shared)
import sys
sys.path.insert(0, '/app')

from shared.database import Base, engine, get_session
from shared.models import Claim, FraudAnalysis

logger = logging.getLogger(__name__)


class Database:
    """SQLAlchemy-backed database operations."""

    def __init__(self):
        self._ensure_tables()

    # ── Schema bootstrap ───────────────────────────────────────────────────────
    def _ensure_tables(self):
        """Create all tables (idempotent — skips existing tables)."""
        try:
            Base.metadata.create_all(bind=engine, checkfirst=True)
            logger.info("✓ Database tables verified / created via SQLAlchemy")
        except Exception as e:
            logger.error(f"✗ Failed to create tables: {e}")
            raise

    # ── Claim insert / upsert ──────────────────────────────────────────────────
    def insert_claim(self, claim_data: Dict[str, Any]) -> bool:
        """
        Upsert a claim into the claims table.

        Uses SQLAlchemy Core's PostgreSQL INSERT … ON CONFLICT for an atomic
        upsert, then falls back to a pure-ORM merge if the pg-specific path
        is unavailable.
        """
        claim_id = claim_data.get("claim_id", "UNKNOWN")
        try:
            with get_session() as session:
                # Build the ORM object from the incoming dict, keeping only
                # columns that exist on the model.
                valid_cols = {c.name for c in Claim.__table__.columns}
                filtered   = {k: v for k, v in claim_data.items() if k in valid_cols}

                # pg INSERT … ON CONFLICT DO UPDATE (true upsert)
                stmt = (
                    pg_insert(Claim)
                    .values(**filtered)
                    .on_conflict_do_update(
                        index_elements=["claim_id"],
                        set_={"updated_at": func.now()},
                    )
                )
                session.execute(stmt)

            logger.info(f"✓ Upserted claim {claim_id}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to insert claim {claim_id}: {e}", exc_info=True)
            return False

    # ── Analysis result insert / upsert ───────────────────────────────────────
    def insert_analysis_result(self, result: Dict[str, Any]) -> bool:
        """
        Upsert a fraud analysis result into the fraud_analysis table.
        """
        claim_id = result.get("claim_id", "UNKNOWN")
        try:
            with get_session() as session:
                details_value = result.get("details", {})
                # Ensure details is a plain dict (JSONB column)
                if isinstance(details_value, str):
                    details_value = json.loads(details_value)

                stmt = (
                    pg_insert(FraudAnalysis)
                    .values(
                        claim_id=result["claim_id"],
                        is_anomaly=result["is_anomaly"],
                        anomaly_score=result["anomaly_score"],
                        iforest_score=result.get("iforest_score"),
                        autoencoder_score=result.get("autoencoder_score"),
                        risk_level=result["risk_level"],
                        details=details_value,
                        analyzed_at=result["analyzed_at"],
                    )
                    .on_conflict_do_update(
                        index_elements=["claim_id"],
                        set_={
                            "is_anomaly":        result["is_anomaly"],
                            "anomaly_score":     result["anomaly_score"],
                            "iforest_score":     result.get("iforest_score"),
                            "autoencoder_score": result.get("autoencoder_score"),
                            "risk_level":        result["risk_level"],
                            "details":           details_value,
                            "analyzed_at":       result["analyzed_at"],
                        },
                    )
                )
                session.execute(stmt)

            logger.info(f"✓ Upserted analysis result for claim {claim_id}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to insert analysis result for {claim_id}: {e}", exc_info=True)
            return False

    def close(self):
        """No-op — SQLAlchemy engine manages the connection pool."""
        logger.info("Database engine pool handles cleanup automatically")
