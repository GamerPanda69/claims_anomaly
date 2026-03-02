"""
SQLAlchemy ORM models — single source of truth for the database schema.
These replace the raw CREATE TABLE statements in database/init.sql.
"""
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey,
    Integer, Numeric, String, Text, func,
)
from sqlalchemy.dialects.postgresql import JSONB, INET
from sqlalchemy.orm import relationship

from shared.database import Base


# ── Users ──────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(100), unique=True, nullable=False, index=True)
    email         = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role          = Column(String(20), nullable=False, default="viewer")   # superuser | analyst | viewer
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, server_default=func.now())
    created_by    = Column(Integer, ForeignKey("users.id"), nullable=True)
    last_login    = Column(DateTime, nullable=True)

    # self-referential: created_by relationship (optional, lazy load)
    creator       = relationship("User", remote_side=[id], foreign_keys=[created_by])

    # back-references
    fraud_reviews = relationship("FraudAnalysis", back_populates="reviewer", foreign_keys="FraudAnalysis.reviewed_by")
    audit_entries = relationship("AuditLog",       back_populates="user",     foreign_keys="AuditLog.user_id")

    def __repr__(self):
        return f"<User id={self.id} username={self.username!r} role={self.role!r}>"


# ── Claims ─────────────────────────────────────────────────────────────────────
class Claim(Base):
    __tablename__ = "claims"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    claim_id         = Column(String(255), unique=True, nullable=False)
    beneficiary_id   = Column(String(255), nullable=False, index=True)
    provider_id      = Column(String(255), nullable=False, index=True)

    # Dates
    claim_start_date = Column(Date, nullable=True)
    claim_end_date   = Column(Date, nullable=True)

    # Financial
    claim_amount         = Column(Numeric(12, 2), nullable=False)
    deductible_amt_paid  = Column(Numeric(12, 2), nullable=True)

    # Diagnosis codes
    primary_diagnosis_code  = Column(String(10), nullable=False)
    diagnosis_code_2        = Column(String(10), nullable=True)
    diagnosis_code_3        = Column(String(10), nullable=True)
    diagnosis_code_4        = Column(String(10), nullable=True)
    diagnosis_code_5        = Column(String(10), nullable=True)
    diagnosis_code_6        = Column(String(10), nullable=True)
    diagnosis_code_7        = Column(String(10), nullable=True)
    diagnosis_code_8        = Column(String(10), nullable=True)
    diagnosis_code_9        = Column(String(10), nullable=True)
    diagnosis_code_10       = Column(String(10), nullable=True)
    admit_diagnosis_code    = Column(String(10), nullable=True)

    # Procedure codes
    procedure_code_1 = Column(String(10), nullable=True)
    procedure_code_2 = Column(String(10), nullable=True)
    procedure_code_3 = Column(String(10), nullable=True)
    procedure_code_4 = Column(String(10), nullable=True)
    procedure_code_5 = Column(String(10), nullable=True)
    procedure_code_6 = Column(String(10), nullable=True)

    # Physicians
    attending_physician = Column(String(20), nullable=True)
    operating_physician = Column(String(20), nullable=True)
    other_physician     = Column(String(20), nullable=True)

    # Demographics
    age    = Column(Integer, nullable=False)
    gender = Column(Integer, nullable=True)
    race   = Column(Integer, nullable=True)
    state  = Column(Integer, nullable=True)
    county = Column(Integer, nullable=True)

    # Coverage
    no_of_months_part_a_cov = Column(Integer, nullable=True)
    no_of_months_part_b_cov = Column(Integer, nullable=True)

    # Chronic conditions (1 = Yes, 2 = No)
    chronic_cond_alzheimer          = Column(Integer, nullable=True)
    chronic_cond_heartfailure       = Column(Integer, nullable=True)
    chronic_cond_kidneydisease      = Column(Integer, nullable=True)
    chronic_cond_cancer             = Column(Integer, nullable=True)
    chronic_cond_obstrpulmonary     = Column(Integer, nullable=True)
    chronic_cond_depression         = Column(Integer, nullable=True)
    chronic_cond_diabetes           = Column(Integer, nullable=True)
    chronic_cond_ischemicheart      = Column(Integer, nullable=True)
    chronic_cond_osteoporasis       = Column(Integer, nullable=True)
    chronic_cond_rheumatoidarthritis = Column(Integer, nullable=True)
    chronic_cond_stroke             = Column(Integer, nullable=True)
    renal_disease_indicator         = Column(String(1), nullable=True)

    # Claim type
    claim_type = Column(String(20), nullable=True)

    # Metadata
    created_at = Column(DateTime, server_default=func.now(), index=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Back-reference to analysis
    analysis = relationship("FraudAnalysis", back_populates="claim", uselist=False,
                            cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Claim claim_id={self.claim_id!r} amount={self.claim_amount}>"


# ── Fraud Analysis ─────────────────────────────────────────────────────────────
class FraudAnalysis(Base):
    __tablename__ = "fraud_analysis"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    claim_id         = Column(String(255), ForeignKey("claims.claim_id", ondelete="CASCADE"),
                              unique=True, nullable=False)

    # Analysis results
    is_anomaly        = Column(Boolean, nullable=False, index=True)
    anomaly_score     = Column(Float, nullable=False)
    iforest_score     = Column(Float, nullable=True)
    autoencoder_score = Column(Float, nullable=True)
    risk_level        = Column(String(20), nullable=False, index=True)  # LOW|MEDIUM|HIGH|CRITICAL

    # Flexible JSON details
    details = Column(JSONB, nullable=True)

    # Review tracking
    reviewed     = Column(Boolean, default=False, index=True)
    reviewed_by  = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at  = Column(DateTime, nullable=True)
    review_notes = Column(Text, nullable=True)

    # Metadata
    analyzed_at = Column(DateTime, server_default=func.now(), index=True)

    # Relationships
    claim    = relationship("Claim",    back_populates="analysis")
    reviewer = relationship("User",     back_populates="fraud_reviews", foreign_keys=[reviewed_by])

    def __repr__(self):
        return (f"<FraudAnalysis claim_id={self.claim_id!r} "
                f"risk={self.risk_level!r} anomaly={self.is_anomaly}>")


# ── Audit Log ──────────────────────────────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_log"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    action        = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id   = Column(String(255), nullable=True)
    details       = Column(JSONB, nullable=True)
    ip_address    = Column(INET, nullable=True)
    created_at    = Column(DateTime, server_default=func.now(), index=True)

    # Relationship
    user = relationship("User", back_populates="audit_entries", foreign_keys=[user_id])

    def __repr__(self):
        return f"<AuditLog user_id={self.user_id} action={self.action!r}>"
