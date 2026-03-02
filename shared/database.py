"""
SQLAlchemy engine, session factory, and base for all services.
Import `engine`, `SessionLocal`, and `get_session` from here.
"""
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ── Connection URL ─────────────────────────────────────────────────────────────
def _build_url() -> str:
    host = os.getenv("DB_HOST", "db")
    port = os.getenv("DB_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "claims_db")
    user = os.getenv("POSTGRES_USER", "scholar")
    pw   = os.getenv("POSTGRES_PASSWORD", "fraud_password")
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"


# ── Engine (shared pool) ───────────────────────────────────────────────────────
engine = create_engine(
    _build_url(),
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,   # validate connections before use
    echo=False,
)

# ── Session factory ────────────────────────────────────────────────────────────
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# ── Declarative base (import this into models.py) ─────────────────────────────
Base = declarative_base()


# ── Convenience context manager ────────────────────────────────────────────────
@contextmanager
def get_session():
    """Yield a SQLAlchemy session and auto-commit / rollback on exit."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
