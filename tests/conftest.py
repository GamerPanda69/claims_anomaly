"""
conftest.py — Project-wide pytest fixtures and import-time mocks.

PROBLEM: `shared.database` is imported by `shared.models` which is imported
by `analysis_worker.*` and `ingestion_api.main`.  At import time it creates a
SQLAlchemy engine that immediately tries to resolve the DB hostname "db" and
ping it — hanging the test run when Docker is not running.

SOLUTION: Stub out the entire `psycopg2`, `shared.database`, and
`shared.models` modules before any test file is collected.  All tests in this
project then use these lightweight fakes instead of real DB connections.
"""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch
import types

# ── 1. Stub psycopg2 so SQLAlchemy doesn't even try to load the driver ────────
_psycopg2_stub = types.ModuleType("psycopg2")
sys.modules.setdefault("psycopg2", _psycopg2_stub)
sys.modules.setdefault("psycopg2.extensions", types.ModuleType("psycopg2.extensions"))
sys.modules.setdefault("psycopg2.extras",     types.ModuleType("psycopg2.extras"))

# ── 2. Stub sqlalchemy.dialects.postgresql so JSONB/INET imports don't fail ───
import sqlalchemy.dialects  # noqa: E402 — real pkg, just patching sub-modules

_pg_types = types.ModuleType("sqlalchemy.dialects.postgresql")
_pg_types.JSONB = MagicMock()
_pg_types.INET  = MagicMock()
sys.modules["sqlalchemy.dialects.postgresql"] = _pg_types

# ── 3. Stub shared.database — avoids the engine-creation + DB ping ────────────
_db_stub            = types.ModuleType("shared.database")
_db_stub.Base       = MagicMock()          # declarative_base() mock
_db_stub.engine     = MagicMock()
_db_stub.SessionLocal = MagicMock()
_get_session_mock   = MagicMock()
from contextlib import contextmanager      # noqa: E402

@contextmanager
def _fake_get_session():
    yield MagicMock()

_db_stub.get_session = _fake_get_session
import shared
sys.modules["shared.database"] = _db_stub

# ── 4. Stub shared.models (depends on shared.database.Base) ──────────────────
_models_stub                = types.ModuleType("shared.models")
_models_stub.User           = MagicMock()
_models_stub.Claim          = MagicMock()
_models_stub.FraudAnalysis  = MagicMock()
_models_stub.AuditLog       = MagicMock()
sys.modules["shared.models"] = _models_stub
