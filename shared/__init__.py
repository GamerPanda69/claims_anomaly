# Shared package
# Each service imports directly from sub-modules:
#   from shared.schemas  import ClaimRequest, ClaimResponse, FraudAnalysisResult
#   from shared.database import get_session, engine, Base, SessionLocal
#   from shared.models   import User, Claim, FraudAnalysis, AuditLog
