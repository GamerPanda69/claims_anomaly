# Healthcare Claims Fraud Detection System

## Quick Start

### 1. Start all services
```bash
docker-compose up -d
```

### 2. Check service health
```bash
# Check all services are running
docker-compose ps

# Check API health
curl http://localhost:8080/health
```

### 3. Submit a test claim
```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "TEST001",
    "user_id": "BENE123",
    "provider_id": "PRV456",
    "amount": 5000.00,
    "icd_code": "E11",
    "age": 65
  }'
```

### 4. Access Dashboard
Open browser to http://localhost:8501

**Default login credentials:**
- Username: `admin`
- Password: `admin123`

⚠️ **IMPORTANT**: Change the default password after first login!

### 5. Monitor worker logs
```bash
docker logs fraud_analysis_worker -f
```

## Services

- **Ingestion API**: http://localhost:8080
  - `POST /ingest` - Submit claims
  - `GET /health` - Health check
  - `GET /docs` - API documentation
  
- **Dashboard**: http://localhost:8501
  - View fraud detection results
  - Manage users (superuser only)
  - Monitor statistics

- **PostgreSQL**: localhost:5432 (internal)
- **Redis**: localhost:6379 (internal)

## Architecture

```
┌─────────────┐      ┌───────┐      ┌─────────────────┐      ┌──────────────┐
│Ingestion API│─────▶│ Redis │─────▶│ Analysis Worker │─────▶│  PostgreSQL  │
│  (FastAPI)  │      │ Queue │      │   (ML Models)   │      │   Database   │
└─────────────┘      └───────┘      └─────────────────┘      └──────────────┘
                                                                       │
                                                                       ▼
                                                              ┌─────────────────┐
                                                              │   Dashboard     │
                                                              │  (Streamlit)    │
                                                              └─────────────────┘
```

## Environment Variables

Edit `.env` file for configuration:
```bash
POSTGRES_USER=scholar
POSTGRES_PASSWORD=fraud_password
POSTGRES_DB=claims_db
REDIS_URL=redis://redis:6379
```

## Stopping Services

```bash
docker-compose down
```

To remove all data:
```bash
docker-compose down -v
```
