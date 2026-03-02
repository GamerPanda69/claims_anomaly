from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from shared.schemas import ClaimRequest, ClaimResponse
import redis
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare Claims Ingestion API",
    description="API for ingesting healthcare claims for fraud detection",
    version="1.0.0"
)

# Add CORS middleware for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup"""
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info(f"✓ Connected to Redis at {REDIS_URL}")
    except Exception as e:
        logger.error(f"✗ Failed to connect to Redis: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on shutdown"""
    if redis_client:
        redis_client.close()
        logger.info("✓ Redis connection closed")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_client.ping()
        queue_size = redis_client.llen('claims_queue')
        
        return {
            "status": "healthy",
            "service": "ingestion_api",
            "redis": "connected",
            "queue_size": queue_size,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/ingest", response_model=ClaimResponse)
async def ingest_claim(claim: ClaimRequest):
    """
    Ingest a healthcare claim for fraud detection
    
    The claim will be queued for asynchronous analysis by the worker service.
    """
    try:
        # Convert Pydantic model to JSON
        claim_data = claim.model_dump(by_alias=False)  # Use field names, not aliases
        claim_json = json.dumps(claim_data, default=str)
        
        # Push to Redis queue (left push, so worker can right pop for FIFO)
        redis_client.lpush("claims_queue", claim_json)
        
        queue_size = redis_client.llen("claims_queue")
        
        logger.info(
            f"✓ Claim {claim.claim_id} queued for analysis "
            f"(queue size: {queue_size})"
        )
        
        return ClaimResponse(
            status="success",
            message=f"Claim {claim.claim_id} queued for analysis",
            claim_id=claim.claim_id,
            queued_at=datetime.now()
        )
        
    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue service unavailable"
        )
    except Exception as e:
        logger.error(f"Error queuing claim: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Healthcare Claims Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ingest": "Submit a claim for fraud detection",
            "GET /health": "Check API health status",
            "GET /docs": "Interactive API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)