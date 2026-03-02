# Shared Package
This directory contains shared schemas used across multiple services (ingestion API, analysis worker, dashboard).

## Files
- `schemas.py`: Pydantic models for claim requests, responses, and fraud analysis results
- `__init__.py`: Package initialization

## Usage
```python
from shared.schemas import ClaimRequest, ClaimResponse, FraudAnalysisResult
```
