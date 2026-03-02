from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List

class ClaimRequest(BaseModel):
    """Extended claim schema matching training data structure"""
    # Basic claim info
    claim_id: str = Field(..., min_length=1, max_length=255)
    beneficiary_id: str = Field(..., min_length=1, max_length=255, alias="user_id")
    provider_id: str = Field(..., min_length=1, max_length=255)
    
    # Dates
    claim_start_date: Optional[str] = None
    claim_end_date: Optional[str] = None
    
    # Financial
    claim_amount: float = Field(..., gt=0, alias="amount")
    deductible_amt_paid: Optional[float] = Field(default=0.0, ge=0)
    
    # Diagnosis codes (primary + additional)
    primary_diagnosis_code: str = Field(..., alias="icd_code")
    diagnosis_code_2: Optional[str] = None
    diagnosis_code_3: Optional[str] = None
    diagnosis_code_4: Optional[str] = None
    diagnosis_code_5: Optional[str] = None
    diagnosis_code_6: Optional[str] = None
    diagnosis_code_7: Optional[str] = None
    diagnosis_code_8: Optional[str] = None
    diagnosis_code_9: Optional[str] = None
    diagnosis_code_10: Optional[str] = None
    admit_diagnosis_code: Optional[str] = None
    
    # Procedure codes
    procedure_code_1: Optional[str] = None
    procedure_code_2: Optional[str] = None
    procedure_code_3: Optional[str] = None
    procedure_code_4: Optional[str] = None
    procedure_code_5: Optional[str] = None
    procedure_code_6: Optional[str] = None
    
    # Physicians
    attending_physician: Optional[str] = None
    operating_physician: Optional[str] = None
    other_physician: Optional[str] = None
    
    # Beneficiary demographics
    age: int = Field(..., ge=0, le=120)
    gender: Optional[int] = Field(default=None, ge=1, le=2)  # 1=Male, 2=Female
    race: Optional[int] = Field(default=None, ge=1, le=5)
    state: Optional[int] = None
    county: Optional[int] = None
    
    # Coverage
    no_of_months_part_a_cov: Optional[int] = Field(default=12, ge=0, le=12)
    no_of_months_part_b_cov: Optional[int] = Field(default=12, ge=0, le=12)
    
    # Chronic conditions (1=Yes, 2=No)
    chronic_cond_alzheimer: Optional[int] = Field(default=2, ge=1, le=2)
    chronic_cond_heartfailure: Optional[int] = Field(default=2, ge=1, le=2)
    chronic_cond_kidneydisease: Optional[int] = Field(default=2, ge=1, le=2)
    chronic_cond_cancer: Optional[int] = Field(default=2, ge=1, le=2)
    chronic_cond_obstrpulmonary: Optional[int] = Field(default=2, ge=1, le=2)
    chronic_cond_depression: Optional[int] = Field(default=2, ge=1, le=2)
    chronic_cond_diabetes: Optional[int] = Field(default=2, ge=1, le=2)
    chronic_cond_ischemicheart: Optional[int] = Field(default=2, ge=1, le=2)
    chronic_cond_osteoporasis: Optional[int] = Field(default=2, ge=1, le=2)
    chronic_cond_rheumatoidarthritis: Optional[int] = Field(default=2, ge=1, le=2)
    chronic_cond_stroke: Optional[int] = Field(default=2, ge=1, le=2)
    renal_disease_indicator: Optional[str] = Field(default="0")
    
    # Claim type
    claim_type: Optional[str] = Field(default="OUTPATIENT")  # INPATIENT or OUTPATIENT
    
    class Config:
        populate_by_name = True  # Allow both field name and alias
    
    @field_validator('claim_amount')
    @classmethod
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Claim amount must be positive')
        if v > 1000000:  # Sanity check
            raise ValueError('Claim amount exceeds reasonable limit')
        return v
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v


class ClaimResponse(BaseModel):
    """Response after ingesting a claim"""
    status: str
    message: str
    claim_id: str
    queued_at: datetime


class FraudAnalysisResult(BaseModel):
    """Results from fraud analysis"""
    claim_id: str
    is_anomaly: bool
    anomaly_score: float
    iforest_score: Optional[float] = None
    autoencoder_score: Optional[float] = None
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    analyzed_at: datetime
    details: Optional[dict] = None