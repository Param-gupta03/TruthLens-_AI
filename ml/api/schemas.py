from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="Claim text to verify")
    evidence: List[str] = Field(..., min_items=1, description="List of candidate evidence passages")
    threshold: Optional[float] = Field(None, description="Optional custom decision threshold for Model A")


class EvidenceResult(BaseModel):
    evidence: str
    relevanceLabel: str
    relevanceScore: float


class VerificationProbabilities(BaseModel):
    SUPPORTS: float
    REFUTES: float
    NOT_ENOUGH_INFO: float


class IndividualVerification(BaseModel):
    evidence: str
    relevanceScore: float
    label: str
    confidence: float
    probabilities: VerificationProbabilities


class VerificationResult(BaseModel):
    label: str
    confidence: float
    probabilities: VerificationProbabilities
    individualVerifications: Optional[List[IndividualVerification]] = []


class PredictResponse(BaseModel):
    claim: str
    evidenceResults: List[EvidenceResult]
    verification: VerificationResult


class HealthResponse(BaseModel):
    status: str
    service: str
    cuda: bool
    device: str
    modelA_loaded: bool = True
    modelB_loaded: bool = True

