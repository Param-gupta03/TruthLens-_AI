import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure api directory is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemas import PredictRequest, PredictResponse, HealthResponse
from service import MLInferenceService

service_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service_instance
    # Pre-load models once during application startup into GPU memory
    service_instance = MLInferenceService.get_instance()
    yield
    # Cleanup if needed on shutdown


app = FastAPI(
    title="TruthLens ML Service",
    description="Inference service for Evidence Relevance (Model A) and Claim Verification (Model B)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local cross-service communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    svc = MLInferenceService.get_instance()
    model_a_ok = svc.pipeline.model_a is not None and svc.pipeline.model_a.model is not None
    model_b_ok = svc.pipeline.model_b is not None and svc.pipeline.model_b.model is not None
    return HealthResponse(
        status="healthy",
        service="TruthLens ML Service",
        cuda=svc.cuda_available,
        device=svc.device_name,
        modelA_loaded=model_a_ok,
        modelB_loaded=model_b_ok
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.claim.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Claim cannot be empty."
        )
    if not request.evidence:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Evidence list cannot be empty."
        )

    try:
        svc = MLInferenceService.get_instance()
        result = svc.predict(claim=request.claim, evidence=request.evidence, threshold=request.threshold)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
