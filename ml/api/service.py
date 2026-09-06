import os
import sys
from typing import Dict, Any, List, Optional
import torch

# Add ml/src to Python path to reuse existing wrappers without duplication
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fact_check_pipeline import FactCheckPipeline


class MLInferenceService:
    _instance = None

    def __init__(self):
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        print("TruthLens ML Service")
        print(f"Device: {device_name}")
        print(f"CUDA: {cuda_available}")

        # Initialize the pipeline (which loads Model A & Model B onto GPU once)
        self.pipeline = FactCheckPipeline.get_instance()
        self.cuda_available = cuda_available
        self.device_name = device_name

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, claim: str, evidence: List[str], threshold: Optional[float] = None) -> Dict[str, Any]:
        raw_result = self.pipeline.run_fact_check(
            claim=claim,
            candidate_evidence=evidence,
            custom_threshold=threshold
        )

        # Format output into camelCase schema expected by API
        evidence_results = []
        for ev in raw_result.get("evidence_results", []):
            evidence_results.append({
                "evidence": ev["evidence"],
                "relevanceLabel": ev["relevance_label"],
                "relevanceScore": float(ev["relevance_score"])
            })

        verif = raw_result.get("verification", {})
        probabilities = verif.get("probabilities", {})
        raw_indiv = verif.get("individual_verifications", [])
        individual_verifications = []
        for iv in raw_indiv:
            probs = iv.get("probabilities", {})
            individual_verifications.append({
                "evidence": iv.get("evidence", ""),
                "relevanceScore": float(iv.get("relevance_score", 0.0)),
                "label": iv.get("label", "NOT_ENOUGH_INFO"),
                "confidence": float(iv.get("confidence", 0.0)),
                "probabilities": {
                    "SUPPORTS": float(probs.get("SUPPORTS", 0.0)),
                    "REFUTES": float(probs.get("REFUTES", 0.0)),
                    "NOT_ENOUGH_INFO": float(probs.get("NOT_ENOUGH_INFO", 0.0)),
                }
            })

        return {
            "claim": raw_result.get("claim", claim),
            "evidenceResults": evidence_results,
            "verification": {
                "label": verif.get("label", "NOT_ENOUGH_INFO"),
                "confidence": float(verif.get("confidence", 0.0)),
                "probabilities": {
                    "SUPPORTS": float(probabilities.get("SUPPORTS", 0.0)),
                    "REFUTES": float(probabilities.get("REFUTES", 0.0)),
                    "NOT_ENOUGH_INFO": float(probabilities.get("NOT_ENOUGH_INFO", 0.0)),
                },
                "individualVerifications": individual_verifications
            }
        }
