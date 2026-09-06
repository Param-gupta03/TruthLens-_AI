"""
predict.py - Inference and Prediction Module for TruthLens AI Claim Verification.

Provides:
- predict_veracity(claim, evidence, model_path=None)
- Returns predicted label, confidence, and 3-class probability distribution.
- Automatically selects CUDA when available.
"""

import os
import sys
import json
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_utils import ID2LABEL, LABEL2ID

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "verification",
    "scifact_deberta"
)

# Global model cache to avoid reloading weights repeatedly
_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_CACHED_DEVICE = None


def load_verification_model(model_dir: str = DEFAULT_MODEL_DIR):
    """Loads and caches the fine-tuned model and tokenizer on CUDA if available."""
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE
    
    if _CACHED_MODEL is not None and _CACHED_TOKENIZER is not None:
        return _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE
        
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Trained verification model not found at: {model_dir}. Please train Model B first.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Loading model from {model_dir} on device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    _CACHED_MODEL = model
    _CACHED_TOKENIZER = tokenizer
    _CACHED_DEVICE = device
    return model, tokenizer, device


def predict_veracity(
    claim: str,
    evidence: str,
    model_dir: str = DEFAULT_MODEL_DIR
) -> Dict[str, Any]:
    """
    Evaluates claim veracity against evidence text.
    
    Input:
        claim (str): The assertion or news claim to evaluate.
        evidence (str): The retrieved evidence passage or rationale.
        model_dir (str): Path to directory with saved model and tokenizer.
        
    Returns:
        dict with schema:
        {
            "label": "SUPPORTS" | "REFUTES" | "NOT_ENOUGH_INFO",
            "confidence": float,
            "probabilities": {
                "SUPPORTS": float,
                "REFUTES": float,
                "NOT_ENOUGH_INFO": float
            }
        }
    """
    model, tokenizer, device = load_verification_model(model_dir)
    
    # Input formatting: [CLAIM] claim text [SEP] evidence text
    claim_clean = str(claim).strip()
    evidence_clean = str(evidence).strip() if evidence else "None"
    input_text = f"[CLAIM] {claim_clean} [SEP] {evidence_clean}"
    
    inputs = tokenizer(
        input_text,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
    pred_idx = int(np.argmax(probs))
    pred_label = ID2LABEL.get(pred_idx, "UNKNOWN")
    confidence = float(probs[pred_idx])
    
    prob_dict = {
        "SUPPORTS": float(round(probs[0], 4)),
        "REFUTES": float(round(probs[1], 4)),
        "NOT_ENOUGH_INFO": float(round(probs[2], 4)),
    }
    
    return {
        "label": pred_label,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TruthLens AI — Claim Veracity Predictor")
    parser.add_argument("--claim", type=str, default="32% of liver transplantation programs required patients to discontinue methadone treatment in 2001.", help="Claim text to verify")
    parser.add_argument("--evidence", type=str, default="Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base for efficacy of long-term replacement therapies.", help="Evidence passage")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Path to fine-tuned model directory")
    args = parser.parse_args()
    
    try:
        res = predict_veracity(claim=args.claim, evidence=args.evidence, model_dir=args.model_dir)
        print("\nPrediction Result:")
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Prediction runner error: {e}")
        sys.exit(1)
