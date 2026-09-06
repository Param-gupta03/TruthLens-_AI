"""
predict_relevance.py - Real-Time Inference Module for Model A (Evidence Relevance).

Input: claim + candidate evidence
Output: RELEVANT (1) or NOT_RELEVANT (0) with calibrated probabilities and decision threshold.
Hardware: Strictly CUDA accelerated.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "evidence_relevance",
    "roberta-base"
)

_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_CACHED_DEVICE = None
_CACHED_THRESHOLD = 0.35


def load_relevance_model(model_dir: str = DEFAULT_MODEL_DIR):
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE, _CACHED_THRESHOLD

    if _CACHED_MODEL is not None and _CACHED_TOKENIZER is not None:
        return _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE, _CACHED_THRESHOLD

    if not torch.cuda.is_available():
        error_msg = (
            "[CRITICAL ERROR] CUDA is NOT available for Model A inference!\n"
            "Per user configuration, inference requires an active NVIDIA GPU with CUDA.\n"
            "Silent fallback to CPU is disabled."
        )
        print(error_msg, file=sys.stderr)
        raise RuntimeError("CUDA is required for TruthLens AI Model A inference.")

    device = torch.device("cuda")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model A directory not found at: {model_dir}")

    # Load selected threshold from training_config.json
    config_path = os.path.join(model_dir, "training_config.json")
    threshold = 0.35
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                threshold = cfg.get("selected_threshold", 0.35)
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_TOKENIZER = tokenizer
    _CACHED_DEVICE = device
    _CACHED_THRESHOLD = threshold

    return model, tokenizer, device, threshold


def predict_relevance(
    claim: str,
    evidence: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Classifies candidate evidence as RELEVANT or NOT_RELEVANT for a given claim.
    """
    model, tokenizer, device, default_threshold = load_relevance_model(model_dir)
    decision_threshold = threshold if threshold is not None else default_threshold

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
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

    prob_not_rel = float(round(float(probs[0]), 4))
    prob_rel = float(round(float(probs[1]), 4))

    is_relevant = prob_rel >= decision_threshold
    predicted_label = "RELEVANT" if is_relevant else "NOT_RELEVANT"
    confidence = prob_rel if is_relevant else prob_not_rel

    return {
        "label": predicted_label,
        "confidence": round(float(confidence), 4),
        "probability_relevant": prob_rel,
        "probability_not_relevant": prob_not_rel,
        "threshold": decision_threshold,
    }


def run_demo_tests():
    print("=" * 70)
    print(" TRUTHLENS AI — MODEL A (EVIDENCE RELEVANCE) INFERENCE TEST")
    print("=" * 70)

    test_cases = [
        {
            "description": "Example 1: Clearly Relevant Evidence",
            "claim": "40mg/day dosage of folic acid and 2mg/day dosage of vitamin B12 does not affect chronic kidney disease (CKD) progression.",
            "evidence": "Treatment with high doses of folic acid and B vitamins did not improve survival or reduce the incidence of vascular disease in patients with advanced chronic kidney disease or end-stage renal disease.",
            "expected": "RELEVANT"
        },
        {
            "description": "Example 2: Clearly Irrelevant Evidence",
            "claim": "40mg/day dosage of folic acid and 2mg/day dosage of vitamin B12 does not affect chronic kidney disease (CKD) progression.",
            "evidence": "Jupiter is the fifth planet from the Sun and the largest in the Solar System, primarily composed of hydrogen.",
            "expected": "NOT_RELEVANT"
        },
        {
            "description": "Example 3: Difficult Hard-Negative Evidence (Same document topic/keywords, but not answering the claim)",
            "claim": "40mg/day dosage of folic acid and 2mg/day dosage of vitamin B12 does not affect chronic kidney disease (CKD) progression.",
            "evidence": "Plasma homocysteine levels are elevated in more than 85% of patients with chronic kidney disease, but the cardiovascular benefits of lowering homocysteine remain an active area of nephrology research.",
            "expected": "NOT_RELEVANT"
        }
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- {tc['description']} ---")
        print(f"Claim    : {tc['claim']}")
        print(f"Evidence : {tc['evidence']}")
        res = predict_relevance(tc["claim"], tc["evidence"])
        print("Prediction Result:")
        print(json.dumps(res, indent=2))
        print(f"Verdict Match: {'CORRECT' if res['label'] == tc['expected'] else 'INCORRECT'} (Expected: {tc['expected']})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthLens AI — Model A Evidence Relevance Predictor")
    parser.add_argument("--claim", type=str, default=None, help="Claim statement")
    parser.add_argument("--evidence", type=str, default=None, help="Candidate evidence passage")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Model directory")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold")
    parser.add_argument("--demo", action="store_true", help="Run 3 standard test cases")
    args = parser.parse_args()

    if args.claim and args.evidence:
        result = predict_relevance(args.claim, args.evidence, model_dir=args.model_dir, threshold=args.threshold)
        print(json.dumps(result, indent=2))
    else:
        run_demo_tests()
