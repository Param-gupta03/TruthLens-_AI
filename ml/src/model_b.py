"""
model_b.py - Model B (Claim Verification) Inference Wrapper for TruthLens AI.

Task:
Input: claim + candidate evidence passage
Output: SUPPORTS / REFUTES / NOT_ENOUGH_INFO with probability distribution.
Hardware: CUDA accelerated with model caching.
"""

import os
import sys
import json
from typing import Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL_B_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "verification",
    "scifact_deberta"
)

ID2LABEL = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT_ENOUGH_INFO",
}


FALLBACK_MODEL_B = "roberta-base"


class ModelB:
    _instance = None

    def __init__(self, model_dir: str = DEFAULT_MODEL_B_DIR):
        has_weights = os.path.exists(model_dir) and (
            os.path.exists(os.path.join(model_dir, "model.safetensors")) or
            os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
        )
        if not has_weights:
            print(f"Model B local weights not found in '{model_dir}'. Falling back to HF Hub: '{FALLBACK_MODEL_B}'")
            model_dir = FALLBACK_MODEL_B

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_name = torch.cuda.get_device_name(0)
            self.cuda_available = True
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU"
            self.cuda_available = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def get_instance(cls, model_dir: str = DEFAULT_MODEL_B_DIR):
        if cls._instance is None:
            cls._instance = cls(model_dir=model_dir)
        return cls._instance

    def predict_verification(self, claim: str, evidence: str) -> Dict[str, Any]:
        claim_clean = str(claim).strip()
        evidence_clean = str(evidence).strip() if evidence else "None"
        input_text = f"[CLAIM] {claim_clean} [SEP] {evidence_clean}"

        inputs = self.tokenizer(
            input_text,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

        prob_supports = float(round(float(probs[0]), 4))
        prob_refutes = float(round(float(probs[1]), 4))
        prob_nei = float(round(float(probs[2]), 4))

        prob_dict = {
            "SUPPORTS": prob_supports,
            "REFUTES": prob_refutes,
            "NOT_ENOUGH_INFO": prob_nei,
        }

        # Winning class
        pred_idx = int(probs.argmax())
        pred_label = ID2LABEL.get(pred_idx, "NOT_ENOUGH_INFO")
        confidence = float(round(float(probs[pred_idx]), 4))

        return {
            "label": pred_label,
            "confidence": confidence,
            "probabilities": prob_dict,
        }


def predict_verification(
    claim: str,
    evidence: str,
    model_dir: str = DEFAULT_MODEL_B_DIR
) -> Dict[str, Any]:
    wrapper = ModelB.get_instance(model_dir=model_dir)
    return wrapper.predict_verification(claim, evidence)


if __name__ == "__main__":
    print("Testing Model B wrapper...")
    c = "Smoking causes lung cancer."
    e = "Cigarette smoking is proven to cause lung cancer and respiratory illness."
    res = predict_verification(c, e)
    print(json.dumps(res, indent=2))
