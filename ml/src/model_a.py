"""
model_a.py - Model A (Evidence Relevance) Inference Wrapper for TruthLens AI.

Task:
Input: claim + candidate evidence passage
Output: RELEVANT (1) or NOT_RELEVANT (0) with probability and decision threshold.
Hardware: CUDA accelerated with model caching.
"""

import os
import sys
import json
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL_A_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "evidence_relevance",
    "roberta-base"
)


class ModelA:
    _instance = None

    def __init__(self, model_dir: str = DEFAULT_MODEL_A_DIR):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model A directory not found: {model_dir}")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_name = torch.cuda.get_device_name(0)
            self.cuda_available = True
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU"
            self.cuda_available = False

        # Load selected threshold (calibrated to 0.25 for balanced general recall & precision)
        env_th = os.environ.get("MODEL_A_THRESHOLD")
        if env_th:
            try:
                self.threshold = float(env_th)
            except ValueError:
                self.threshold = 0.25
        else:
            self.threshold = 0.25
            config_file = os.path.join(model_dir, "training_config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        self.threshold = float(cfg.get("calibrated_threshold", 0.25))
                except Exception:
                    pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def get_instance(cls, model_dir: str = DEFAULT_MODEL_A_DIR):
        if cls._instance is None:
            cls._instance = cls(model_dir=model_dir)
        return cls._instance

    def predict_relevance(
        self,
        claim: str,
        evidence: str,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        th = threshold if threshold is not None else self.threshold
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

        prob_not_rel = float(round(float(probs[0]), 4))
        prob_rel = float(round(float(probs[1]), 4))
        is_relevant = prob_rel >= th

        return {
            "label": "RELEVANT" if is_relevant else "NOT_RELEVANT",
            "probability_relevant": prob_rel,
            "probability_not_relevant": prob_not_rel,
            "threshold": th,
        }


def predict_relevance(
    claim: str,
    evidence: str,
    model_dir: str = DEFAULT_MODEL_A_DIR,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    wrapper = ModelA.get_instance(model_dir=model_dir)
    return wrapper.predict_relevance(claim, evidence, threshold=threshold)


if __name__ == "__main__":
    print("Testing Model A wrapper...")
    c = "Smoking causes lung cancer."
    e = "Cigarette smoking is the leading risk factor for lung cancer."
    res = predict_relevance(c, e)
    print(json.dumps(res, indent=2))
