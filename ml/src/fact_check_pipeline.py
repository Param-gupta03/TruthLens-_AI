"""
fact_check_pipeline.py - Integrated Fact-Checking Pipeline for TruthLens AI (Phase 12).

Architecture:
User Claim + Candidate Evidence Passages
  ↓
MODEL A (Evidence Relevance Filter / Ranker, calibrated threshold = 0.25)
  + General Domain High-Fidelity Encyclopedic Overlap (>= 80% overlap & >= 3 matches)
  ↓
Filtered & Ranked Diverse Relevant Evidence Passages (Max 5, >= 0.25)
  ↓
MODEL B (Authoritative Claim Verification Predictor)
  ↓
Relevance-Weighted Veracity Verdict (SUPPORTS / REFUTES / NOT_ENOUGH_INFO)
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Optional
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_a import ModelA, DEFAULT_MODEL_A_DIR
from model_b import ModelB, DEFAULT_MODEL_B_DIR


def compute_keyword_overlap(claim: str, text: str):
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about',
        'into', 'over', 'after', 'beneath', 'under', 'above', 'that', 'this',
        'these', 'those', 'it', 'its', 'as', 'and', 'or', 'but', 'if', 'then',
        'so', 'than', 'too', 'very', 'just', 'now', 'such', 'why', 'what', 'how'
    }
    import re
    claim_words = [w for w in re.sub(r'[^\w\s]', ' ', claim.lower()).split() if len(w) > 1 and w not in stop_words]
    if not claim_words:
        return 0.0, 0
    words_in_text = set(re.sub(r'[^\w\s]', ' ', text.lower()).split())
    matches = 0
    for w in claim_words:
        if (w in words_in_text or
            (w.endswith('s') and w[:-1] in words_in_text) or
            (w.endswith('es') and w[:-2] in words_in_text) or
            (w.endswith('ing') and w[:-3] in words_in_text) or
            (w.endswith('ed') and w[:-2] in words_in_text)):
            matches += 1
    return round(matches / len(claim_words), 4), matches


class FactCheckPipeline:

    _instance = None

    def __init__(
        self,
        model_a_dir: str = DEFAULT_MODEL_A_DIR,
        model_b_dir: str = DEFAULT_MODEL_B_DIR
    ):
        if torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name(0)
            self.cuda_available = True
            print(f"Initializing TruthLens FactCheckPipeline...")
            print(f"Device: {self.device_name} (CUDA Runtime Active)")
        else:
            self.device_name = "CPU"
            self.cuda_available = False
            print(f"Initializing TruthLens FactCheckPipeline...")
            print(f"Device: CPU (CUDA unavailable, CPU fallback active)")

        self.model_a = ModelA.get_instance(model_dir=model_a_dir)
        self.model_b = ModelB.get_instance(model_dir=model_b_dir)
        print("Model A (Evidence Relevance) and Model B (Claim Verification) loaded successfully.")

    @classmethod
    def get_instance(
        cls,
        model_a_dir: str = DEFAULT_MODEL_A_DIR,
        model_b_dir: str = DEFAULT_MODEL_B_DIR
    ):
        if cls._instance is None:
            cls._instance = cls(model_a_dir=model_a_dir, model_b_dir=model_b_dir)
        return cls._instance

    def run_fact_check(
        self,
        claim: str,
        candidate_evidence: List[str],
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end fact checking on a claim with candidate evidence passages.
        """
        pipeline_start = time.time()
        claim_str = str(claim).strip()
        operating_threshold = custom_threshold if custom_threshold is not None else self.model_a.threshold

        # Step 1: Filter & Score Candidate Evidence with Model A (threshold = 0.25)
        model_a_start = time.time()
        evidence_results = []
        relevant_passages = []

        for ev in candidate_evidence:
            ev_str = str(ev).strip()
            if not ev_str:
                continue
            pred_a = self.model_a.predict_relevance(claim=claim_str, evidence=ev_str, threshold=operating_threshold)
            raw_rel = pred_a["probability_relevant"]

            # Step 12 Domain Generalization:
            # Model A was fine-tuned on SciFact (biomedical). For non-biomedical encyclopedic definitions
            # (e.g. Paris capital of France, Apollo 11 Moon landing), direct definitions can receive low raw scores.
            # Only boost if there is near-complete encyclopedic overlap (>= 80% and >= 3 matches).
            overlap, matches = compute_keyword_overlap(claim_str, ev_str)
            if raw_rel < operating_threshold and matches >= 3 and overlap >= 0.80:
                combined_rel = round(max(raw_rel, 0.45 * overlap + 0.55 * raw_rel, 0.35 * overlap), 4)
            else:
                combined_rel = raw_rel

            is_relevant = combined_rel >= operating_threshold
            res_item = {
                "evidence": ev_str,
                "relevance_label": "RELEVANT" if is_relevant else "NOT_RELEVANT",
                "relevance_score": combined_rel,
                "probability_not_relevant": round(1.0 - combined_rel, 4),
            }
            evidence_results.append(res_item)
            if is_relevant:
                relevant_passages.append(res_item)

        model_a_duration = time.time() - model_a_start

        # Rank relevant evidence by relevance score descending
        relevant_passages.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Step 2: Honest NOT_ENOUGH_INFO Handling (Step 14)
        if len(relevant_passages) == 0:
            if len(evidence_results) > 0:
                max_rel_score = max(r["relevance_score"] for r in evidence_results)
                nei_confidence = round(1.0 - max_rel_score, 4)
            else:
                nei_confidence = 1.0

            total_duration = time.time() - pipeline_start
            return {
                "claim": claim_str,
                "evidence_results": evidence_results,
                "relevant_evidence_count": 0,
                "total_evidence_count": len(evidence_results),
                "verification": {
                    "label": "NOT_ENOUGH_INFO",
                    "confidence": max(nei_confidence, 0.50),
                    "reason": "No sufficiently relevant evidence found meeting the calibrated relevance threshold.",
                    "probabilities": {
                        "SUPPORTS": 0.0,
                        "REFUTES": 0.0,
                        "NOT_ENOUGH_INFO": 1.0
                    },
                    "individual_verifications": []
                },
                "timings": {
                    "model_a_time_sec": round(model_a_duration, 4),
                    "model_b_time_sec": 0.0,
                    "pipeline_time_sec": round(total_duration, 4),
                }
            }

        # Step 3: Select Top K Passages for Model B (K = 3 to 5) (Step 8 & 10)
        top_k_passages = relevant_passages[:5]

        # Step 4: Run Model B on each selected relevant evidence passage
        model_b_start = time.time()
        b_verifications = []
        for rel_item in top_k_passages:
            ev_text = rel_item["evidence"]
            pred_b = self.model_b.predict_verification(claim=claim_str, evidence=ev_text)
            b_verifications.append({
                "evidence": ev_text,
                "relevance_score": rel_item["relevance_score"],
                "label": pred_b["label"],
                "confidence": pred_b["confidence"],
                "probabilities": pred_b["probabilities"],
            })

        model_b_duration = time.time() - model_b_start

        # Step 5: Relevance-Weighted Aggregation of Model B Probabilities (Step 11)
        weights = [max(item["relevance_score"], 0.1) for item in b_verifications]
        total_weight = sum(weights)

        weighted_supports = sum(w * item["probabilities"]["SUPPORTS"] for w, item in zip(weights, b_verifications)) / total_weight
        weighted_refutes = sum(w * item["probabilities"]["REFUTES"] for w, item in zip(weights, b_verifications)) / total_weight
        weighted_nei = sum(w * item["probabilities"]["NOT_ENOUGH_INFO"] for w, item in zip(weights, b_verifications)) / total_weight

        agg_probabilities = {
            "SUPPORTS": round(float(weighted_supports), 4),
            "REFUTES": round(float(weighted_refutes), 4),
            "NOT_ENOUGH_INFO": round(float(weighted_nei), 4),
        }

        # Conflict detection (Step 11)
        has_strong_supports = any(iv["label"] == "SUPPORTS" and iv["confidence"] >= 0.70 for iv in b_verifications)
        has_strong_refutes = any(iv["label"] == "REFUTES" and iv["confidence"] >= 0.70 for iv in b_verifications)
        is_contradictory = has_strong_supports and has_strong_refutes
        margin = abs(agg_probabilities["SUPPORTS"] - agg_probabilities["REFUTES"])

        # Determine winning verdict
        if is_contradictory and margin < 0.15:
            winning_label = "NOT_ENOUGH_INFO"
            final_confidence = round(max(agg_probabilities["NOT_ENOUGH_INFO"], 0.55), 4)
        else:
            if agg_probabilities["SUPPORTS"] > agg_probabilities["REFUTES"] and agg_probabilities["SUPPORTS"] > agg_probabilities["NOT_ENOUGH_INFO"]:
                winning_label = "SUPPORTS"
                final_confidence = agg_probabilities["SUPPORTS"]
            elif agg_probabilities["REFUTES"] > agg_probabilities["SUPPORTS"] and agg_probabilities["REFUTES"] > agg_probabilities["NOT_ENOUGH_INFO"]:
                winning_label = "REFUTES"
                final_confidence = agg_probabilities["REFUTES"]
            else:
                winning_label = "NOT_ENOUGH_INFO"
                final_confidence = agg_probabilities["NOT_ENOUGH_INFO"]

        total_duration = time.time() - pipeline_start

        return {
            "claim": claim_str,
            "evidence_results": evidence_results,
            "relevant_evidence_count": len(relevant_passages),
            "total_evidence_count": len(evidence_results),
            "verification": {
                "label": winning_label,
                "confidence": final_confidence,
                "probabilities": agg_probabilities,
                "individual_verifications": b_verifications,
                "contradictory_evidence": is_contradictory
            },
            "timings": {
                "model_a_time_sec": round(model_a_duration, 4),
                "model_b_time_sec": round(model_b_duration, 4),
                "pipeline_time_sec": round(total_duration, 4),
            }
        }


def print_cli_report(result: Dict[str, Any]):
    print("\n" + "=" * 29)
    print("TRUTHLENS AI")
    print("=" * 29)
    print("\nClaim:")
    print(result["claim"])

    print("\nEvidence Analysis:\n")
    for i, ev_res in enumerate(result["evidence_results"], 1):
        print(f"{i}. {ev_res['relevance_label']} — {ev_res['relevance_score']:.2f}")
        print(f"   Excerpt: \"{ev_res['evidence'][:80]}...\"")

    verif = result["verification"]
    print("\nFinal Verification:\n")
    print(verif["label"])

    print("\nConfidence:")
    print(f"{verif['confidence']:.2f}")

    if "reason" in verif:
        print(f"\nReason: {verif['reason']}")

    print("\nProbabilities:\n")
    for k, v in verif["probabilities"].items():
        print(f"{k}: {v:.2f}")

    print(f"\nPipeline Time:")
    print(f"{result['timings']['pipeline_time_sec']:.3f} seconds (Model A: {result['timings']['model_a_time_sec']:.3f}s | Model B: {result['timings']['model_b_time_sec']:.3f}s)")
    print("=" * 29)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthLens AI — Fact Checking Pipeline CLI")
    parser.add_argument("--claim", type=str, default=None, help="Claim to fact check")
    parser.add_argument("--evidence", nargs="+", default=None, help="Candidate evidence passages")
    args = parser.parse_args()

    pipeline = FactCheckPipeline.get_instance()

    if args.claim and args.evidence:
        res = pipeline.run_fact_check(args.claim, args.evidence)
        print_cli_report(res)
    else:
        demo_claim = "Paris is the capital of France"
        demo_evidence = [
            "Paris is the capital and most populous city of France, situated along the Seine River.",
            "Water freezes at 0 degrees Celsius under standard atmospheric pressure."
        ]
        res = pipeline.run_fact_check(demo_claim, demo_evidence)
        print_cli_report(res)
