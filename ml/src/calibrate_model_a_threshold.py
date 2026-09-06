"""
calibrate_model_a_threshold.py - Systematic evaluation of Model A decision thresholds.
Evaluates: 0.20, 0.25, 0.30, 0.35, 0.40, 0.45
Computes: precision, recall, F1, false positives, false negatives on validation set.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_a import ModelA, DEFAULT_MODEL_A_DIR

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "final")
VALID_CSV = os.path.join(DATA_DIR, "evidence_relevance_valid.csv")
TEST_CSV = os.path.join(DATA_DIR, "evidence_relevance_test.csv")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "reports")


def evaluate_thresholds(csv_path: str, split_name: str = "Validation"):
    print(f"\n=======================================================")
    print(f"  EVALUATING MODEL A THRESHOLDS ON {split_name.upper()} SET")
    print(f"=======================================================")

    df = pd.read_csv(csv_path)
    print(f"Total samples in {split_name}: {len(df)}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")

    model_wrapper = ModelA.get_instance()
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    device = model_wrapper.device

    claims = df["claim"].astype(str).tolist()
    evidences = df["evidence"].fillna("None").astype(str).tolist()
    y_true = df["label"].astype(int).values

    # Batch inference for speed
    batch_size = 32
    probabilities = []

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(claims), batch_size):
                b_claims = claims[i : i + batch_size]
                b_evidences = evidences[i : i + batch_size]
                texts = [f"[CLAIM] {c.strip()} [SEP] {e.strip()}" for c, e in zip(b_claims, b_evidences)]

                inputs = tokenizer(
                    texts,
                    max_length=256,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                probabilities.extend(probs)

    probabilities = np.array(probabilities)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    results = []

    print(f"\n{'Threshold':<10} | {'Accuracy':<9} | {'Precision':<10} | {'Recall':<9} | {'F1 Score':<9} | {'FP':<6} | {'FN':<6} | {'TP':<6} | {'TN':<6}")
    print("-" * 85)

    for th in thresholds:
        y_pred = (probabilities >= th).astype(int)

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "split": split_name,
            "threshold": th,
            "accuracy": round(float(acc), 4),
            "relevant_precision": round(float(p), 4),
            "relevant_recall": round(float(r), 4),
            "relevant_f1": round(float(f1), 4),
            "macro_f1": round(float(macro_f1), 4),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        })

        print(f"{th:<10.2f} | {acc * 100:<8.2f}% | {p * 100:<9.2f}% | {r * 100:<8.2f}% | {f1 * 100:<8.2f}% | {fp:<6} | {fn:<6} | {tp:<6} | {tn:<6}")

    return results


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    val_results = evaluate_thresholds(VALID_CSV, split_name="Validation")
    test_results = evaluate_thresholds(TEST_CSV, split_name="Test")

    all_results = val_results + test_results
    res_df = pd.DataFrame(all_results)
    out_csv = os.path.join(REPORTS_DIR, "phase12_model_a_threshold_calibration.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved calibration results to: {out_csv}")


if __name__ == "__main__":
    main()
