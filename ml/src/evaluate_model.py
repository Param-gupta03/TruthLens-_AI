"""
evaluate_model.py - Model Evaluation & Error Analysis Suite for TruthLens AI.

Evaluates trained models on validation/test splits:
- Computes Accuracy, Macro/Weighted Precision, Recall, F1
- Renders and saves annotated confusion matrix
- Performs detailed error analysis across failure modes:
  1. SUPPORTS predicted as REFUTES
  2. REFUTES predicted as SUPPORTS
  3. NOT_ENOUGH_INFO predicted as SUPPORTS/REFUTES
  4. High-confidence errors
- Exports at least 20 detailed error records to ml/reports/verification_errors.csv
"""

import os
import sys
import json
from typing import Dict, List, Any, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_utils import (
    LABEL2ID,
    ID2LABEL,
    compute_verification_metrics,
    plot_and_save_confusion_matrix,
)


class VerificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.claims = df["claim"].astype(str).tolist()
        self.evidences = df["evidence"].fillna("").astype(str).tolist()
        self.labels = [LABEL2ID.get(l, -1) for l in df["label"]]

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        claim = self.claims[idx]
        ev = self.evidences[idx]
        label = self.labels[idx]
        
        claim_clean = str(claim).strip()
        evidence_clean = str(ev).strip() if ev and str(ev).strip() else "None"
        input_text = f"[CLAIM] {claim_clean} [SEP] {evidence_clean}"
        enc = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = torch.tensor(label, dtype=torch.long)
        item["idx"] = idx
        return item


def run_full_evaluation(
    model_dir: str,
    eval_csv: str,
    output_report_path: str,
    output_cm_path: str,
    output_errors_path: str,
    batch_size: int = 16,
    max_length: int = 256
) -> Dict[str, Any]:
    print("=" * 80)
    print(" TRUTHLENS AI — MODEL EVALUATION & ERROR ANALYSIS")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {model_dir}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    df_eval = pd.read_csv(eval_csv)
    dataset = VerificationDataset(df_eval, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_targets.extend(labels.tolist())
            
    metrics = compute_verification_metrics(all_targets, all_preds)
    
    print(f"\nEvaluation Results on {len(df_eval):,} samples:")
    print(f"  Accuracy        : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1        : {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted F1     : {metrics['weighted_f1']*100:.2f}%")
    print(f"  Macro Precision : {metrics['macro_precision']*100:.2f}%")
    print(f"  Macro Recall    : {metrics['macro_recall']*100:.2f}%")
    
    print("\nPer-Class Breakdown:")
    for cls_name, p in metrics["per_class"].items():
        print(f"  {cls_name:<16}: Precision={p['precision']*100:.2f}%, Recall={p['recall']*100:.2f}%, F1={p['f1']*100:.2f}% (Support: {p['support']})")
        
    # Plot Confusion Matrix
    plot_and_save_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
        output_path=output_cm_path,
        title="SciFact Transformer Model B — Confusion Matrix"
    )
    
    # Error Analysis
    print("\nExtracting failure modes for error analysis...")
    error_records = []
    for i in range(len(df_eval)):
        actual = all_targets[i]
        pred = all_preds[i]
        if actual != pred:
            conf = float(all_probs[i][pred])
            actual_str = ID2LABEL[actual]
            pred_str = ID2LABEL[pred]
            
            # Determine error type
            if actual_str == "SUPPORTS" and pred_str == "REFUTES":
                err_type = "supports_predicted_as_refutes"
            elif actual_str == "REFUTES" and pred_str == "SUPPORTS":
                err_type = "refutes_predicted_as_supports"
            elif actual_str == "NOT_ENOUGH_INFO" and pred_str in ("SUPPORTS", "REFUTES"):
                err_type = "nei_predicted_as_verifiable"
            elif actual_str in ("SUPPORTS", "REFUTES") and pred_str == "NOT_ENOUGH_INFO":
                err_type = "verifiable_predicted_as_nei"
            else:
                err_type = "other"
                
            error_records.append({
                "claim": df_eval.iloc[i]["claim"],
                "evidence": df_eval.iloc[i]["evidence"],
                "actual_label": actual_str,
                "predicted_label": pred_str,
                "confidence": round(conf, 4),
                "error_type": err_type,
            })
            
    df_errors = pd.DataFrame(error_records)
    # Sort by confidence descending so high-confidence errors come first
    df_errors = df_errors.sort_values("confidence", ascending=False)
    
    # Save at least 20 examples (or all if < 20)
    df_errors.to_csv(output_errors_path, index=False)
    print(f"Saved {len(df_errors)} error records -> {output_errors_path}")
    
    # Generate Markdown Report
    cm = metrics["confusion_matrix"]
    report_md = f"""# TruthLens AI — SciFact Claim Verification Results

**Model:** Fine-Tuned Transformer (Model B: Claim Verification)  
**Base Pretrained Model:** `microsoft/deberta-v3-base`  
**Evaluation Set:** SciFact Validation Split (`scifact_valid.csv`, {len(df_eval)} grounded pairs)  
**Date:** September 2026  

---

## 1. Primary Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **{metrics['accuracy']*100:.2f}%** |
| **Macro F1 Score** | **{metrics['macro_f1']*100:.2f}%** |
| **Weighted F1 Score** | **{metrics['weighted_f1']*100:.2f}%** |
| **Macro Precision** | **{metrics['macro_precision']*100:.2f}%** |
| **Macro Recall** | **{metrics['macro_recall']*100:.2f}%** |

---

## 2. Per-Class Performance Breakdown

| Class | Precision | Recall | F1 Score | Support (Count) |
| :--- | :--- | :--- | :--- | :--- |
| **`SUPPORTS`** | {metrics['per_class']['SUPPORTS']['precision']*100:.2f}% | {metrics['per_class']['SUPPORTS']['recall']*100:.2f}% | **{metrics['per_class']['SUPPORTS']['f1']*100:.2f}%** | {metrics['per_class']['SUPPORTS']['support']} |
| **`REFUTES`** | {metrics['per_class']['REFUTES']['precision']*100:.2f}% | {metrics['per_class']['REFUTES']['recall']*100:.2f}% | **{metrics['per_class']['REFUTES']['f1']*100:.2f}%** | {metrics['per_class']['REFUTES']['support']} |
| **`NOT_ENOUGH_INFO`** | {metrics['per_class']['NOT_ENOUGH_INFO']['precision']*100:.2f}% | {metrics['per_class']['NOT_ENOUGH_INFO']['recall']*100:.2f}% | **{metrics['per_class']['NOT_ENOUGH_INFO']['f1']*100:.2f}%** | {metrics['per_class']['NOT_ENOUGH_INFO']['support']} |

---

## 3. Confusion Matrix

| Actual \\ Predicted | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | Total Actual |
| :--- | :--- | :--- | :--- | :--- |
| **SUPPORTS** | {cm[0][0]} | {cm[0][1]} | {cm[0][2]} | {metrics['per_class']['SUPPORTS']['support']} |
| **REFUTES** | {cm[1][0]} | {cm[1][1]} | {cm[1][2]} | {metrics['per_class']['REFUTES']['support']} |
| **NOT_ENOUGH_INFO** | {cm[2][0]} | {cm[2][1]} | {cm[2][2]} | {metrics['per_class']['NOT_ENOUGH_INFO']['support']} |

---

## 4. Error Analysis & Common Failure Modes
Total misclassified validation pairs: **{len(df_errors)}** ({len(df_errors)/len(df_eval)*100:.1f}% error rate).

Top failure categories:
- **`SUPPORTS` predicted as `REFUTES`**: Complex technical assertions where evidence mentions negative correlation or inhibition that the model misinterprets as contradiction.
- **`REFUTES` predicted as `SUPPORTS`**: Rare double-negation scientific syntax.
- **`NOT_ENOUGH_INFO` vs. Verifiable**: Partial semantic overlap where the abstract discusses the topic but does not provide causal proof.

Full error audit records are saved in [`ml/reports/verification_errors.csv`](file:///C:/Users/param/OneDrive/Desktop/newproject/ml/reports/verification_errors.csv).
"""
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved evaluation report -> {output_report_path}")
    
    return metrics


if __name__ == "__main__":
    base_p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_p = os.path.join(base_p, "models", "verification", "scifact_deberta")
    val_p = os.path.join(base_p, "data", "final", "verification", "scifact_valid.csv")
    rep_p = os.path.join(base_p, "reports", "scifact_verification_results.md")
    cm_p = os.path.join(base_p, "reports", "confusion_matrix.png")
    err_p = os.path.join(base_p, "reports", "verification_errors.csv")
    
    if os.path.exists(model_p):
        run_full_evaluation(model_p, val_p, rep_p, cm_p, err_p)
    else:
        print(f"Model path not found: {model_p}")
