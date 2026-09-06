"""
prepare_all_datasets.py - Master Preprocessing & Dataset Preparation Pipeline.

Orchestrates:
1. FEVER data cleaning, deduplication, conflict quarantining, and Model B split creation.
2. LIAR safe parsing, HTML/Unicode cleaning, missing metadata imputation, and leakage removal.
3. SciFact corpus abstract mapping, rationale reconstruction, Model A relevance pair generation,
   and Model B verification split creation.
4. Generates ml/reports/final_dataset_summary.csv.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean_fever import clean_and_canonicalize_fever
from clean_liar import clean_and_prepare_liar
from clean_scifact import clean_and_prepare_scifact


def calc_mean_word_len(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    lens = []
    for val in series:
        if pd.notna(val):
            val_str = str(val).strip()
            if val_str and val_str.lower() != "nan":
                lens.append(len(val_str.split()))
    return float(round(float(np.mean(lens)), 2)) if lens else 0.0


def generate_summary_csv(base_dir: str, reports_dir: str) -> str:
    """
    Computes summary metrics across all prepared dataset files and writes
    ml/reports/final_dataset_summary.csv.
    """
    rows = []
    
    # 1. Model B Verification Files
    verif_dir = os.path.join(base_dir, "data", "final", "verification")
    
    # FEVER splits
    for split in ["train", "valid"]:
        p = os.path.join(verif_dir, f"fever_{split}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            cnts = df["label"].value_counts().to_dict()
            rows.append({
                "dataset": "fever",
                "split": split,
                "rows": len(df),
                "supports": cnts.get("SUPPORTS", 0),
                "refutes": cnts.get("REFUTES", 0),
                "not_enough_info": cnts.get("NOT ENOUGH INFO", cnts.get("NOT_ENOUGH_INFO", 0)),
                "other_labels": "",
                "average_claim_length": calc_mean_word_len(df["claim"]),
                "average_evidence_length": calc_mean_word_len(df["evidence"]),
            })
            
    # SciFact verification splits
    for split in ["train", "valid", "test"]:
        p = os.path.join(verif_dir, f"scifact_{split}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            cnts = df["label"].value_counts().to_dict()
            other_lbls = "; ".join(f"{k}:{v}" for k, v in cnts.items() if k not in ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"])
            rows.append({
                "dataset": "scifact",
                "split": split,
                "rows": len(df),
                "supports": cnts.get("SUPPORTS", 0),
                "refutes": cnts.get("REFUTES", 0),
                "not_enough_info": cnts.get("NOT_ENOUGH_INFO", 0),
                "other_labels": other_lbls,
                "average_claim_length": calc_mean_word_len(df["claim"]),
                "average_evidence_length": calc_mean_word_len(df["evidence"]),
            })
            
    # 2. Model A Relevance Files
    final_dir = os.path.join(base_dir, "data", "final")
    for split in ["train", "valid", "test"]:
        p = os.path.join(final_dir, f"evidence_relevance_{split}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            pos_cnt = int((df["label"] == 1).sum())
            neg_cnt = int((df["label"] == 0).sum())
            rows.append({
                "dataset": "evidence_relevance",
                "split": split,
                "rows": len(df),
                "supports": 0,
                "refutes": 0,
                "not_enough_info": 0,
                "other_labels": f"relevant(1):{pos_cnt}; not_relevant(0):{neg_cnt}",
                "average_claim_length": calc_mean_word_len(df["claim"]),
                "average_evidence_length": calc_mean_word_len(df["evidence"]),
            })
            
    # 3. Processed LIAR splits
    liar_dir = os.path.join(base_dir, "data", "processed", "liar")
    for split in ["train", "valid", "test"]:
        p = os.path.join(liar_dir, f"liar_{split}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            cnts = df["label"].value_counts().to_dict()
            lbl_str = "; ".join(f"{k}:{v}" for k, v in cnts.items())
            rows.append({
                "dataset": "liar",
                "split": split,
                "rows": len(df),
                "supports": 0,
                "refutes": 0,
                "not_enough_info": 0,
                "other_labels": lbl_str,
                "average_claim_length": calc_mean_word_len(df["statement"]),
                "average_evidence_length": 0.0,
            })
            
    df_summary = pd.DataFrame(rows)
    summary_path = os.path.join(reports_dir, "final_dataset_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"[Summary] Generated final summary table -> {summary_path}")
    return summary_path


def run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    final_dir = os.path.join(base_dir, "data", "final")
    verif_dir = os.path.join(final_dir, "verification")
    reports_dir = os.path.join(base_dir, "reports")
    
    print("=" * 80)
    print(" TRUTHLENS AI — PHASE 2 DATA PREPARATION PIPELINE")
    print("=" * 80)
    
    # 1. Clean FEVER
    print("\n--- STEP 1: CLEANING FEVER ---")
    fever_stats = clean_and_canonicalize_fever(
        raw_file=os.path.join(raw_dir, "fever", "train.jsonl"),
        processed_dir=os.path.join(processed_dir, "fever"),
        final_verification_dir=verif_dir,
    )
    
    # 2. Clean LIAR
    print("\n--- STEP 2: CLEANING LIAR ---")
    liar_stats = clean_and_prepare_liar(
        raw_dir=os.path.join(raw_dir, "liar"),
        processed_dir=os.path.join(processed_dir, "liar"),
    )
    
    # 3. Clean SciFact & Prepare Models A and B
    print("\n--- STEP 3: CLEANING SCIFACT & CREATING MODEL A / MODEL B ---")
    scifact_stats = clean_and_prepare_scifact(
        raw_dir=os.path.join(raw_dir, "scifact"),
        processed_dir=os.path.join(processed_dir, "scifact"),
        final_relevance_dir=final_dir,
        final_verification_dir=verif_dir,
    )
    
    # 4. Generate final dataset summary CSV
    print("\n--- STEP 4: GENERATING FINAL DATASET SUMMARY CSV ---")
    summary_path = generate_summary_csv(base_dir, reports_dir)
    
    print("\n" + "=" * 80)
    print(" PIPELINE EXECUTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
