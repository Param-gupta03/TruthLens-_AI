"""
clean_liar.py - Data Cleaning and Leakage Removal for LIAR.

Handles:
1. Robust parsing using csv.QUOTE_NONE to prevent unescaped quote misalignments.
2. Safe text cleaning (HTML entity unescaping, Unicode quote/whitespace normalization).
3. Metadata imputation:
   - speaker_job_title -> [UNKNOWN_JOB]
   - state_info        -> [UNKNOWN_STATE]
   - context           -> [UNKNOWN_CONTEXT]
4. Duplicate removal:
   - Drops 17 internal train statement duplicates (keeps first occurrence).
5. Cross-split leakage prevention:
   - Removes 5 train/valid overlapping statements from valid.tsv.
   - Removes 4 train/test overlapping statements from test.tsv.
6. Preserves original 6 labels strictly (pants-fire, false, barely-true, half-true, mostly-true, true).
7. Logs all removed records into ml/data/processed/liar/liar_removed_records.csv.
"""

import os
import sys
import csv
from typing import Dict, List, Any, Tuple
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from liar_loader import load_all_liar_splits, LIAR_COLUMNS, LIAR_6_LABELS
from preprocessing import clean_text


def clean_liar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans statements and imputes sentinel tokens for missing metadata in LIAR.
    """
    df = df.copy()
    
    # Safe text normalization
    df["statement"] = df["statement"].apply(clean_text)
    df["context"] = df["context"].apply(clean_text)
    
    # Sentinel token imputation for missing/blank strings
    df["speaker_job_title"] = df["speaker_job_title"].astype(str).str.strip()
    df["speaker_job_title"] = df["speaker_job_title"].replace({"": "[UNKNOWN_JOB]", "nan": "[UNKNOWN_JOB]"})
    
    df["state_info"] = df["state_info"].astype(str).str.strip()
    df["state_info"] = df["state_info"].replace({"": "[UNKNOWN_STATE]", "nan": "[UNKNOWN_STATE]"})
    
    df["context"] = df["context"].astype(str).str.strip()
    df["context"] = df["context"].replace({"": "[UNKNOWN_CONTEXT]", "nan": "[UNKNOWN_CONTEXT]"})
    
    return df


def clean_and_prepare_liar(
    raw_dir: str,
    processed_dir: str
) -> Dict[str, Any]:
    """
    Executes end-to-end cleaning and deduplication on LIAR splits.
    """
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Loading raw LIAR splits from: {raw_dir}")
    splits = load_all_liar_splits(raw_dir)
    
    train_raw = clean_liar_dataframe(splits["train"])
    valid_raw = clean_liar_dataframe(splits["valid"])
    test_raw = clean_liar_dataframe(splits["test"])
    
    removed_log = []
    
    # 1. Deduplicate train internally
    train_raw["stmt_lower"] = train_raw["statement"].str.strip().str.lower()
    dup_mask = train_raw.duplicated(subset=["stmt_lower"], keep="first")
    train_dups = train_raw[dup_mask]
    
    for _, r in train_dups.iterrows():
        removed_log.append({
            "split": "train",
            "id": r["id"],
            "statement": r["statement"],
            "label": r["label"],
            "reason": "internal_train_duplicate_statement",
        })
        
    train_clean = train_raw[~dup_mask].copy()
    print(f"[LIAR] Removed {len(train_dups)} duplicate statements from train (now {len(train_clean):,} rows)")
    
    # Set of clean training statements for leakage check
    clean_train_stmts = set(train_clean["stmt_lower"])
    
    # 2. Check and remove leakage from valid
    valid_raw["stmt_lower"] = valid_raw["statement"].str.strip().str.lower()
    valid_leak_mask = valid_raw["stmt_lower"].isin(clean_train_stmts)
    valid_leaks = valid_raw[valid_leak_mask]
    
    for _, r in valid_leaks.iterrows():
        removed_log.append({
            "split": "valid",
            "id": r["id"],
            "statement": r["statement"],
            "label": r["label"],
            "reason": "cross_split_leakage_with_train",
        })
        
    valid_clean = valid_raw[~valid_leak_mask].copy()
    print(f"[LIAR] Removed {len(valid_leaks)} leakage statements from valid (now {len(valid_clean):,} rows)")
    
    # 3. Check and remove leakage from test
    test_raw["stmt_lower"] = test_raw["statement"].str.strip().str.lower()
    test_leak_mask = test_raw["stmt_lower"].isin(clean_train_stmts)
    test_leaks = test_raw[test_leak_mask]
    
    for _, r in test_leaks.iterrows():
        removed_log.append({
            "split": "test",
            "id": r["id"],
            "statement": r["statement"],
            "label": r["label"],
            "reason": "cross_split_leakage_with_train",
        })
        
    test_clean = test_raw[~test_leak_mask].copy()
    print(f"[LIAR] Removed {len(test_leaks)} leakage statements from test (now {len(test_clean):,} rows)")
    
    # Drop temp lower column
    train_clean = train_clean.drop(columns=["stmt_lower"])
    valid_clean = valid_clean.drop(columns=["stmt_lower"])
    test_clean = test_clean.drop(columns=["stmt_lower"])
    
    # Save clean splits
    train_clean.to_csv(os.path.join(processed_dir, "liar_train.csv"), index=False)
    valid_clean.to_csv(os.path.join(processed_dir, "liar_valid.csv"), index=False)
    test_clean.to_csv(os.path.join(processed_dir, "liar_test.csv"), index=False)
    
    # Save removed log
    df_removed = pd.DataFrame(removed_log)
    df_removed.to_csv(os.path.join(processed_dir, "liar_removed_records.csv"), index=False)
    print(f"[LIAR] Logged {len(df_removed)} removed records -> {os.path.join(processed_dir, 'liar_removed_records.csv')}")
    
    return {
        "raw_counts": {"train": len(train_raw), "valid": len(valid_raw), "test": len(test_raw)},
        "clean_counts": {"train": len(train_clean), "valid": len(valid_clean), "test": len(test_clean)},
        "removed_counts": {
            "train_duplicates": len(train_dups),
            "valid_leakage": len(valid_leaks),
            "test_leakage": len(test_leaks),
            "total_removed": len(df_removed),
        },
        "train_label_distribution": train_clean["label"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_and_prepare_liar(
        raw_dir=os.path.join(base_dir, "data", "raw", "liar"),
        processed_dir=os.path.join(base_dir, "data", "processed", "liar"),
    )
