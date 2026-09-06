"""
clean_fever.py - Data Cleaning and Canonicalization for FEVER.

Handles:
1. Identifying duplicate claims and claims with conflicting labels.
2. Quarantining conflicting claims (753 unique claims, 2,557 rows) into
   ml/data/processed/fever/fever_quarantined_conflicts.csv.
3. Consolidating duplicate claims with identical labels into one canonical record,
   merging evidence sets without losing evidence pointers.
4. Preserving original Wikipedia page titles and sentence IDs without fabricating text.
5. Creating stratified train/validation splits (90/10) for Model B verification.
6. Logging all removed and quarantined records.
"""

import os
import sys
import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fever_loader import load_fever_raw, parse_fever_evidence
from preprocessing import clean_text


def extract_fever_coordinates(evidence_field: Any) -> List[Tuple[str, int]]:
    """
    Extracts unique (wikipedia_page, sentence_id) pairs from FEVER evidence.
    """
    coords = []
    if not isinstance(evidence_field, list):
        return coords
        
    for ev_set in evidence_field:
        if isinstance(ev_set, list):
            for ev_item in ev_set:
                if isinstance(ev_item, list) and len(ev_item) >= 4:
                    _, _, page, sent_id = ev_item[:4]
                    if page is not None and sent_id is not None:
                        coords.append((str(page), int(sent_id)))
    # Deduplicate while preserving order
    seen = set()
    unique_coords = []
    for c in coords:
        if c not in seen:
            seen.add(c)
            unique_coords.append(c)
    return unique_coords


def clean_and_canonicalize_fever(
    raw_file: str,
    processed_dir: str,
    final_verification_dir: str
) -> Dict[str, Any]:
    """
    Cleans raw FEVER, isolates conflicting annotations, canonicalizes duplicate
    claims, and produces stratified splits for Model B.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(final_verification_dir, exist_ok=True)
    
    print(f"Loading raw FEVER from: {raw_file}")
    df_raw, malformed_count = load_fever_raw(raw_file)
    total_raw_rows = len(df_raw)
    
    # 1. Clean claim text safely
    df_raw["claim_clean"] = df_raw["claim"].apply(clean_text)
    df_raw["claim_norm"] = df_raw["claim_clean"].str.strip().str.lower()
    
    # 2. Identify claims with conflicting labels
    label_group = df_raw.groupby("claim_norm")["label"].unique()
    conflicting_claims_set = set(label_group[label_group.apply(len) > 1].index)
    
    df_conflicts = df_raw[df_raw["claim_norm"].isin(conflicting_claims_set)].copy()
    df_clean_pool = df_raw[~df_raw["claim_norm"].isin(conflicting_claims_set)].copy()
    
    # 3. Quarantine conflicting claims
    quarantined_csv = os.path.join(processed_dir, "fever_quarantined_conflicts.csv")
    df_conflicts.to_csv(quarantined_csv, index=False)
    print(f"[FEVER] Quarantined {len(df_conflicts):,} rows across {len(conflicting_claims_set):,} conflicting claims -> {quarantined_csv}")
    
    # 4. Canonicalize duplicate claims with identical labels
    canonical_records = []
    claim_groups = defaultdict(list)
    for idx, row in df_clean_pool.iterrows():
        claim_groups[row["claim_norm"]].append(row)
        
    merged_rows_count = 0
    
    for norm_text, row_list in claim_groups.items():
        first_row = row_list[0]
        claim_text = first_row["claim_clean"]
        raw_label = first_row["label"]
        label = "NOT_ENOUGH_INFO" if raw_label == "NOT ENOUGH INFO" else raw_label
        all_ids = [r["id"] for r in row_list]
        canonical_id = f"fever_{all_ids[0]}"
        
        # Merge all evidence coordinates across annotations
        all_coords = []
        for r in row_list:
            all_coords.extend(extract_fever_coordinates(r.get("evidence", [])))
            
        # Deduplicate coordinates
        unique_coords = []
        seen = set()
        for c in all_coords:
            if c not in seen:
                seen.add(c)
                unique_coords.append(c)
                
        if len(row_list) > 1:
            merged_rows_count += (len(row_list) - 1)
            
        # Format evidence string
        # Since Wikipedia raw text is NOT in local storage, preserve exact pointers
        if label == "NOT_ENOUGH_INFO" or len(unique_coords) == 0:
            evidence_str = ""
            source_str = ""
        else:
            evidence_str = "Wikipedia: " + "; ".join(f"{page} (sentence {sid})" for page, sid in unique_coords)
            source_str = ", ".join(sorted(list(set(page for page, _ in unique_coords))))
            
        metadata_dict = {
            "unresolved_evidence_text": True if unique_coords else False,
            "evidence_references": [{"page": p, "sentence_id": s} for p, s in unique_coords],
            "original_ids": all_ids,
            "annotation_count": len(row_list),
            "verifiable": first_row.get("verifiable", "VERIFIABLE"),
        }
        
        canonical_records.append({
            "claim_id": canonical_id,
            "claim": claim_text,
            "evidence": evidence_str,
            "label": label,
            "dataset": "fever",
            "source": source_str,
            "metadata": json.dumps(metadata_dict),
        })
        
    df_canonical = pd.DataFrame(canonical_records)
    canonical_csv = os.path.join(processed_dir, "fever_cleaned_canonical.csv")
    df_canonical.to_csv(canonical_csv, index=False)
    print(f"[FEVER] Consolidated {len(df_clean_pool):,} clean rows into {len(df_canonical):,} canonical claims (merged {merged_rows_count:,} duplicate rows)")
    
    # 5. Log removed / quarantined records
    removed_records = []
    for c_text in conflicting_claims_set:
        sub = df_conflicts[df_conflicts["claim_clean"] == c_text]
        removed_records.append({
            "claim": c_text,
            "reason": "quarantined_conflicting_labels",
            "count_rows": len(sub),
            "labels_observed": "|".join(sub["label"].unique()),
        })
    removed_records.append({
        "claim": "[DUPLICATE_ROWS_MERGED]",
        "reason": "duplicate_claims_identical_labels_consolidated",
        "count_rows": merged_rows_count,
        "labels_observed": "various",
    })
    pd.DataFrame(removed_records).to_csv(os.path.join(processed_dir, "fever_removed_records.csv"), index=False)
    
    # 6. Stratified Train / Validation Split (90% / 10%)
    train_df, valid_df = train_test_split(
        df_canonical,
        test_size=0.10,
        random_state=42,
        stratify=df_canonical["label"]
    )
    
    train_file = os.path.join(final_verification_dir, "fever_train.csv")
    valid_file = os.path.join(final_verification_dir, "fever_valid.csv")
    
    train_df.to_csv(train_file, index=False)
    valid_df.to_csv(valid_file, index=False)
    print(f"[FEVER] Created Model B splits: Train={len(train_df):,}, Valid={len(valid_df):,} -> {final_verification_dir}")
    
    return {
        "raw_total": total_raw_rows,
        "quarantined_rows": len(df_conflicts),
        "quarantined_unique_claims": len(conflicting_claims_set),
        "merged_duplicate_rows": merged_rows_count,
        "clean_canonical_claims": len(df_canonical),
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "label_distribution": df_canonical["label"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_and_canonicalize_fever(
        raw_file=os.path.join(base_dir, "data", "raw", "fever", "train.jsonl"),
        processed_dir=os.path.join(base_dir, "data", "processed", "fever"),
        final_verification_dir=os.path.join(base_dir, "data", "final", "verification"),
    )
