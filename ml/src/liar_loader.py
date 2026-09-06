"""
liar_loader.py - LIAR Dataset Loader and Parser for TruthLens AI.

This module handles loading, parsing, and analyzing the LIAR benchmark dataset
(Wang, ACL 2017) across train, valid, and test splits. It robustly parses the
14 tab-delimited columns and computes statistics on statements, speakers,
metadata, and 6-class truth labels.
"""

import csv
import os
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

NUMERIC_COUNT_COLS = [
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
]

# Standard 6-point scale defined by PolitiFact and LIAR
LIAR_6_LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]


def load_liar_split(filepath: str) -> Tuple[pd.DataFrame, int]:
    """
    Loads a single LIAR TSV split file.
    
    Uses csv.QUOTE_NONE to ensure unescaped quotes inside speech statements
    do not cause row misalignment or multi-line consumption.
    
    Args:
        filepath: Path to train.tsv, valid.tsv, or test.tsv.
        
    Returns:
        Tuple of (DataFrame of parsed records, count of malformed lines).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"LIAR file not found: {filepath}")
        
    malformed_count = 0
    valid_rows = []
    
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line_idx, row in enumerate(reader, start=1):
            if len(row) == 14:
                valid_rows.append(row)
            elif len(row) > 14:
                # Tab within context or statement; merge trailing elements into context
                merged_row = row[:13] + ["\t".join(row[13:])]
                valid_rows.append(merged_row)
                malformed_count += 1
            else:
                malformed_count += 1
                
    df = pd.DataFrame(valid_rows, columns=LIAR_COLUMNS)
    
    # Cast count columns to numeric
    for col in NUMERIC_COUNT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        
    return df, malformed_count


def load_all_liar_splits(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Loads all available LIAR splits from a directory (train, valid, test).
    """
    splits = {}
    for split_name, filename in [("train", "train.tsv"), ("valid", "valid.tsv"), ("test", "test.tsv")]:
        full_path = os.path.join(data_dir, filename)
        if os.path.exists(full_path):
            df, _ = load_liar_split(full_path)
            splits[split_name] = df
    return splits


def get_liar_statistics(df: pd.DataFrame, split_name: str = "dataset") -> Dict[str, Any]:
    """
    Calculates detailed statistics for a LIAR DataFrame.
    """
    total_records = len(df)
    if total_records == 0:
        return {"total_records": 0}
        
    statement_words = df["statement"].astype(str).apply(lambda s: len(s.split()))
    statement_chars = df["statement"].astype(str).apply(len)
    
    # Missing / empty string counts
    missing_counts = {}
    for col in df.columns:
        null_cnt = int(df[col].isnull().sum())
        empty_cnt = int((df[col].astype(str).str.strip() == "").sum())
        missing_counts[col] = null_cnt + empty_cnt
        
    # Duplicates
    unique_stmts = df["statement"].nunique()
    duplicate_stmts = total_records - unique_stmts
    
    # Label counts
    label_counts = df["label"].value_counts().to_dict()
    
    # Speaker & party metrics
    top_speakers = df["speaker"].value_counts().head(10).to_dict()
    top_parties = df["party_affiliation"].value_counts().head(5).to_dict()
    top_subjects = df["subject"].value_counts().head(10).to_dict()
    
    return {
        "split": split_name,
        "total_records": total_records,
        "columns": list(df.columns),
        "label_counts": label_counts,
        "missing_or_empty_values": missing_counts,
        "unique_statements": unique_stmts,
        "duplicate_statements": duplicate_stmts,
        "statement_word_length": {
            "min": int(statement_words.min()),
            "max": int(statement_words.max()),
            "mean": float(round(statement_words.mean(), 2)),
            "median": float(round(statement_words.median(), 2)),
            "std": float(round(statement_words.std(), 2)),
        },
        "statement_char_length": {
            "min": int(statement_chars.min()),
            "max": int(statement_chars.max()),
            "mean": float(round(statement_chars.mean(), 2)),
            "median": float(round(statement_chars.median(), 2)),
        },
        "top_speakers": top_speakers,
        "top_parties": top_parties,
        "top_subjects": top_subjects,
    }
