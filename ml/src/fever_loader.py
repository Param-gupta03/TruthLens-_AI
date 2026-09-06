"""
fever_loader.py - FEVER Dataset Loader and Parser for TruthLens AI.

This module handles loading, parsing, and analyzing the FEVER (Fact Extraction
and VERification) dataset. It provides utilities to inspect claim lengths,
evidence structures, label distributions, and potential anomalies.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


def load_fever_raw(filepath: str) -> Tuple[pd.DataFrame, int]:
    """
    Loads raw FEVER JSONL file into a pandas DataFrame.
    
    Args:
        filepath: Absolute or relative path to train.jsonl / dev.jsonl.
        
    Returns:
        Tuple of (DataFrame of parsed records, count of malformed lines).
    """
    records: List[Dict[str, Any]] = []
    malformed_count: int = 0
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FEVER file not found: {filepath}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                records.append(data)
            except json.JSONDecodeError:
                malformed_count += 1
                
    df = pd.DataFrame(records)
    return df, malformed_count


def parse_fever_evidence(evidence_field: Any) -> Dict[str, Any]:
    """
    Parses the 3-level nested evidence structure in FEVER:
    List of evidence sets -> List of evidence tuples:
    [annotation_id, evidence_id, wikipedia_page_title, sentence_id]
    
    Returns structured metrics about the evidence.
    """
    if not isinstance(evidence_field, list):
        return {
            "num_evidence_sets": 0,
            "num_evidence_sentences": 0,
            "wiki_pages": [],
            "has_evidence": False,
        }
    
    wiki_pages = set()
    total_sentences = 0
    num_sets = len(evidence_field)
    
    for ev_set in evidence_field:
        if isinstance(ev_set, list):
            for ev_item in ev_set:
                if isinstance(ev_item, list) and len(ev_item) >= 4:
                    annot_id, ev_id, page_title, sent_id = ev_item[:4]
                    if page_title is not None and sent_id is not None:
                        wiki_pages.add(str(page_title))
                        total_sentences += 1
                        
    return {
        "num_evidence_sets": num_sets,
        "num_evidence_sentences": total_sentences,
        "wiki_pages": sorted(list(wiki_pages)),
        "has_evidence": total_sentences > 0,
    }


def get_fever_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates comprehensive statistics for a FEVER DataFrame.
    """
    total_examples = len(df)
    if total_examples == 0:
        return {"total_examples": 0}
        
    # Claim length statistics (words and chars)
    claim_words = df["claim"].astype(str).apply(lambda s: len(s.split()))
    claim_chars = df["claim"].astype(str).apply(len)
    
    # Missing values
    missing_vals = df.isnull().sum().to_dict()
    
    # Duplicates
    num_unique_claims = df["claim"].nunique()
    duplicate_claims_count = total_examples - num_unique_claims
    
    # Conflicting labels for identical claims
    label_per_claim = df.groupby("claim")["label"].nunique()
    conflicting_claims = int((label_per_claim > 1).sum())
    
    # Evidence analysis
    ev_metrics = df["evidence"].apply(parse_fever_evidence)
    has_evidence_series = ev_metrics.apply(lambda m: m["has_evidence"])
    num_sents_series = ev_metrics.apply(lambda m: m["num_evidence_sentences"])
    num_sets_series = ev_metrics.apply(lambda m: m["num_evidence_sets"])
    
    # Label counts
    label_counts = df["label"].value_counts().to_dict()
    verifiable_counts = df["verifiable"].value_counts().to_dict() if "verifiable" in df.columns else {}
    
    return {
        "total_examples": total_examples,
        "columns": list(df.columns),
        "label_counts": label_counts,
        "verifiable_counts": verifiable_counts,
        "missing_values": missing_vals,
        "unique_claims": num_unique_claims,
        "duplicate_claim_rows": duplicate_claims_count,
        "conflicting_claim_annotations": conflicting_claims,
        "claim_word_length": {
            "min": int(claim_words.min()),
            "max": int(claim_words.max()),
            "mean": float(round(claim_words.mean(), 2)),
            "median": float(round(claim_words.median(), 2)),
            "std": float(round(claim_words.std(), 2)),
        },
        "claim_char_length": {
            "min": int(claim_chars.min()),
            "max": int(claim_chars.max()),
            "mean": float(round(claim_chars.mean(), 2)),
            "median": float(round(claim_chars.median(), 2)),
        },
        "evidence_stats": {
            "claims_with_evidence": int(has_evidence_series.sum()),
            "claims_without_evidence": int((~has_evidence_series).sum()),
            "mean_sentences_per_claim": float(round(num_sents_series.mean(), 2)),
            "max_sentences_per_claim": int(num_sents_series.max()),
            "mean_sets_per_claim": float(round(num_sets_series.mean(), 2)),
        },
    }
