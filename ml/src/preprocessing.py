"""
preprocessing.py - Text Cleaning and Schema Normalization for TruthLens AI.

This module provides preprocessing utilities for the TruthLens AI pipeline,
including text normalization, label mapping, and schema formatting across
FEVER, LIAR, and SciFact.
"""

import re
import html
from typing import Dict, List, Any, Optional
import pandas as pd


def clean_text(text: str) -> str:
    """
    Cleans raw text by normalizing unicode characters, removing unescaped entities,
    collapsing redundant whitespace, and handling quotes.
    """
    if not isinstance(text, str):
        return ""
        
    # Unescape HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)
    
    # Replace non-breaking spaces and other special spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    
    # Standardize curly single/double quotes to standard quotes
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
    
    # Standardize dashes
    text = re.sub(r"[\u2013\u2014]", "-", text)
    
    # Remove control characters except standard tabs/newlines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def map_scifact_label(raw_label: str) -> str:
    """
    Maps SciFact raw labels ('SUPPORT', 'CONTRADICT', '') to standardized
    verification labels ('SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO').
    """
    label_norm = str(raw_label).strip().upper()
    mapping = {
        "SUPPORT": "SUPPORTS",
        "CONTRADICT": "REFUTES",
        "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
        "NEI": "NOT_ENOUGH_INFO",
        "": "NOT_ENOUGH_INFO",
    }
    return mapping.get(label_norm, label_norm)


def map_liar_label_to_3way(raw_label: str) -> str:
    """
    Proposes a mapping of LIAR's 6-point scale into a 3-way verification schema:
    - 'true', 'mostly-true' -> 'SUPPORTS'
    - 'barely-true', 'false', 'pants-fire' -> 'REFUTES'
    - 'half-true' -> 'NOT_ENOUGH_INFO' (ambiguous / mixed claims)
    
    NOTE: LIAR does NOT contain evidence text in its raw tsv files.
    This mapping is intended for downstream claim classification or when
    retrieving external news evidence.
    """
    label_norm = str(raw_label).strip().lower()
    mapping = {
        "true": "SUPPORTS",
        "mostly-true": "SUPPORTS",
        "half-true": "NOT_ENOUGH_INFO",
        "barely-true": "REFUTES",
        "false": "REFUTES",
        "pants-fire": "REFUTES",
    }
    return mapping.get(label_norm, "UNKNOWN")


def map_liar_label_to_binary(raw_label: str) -> Optional[str]:
    """
    Binary mapping commonly used in political fact-checking literature:
    - 'true', 'mostly-true' -> 'TRUE'
    - 'barely-true', 'false', 'pants-fire' -> 'FALSE'
    - 'half-true' -> Ambiguous (often excluded or evaluated separately)
    """
    label_norm = str(raw_label).strip().lower()
    if label_norm in ("true", "mostly-true"):
        return "TRUE"
    elif label_norm in ("barely-true", "false", "pants-fire"):
        return "FALSE"
    return None


def format_to_unified_schema(
    claim_id: str,
    claim: str,
    evidence: str,
    label: str,
    source: str,
    domain: str,
    dataset: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Formats a fact-checking record into the unified TruthLens schema.
    
    Schema Fields:
    - claim_id (str): Unique globally qualified identifier (e.g., 'fever_75397')
    - claim (str): Normalized claim statement
    - evidence (str): Concatenated text of verified evidence or rationale
    - label (str): Normalized 3-way label ('SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO')
    - source (str): Citation, document ID, Wikipedia title, or speaker
    - domain (str): Subject domain ('encyclopedic', 'political', 'biomedical')
    - dataset (str): Origin dataset ('fever', 'liar', 'scifact')
    - metadata (dict): Dataset-specific attributes preserved in full
    """
    return {
        "claim_id": str(claim_id),
        "claim": clean_text(claim),
        "evidence": clean_text(evidence),
        "label": label,
        "source": str(source) if source else "",
        "domain": domain,
        "dataset": dataset,
        "metadata": metadata or {},
    }
