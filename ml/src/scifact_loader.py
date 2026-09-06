"""
scifact_loader.py - SciFact Dataset Loader and Evidence Extractor for TruthLens AI.

This module handles loading and parsing the SciFact scientific fact-checking
benchmark (Wadden et al., EMNLP 2020). It loads the research paper corpus and
claim annotations, reconstructing grounded (claim, evidence_text, label) triples.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


def load_scifact_corpus(corpus_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Loads the SciFact corpus of scientific papers from corpus.jsonl.
    
    Returns:
        Dict mapping doc_id (int) to document dictionary with:
        {'doc_id': int, 'title': str, 'abstract': List[str], 'structured': bool}
    """
    corpus = {}
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"SciFact corpus not found at: {corpus_path}")
        
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                doc = json.loads(line_str)
                corpus[int(doc["doc_id"])] = doc
                
    return corpus


def load_scifact_claims(claims_path: str) -> List[Dict[str, Any]]:
    """
    Loads SciFact claim annotations from a JSONL file.
    
    Returns:
        List of claim dictionaries.
    """
    if not os.path.exists(claims_path):
        raise FileNotFoundError(f"SciFact claims file not found: {claims_path}")
        
    claims = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                claims.append(json.loads(line_str))
                
    return claims


def reconstruct_scifact_triples(
    claims: List[Dict[str, Any]],
    corpus: Dict[int, Dict[str, Any]],
    include_nei: bool = True
) -> pd.DataFrame:
    """
    Reconstructs explicit (claim, evidence_text, label) triples from SciFact annotations.
    
    For claims with rationales:
        Extracts the exact sentence text from the cited document's abstract.
    For claims without evidence (empty evidence dict):
        Label is set to 'NOT_ENOUGH_INFO' with empty evidence.
        
    Returns:
        DataFrame with columns:
        ['claim_id', 'claim', 'doc_id', 'doc_title', 'sentence_indices', 'evidence_text', 'label']
    """
    rows = []
    
    for claim_obj in claims:
        claim_id = claim_obj["id"]
        claim_text = claim_obj["claim"]
        evidence_dict = claim_obj.get("evidence", {})
        
        if not evidence_dict:
            if include_nei:
                rows.append({
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "doc_id": None,
                    "doc_title": None,
                    "sentence_indices": [],
                    "evidence_text": "",
                    "label": "NOT_ENOUGH_INFO",
                })
            continue
            
        for doc_id_str, rationale_list in evidence_dict.items():
            doc_id = int(doc_id_str)
            doc = corpus.get(doc_id)
            doc_title = doc["title"] if doc else ""
            abstract_sents = doc["abstract"] if doc else []
            
            for item in rationale_list:
                label = item.get("label", "UNKNOWN")
                sentence_indices = item.get("sentences", [])
                
                # Extract sentence text
                ev_sentences = [
                    abstract_sents[idx]
                    for idx in sentence_indices
                    if idx < len(abstract_sents)
                ]
                evidence_text = " ".join(ev_sentences)
                
                rows.append({
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "sentence_indices": sentence_indices,
                    "evidence_text": evidence_text,
                    "label": label,
                })
                
    return pd.DataFrame(rows)


def get_scifact_statistics(
    claims: List[Dict[str, Any]],
    corpus: Optional[Dict[int, Dict[str, Any]]] = None,
    split_name: str = "split"
) -> Dict[str, Any]:
    """
    Computes summary statistics for a SciFact split.
    """
    total_claims = len(claims)
    if total_claims == 0:
        return {"total_claims": 0}
        
    df_claims = pd.DataFrame(claims)
    claim_words = df_claims["claim"].astype(str).apply(lambda s: len(s.split()))
    
    has_evidence_list = []
    evidence_item_labels = []
    cited_docs = set()
    
    for c in claims:
        ev = c.get("evidence", {})
        has_ev = bool(ev)
        has_evidence_list.append(has_ev)
        if "cited_doc_ids" in c:
            cited_docs.update(c["cited_doc_ids"])
            
        if ev:
            for doc_id, ev_list in ev.items():
                for item in ev_list:
                    evidence_item_labels.append(item.get("label", "UNKNOWN"))
                    
    has_ev_series = pd.Series(has_evidence_list)
    label_series = pd.Series(evidence_item_labels)
    
    stats = {
        "split": split_name,
        "total_claims": total_claims,
        "claims_with_evidence": int(has_ev_series.sum()),
        "claims_without_evidence_nei": int((~has_ev_series).sum()),
        "evidence_label_counts": label_series.value_counts().to_dict() if len(label_series) > 0 else {},
        "unique_cited_docs": len(cited_docs),
        "claim_word_length": {
            "min": int(claim_words.min()),
            "max": int(claim_words.max()),
            "mean": float(round(claim_words.mean(), 2)),
            "median": float(round(claim_words.median(), 2)),
        },
    }
    
    if corpus is not None:
        stats["corpus_total_docs"] = len(corpus)
        abstract_lens = [len(d.get("abstract", [])) for d in corpus.values()]
        stats["corpus_total_sentences"] = sum(abstract_lens)
        stats["mean_abstract_sentences"] = float(round(np.mean(abstract_lens), 2))
        
    return stats
