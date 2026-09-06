"""
clean_scifact.py - Data Cleaning, Grounded Triple Reconstruction, and Dataset
Preparation for SciFact.

Handles:
1. Loading corpus.jsonl and clean claim annotations across train/dev/test.
2. Resolving rationale sentence coordinates into actual text passages.
3. Reconstructing clean (claim, evidence_text, label) triples with label mapping:
   SUPPORT     -> SUPPORTS
   CONTRADICT  -> REFUTES
   empty       -> NOT_ENOUGH_INFO
4. Creating Model A (Evidence Relevance) datasets:
   - Positives: claim + rationale sentence -> label 1
   - Hard Negatives: claim + non-rationale sentence from same cited abstract -> label 0
   - Generates: evidence_relevance_train.csv, evidence_relevance_valid.csv, evidence_relevance_test.csv
5. Creating Model B (Claim Verification) datasets:
   - scifact_train.csv (from train triples)
   - scifact_valid.csv (from dev triples)
   - scifact_test.csv (from blind test claims)
"""

import os
import sys
import json
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scifact_loader import load_scifact_corpus, load_scifact_claims
from preprocessing import clean_text, map_scifact_label


def reconstruct_scifact_verification_records(
    claims: List[Dict[str, Any]],
    corpus: Dict[int, Dict[str, Any]],
    split_name: str
) -> pd.DataFrame:
    """
    Reconstructs verification triples adhering to the unified Model B schema:
    [claim_id, claim, evidence, label, dataset, source, metadata]
    """
    records = []
    
    for c in claims:
        claim_id = f"scifact_{split_name}_{c['id']}"
        claim_text = clean_text(c["claim"])
        evidence_dict = c.get("evidence", {})
        
        if not evidence_dict:
            # NOT_ENOUGH_INFO claim
            meta = {
                "original_id": c["id"],
                "cited_doc_ids": c.get("cited_doc_ids", []),
                "is_grounded": False,
            }
            records.append({
                "claim_id": claim_id,
                "claim": claim_text,
                "evidence": "",
                "label": "NOT_ENOUGH_INFO",
                "dataset": "scifact",
                "source": f"Docs: {','.join(map(str, c.get('cited_doc_ids', [])))}" if c.get("cited_doc_ids") else "",
                "metadata": json.dumps(meta),
            })
            continue
            
        # Grounded claim with one or more cited documents and rationales
        for doc_id_str, rationale_list in evidence_dict.items():
            doc_id = int(doc_id_str)
            doc = corpus.get(doc_id)
            doc_title = doc["title"] if doc else ""
            abstract_sents = doc["abstract"] if doc else []
            
            for rat_idx, rat in enumerate(rationale_list):
                mapped_lbl = map_scifact_label(rat.get("label", "NOT_ENOUGH_INFO"))
                sent_indices = rat.get("sentences", [])
                
                # Extract sentence text from abstract
                ev_sents = [
                    clean_text(abstract_sents[idx])
                    for idx in sent_indices
                    if idx < len(abstract_sents)
                ]
                ev_text = " ".join(ev_sents)
                
                triple_id = f"{claim_id}_doc{doc_id}_rat{rat_idx}"
                meta = {
                    "original_id": c["id"],
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "sentence_indices": sent_indices,
                    "is_grounded": True,
                }
                
                records.append({
                    "claim_id": triple_id,
                    "claim": claim_text,
                    "evidence": ev_text,
                    "label": mapped_lbl,
                    "dataset": "scifact",
                    "source": f"PubMed_{doc_id}: {doc_title}",
                    "metadata": json.dumps(meta),
                })
                
    df = pd.DataFrame(records)
    # Deduplicate in case identical (claim, evidence) exists
    df = df.drop_duplicates(subset=["claim", "evidence"], keep="first")
    return df


def generate_relevance_pairs(
    claims: List[Dict[str, Any]],
    corpus: Dict[int, Dict[str, Any]],
    split_name: str
) -> pd.DataFrame:
    """
    Generates sentence-level relevance pairs for Model A:
    - Positive (label=1): claim + rationale sentence
    - Hard Negative (label=0): claim + non-rationale sentence from SAME cited document abstract
    """
    pairs = []
    
    for c in claims:
        claim_id = f"scifact_{c['id']}"
        claim_text = clean_text(c["claim"])
        evidence_dict = c.get("evidence", {})
        
        if not evidence_dict:
            # Check if there are cited doc IDs where all sentences are negative
            for doc_id in c.get("cited_doc_ids", []):
                doc = corpus.get(int(doc_id))
                if not doc:
                    continue
                for s_idx, s_text in enumerate(doc.get("abstract", [])):
                    pairs.append({
                        "claim_id": claim_id,
                        "claim": claim_text,
                        "evidence": clean_text(s_text),
                        "label": 0,
                        "document_id": doc_id,
                        "dataset": "scifact",
                    })
            continue
            
        for doc_id_str, rationale_list in evidence_dict.items():
            doc_id = int(doc_id_str)
            doc = corpus.get(doc_id)
            if not doc:
                continue
            abstract = doc.get("abstract", [])
            
            # Collect all positive sentence indices for this document
            pos_indices = set()
            for rat in rationale_list:
                pos_indices.update(rat.get("sentences", []))
                
            # Emit positives
            for p_idx in pos_indices:
                if p_idx < len(abstract):
                    pairs.append({
                        "claim_id": claim_id,
                        "claim": claim_text,
                        "evidence": clean_text(abstract[p_idx]),
                        "label": 1,
                        "document_id": doc_id,
                        "dataset": "scifact",
                    })
                    
            # Emit hard negatives from the same document abstract
            for s_idx, s_text in enumerate(abstract):
                if s_idx not in pos_indices:
                    pairs.append({
                        "claim_id": claim_id,
                        "claim": claim_text,
                        "evidence": clean_text(s_text),
                        "label": 0,
                        "document_id": doc_id,
                        "dataset": "scifact",
                    })
                    
    df = pd.DataFrame(pairs)
    if not df.empty:
        df = df.drop_duplicates(subset=["claim", "evidence"], keep="first")
    return df


def clean_and_prepare_scifact(
    raw_dir: str,
    processed_dir: str,
    final_relevance_dir: str,
    final_verification_dir: str
) -> Dict[str, Any]:
    """
    Executes end-to-end cleaning and dataset generation for SciFact.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(final_relevance_dir, exist_ok=True)
    os.makedirs(final_verification_dir, exist_ok=True)
    
    corpus_p = os.path.join(raw_dir, "corpus.jsonl")
    claims_tr_p = os.path.join(raw_dir, "claims_train.jsonl")
    claims_dv_p = os.path.join(raw_dir, "claims_dev.jsonl")
    claims_te_p = os.path.join(raw_dir, "claims_test.jsonl")
    
    print(f"Loading SciFact raw files from: {raw_dir}")
    corpus = load_scifact_corpus(corpus_p)
    claims_train = load_scifact_claims(claims_tr_p)
    claims_dev_raw = load_scifact_claims(claims_dv_p)
    claims_test = load_scifact_claims(claims_te_p)
    
    # Check and remove train/dev claim leakage (claims 870 and 1292 duplicate 871 and 1291)
    train_claims_set = set(c["claim"].strip().lower() for c in claims_train)
    claims_dev = []
    scifact_removed = []
    
    for c in claims_dev_raw:
        c_norm = c["claim"].strip().lower()
        if c_norm in train_claims_set:
            scifact_removed.append({
                "split": "dev",
                "id": c["id"],
                "claim": c["claim"],
                "reason": "cross_split_claim_leakage_with_train",
            })
        else:
            claims_dev.append(c)
            
    df_sci_removed = pd.DataFrame(scifact_removed)
    df_sci_removed.to_csv(os.path.join(processed_dir, "scifact_removed_records.csv"), index=False)
    print(f"[SciFact] Removed {len(scifact_removed)} leaking claims from dev (logged to scifact_removed_records.csv)")
    
    # 1. Reconstruct Model B Verification datasets
    print("[SciFact] Reconstructing verification triples for Model B...")
    df_verif_train = reconstruct_scifact_verification_records(claims_train, corpus, "train")
    df_verif_valid = reconstruct_scifact_verification_records(claims_dev, corpus, "valid")
    
    # Test split (blind evaluation)
    test_records = []
    for c in claims_test:
        test_records.append({
            "claim_id": f"scifact_test_{c['id']}",
            "claim": clean_text(c["claim"]),
            "evidence": "",
            "label": "UNLABELED",
            "dataset": "scifact",
            "source": "",
            "metadata": json.dumps({"original_id": c["id"], "task": "blind_test_evaluation"}),
        })
    df_verif_test = pd.DataFrame(test_records)
    
    # Save Model B verification CSVs
    df_verif_train.to_csv(os.path.join(final_verification_dir, "scifact_train.csv"), index=False)
    df_verif_valid.to_csv(os.path.join(final_verification_dir, "scifact_valid.csv"), index=False)
    df_verif_test.to_csv(os.path.join(final_verification_dir, "scifact_test.csv"), index=False)
    print(f"[SciFact] Saved Model B verification files: Train={len(df_verif_train):,}, Valid={len(df_verif_valid):,}, Test={len(df_verif_test):,}")
    
    # Also save to processed/scifact
    df_verif_train.to_csv(os.path.join(processed_dir, "scifact_train_triples.csv"), index=False)
    df_verif_valid.to_csv(os.path.join(processed_dir, "scifact_dev_triples.csv"), index=False)
    df_verif_test.to_csv(os.path.join(processed_dir, "scifact_test_claims.csv"), index=False)
    
    # 2. Generate Model A Evidence Relevance datasets
    print("[SciFact] Generating Model A evidence relevance datasets (with hard negatives)...")
    df_rel_train = generate_relevance_pairs(claims_train, corpus, "train")
    
    # Split dev claims (300 claims) into 150 valid / 150 test claims to evaluate Model A with ground truth
    has_ev_list = [bool(c.get("evidence", {})) for c in claims_dev]
    claims_dev_valid, claims_dev_test = train_test_split(
        claims_dev,
        test_size=0.50,
        random_state=42,
        stratify=has_ev_list
    )
    
    df_rel_valid = generate_relevance_pairs(claims_dev_valid, corpus, "valid")
    df_rel_test = generate_relevance_pairs(claims_dev_test, corpus, "test")
    
    # Save Model A relevance CSVs
    rel_train_file = os.path.join(final_relevance_dir, "evidence_relevance_train.csv")
    rel_valid_file = os.path.join(final_relevance_dir, "evidence_relevance_valid.csv")
    rel_test_file = os.path.join(final_relevance_dir, "evidence_relevance_test.csv")
    
    df_rel_train.to_csv(rel_train_file, index=False)
    df_rel_valid.to_csv(rel_valid_file, index=False)
    df_rel_test.to_csv(rel_test_file, index=False)
    print(f"[SciFact] Saved Model A relevance files:")
    print(f"  Train: {len(df_rel_train):,} pairs (Pos: {(df_rel_train['label']==1).sum()}, Neg: {(df_rel_train['label']==0).sum()})")
    print(f"  Valid: {len(df_rel_valid):,} pairs (Pos: {(df_rel_valid['label']==1).sum()}, Neg: {(df_rel_valid['label']==0).sum()})")
    print(f"  Test : {len(df_rel_test):,} pairs (Pos: {(df_rel_test['label']==1).sum()}, Neg: {(df_rel_test['label']==0).sum()})")
    
    return {
        "verification_counts": {
            "train": len(df_verif_train),
            "valid": len(df_verif_valid),
            "test": len(df_verif_test),
        },
        "relevance_counts": {
            "train": len(df_rel_train),
            "valid": len(df_rel_valid),
            "test": len(df_rel_test),
        },
        "train_verification_label_distribution": df_verif_train["label"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_and_prepare_scifact(
        raw_dir=os.path.join(base_dir, "data", "raw", "scifact"),
        processed_dir=os.path.join(base_dir, "data", "processed", "scifact"),
        final_relevance_dir=os.path.join(base_dir, "data", "final"),
        final_verification_dir=os.path.join(base_dir, "data", "final", "verification"),
    )
