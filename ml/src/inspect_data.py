"""
inspect_data.py - Comprehensive Data Inspection Script for TruthLens AI.

This script inspects FEVER, LIAR, and SciFact datasets locally:
- Validates file integrity and parses formats
- Calculates dataset sizes, columns, and data types
- Computes label distributions and percentages
- Computes claim length statistics
- Analyzes evidence availability and representation
- Detects missing/empty values and malformed rows
- Detects duplicates and cross-split leakage
- Prints sample records from each dataset
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add src directory to path if running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fever_loader import load_fever_raw, get_fever_statistics
from liar_loader import load_all_liar_splits, get_liar_statistics, LIAR_COLUMNS
from scifact_loader import (
    load_scifact_corpus,
    load_scifact_claims,
    reconstruct_scifact_triples,
    get_scifact_statistics,
)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def print_sub_header(title: str):
    print("\n" + "-" * 60)
    print(f" {title}")
    print("-" * 60)


def inspect_fever(raw_dir: str):
    print_header("1. FEVER DATASET INSPECTION")
    fever_path = os.path.join(raw_dir, "fever", "train.jsonl")
    
    if not os.path.exists(fever_path):
        print(f"File not found: {fever_path}")
        return
        
    print(f"File path: {fever_path}")
    print(f"File size: {os.path.getsize(fever_path):,} bytes ({os.path.getsize(fever_path) / (1024*1024):.2f} MB)")
    print(f"Format: JSON Lines (.jsonl)")
    
    df_fever, malformed = load_fever_raw(fever_path)
    stats = get_fever_statistics(df_fever)
    
    print(f"Total rows parsed: {stats['total_examples']:,}")
    print(f"Malformed rows: {malformed}")
    print(f"Columns: {stats['columns']}")
    print(f"Data types:\n{df_fever.dtypes.to_dict()}")
    
    print_sub_header("FEVER Label Distribution")
    for label, count in stats["label_counts"].items():
        pct = (count / stats["total_examples"]) * 100
        print(f"  {label:<18}: {count:>7,} ({pct:>5.2f}%)")
        
    print_sub_header("FEVER Verifiable Flag Distribution")
    for ver, count in stats["verifiable_counts"].items():
        pct = (count / stats["total_examples"]) * 100
        print(f"  {ver:<18}: {count:>7,} ({pct:>5.2f}%)")
        
    print_sub_header("FEVER Missing Values")
    for col, nulls in stats["missing_values"].items():
        print(f"  {col:<15}: {nulls} missing")
        
    print_sub_header("FEVER Duplicates & Annotation Consistency")
    print(f"  Unique claims: {stats['unique_claims']:,}")
    print(f"  Duplicate claim rows: {stats['duplicate_claim_rows']:,}")
    print(f"  Claims with conflicting labels across annotations: {stats['conflicting_claim_annotations']:,}")
    
    print_sub_header("FEVER Claim Length Statistics")
    w = stats["claim_word_length"]
    print(f"  Word count  - Min: {w['min']}, Max: {w['max']}, Mean: {w['mean']}, Median: {w['median']}, Std: {w['std']}")
    c = stats["claim_char_length"]
    print(f"  Char count  - Min: {c['min']}, Max: {c['max']}, Mean: {c['mean']}, Median: {c['median']}")
    
    print_sub_header("FEVER Evidence Availability & Structure")
    ev = stats["evidence_stats"]
    print(f"  Claims with evidence references    : {ev['claims_with_evidence']:,} ({ev['claims_with_evidence']/stats['total_examples']*100:.1f}%)")
    print(f"  Claims without evidence references : {ev['claims_without_evidence']:,} ({ev['claims_without_evidence']/stats['total_examples']*100:.1f}%)")
    print(f"  Mean evidence sentences per claim  : {ev['mean_sentences_per_claim']}")
    print(f"  Max evidence sentences for a claim : {ev['max_sentences_per_claim']}")
    print(f"  Evidence representation format     : 3-level nested list [Sets -> Sentences -> [annot_id, ev_id, wiki_page, sent_id]]")
    print(f"  IMPORTANT NOTE: Wikipedia raw text is NOT included inside train.jsonl (only title + sentence index).")
    
    print_sub_header("FEVER Sample Records (one per label)")
    for target_label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        sample = df_fever[df_fever["label"] == target_label].iloc[0]
        print(f"\n  [Label: {target_label}]")
        print(f"    ID       : {sample['id']}")
        print(f"    Claim    : {sample['claim']}")
        print(f"    Evidence : {sample['evidence'][:2]}")


def inspect_liar(raw_dir: str):
    print_header("2. LIAR DATASET INSPECTION")
    liar_dir = os.path.join(raw_dir, "liar")
    splits = load_all_liar_splits(liar_dir)
    
    if not splits:
        print(f"No LIAR splits found in: {liar_dir}")
        return
        
    print(f"Directory: {liar_dir}")
    print(f"Format: Tab-Separated Values (TSV, headerless, 14 columns)")
    print(f"Splits found: {list(splits.keys())}")
    
    total_liar_rows = sum(len(df) for df in splits.values())
    print(f"Total LIAR examples across all splits: {total_liar_rows:,}")
    
    for split_name, df_split in splits.items():
        print_sub_header(f"LIAR Split: {split_name.upper()}")
        fpath = os.path.join(liar_dir, f"{split_name}.tsv")
        fsize = os.path.getsize(fpath) if os.path.exists(fpath) else 0
        stats = get_liar_statistics(df_split, split_name)
        
        print(f"  Rows: {len(df_split):,} | Size: {fsize:,} bytes | Columns: {len(df_split.columns)}")
        print(f"  Label distribution in {split_name}:")
        for label, count in stats["label_counts"].items():
            pct = (count / len(df_split)) * 100
            print(f"    {label:<15}: {count:>5} ({pct:>5.2f}%)")
            
        w = stats["statement_word_length"]
        print(f"  Statement words - Min: {w['min']}, Max: {w['max']}, Mean: {w['mean']}, Median: {w['median']}")
        print(f"  Duplicate statements in split: {stats['duplicate_statements']}")
        
        # Missing values
        empty_cols = {k: v for k, v in stats["missing_or_empty_values"].items() if v > 0}
        if empty_cols:
            print(f"  Empty/Missing fields: {empty_cols}")
            
    # Cross-split leakage in LIAR
    if "train" in splits and "valid" in splits and "test" in splits:
        print_sub_header("LIAR Cross-Split Statement Leakage")
        tr_stmts = set(splits["train"]["statement"].str.strip().str.lower())
        val_stmts = set(splits["valid"]["statement"].str.strip().str.lower())
        te_stmts = set(splits["test"]["statement"].str.strip().str.lower())
        
        tr_val_leak = tr_stmts.intersection(val_stmts)
        tr_te_leak = tr_stmts.intersection(te_stmts)
        val_te_leak = val_stmts.intersection(te_stmts)
        
        print(f"  Train & Valid overlapping statements: {len(tr_val_leak)}")
        print(f"  Train & Test overlapping statements : {len(tr_te_leak)}")
        print(f"  Valid & Test overlapping statements : {len(val_te_leak)}")
        if tr_val_leak:
            print(f"  Sample leakage (train/valid): {list(tr_val_leak)[:2]}")
            
    print_sub_header("LIAR Metadata & Evidence Availability")
    print(f"  Evidence availability : NO factual evidence / document text is provided in LIAR TSVs.")
    print(f"  Available metadata    : Speaker, party, job, state, context/venue, 5 credit-history counts.")
    print(f"  Suitability           : Useful for speaker credibility & statement classification, but")
    print(f"                          CANNOT directly train Claim+Evidence verification without external retrieval.")
    
    print_sub_header("LIAR Sample Records (one per label)")
    df_train = splits.get("train", pd.DataFrame())
    if not df_train.empty:
        for lbl in ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]:
            subset = df_train[df_train["label"] == lbl]
            if not subset.empty:
                s = subset.iloc[0]
                print(f"\n  [Label: {lbl}]")
                print(f"    Speaker   : {s['speaker']} ({s['party_affiliation']}, {s['state_info']})")
                print(f"    Context   : {s['context']}")
                print(f"    Statement : {s['statement']}")


def inspect_scifact(raw_dir: str):
    print_header("3. SCIFACT DATASET INSPECTION")
    scifact_dir = os.path.join(raw_dir, "scifact")
    
    corpus_path = os.path.join(scifact_dir, "corpus.jsonl")
    claims_train_p = os.path.join(scifact_dir, "claims_train.jsonl")
    claims_dev_p = os.path.join(scifact_dir, "claims_dev.jsonl")
    claims_test_p = os.path.join(scifact_dir, "claims_test.jsonl")
    
    corpus = load_scifact_corpus(corpus_path)
    claims_tr = load_scifact_claims(claims_train_p)
    claims_dv = load_scifact_claims(claims_dev_p)
    claims_te = load_scifact_claims(claims_test_p)
    
    print(f"Directory: {scifact_dir}")
    print(f"Corpus documents loaded: {len(corpus):,} research paper abstracts")
    
    # Corpus sentence analysis
    abstract_lens = [len(doc.get("abstract", [])) for doc in corpus.values()]
    total_sentences = sum(abstract_lens)
    print(f"Total corpus sentences: {total_sentences:,}")
    print(f"Sentences per abstract: Min={min(abstract_lens)}, Max={max(abstract_lens)}, Mean={np.mean(abstract_lens):.2f}")
    
    # Claims analysis
    for name, c_list in [("claims_train", claims_tr), ("claims_dev", claims_dv), ("claims_test", claims_te)]:
        print_sub_header(f"SciFact Split: {name.upper()}")
        stats = get_scifact_statistics(c_list, split_name=name)
        print(f"  Total claims: {stats['total_claims']}")
        print(f"  Claims with evidence: {stats['claims_with_evidence']} ({stats['claims_with_evidence']/stats['total_claims']*100:.1f}%)")
        print(f"  Claims without evidence (NEI): {stats['claims_without_evidence_nei']} ({stats['claims_without_evidence_nei']/stats['total_claims']*100:.1f}%)")
        if stats["evidence_label_counts"]:
            print(f"  Evidence rationale labels: {stats['evidence_label_counts']}")
        w = stats["claim_word_length"]
        print(f"  Claim words: Min={w['min']}, Max={w['max']}, Mean={w['mean']}, Median={w['median']}")
        
    # Reconstructed triples demonstration
    triples_train = reconstruct_scifact_triples(claims_tr, corpus, include_nei=True)
    print_sub_header("SciFact Reconstructed Triples (Train)")
    print(f"  Total reconstructed claim-evidence pairs: {len(triples_train):,}")
    print(f"  Label distribution:\n{triples_train['label'].value_counts().to_dict()}")
    
    # Cross-split claim & doc overlap
    print_sub_header("SciFact Cross-Split Overlap & Leakage")
    tr_claims = set(c["claim"].strip().lower() for c in claims_tr)
    dv_claims = set(c["claim"].strip().lower() for c in claims_dv)
    te_claims = set(c["claim"].strip().lower() for c in claims_te)
    
    print(f"  Train & Dev overlapping claims: {len(tr_claims.intersection(dv_claims))}")
    print(f"  Train & Test overlapping claims: {len(tr_claims.intersection(te_claims))}")
    
    tr_docs = set(d for c in claims_tr for d in c.get("cited_doc_ids", []))
    dv_docs = set(d for c in claims_dv for d in c.get("cited_doc_ids", []))
    print(f"  Train cited documents: {len(tr_docs):,}")
    print(f"  Dev cited documents: {len(dv_docs):,}")
    print(f"  Train & Dev cited doc overlap: {len(tr_docs.intersection(dv_docs)):,} documents (potential corpus leakage)")
    
    print_sub_header("SciFact Sample Reconstructed Grounded Triples")
    for lbl in ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]:
        sub = triples_train[triples_train["label"] == lbl]
        if not sub.empty:
            row = sub.iloc[0]
            print(f"\n  [Label: {lbl}]")
            print(f"    Claim ID : {row['claim_id']}")
            print(f"    Claim    : {row['claim']}")
            print(f"    Doc ID   : {row['doc_id']} | Title: {row['doc_title']}")
            print(f"    Evidence : {row['evidence_text'][:160]}...")


def main():
    raw_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    print("TruthLens AI - Full Dataset Inspection Starting...")
    print(f"Raw data base directory: {raw_base_dir}")
    
    inspect_fever(raw_base_dir)
    inspect_liar(raw_base_dir)
    inspect_scifact(raw_base_dir)
    
    print_header("INSPECTION SUMMARY COMPLETED")


if __name__ == "__main__":
    main()
