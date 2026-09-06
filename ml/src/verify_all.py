"""
verify_all.py - Verification script to confirm the entire TruthLens AI ML stage 1 setup.
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fever_loader import load_fever_raw, get_fever_statistics
from liar_loader import load_all_liar_splits, get_liar_statistics
from scifact_loader import load_scifact_corpus, load_scifact_claims, reconstruct_scifact_triples, get_scifact_statistics
from preprocessing import clean_text, map_scifact_label, map_liar_label_to_3way, format_to_unified_schema


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    reports_dir = os.path.join(base_dir, "reports")
    notebooks_dir = os.path.join(base_dir, "notebooks")
    
    print("=== TRUTHLENS AI PIPELINE VERIFICATION ===")
    
    # 1. Verify loaders
    df_fever, mal_fever = load_fever_raw(os.path.join(raw_dir, "fever", "train.jsonl"))
    assert len(df_fever) == 145449, f"FEVER count mismatch: {len(df_fever)}"
    print(f"[OK] FEVER Loader: (145,449 rows, {mal_fever} malformed)")
    
    liar_splits = load_all_liar_splits(os.path.join(raw_dir, "liar"))
    assert len(liar_splits["train"]) == 10269, f"LIAR train mismatch: {len(liar_splits['train'])}"
    assert len(liar_splits["valid"]) == 1284, f"LIAR valid mismatch: {len(liar_splits['valid'])}"
    assert len(liar_splits["test"]) == 1283, f"LIAR test mismatch: {len(liar_splits['test'])}"
    print(f"[OK] LIAR Loader: (12,836 total rows across train/valid/test)")
    
    corpus = load_scifact_corpus(os.path.join(raw_dir, "scifact", "corpus.jsonl"))
    claims_tr = load_scifact_claims(os.path.join(raw_dir, "scifact", "claims_train.jsonl"))
    assert len(corpus) == 5183, f"SciFact corpus mismatch: {len(corpus)}"
    assert len(claims_tr) == 809, f"SciFact train mismatch: {len(claims_tr)}"
    triples = reconstruct_scifact_triples(claims_tr, corpus, include_nei=True)
    assert len(triples) == 1261, f"SciFact triples mismatch: {len(triples)}"
    print(f"[OK] SciFact Loader & Triples: (5,183 corpus docs, 809 claims -> 1,261 grounded triples)")
    
    # 2. Verify preprocessing
    text_sample = ' "The economic growth was 3.5% &amp; inflation fell" '
    assert clean_text(text_sample) == '"The economic growth was 3.5% & inflation fell"'
    assert map_scifact_label("SUPPORT") == "SUPPORTS"
    assert map_scifact_label("CONTRADICT") == "REFUTES"
    assert map_liar_label_to_3way("pants-fire") == "REFUTES"
    assert map_liar_label_to_3way("mostly-true") == "SUPPORTS"
    assert map_liar_label_to_3way("half-true") == "NOT_ENOUGH_INFO"
    unified = format_to_unified_schema("1", "Claim", "Evidence", "SUPPORTS", "Source", "political", "liar")
    assert "claim_id" in unified and unified["dataset"] == "liar"
    print("[OK] Preprocessing functions")
    
    # 3. Verify notebooks
    expected_nbs = [
        "01_dataset_inspection.ipynb",
        "02_fever_eda.ipynb",
        "03_liar_eda.ipynb",
        "04_scifact_eda.ipynb"
    ]
    for nb_name in expected_nbs:
        nb_path = os.path.join(notebooks_dir, nb_name)
        assert os.path.exists(nb_path), f"Notebook missing: {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            assert "cells" in nb and len(nb["cells"]) > 0
    print("[OK] Notebooks (all 4 valid and populated)")
    
    # 4. Verify reports & plots
    expected_reports = [
        "dataset_comparison.md",
        "data_quality_report.md",
        "dataset_report.md",
        "cleaning_report.md",
        "final_dataset_summary.csv",
        "dataset_size_comparison.png",
        "fever_label_dist.png",
        "liar_label_dist.png",
        "scifact_label_dist.png",
        "claim_length_comparison.png",
        "evidence_length_distribution.png",
        "liar_missing_values.png",
        "liar_top_speakers.png"
    ]
    for rep in expected_reports:
        rep_p = os.path.join(reports_dir, rep)
        assert os.path.exists(rep_p), f"Report/Plot missing: {rep_p}"
    print("[OK] Reports & Plots (4 markdown reports + 1 summary CSV + 8 EDA plots)")
    
    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
