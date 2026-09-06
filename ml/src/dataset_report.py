"""
dataset_report.py - EDA and Visualization Generator for TruthLens AI.

Generates high-resolution EDA plots and saves them in ml/reports/:
1. dataset_size_comparison.png
2. fever_label_dist.png
3. liar_label_dist.png
4. scifact_label_dist.png
5. claim_length_comparison.png
6. evidence_length_distribution.png
7. liar_missing_values.png
8. liar_top_speakers.png
"""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fever_loader import load_fever_raw, parse_fever_evidence
from liar_loader import load_all_liar_splits, LIAR_6_LABELS
from scifact_loader import load_scifact_corpus, load_scifact_claims, reconstruct_scifact_triples

# Set global style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def generate_all_eda_plots(raw_dir: str, reports_dir: str):
    os.makedirs(reports_dir, exist_ok=True)
    print(f"Generating EDA visual plots into: {reports_dir}")
    
    # 1. Load Datasets
    print("Loading FEVER...")
    df_fever, _ = load_fever_raw(os.path.join(raw_dir, "fever", "train.jsonl"))
    
    print("Loading LIAR...")
    liar_splits = load_all_liar_splits(os.path.join(raw_dir, "liar"))
    df_liar_train = liar_splits.get("train", pd.DataFrame())
    
    print("Loading SciFact...")
    corpus = load_scifact_corpus(os.path.join(raw_dir, "scifact", "corpus.jsonl"))
    scifact_tr = load_scifact_claims(os.path.join(raw_dir, "scifact", "claims_train.jsonl"))
    scifact_dv = load_scifact_claims(os.path.join(raw_dir, "scifact", "claims_dev.jsonl"))
    scifact_te = load_scifact_claims(os.path.join(raw_dir, "scifact", "claims_test.jsonl"))
    scifact_triples = reconstruct_scifact_triples(scifact_tr, corpus, include_nei=True)
    
    # Plot 1: Dataset Size Comparison
    print("Generating: dataset_size_comparison.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    sizes = {
        "FEVER (Train)": len(df_fever),
        "LIAR (Train)": len(df_liar_train),
        "LIAR (Valid)": len(liar_splits.get("valid", [])),
        "LIAR (Test)": len(liar_splits.get("test", [])),
        "SciFact (Train Claims)": len(scifact_tr),
        "SciFact (Dev Claims)": len(scifact_dv),
        "SciFact (Test Claims)": len(scifact_te),
        "SciFact (Corpus Papers)": len(corpus),
    }
    y_pos = np.arange(len(sizes))
    colors = ["#2b5c8f", "#d95f02", "#e6ab02", "#a6761d", "#7570b3", "#1b9e77", "#66a61e", "#e7298a"]
    bars = ax.barh(y_pos, list(sizes.values()), color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(sizes.keys()))
    ax.set_xscale("log")
    ax.set_xlabel("Number of Examples / Documents (Log Scale)")
    ax.set_title("TruthLens AI - Dataset Size Comparison Across Benchmarks")
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.15, bar.get_y() + bar.get_height() / 2, f"{int(width):,}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "dataset_size_comparison.png"), dpi=300)
    plt.close()
    
    # Plot 2: FEVER Label Distribution
    print("Generating: fever_label_dist.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fever_lbl_counts = df_fever["label"].value_counts()
    colors_fever = ["#2ca02c", "#7f7f7f", "#d62728"]
    sns.barplot(x=fever_lbl_counts.index, y=fever_lbl_counts.values, palette=colors_fever, ax=ax)
    ax.set_title("FEVER Label Distribution (Train Set)")
    ax.set_ylabel("Count")
    ax.set_xlabel("Label")
    for p in ax.patches:
        height = p.get_height()
        pct = (height / len(df_fever)) * 100
        ax.annotate(f"{int(height):,}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2.0, height / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "fever_label_dist.png"), dpi=300)
    plt.close()
    
    # Plot 3: LIAR Label Distribution
    print("Generating: liar_label_dist.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    liar_lbl_counts = df_liar_train["label"].value_counts()
    palette_liar = {
        "true": "#2ca02c",
        "mostly-true": "#74c476",
        "half-true": "#fed976",
        "barely-true": "#fd8d3c",
        "false": "#e31a1c",
        "pants-fire": "#800026",
    }
    ordered_labels = [l for l in LIAR_6_LABELS if l in liar_lbl_counts.index]
    counts_ordered = [liar_lbl_counts[l] for l in ordered_labels]
    bar_colors = [palette_liar.get(l, "#333333") for l in ordered_labels]
    bars = ax.bar(ordered_labels, counts_ordered, color=bar_colors)
    ax.set_title("LIAR 6-Class Label Distribution (Train Split)")
    ax.set_ylabel("Count")
    ax.set_xlabel("PolitiFact Truth-O-Meter Rating")
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df_liar_train)) * 100
        ax.annotate(f"{int(height):,}\n({pct:.1f}%)", (bar.get_x() + bar.get_width() / 2.0, height / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "liar_label_dist.png"), dpi=300)
    plt.close()
    
    # Plot 4: SciFact Label Distribution
    print("Generating: scifact_label_dist.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sci_counts = scifact_triples["label"].value_counts()
    palette_sci = {"SUPPORT": "#2ca02c", "CONTRADICT": "#d62728", "NOT_ENOUGH_INFO": "#7f7f7f"}
    colors_sci = [palette_sci.get(l, "#1f77b4") for l in sci_counts.index]
    bars = ax.bar(sci_counts.index, sci_counts.values, color=colors_sci)
    ax.set_title("SciFact Grounded Triples Label Distribution (Train)")
    ax.set_ylabel("Number of Grounded Pairs")
    ax.set_xlabel("Label")
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(scifact_triples)) * 100
        ax.annotate(f"{int(height):,}\n({pct:.1f}%)", (bar.get_x() + bar.get_width() / 2.0, height / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "scifact_label_dist.png"), dpi=300)
    plt.close()
    
    # Plot 5: Claim Length Comparison
    print("Generating: claim_length_comparison.png")
    fever_wlens = df_fever["claim"].astype(str).apply(lambda s: len(s.split()))
    liar_wlens = df_liar_train["statement"].astype(str).apply(lambda s: len(s.split()))
    sci_wlens = pd.Series([len(c["claim"].split()) for c in scifact_tr])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    
    axes[0].hist(fever_wlens, bins=25, range=(0, 40), color="#2b5c8f", edgecolor="black", alpha=0.8)
    axes[0].set_title(f"FEVER Claim Length (Words)\nMean: {fever_wlens.mean():.1f} | Med: {fever_wlens.median():.1f}")
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Number of Claims")
    
    axes[1].hist(liar_wlens, bins=25, range=(0, 60), color="#d95f02", edgecolor="black", alpha=0.8)
    axes[1].set_title(f"LIAR Statement Length (Words)\nMean: {liar_wlens.mean():.1f} | Med: {liar_wlens.median():.1f}")
    axes[1].set_xlabel("Word Count")
    
    axes[2].hist(sci_wlens, bins=20, range=(0, 40), color="#7570b3", edgecolor="black", alpha=0.8)
    axes[2].set_title(f"SciFact Claim Length (Words)\nMean: {sci_wlens.mean():.1f} | Med: {sci_wlens.median():.1f}")
    axes[2].set_xlabel("Word Count")
    
    plt.suptitle("TruthLens AI - Claim Length Distributions Across Datasets", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "claim_length_comparison.png"), dpi=300)
    plt.close()
    
    # Plot 6: Evidence Sentence Length Distribution (SciFact vs FEVER)
    print("Generating: evidence_length_distribution.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # FEVER evidence sentences per claim
    fever_ev_sents = df_fever["evidence"].apply(lambda ev: parse_fever_evidence(ev)["num_evidence_sentences"])
    ax1.hist(fever_ev_sents[fever_ev_sents <= 10], bins=11, range=(0, 10), color="#2b5c8f", edgecolor="black")
    ax1.set_title(f"FEVER Evidence Sentences / Claim\n(Mean: {fever_ev_sents.mean():.2f})")
    ax1.set_xlabel("Number of Referenced Sentences")
    ax1.set_ylabel("Count of Claims")
    
    # SciFact rationale sentence length
    scifact_ev_sents = [
        len(ev["sentences"])
        for c in scifact_tr if c.get("evidence")
        for doc_id, ev_list in c["evidence"].items()
        for ev in ev_list
    ]
    ax2.hist(scifact_ev_sents, bins=6, range=(1, 7), color="#7570b3", edgecolor="black")
    ax2.set_title(f"SciFact Sentences / Evidence Rationale\n(Mean: {np.mean(scifact_ev_sents):.2f})")
    ax2.set_xlabel("Sentence Count in Rationale")
    ax2.set_ylabel("Count of Rationales")
    
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "evidence_length_distribution.png"), dpi=300)
    plt.close()
    
    # Plot 7: LIAR Missing/Empty Values
    print("Generating: liar_missing_values.png")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    empty_rates = {}
    for col in df_liar_train.columns:
        empty_cnt = (df_liar_train[col].astype(str).str.strip() == "").sum()
        empty_rates[col] = (empty_cnt / len(df_liar_train)) * 100
        
    sorted_empty = pd.Series(empty_rates).sort_values(ascending=False)
    ax.bar(sorted_empty.index, sorted_empty.values, color="#e41a1c")
    ax.set_xticklabels(sorted_empty.index, rotation=45, ha="right")
    ax.set_ylabel("Missing / Empty Percentage (%)")
    ax.set_title("LIAR Metadata Field Sparsity (Train Split)")
    for p in ax.patches:
        h = p.get_height()
        if h > 0.1:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2.0, h + 0.8), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "liar_missing_values.png"), dpi=300)
    plt.close()
    
    # Plot 8: LIAR Top Speakers
    print("Generating: liar_top_speakers.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    top_speakers = df_liar_train["speaker"].value_counts().head(10)
    ax.barh(top_speakers.index[::-1], top_speakers.values[::-1], color="#377eb8")
    ax.set_xlabel("Number of Statements Fact-Checked")
    ax.set_title("LIAR - Top 10 Most Frequent Speakers (Train)")
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(f"{int(w)}", (w + 5, p.get_y() + p.get_height() / 2), va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "liar_top_speakers.png"), dpi=300)
    plt.close()
    
    print("All 8 EDA plots generated successfully in ml/reports/")


if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    reports_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    generate_all_eda_plots(raw_path, reports_path)
