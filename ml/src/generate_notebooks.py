"""
generate_notebooks.py - Generates the 4 required Jupyter Notebooks for TruthLens AI EDA:
1. 01_dataset_inspection.ipynb
2. 02_fever_eda.ipynb
3. 03_liar_eda.ipynb
4. 04_scifact_eda.ipynb
"""

import json
import os


def make_cell(cell_type: str, source_text: str):
    # Split into lines preserving newlines
    lines = [line + "\n" for line in source_text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.6",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }


def create_01_inspection_notebook(output_path: str):
    cells = [
        make_cell("markdown", """# TruthLens AI — Dataset Inspection & Discovery
This notebook performs systematic data discovery and inspection across all three raw datasets:
1. **FEVER** (Fact Extraction and VERification)
2. **LIAR** (PolitiFact Political Fact-Checking Benchmark)
3. **SciFact** (Scientific Claims & Rationales)

### Key Objectives:
- Confirm file formats, sizes, and record counts
- Inspect columns, schemas, and data types
- Analyze missing values and malformed records
- Verify evidence representation mechanisms"""),
        make_cell("code", """import os
import sys
import json
import pandas as pd
import numpy as np

# Add src to python path
sys.path.insert(0, os.path.abspath("../src"))

from fever_loader import load_fever_raw, get_fever_statistics
from liar_loader import load_all_liar_splits, get_liar_statistics, LIAR_COLUMNS
from scifact_loader import load_scifact_corpus, load_scifact_claims, reconstruct_scifact_triples, get_scifact_statistics

RAW_DIR = os.path.abspath("../data/raw")
print(f"Loading raw datasets from: {RAW_DIR}")"""),
        make_cell("markdown", """## 1. Load FEVER Dataset
FEVER is stored in JSON Lines format containing claim assertions labeled as `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`."""),
        make_cell("code", """fever_path = os.path.join(RAW_DIR, "fever", "train.jsonl")
df_fever, malformed_fever = load_fever_raw(fever_path)
print(f"FEVER Train: {len(df_fever):,} rows parsed (Malformed: {malformed_fever})")
print("Columns:", list(df_fever.columns))
display(df_fever.head(3))"""),
        make_cell("markdown", """## 2. Load LIAR Dataset
LIAR contains political statements rated on PolitiFact's 6-point Truth-O-Meter scale, accompanied by speaker and venue metadata."""),
        make_cell("code", """liar_dir = os.path.join(RAW_DIR, "liar")
liar_splits = load_all_liar_splits(liar_dir)

for split, df_s in liar_splits.items():
    print(f"LIAR {split.upper()}: {len(df_s):,} rows, {len(df_s.columns)} columns")

df_liar_train = liar_splits["train"]
display(df_liar_train.head(3))"""),
        make_cell("markdown", """## 3. Load SciFact Dataset
SciFact contains expert scientific claims paired with PubMed research paper abstracts and annotated rationale sentence IDs."""),
        make_cell("code", """scifact_dir = os.path.join(RAW_DIR, "scifact")
corpus = load_scifact_corpus(os.path.join(scifact_dir, "corpus.jsonl"))
scifact_train_claims = load_scifact_claims(os.path.join(scifact_dir, "claims_train.jsonl"))
scifact_dev_claims = load_scifact_claims(os.path.join(scifact_dir, "claims_dev.jsonl"))
scifact_test_claims = load_scifact_claims(os.path.join(scifact_dir, "claims_test.jsonl"))

print(f"SciFact Corpus: {len(corpus):,} research paper abstracts")
print(f"SciFact Claims - Train: {len(scifact_train_claims)}, Dev: {len(scifact_dev_claims)}, Test: {len(scifact_test_claims)}")"""),
        make_cell("markdown", """## 4. Reconstruct Grounded Triples for SciFact
Using the document IDs and sentence indices, we reconstruct explicit `(claim, evidence_text, label)` triples."""),
        make_cell("code", """scifact_triples = reconstruct_scifact_triples(scifact_train_claims, corpus, include_nei=True)
print(f"SciFact Reconstructed Triples: {len(scifact_triples):,}")
display(scifact_triples.head(5))"""),
        make_cell("markdown", """## 5. Dataset Summary Matrix
Comparison of key attributes across all three raw datasets."""),
        make_cell("code", """summary_data = [
    {"Dataset": "FEVER", "Domain": "General / Wikipedia", "Examples": len(df_fever), "Evidence Included": "No (Title + Sent IDs only)", "Labels": "SUPPORTS, REFUTES, NOT ENOUGH INFO (3 classes)"},
    {"Dataset": "LIAR", "Domain": "Political Speech / PolitiFact", "Examples": sum(len(d) for d in liar_splits.values()), "Evidence Included": "No (Metadata only: speaker, party, context)", "Labels": "6-point scale (pants-fire to true)"},
    {"Dataset": "SciFact", "Domain": "Biomedical / Scientific", "Examples": f"{len(scifact_train_claims)} train claims / 5,183 corpus docs", "Evidence Included": "Yes (Abstract sentences in corpus.jsonl)", "Labels": "SUPPORT, CONTRADICT, NOT ENOUGH INFO"},
]
summary_df = pd.DataFrame(summary_data)
display(summary_df)"""),
    ]
    nb = make_notebook(cells)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {output_path}")


def create_02_fever_notebook(output_path: str):
    cells = [
        make_cell("markdown", """# TruthLens AI — FEVER Dataset Deep-Dive EDA
This notebook performs comprehensive exploratory data analysis on the FEVER (Fact Extraction and VERification) dataset.

### Topics Covered:
- Label distribution (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`)
- Verifiable vs. Not Verifiable claims
- Claim length analysis (word and character distributions)
- Evidence structure (evidence sets, sentence pointers, Wikipedia page references)
- Annotation duplicates & contradictory labeling analysis"""),
        make_cell("code", """import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath("../src"))
from fever_loader import load_fever_raw, parse_fever_evidence, get_fever_statistics

sns.set_theme(style="whitegrid")
df_fever, malformed = load_fever_raw("../data/raw/fever/train.jsonl")
print(f"Loaded FEVER train: {len(df_fever):,} rows")"""),
        make_cell("markdown", """## 1. Label Distribution & Verifiable Status"""),
        make_cell("code", """lbl_counts = df_fever["label"].value_counts()
print(lbl_counts)

plt.figure(figsize=(7, 4.5))
sns.barplot(x=lbl_counts.index, y=lbl_counts.values, palette=["#2ca02c", "#7f7f7f", "#d62728"])
plt.title("FEVER Train Label Distribution")
plt.ylabel("Count")
plt.xlabel("Verification Label")
for i, v in enumerate(lbl_counts.values):
    plt.text(i, v / 2, f"{v:,}\\n({v/len(df_fever)*100:.1f}%)", ha="center", va="center", color="white", fontweight="bold")
plt.tight_layout()
plt.show()"""),
        make_cell("markdown", """## 2. Claim Length Distribution"""),
        make_cell("code", """df_fever["word_count"] = df_fever["claim"].apply(lambda c: len(c.split()))
df_fever["char_count"] = df_fever["claim"].apply(len)

print(df_fever[["word_count", "char_count"]].describe())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(df_fever["word_count"], bins=30, ax=ax1, color="#2b5c8f", kde=True)
ax1.set_title("Claim Word Count Distribution")
ax1.set_xlim(0, 35)

sns.boxplot(x=df_fever["label"], y=df_fever["word_count"], ax=ax2, palette="muted")
ax2.set_title("Claim Word Count by Label")
ax2.set_ylim(0, 35)
plt.tight_layout()
plt.show()"""),
        make_cell("markdown", """## 3. Evidence Structure Analysis
In FEVER, evidence is formatted as a 3-level list: `[Sets -> Sentences -> [annot_id, ev_id, wiki_page, sent_id]]`."""),
        make_cell("code", """ev_parsed = df_fever["evidence"].apply(parse_fever_evidence)
df_fever["num_ev_sents"] = ev_parsed.apply(lambda x: x["num_evidence_sentences"])
df_fever["num_ev_sets"] = ev_parsed.apply(lambda x: x["num_evidence_sets"])

print("Evidence sentences per claim:")
print(df_fever["num_ev_sents"].value_counts().head(8))

print(f"Total claims with at least 1 evidence reference: {(df_fever['num_ev_sents'] > 0).sum():,} ({(df_fever['num_ev_sents'] > 0).sum()/len(df_fever)*100:.1f}%)")
print(f"Total claims with 0 evidence references (NEI): {(df_fever['num_ev_sents'] == 0).sum():,} ({(df_fever['num_ev_sents'] == 0).sum()/len(df_fever)*100:.1f}%)")"""),
        make_cell("markdown", """## 4. Duplicate Claims & Label Conflicts
FEVER contains duplicate claim strings from distinct crowd-source annotators."""),
        make_cell("code", """unique_claims = df_fever["claim"].nunique()
print(f"Unique claims: {unique_claims:,} out of {len(df_fever):,} rows ({len(df_fever) - unique_claims:,} duplicates)")

# Conflicting annotations
claim_labels = df_fever.groupby("claim")["label"].nunique()
conflicting = claim_labels[claim_labels > 1]
print(f"Claims with conflicting annotations: {len(conflicting):,}")

# Show examples of conflicting annotations
sample_conflicts = conflicting.index[:3]
for c in sample_conflicts:
    print(f"Claim: {c}")
    display(df_fever[df_fever["claim"] == c][["id", "label", "evidence"]])"""),
    ]
    nb = make_notebook(cells)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {output_path}")


def create_03_liar_notebook(output_path: str):
    cells = [
        make_cell("markdown", """# TruthLens AI — LIAR Dataset Deep-Dive EDA
This notebook performs exploratory data analysis on the LIAR dataset (William Yang Wang, ACL 2017).

### Topics Covered:
- 6-class truth rating distribution across train, valid, and test splits
- Statement text length distributions
- Metadata analysis: Speaker distributions, party affiliation, contexts/venues
- Historical truth credit metrics
- Cross-split statement leakage
- Proposed 3-way label mappings for TruthLens AI"""),
        make_cell("code", """import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath("../src"))
from liar_loader import load_all_liar_splits, LIAR_COLUMNS, LIAR_6_LABELS
from preprocessing import map_liar_label_to_3way

sns.set_theme(style="whitegrid")
liar_splits = load_all_liar_splits("../data/raw/liar")
df_train = liar_splits["train"]
df_valid = liar_splits["valid"]
df_test = liar_splits["test"]

print(f"LIAR loaded: Train={len(df_train):,}, Valid={len(df_valid):,}, Test={len(df_test):,}")"""),
        make_cell("markdown", """## 1. 6-Class Label Distribution Across Splits"""),
        make_cell("code", """split_dfs = []
for name, df in liar_splits.items():
    s_counts = df["label"].value_counts(normalize=True).rename(name)
    split_dfs.append(s_counts)

df_label_comp = pd.concat(split_dfs, axis=1).reindex(LIAR_6_LABELS) * 100
display(df_label_comp.round(2))

df_label_comp.plot(kind="bar", figsize=(10, 5), colormap="Set2")
plt.title("LIAR Label Percentage Across Splits")
plt.ylabel("Percentage (%)")
plt.xlabel("Truth-O-Meter Rating")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()"""),
        make_cell("markdown", """## 2. Statement Length Statistics"""),
        make_cell("code", """df_train["word_count"] = df_train["statement"].apply(lambda s: len(s.split()))
print(df_train["word_count"].describe())

plt.figure(figsize=(8, 4))
sns.histplot(df_train["word_count"], bins=30, color="#d95f02", kde=True)
plt.title("LIAR Train Statement Word Count Distribution")
plt.xlim(0, 60)
plt.xlabel("Word Count")
plt.tight_layout()
plt.show()"""),
        make_cell("markdown", """## 3. Top Speakers and Party Distribution"""),
        make_cell("code", """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Top 10 speakers
top_speakers = df_train["speaker"].value_counts().head(10)
sns.barplot(x=top_speakers.values, y=top_speakers.index, ax=ax1, palette="Blues_r")
ax1.set_title("Top 10 Speakers in Train Split")
ax1.set_xlabel("Number of Statements")

# Top parties
top_parties = df_train["party_affiliation"].value_counts().head(5)
sns.barplot(x=top_parties.index, y=top_parties.values, ax=ax2, palette="Greens_r")
ax2.set_title("Statements by Political Party")
ax2.set_ylabel("Count")
plt.tight_layout()
plt.show()"""),
        make_cell("markdown", """## 4. Metadata Completeness & Missing Fields"""),
        make_cell("code", """empty_counts = {}
for col in df_train.columns:
    empty_cnt = (df_train[col].astype(str).str.strip() == "").sum()
    empty_counts[col] = (empty_cnt / len(df_train)) * 100

pd.Series(empty_counts).sort_values(ascending=False).plot(kind="bar", figsize=(10, 4), color="#e41a1c")
plt.title("LIAR Metadata Sparsity (Missing / Empty % in Train)")
plt.ylabel("Percentage Missing (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()"""),
        make_cell("markdown", """## 5. Cross-Split Statement Leakage"""),
        make_cell("code", """tr_stmts = set(df_train["statement"].str.strip().str.lower())
val_stmts = set(df_valid["statement"].str.strip().str.lower())
te_stmts = set(df_test["statement"].str.strip().str.lower())

print("Train & Valid overlap:", len(tr_stmts.intersection(val_stmts)))
print("Train & Test overlap:", len(tr_stmts.intersection(te_stmts)))
print("Valid & Test overlap:", len(val_stmts.intersection(te_stmts)))

overlap_sample = list(tr_stmts.intersection(val_stmts))
for s in overlap_sample:
    print(f"- Overlapping statement: {s}")"""),
        make_cell("markdown", """## 6. Mapping LIAR to 3-Way Schema"""),
        make_cell("code", """df_train["mapped_label_3way"] = df_train["label"].apply(map_liar_label_to_3way)
print("Mapped 3-Way Distribution:")
print(df_train["mapped_label_3way"].value_counts())
print(df_train["mapped_label_3way"].value_counts(normalize=True) * 100)"""),
    ]
    nb = make_notebook(cells)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {output_path}")


def create_04_scifact_notebook(output_path: str):
    cells = [
        make_cell("markdown", """# TruthLens AI — SciFact Dataset Deep-Dive EDA
This notebook explores the SciFact scientific fact-checking benchmark (EMNLP 2020).

### Topics Covered:
- Corpus document distribution (research paper abstracts, sentence counts)
- Claim verification structure (claims with evidence vs. claims without evidence / NEI)
- Grounded `(claim, evidence_text, label)` triple reconstruction
- Rationale sentence length distributions
- Cross-validation fold consistency
- Cross-split document leakage analysis"""),
        make_cell("code", """import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath("../src"))
from scifact_loader import load_scifact_corpus, load_scifact_claims, reconstruct_scifact_triples, get_scifact_statistics

sns.set_theme(style="whitegrid")
corpus = load_scifact_corpus("../data/raw/scifact/corpus.jsonl")
claims_tr = load_scifact_claims("../data/raw/scifact/claims_train.jsonl")
claims_dv = load_scifact_claims("../data/raw/scifact/claims_dev.jsonl")
claims_te = load_scifact_claims("../data/raw/scifact/claims_test.jsonl")

print(f"Corpus: {len(corpus):,} papers")
print(f"Claims: Train={len(claims_tr)}, Dev={len(claims_dv)}, Test={len(claims_te)}")"""),
        make_cell("markdown", """## 1. Corpus Abstract Statistics"""),
        make_cell("code", """abstract_sents = [len(d.get("abstract", [])) for d in corpus.values()]
print(f"Total sentences across corpus: {sum(abstract_sents):,}")
print(f"Sentences per abstract: Min={min(abstract_sents)}, Max={max(abstract_sents)}, Mean={np.mean(abstract_sents):.2f}")

plt.figure(figsize=(8, 4))
sns.histplot(abstract_sents, bins=30, range=(1, 25), color="#7570b3")
plt.title("SciFact Corpus: Abstract Sentence Count Distribution")
plt.xlabel("Sentences in Abstract")
plt.tight_layout()
plt.show()"""),
        make_cell("markdown", """## 2. Claim Evidence Availability"""),
        make_cell("code", """tr_has_ev = [bool(c.get("evidence", {})) for c in claims_tr]
dv_has_ev = [bool(c.get("evidence", {})) for c in claims_dv]

print(f"Train claims with evidence: {sum(tr_has_ev)} / {len(claims_tr)} ({sum(tr_has_ev)/len(claims_tr)*100:.1f}%)")
print(f"Dev claims with evidence: {sum(dv_has_ev)} / {len(claims_dv)} ({sum(dv_has_ev)/len(claims_dv)*100:.1f}%)")"""),
        make_cell("markdown", """## 3. Grounded Triples Reconstruction"""),
        make_cell("code", """triples_tr = reconstruct_scifact_triples(claims_tr, corpus, include_nei=True)
print(f"Total reconstructed triples (train): {len(triples_tr)}")
print("Label counts:")
print(triples_tr["label"].value_counts())

display(triples_tr.head(5))"""),
        make_cell("markdown", """## 4. Evidence Rationale Sentence Lengths"""),
        make_cell("code", """triples_tr["ev_sentence_count"] = triples_tr["sentence_indices"].apply(len)
rationale_sub = triples_tr[triples_tr["ev_sentence_count"] > 0]

plt.figure(figsize=(7, 4))
sns.countplot(x=rationale_sub["ev_sentence_count"], palette="Purples_r")
plt.title("Sentences per Evidence Rationale in SciFact")
plt.xlabel("Number of Sentences")
plt.ylabel("Count")
plt.tight_layout()
plt.show()"""),
        make_cell("markdown", """## 5. Cross-Split Document Leakage"""),
        make_cell("code", """tr_docs = set(d for c in claims_tr for d in c.get("cited_doc_ids", []))
dv_docs = set(d for c in claims_dv for d in c.get("cited_doc_ids", []))

print(f"Train unique cited docs: {len(tr_docs)}")
print(f"Dev unique cited docs: {len(dv_docs)}")
overlap_docs = tr_docs.intersection(dv_docs)
print(f"Cited docs appearing in BOTH train and dev: {len(overlap_docs)} ({len(overlap_docs)/len(dv_docs)*100:.1f}% of dev docs)")"""),
    ]
    nb = make_notebook(cells)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {output_path}")


def main():
    nb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
    os.makedirs(nb_dir, exist_ok=True)
    
    create_01_inspection_notebook(os.path.join(nb_dir, "01_dataset_inspection.ipynb"))
    create_02_fever_notebook(os.path.join(nb_dir, "02_fever_eda.ipynb"))
    create_03_liar_notebook(os.path.join(nb_dir, "03_liar_eda.ipynb"))
    create_04_scifact_notebook(os.path.join(nb_dir, "04_scifact_eda.ipynb"))
    print("All 4 notebooks successfully generated!")


if __name__ == "__main__":
    main()
