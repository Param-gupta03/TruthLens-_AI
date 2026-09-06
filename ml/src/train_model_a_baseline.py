"""
train_model_a_baseline.py - TF-IDF + Logistic Regression Baseline for Model A (Evidence Relevance).
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_utils import set_seed


def evaluate_binary_predictions(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    
    roc_auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if y_prob is not None else None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    
    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "not_relevant_f1": round(float(f1_per[0]), 4),
        "not_relevant_precision": round(float(p_per[0]), 4),
        "not_relevant_recall": round(float(r_per[0]), 4),
        "not_relevant_support": int(sup_per[0]),
        "relevant_f1": round(float(f1_per[1]), 4),
        "relevant_precision": round(float(p_per[1]), 4),
        "relevant_recall": round(float(r_per[1]), 4),
        "relevant_support": int(sup_per[1]),
        "roc_auc": round(roc_auc, 4) if roc_auc else None,
        "pr_auc": round(pr_auc, 4) if pr_auc else None,
        "confusion_matrix": cm,
    }


def plot_binary_cm(cm, output_path, title="Model A Baseline — Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(cm)
    row_sums = cm_arr.sum(axis=1)[:, np.newaxis]
    norm_cm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr, dtype=float), where=row_sums != 0)
    
    annot = np.empty_like(cm_arr, dtype=object)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            annot[i, j] = f"{cm_arr[i, j]:,}\n({norm_cm[i, j]*100:.1f}%)"
            
    sns.heatmap(
        cm_arr,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["NOT_RELEVANT (0)", "RELEVANT (1)"],
        yticklabels=["NOT_RELEVANT (0)", "RELEVANT (1)"],
        cbar=True
    )
    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("Actual Ground Truth", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_model_a_baseline():
    print("=" * 80)
    print(" TRUTHLENS AI — MODEL A (EVIDENCE RELEVANCE) BASELINE TRAINING")
    print("=" * 80)
    
    set_seed(42)
    start_time = time.time()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "final")
    models_dir = os.path.join(base_dir, "models", "baseline", "model_a")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "evidence_relevance_train.csv")
    valid_path = os.path.join(data_dir, "evidence_relevance_valid.csv")
    
    print(f"Loading Train data: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"Loading Valid data: {valid_path}")
    df_valid = pd.read_csv(valid_path)
    
    # Input format: [CLAIM] claim [SEP] evidence
    X_train_text = [f"[CLAIM] {str(c).strip()} [SEP] {str(e).strip()}" for c, e in zip(df_train["claim"], df_train["evidence"])]
    y_train = df_train["label"].astype(int).tolist()
    
    X_valid_text = [f"[CLAIM] {str(c).strip()} [SEP] {str(e).strip()}" for c, e in zip(df_valid["claim"], df_valid["evidence"])]
    y_valid = df_valid["label"].astype(int).tolist()
    
    print(f"Train samples: {len(X_train_text):,} | Valid samples: {len(X_valid_text):,}")
    
    # 1. TF-IDF Vectorizer
    print("\nFitting TF-IDF Vectorizer (unigrams + bigrams, 25,000 features)...")
    tfidf = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode"
    )
    X_tr_tfidf = tfidf.fit_transform(X_train_text)
    X_va_tfidf = tfidf.transform(X_valid_text)
    print(f"Vocabulary size: {len(tfidf.vocabulary_):,} features")
    
    # 2. Logistic Regression (balanced class weights)
    print("Training Logistic Regression (class_weight='balanced', solver='lbfgs')...")
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )
    clf.fit(X_tr_tfidf, y_train)
    
    # 3. Evaluation on Validation Set
    y_pred = clf.predict(X_va_tfidf)
    y_prob = clf.predict_proba(X_va_tfidf)[:, 1]
    
    metrics = evaluate_binary_predictions(y_valid, y_pred, y_prob)
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"BASELINE VALIDATION RESULTS (Runtime: {elapsed:.2f}s)")
    print(f"Accuracy        : {metrics['accuracy']*100:.2f}%")
    print(f"Macro F1        : {metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1     : {metrics['weighted_f1']*100:.2f}%")
    print(f"RELEVANT F1     : {metrics['relevant_f1']*100:.2f}% (Precision: {metrics['relevant_precision']*100:.2f}%, Recall: {metrics['relevant_recall']*100:.2f}%)")
    print(f"NOT_RELEVANT F1 : {metrics['not_relevant_f1']*100:.2f}% (Precision: {metrics['not_relevant_precision']*100:.2f}%, Recall: {metrics['not_relevant_recall']*100:.2f}%)")
    print(f"PR-AUC          : {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC         : {metrics['roc_auc']:.4f}")
    print("=" * 60)
    
    # 4. Save Artifacts
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(models_dir, "logistic_regression_model.joblib"))
    
    config = {
        "model_type": "TF-IDF + Logistic Regression",
        "task": "Model A Evidence Relevance Baseline",
        "max_features": 25000,
        "class_weight": "balanced",
        "train_samples": len(df_train),
        "valid_samples": len(df_valid),
        "runtime_seconds": round(elapsed, 2),
        "metrics": metrics,
    }
    with open(os.path.join(models_dir, "baseline_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    cm_path = os.path.join(reports_dir, "model_a_baseline_confusion_matrix.png")
    plot_binary_cm(metrics["confusion_matrix"], cm_path, title="Model A Baseline (TF-IDF + LR) — Confusion Matrix")
    
    # 5. Generate Markdown Report
    cm = metrics["confusion_matrix"]
    rep = f"""# TruthLens AI — Model A Baseline Results

**Model:** TF-IDF (25,000 features, n-grams 1-2) + Logistic Regression (Balanced)  
**Task:** Model A (Evidence Relevance: `claim + candidate evidence -> RELEVANT (1) / NOT_RELEVANT (0)`)  
**Evaluation Set:** `evidence_relevance_valid.csv` ({len(df_valid)} samples)  
**Date:** September 2026  
**Runtime:** {elapsed:.2f} seconds  

---

## 1. Quantitative Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **{metrics['accuracy']*100:.2f}%** |
| **Macro F1 Score** | **{metrics['macro_f1']*100:.2f}%** |
| **Weighted F1 Score** | **{metrics['weighted_f1']*100:.2f}%** |
| **Macro Precision** | **{metrics['macro_precision']*100:.2f}%** |
| **Macro Recall** | **{metrics['macro_recall']*100:.2f}%** |
| **PR-AUC (Precision-Recall Area)** | **{metrics['pr_auc']:.4f}** |
| **ROC-AUC** | **{metrics['roc_auc']:.4f}** |

---

## 2. Per-Class Metrics

| Class | Precision | Recall | F1 Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **`NOT_RELEVANT (0)`** | {metrics['not_relevant_precision']*100:.2f}% | {metrics['not_relevant_recall']*100:.2f}% | **{metrics['not_relevant_f1']*100:.2f}%** | {metrics['not_relevant_support']} |
| **`RELEVANT (1)`** | {metrics['relevant_precision']*100:.2f}% | {metrics['relevant_recall']*100:.2f}% | **{metrics['relevant_f1']*100:.2f}%** | {metrics['relevant_support']} |

---

## 3. Confusion Matrix

| Actual \\ Predicted | NOT_RELEVANT (0) | RELEVANT (1) | Total Actual |
| :--- | :--- | :--- | :--- |
| **NOT_RELEVANT (0)** | {cm[0][0]} ({cm[0][0]/metrics['not_relevant_support']*100:.1f}%) | {cm[0][1]} ({cm[0][1]/metrics['not_relevant_support']*100:.1f}%) | {metrics['not_relevant_support']} |
| **RELEVANT (1)** | {cm[1][0]} ({cm[1][0]/metrics['relevant_support']*100:.1f}%) | {cm[1][1]} ({cm[1][1]/metrics['relevant_support']*100:.1f}%) | {metrics['relevant_support']} |

---

## 4. Key Takeaways
1. **High Recall, Low Precision:** Balanced Logistic Regression achieves {metrics['relevant_recall']*100:.2f}% recall on the minority positive class, but precision is low ({metrics['relevant_precision']*100:.2f}%) resulting in an F1 of **{metrics['relevant_f1']*100:.2f}%**.
2. **Hard Negatives Challenge TF-IDF:** Because candidate evidence sentences share PubMed vocabulary with the claim, word co-occurrence alone leads to frequent false positives ({cm[0][1]} false alarms).
3. **Benchmark for Transformer:** RoBERTa will need to leverage deep contextual embeddings to disambiguate topical relatedness from genuine evidential relevance.
"""
    rep_path = os.path.join(reports_dir, "model_a_baseline_results.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep)
    print(f"Saved baseline report -> {rep_path}")
    print(f"Saved baseline confusion matrix -> {cm_path}")


if __name__ == "__main__":
    train_model_a_baseline()
