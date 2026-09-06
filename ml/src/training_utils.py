"""
training_utils.py - Reusable Training, Metrics, and Hardware Logging Utilities
for TruthLens AI.
"""

import os
import random
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

LABEL2ID = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT_ENOUGH_INFO": 2,
}

ID2LABEL = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT_ENOUGH_INFO",
}


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def compute_class_weights(labels: List[int], num_classes: int = 3) -> np.ndarray:
    """Computes balanced inverse-frequency class weights."""
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # Avoid zero division
    weights = np.where(counts > 0, total / (num_classes * counts), 1.0)
    return weights.astype(np.float32)


def compute_verification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """
    Computes comprehensive verification metrics:
    Accuracy, Macro/Weighted Precision, Recall, and F1, plus per-class metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Per-class scores
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    
    per_class = {}
    for idx, name in ID2LABEL.items():
        per_class[name] = {
            "precision": float(round(p_per[idx], 4)),
            "recall": float(round(r_per[idx], 4)),
            "f1": float(round(f1_per[idx], 4)),
            "support": int(sup_per[idx]),
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def plot_and_save_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    output_path: str,
    title: str = "TruthLens AI — Confusion Matrix"
):
    """Renders and saves a high-resolution annotated confusion matrix."""
    cm_arr = np.array(cm)
    plt.figure(figsize=(7, 5.5))
    
    # Annotation text with count and normalized percentage
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
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=13, pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Ground Truth", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Confusion matrix -> {output_path}")


def log_hardware_metrics(
    output_path: str,
    gpu_info: Dict[str, Any],
    training_time_sec: float,
    total_samples: int,
    epochs: int,
    peak_vram_mb: Optional[float] = None,
    extra_notes: str = ""
):
    """Writes detailed hardware and throughput metrics to markdown."""
    samples_per_sec = total_samples * epochs / max(training_time_sec, 0.001)
    sec_per_epoch = training_time_sec / max(epochs, 1)
    
    content = f"""# TruthLens AI — Hardware & Training Performance Log

**Date:** September 2026  
**Host System:** ASUS TUF F17 Gaming Laptop  
**Target Hardware:** NVIDIA GPU with CUDA Acceleration  

---

## 1. Hardware Specifications

| Property | Value |
| :--- | :--- |
| **GPU Model** | {gpu_info.get('gpu_name', 'NVIDIA GPU')} |
| **Compute Capability** | {gpu_info.get('compute_capability', 'N/A')} |
| **Total Dedicated VRAM** | {gpu_info.get('total_vram_gb', 'N/A')} GB |
| **CUDA Runtime Version** | {gpu_info.get('cuda_version', 'N/A')} |
| **cuDNN Version** | {gpu_info.get('cudnn_version', 'N/A')} |
| **Device Identifier** | `device = {gpu_info.get('device', 'cuda')}` |

---

## 2. Training Throughput & Resource Utilization

| Metric | Measured Value |
| :--- | :--- |
| **Training Execution Time** | **{training_time_sec:.2f} seconds** ({training_time_sec/60:.2f} minutes) |
| **Epochs Trained** | **{epochs}** |
| **Average Time Per Epoch** | **{sec_per_epoch:.2f} seconds** |
| **Dataset Size (Samples)** | **{total_samples:,}** |
| **Training Throughput** | **{samples_per_sec:.2f} samples/second** |
| **Peak VRAM Allocated** | **{f'{peak_vram_mb:.2f} MB' if peak_vram_mb else 'N/A'}** |

---

## 3. Performance Notes & Reproducibility
{extra_notes}
- Mixed precision FP16 was utilized to respect the 4.0 GB VRAM envelope of the RTX 2050.
- Fixed seed `42` applied across Python, NumPy, and PyTorch CUDA.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"[Hardware Log Saved] -> {output_path}")
