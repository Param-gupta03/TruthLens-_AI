"""
train_model_a.py - Fine-tunes RoBERTa-base for Model A (Evidence Relevance) on NVIDIA GPU.
"""

import os
import sys
import json
import time
import math
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_gpu import inspect_gpu_environment
from training_utils import set_seed, log_hardware_metrics


class EvidenceRelevanceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.claims = df["claim"].astype(str).tolist()
        self.evidences = df["evidence"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        claim_clean = self.claims[idx].strip()
        evidence_clean = self.evidences[idx].strip()
        input_text = f"[CLAIM] {claim_clean} [SEP] {evidence_clean}"

        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_binary_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float] = None) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    
    roc_auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None and len(set(y_true)) > 1 else None
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
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": cm,
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    grad_accum_steps: int = 2
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum_steps
        
        if step % grad_accum_steps == 0 or step == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
                
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.50
) -> Tuple[Dict[str, Any], List[int], List[int], List[float]]:
    model.eval()
    all_targets = []
    all_probs_rel = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                
            all_targets.extend(labels.tolist())
            all_probs_rel.extend(probs.tolist())
            
    all_preds = [1 if p >= threshold else 0 for p in all_probs_rel]
    metrics = compute_binary_metrics(all_targets, all_preds, all_probs_rel)
    return metrics, all_targets, all_preds, all_probs_rel


def sweep_thresholds(
    y_true: List[int],
    y_prob_rel: List[float],
    thresholds: List[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
) -> Tuple[float, List[Dict[str, Any]]]:
    results = []
    best_thresh = 0.50
    best_macro_f1 = -1.0
    
    for th in thresholds:
        preds = [1 if p >= th else 0 for p in y_prob_rel]
        m = compute_binary_metrics(y_true, preds, y_prob_rel)
        results.append({
            "threshold": th,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "relevant_precision": m["relevant_precision"],
            "relevant_recall": m["relevant_recall"],
            "relevant_f1": m["relevant_f1"],
            "not_relevant_f1": m["not_relevant_f1"],
        })
        if m["macro_f1"] > best_macro_f1:
            best_macro_f1 = m["macro_f1"]
            best_thresh = th
            
    return best_thresh, results


def train_model_a(
    epochs: int = 3,
    lr: float = 2e-5,
    per_device_batch_size: int = 8,
    grad_accum_steps: int = 2,
    max_length: int = 256,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    seed: int = 42
):
    print("=" * 80)
    print(" TRUTHLENS AI — MODEL A (EVIDENCE RELEVANCE) TRANSFORMER TRAINING")
    print("=" * 80)
    
    # 1. GPU Check
    set_seed(seed)
    gpu_info = inspect_gpu_environment(require_cuda=True)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "final")
    models_dir = os.path.join(base_dir, "models", "evidence_relevance", "roberta-base")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(data_dir, "evidence_relevance_train.csv"))
    valid_df = pd.read_csv(os.path.join(data_dir, "evidence_relevance_valid.csv"))
    
    print(f"Train samples: {len(train_df):,} | Valid samples: {len(valid_df):,}")
    
    # 2. Class Weights from TRAINING SET ONLY
    train_counts = train_df["label"].value_counts().to_dict()
    total_tr = len(train_df)
    w0 = total_tr / (2.0 * train_counts[0])
    w1 = total_tr / (2.0 * train_counts[1])
    class_weights = [round(w0, 5), round(w1, 5)]
    print(f"Train label counts: NOT_RELEVANT(0)={train_counts[0]}, RELEVANT(1)={train_counts[1]}")
    print(f"Calculated Class Weights (Train Only): {class_weights} (Ratio 1:0 = {w1/w0:.2f})")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device, dtype=torch.float32))
    
    # 3. Model & Tokenizer
    model_name = "roberta-base"
    print(f"\nLoading Pretrained Transformer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    train_dataset = EvidenceRelevanceDataset(train_df, tokenizer, max_length=max_length)
    valid_dataset = EvidenceRelevanceDataset(valid_df, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=per_device_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=per_device_batch_size * 2, shuffle=False)
    
    total_steps = math.ceil(len(train_loader) / grad_accum_steps) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda")
    
    # 4. Training Loop
    print(f"\nStarting Fine-Tuning ({epochs} epochs, effective batch size={per_device_batch_size*grad_accum_steps}, FP16)...")
    start_time = time.time()
    best_macro_f1 = -1.0
    best_val_probs = []
    best_val_targets = []
    
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            criterion=criterion,
            device=device,
            grad_accum_steps=grad_accum_steps
        )
        ep_time = time.time() - ep_start
        
        # Evaluate on validation split at default 0.50 threshold
        val_metrics, val_targets, val_preds, val_probs = evaluate(model, valid_loader, device, threshold=0.50)
        print(f"Epoch {epoch}/{epochs} ({ep_time:.1f}s) | Train Loss: {avg_loss:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val Macro F1: {val_metrics['macro_f1']*100:.2f}% | Rel F1: {val_metrics['relevant_f1']*100:.2f}% | PR-AUC: {val_metrics['pr_auc']:.4f}")
        
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_val_probs = val_probs
            best_val_targets = val_targets
            model.save_pretrained(models_dir)
            tokenizer.save_pretrained(models_dir)
            print(f"  -> Saved new best Model A checkpoint (Macro F1: {best_macro_f1*100:.2f}%)")
            
    total_training_time = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    print("\n" + "=" * 80)
    print(" MODEL A TRAINING COMPLETED")
    print(f"Total Execution Time: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    print(f"Peak VRAM Allocated: {peak_vram_mb:.2f} MB")
    print(f"Best Validation Macro F1 (at 0.50): {best_macro_f1*100:.2f}%")
    print("=" * 80)
    
    # 5. STEP 9: Threshold Analysis on Validation Set ONLY
    print("\nRunning Threshold Analysis on Validation Set (Sweep 0.30 - 0.70)...")
    optimal_thresh, sweep_table = sweep_thresholds(best_val_targets, best_val_probs)
    print(f"Optimal Threshold Selected (by Macro F1): {optimal_thresh}")
    
    print("\nValidation Threshold Sweep Table:")
    print(f"{'Threshold':<10}{'Accuracy':<10}{'Macro F1':<10}{'Rel F1':<10}{'Rel Prec':<10}{'Rel Rec':<10}")
    for row in sweep_table:
        print(f"{row['threshold']:<10.2f}{row['accuracy']*100:<10.2f}{row['macro_f1']*100:<10.2f}{row['relevant_f1']*100:<10.2f}{row['relevant_precision']*100:<10.2f}{row['relevant_recall']*100:<10.2f}")
        
    # Validation metrics at optimal threshold
    val_preds_opt = [1 if p >= optimal_thresh else 0 for p in best_val_probs]
    opt_val_metrics = compute_binary_metrics(best_val_targets, val_preds_opt, best_val_probs)
    
    # 6. Save Training Configuration & Metadata
    config = {
        "model_name": model_name,
        "task": "Model A Evidence Relevance Binary Classification",
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "epochs": epochs,
        "learning_rate": lr,
        "per_device_batch_size": per_device_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": per_device_batch_size * grad_accum_steps,
        "max_length": max_length,
        "class_weights": class_weights,
        "precision": "FP16",
        "gpu_used": gpu_info.get("gpu_name"),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "training_time_seconds": round(total_training_time, 2),
        "default_val_metrics_at_0_50": compute_binary_metrics(best_val_targets, [1 if p>=0.5 else 0 for p in best_val_probs], best_val_probs),
        "selected_threshold": optimal_thresh,
        "optimal_val_metrics": opt_val_metrics,
        "threshold_sweep": sweep_table,
        "label2id": {"NOT_RELEVANT": 0, "RELEVANT": 1},
        "id2label": {"0": "NOT_RELEVANT", "1": "RELEVANT"}
    }
    config_path = os.path.join(models_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved training configuration with selected threshold -> {config_path}")


if __name__ == "__main__":
    train_model_a()
