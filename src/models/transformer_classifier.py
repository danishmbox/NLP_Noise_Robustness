"""
Transformer Training and Validation Pipeline for Tier 1 Models:
- Model B: MuRIL (google/muril-base-cased)
- Model C: XLM-RoBERTa (xlm-roberta-base)
- Model D: mBERT (bert-base-multilingual-cased)

Provides PyTorch / HuggingFace Transformers fine-tuning with:
- Configurable checkpoint, max_length, learning rate, batch size, epochs
- Truncation analysis on train and dev splits
- Epoch-wise validation on dev.csv
- Best checkpoint saving and early stopping governed strictly by validation Macro-F1
- Robust OOM handling and environment dependency validation
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import LABEL2ID, ID2LABEL, ORDERED_LABELS
from src.utils.reproducibility import set_seed, get_runtime_metadata, get_reproducibility_caveats
from src.evaluation.metrics import (
    compute_evaluation_metrics,
    save_validation_results,
    save_confusion_matrix_plot
)


DEFAULT_TRANSFORMER_CONFIG: Dict[str, Any] = {
    "model_family": "transformer",
    "model_name": "muril",
    "checkpoint": "google/muril-base-cased",
    "max_length": 128,
    "learning_rate": 2e-5,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "num_epochs": 3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "random_seed": 42,
    "early_stopping_patience": 2,
    "fp16": True,
    "output_dir": "results/models"
}

TIER1_CHECKPOINTS: Dict[str, str] = {
    "muril": "google/muril-base-cased",
    "xlm_roberta": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased"
}


def check_transformer_dependencies() -> None:
    """
    Checks if PyTorch and Transformers are installed; raises descriptive error if missing.
    """
    missing = []
    for pkg in ["torch", "transformers", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        raise ImportError(
            f"Missing required transformer acceleration dependencies: {missing}.\n"
            f"To train transformer models (MuRIL, XLM-R, mBERT):\n"
            f"1. Open the provided Google Colab notebook: notebooks/02_Train_Tier1_Transformers_Colab.ipynb\n"
            f"2. Or install dependencies with GPU support: pip install torch transformers accelerate datasets\n"
            f"Refer to docs/tier1_training_guide.md for the complete execution guide."
        )


def analyze_tokenization_truncation(
    texts: List[str],
    tokenizer: Any,
    max_length: int,
    split_name: str
) -> Dict[str, Any]:
    """
    Analyzes tokenization length distribution and calculates truncation statistics.

    Args:
        texts: List of text samples.
        tokenizer: Pretrained HuggingFace tokenizer.
        max_length: Selected maximum sequence length.
        split_name: Name of the split ('train' or 'dev').

    Returns:
        Dict of length statistics and truncation percentage.
    """
    lengths = []
    truncated_count = 0

    for text in texts:
        tokens = tokenizer.encode(str(text), add_special_tokens=True, truncation=False)
        tok_len = len(tokens)
        lengths.append(tok_len)
        if tok_len > max_length:
            truncated_count += 1

    lengths_arr = np.array(lengths)
    total = len(texts)
    trunc_pct = (truncated_count / total * 100.0) if total > 0 else 0.0

    stats = {
        "split": split_name,
        "sample_count": total,
        "max_length_configured": max_length,
        "truncated_count": truncated_count,
        "truncation_percentage": round(trunc_pct, 2),
        "min_token_length": int(lengths_arr.min()) if total > 0 else 0,
        "mean_token_length": round(float(lengths_arr.mean()), 1) if total > 0 else 0.0,
        "median_token_length": float(np.median(lengths_arr)) if total > 0 else 0.0,
        "p95_token_length": float(np.percentile(lengths_arr, 95)) if total > 0 else 0.0,
        "max_token_length": int(lengths_arr.max()) if total > 0 else 0
    }

    return stats


def build_transformer_pipeline(
    checkpoint: str,
    num_labels: int = 3
) -> Tuple[Any, Any]:
    """
    Loads HuggingFace AutoTokenizer and AutoModelForSequenceClassification.

    Args:
        checkpoint: HuggingFace model checkpoint string.
        num_labels: Number of output sentiment classes (default: 3).

    Returns:
        Tuple of (tokenizer, model).
    """
    check_transformer_dependencies()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"Loading tokenizer for checkpoint: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    print(f"Loading sequence classification model for checkpoint: {checkpoint}")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    return tokenizer, model


def train_transformer_classifier(
    model_name: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None,
    val_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Trains a transformer classifier with epoch-wise dev validation and Macro-F1 checkpointing.

    Args:
        model_name: Identifier ('muril', 'xlm_roberta', 'mbert', or custom).
        train_df: Training DataFrame containing 'text' and 'sentiment'.
        dev_df: Validation DataFrame containing 'text' and 'sentiment'.
        config: Configuration dictionary.
        models_dir: Directory to store model checkpoints.
        val_dir: Directory to store validation metrics and plots.

    Returns:
        Dictionary of final validation metrics and training metadata.
    """
    check_transformer_dependencies()
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import get_linear_schedule_with_warmup

    cfg = dict(DEFAULT_TRANSFORMER_CONFIG)
    if model_name in TIER1_CHECKPOINTS:
        cfg["checkpoint"] = TIER1_CHECKPOINTS[model_name]
        cfg["model_name"] = model_name
    if config:
        cfg.update(config)

    # Set seeds
    seed = cfg.get("random_seed", 42)
    seed_info = set_seed(seed)

    if models_dir is None:
        models_dir = Path("results/models")
    if val_dir is None:
        val_dir = Path("results/validation")

    models_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = cfg["checkpoint"]
    max_length = cfg.get("max_length", 128)
    batch_size = cfg.get("batch_size", 16)
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    num_epochs = cfg.get("num_epochs", 3)
    lr = cfg.get("learning_rate", 2e-5)
    weight_decay = cfg.get("weight_decay", 0.01)
    warmup_ratio = cfg.get("warmup_ratio", 0.1)
    patience = cfg.get("early_stopping_patience", 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.get("fp16", True) and torch.cuda.is_available()

    print("=" * 70)
    print(f"   TRANSFORMER TRAINING: {cfg['model_name'].upper()} ({checkpoint})   ")
    print("=" * 70)
    print(f"Device: {device} | Mixed Precision (FP16): {use_fp16}")
    print(f"Train samples: {len(train_df):,} | Dev samples: {len(dev_df):,}")
    print(f"Max length: {max_length} | Batch size: {batch_size} | Grad Accum: {grad_accum}")
    print(f"Epochs: {num_epochs} | LR: {lr} | Weight Decay: {weight_decay} | Seed: {seed}")

    tokenizer, model = build_transformer_pipeline(checkpoint, num_labels=3)
    model.to(device)

    # Truncation Analysis
    print("\nRunning tokenization & truncation analysis...")
    train_trunc = analyze_tokenization_truncation(train_df["text"].tolist(), tokenizer, max_length, "train")
    dev_trunc = analyze_tokenization_truncation(dev_df["text"].tolist(), tokenizer, max_length, "dev")
    print(f"  Train truncation at L={max_length}: {train_trunc['truncated_count']} / {train_trunc['sample_count']} ({train_trunc['truncation_percentage']}%)")
    print(f"  Dev truncation at L={max_length}:   {dev_trunc['truncated_count']} / {dev_trunc['sample_count']} ({dev_trunc['truncation_percentage']}%)")

    # Dataset definition
    class TextSentimentDataset(Dataset):
        def __init__(self, texts, labels, tok, max_len):
            self.texts = list(texts)
            self.labels = [LABEL2ID[l] if isinstance(l, str) else int(l) for l in labels]
            self.tok = tok
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tok(
                str(self.texts[idx]),
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset = TextSentimentDataset(train_df["text"], train_df["sentiment"], tokenizer, max_length)
    dev_dataset = TextSentimentDataset(dev_df["text"], dev_df["sentiment"], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = (len(train_loader) + grad_accum - 1) // grad_accum
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=True) if (use_fp16 and torch.cuda.is_available()) else None

    best_dev_macro_f1 = -1.0
    best_checkpoint_dir = models_dir / f"{cfg['model_name']}_best_checkpoint"
    best_metrics = {}
    epoch_history = []
    patience_counter = 0

    print("\nStarting model training...")
    start_total_time = time.time()

    try:
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            model.train()
            train_loss_total = 0.0

            for step, batch in enumerate(train_loader, 1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=use_fp16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / grad_accum

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                train_loss_total += loss.item() * grad_accum

                if step % grad_accum == 0 or step == len(train_loader):
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            avg_train_loss = train_loss_total / len(train_loader)

            # Evaluate on dev set
            model.eval()
            dev_preds = []
            dev_trues = []
            dev_loss_total = 0.0

            with torch.no_grad():
                for batch in dev_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        dev_loss_total += outputs.loss.item()

                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    dev_preds.extend(preds)
                    dev_trues.extend(labels.cpu().numpy())

            avg_dev_loss = dev_loss_total / len(dev_loader)
            epoch_duration = time.time() - epoch_start

            val_metrics = compute_evaluation_metrics(
                y_true=dev_trues,
                y_pred=dev_preds,
                model_name=cfg["model_name"],
                split_name="dev"
            )

            current_macro_f1 = val_metrics["macro_f1"]

            log_entry = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "dev_loss": round(avg_dev_loss, 4),
                "dev_macro_f1": current_macro_f1,
                "dev_accuracy": val_metrics["accuracy"],
                "dev_weighted_f1": val_metrics["weighted_f1"],
                "epoch_duration_seconds": round(epoch_duration, 2)
            }
            epoch_history.append(log_entry)

            print(
                f"Epoch {epoch}/{num_epochs} [{epoch_duration:.1f}s] | "
                f"Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f} | "
                f"Dev Macro-F1: {current_macro_f1:.4f} | Dev Acc: {val_metrics['accuracy']:.4f}"
            )

            # Save best checkpoint strictly based on Dev Macro-F1
            if current_macro_f1 > best_dev_macro_f1:
                print(f"  * New best validation Macro-F1: {current_macro_f1:.4f} (improved from {best_dev_macro_f1:.4f}). Saving checkpoint...")
                best_dev_macro_f1 = current_macro_f1
                best_metrics = val_metrics
                patience_counter = 0

                best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_checkpoint_dir)
                tokenizer.save_pretrained(best_checkpoint_dir)
            else:
                patience_counter += 1
                print(f"  No improvement in Macro-F1. Early stopping patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

    except torch.cuda.OutOfMemoryError as oom_err:
        print("\n" + "!" * 70)
        print("GPU OUT-OF-MEMORY (OOM) ERROR ENCOUNTERED")
        print("!" * 70)
        print("Recommendation to resolve OOM on Colab / GPU:")
        print(f"1. Reduce batch_size: currently {batch_size} -> try 8 or 4.")
        print(f"2. Increase gradient_accumulation_steps: currently {grad_accum} -> try {grad_accum * 2}.")
        print("3. Ensure mixed precision (fp16: true) is active.")
        print("!" * 70)
        raise oom_err

    total_train_duration = time.time() - start_total_time
    print(f"\nTraining completed in {total_train_duration:.2f} seconds.")
    print(f"Best Validation Macro-F1: {best_dev_macro_f1:.4f}")

    # Save final best validation metrics
    model_slug = cfg["model_name"]
    json_path, csv_path = save_validation_results(best_metrics, val_dir, model_slug)
    print(f"\nSaved validation metrics JSON -> {json_path}")
    print(f"Saved validation metrics CSV  -> {csv_path}")

    # Save confusion matrix plot
    cm_path = val_dir / f"{model_slug}_confusion_matrix.png"
    save_confusion_matrix_plot(
        cm=best_metrics["confusion_matrix"],
        model_name=f"Model: {cfg['model_name'].upper()}",
        output_path=cm_path,
        macro_f1=best_metrics["macro_f1"]
    )
    print(f"Saved confusion matrix plot   -> {cm_path}")

    # Save epoch history log
    history_path = val_dir / f"{model_slug}_epoch_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(epoch_history, f, indent=2)
    print(f"Saved epoch history log       -> {history_path}")

    # Save metadata & configuration
    metadata = {
        "model_slug": model_slug,
        "checkpoint": checkpoint,
        "config": cfg,
        "seed_info": seed_info,
        "truncation_analysis": {
            "train": train_trunc,
            "dev": dev_trunc
        },
        "total_train_duration_seconds": round(total_train_duration, 2),
        "runtime_meta": get_runtime_metadata(),
        "reproducibility_caveats": get_reproducibility_caveats(),
        "best_validation_metrics": best_metrics,
        "epoch_history": epoch_history
    }
    meta_path = val_dir / f"{model_slug}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved execution metadata      -> {meta_path}")

    return {
        "best_metrics": best_metrics,
        "checkpoint_dir": str(best_checkpoint_dir),
        "epoch_history": epoch_history,
        "metadata_path": str(meta_path)
    }
