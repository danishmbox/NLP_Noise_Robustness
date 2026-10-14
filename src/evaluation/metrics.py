"""
Standardized evaluation metrics for NLP Noise Robustness research.

Calculates:
- Macro-F1 (primary competition/evaluation metric)
- Accuracy
- Weighted-F1
- Per-class Precision, Recall, and F1
- Confusion Matrix (data and plot export)

All metrics are explicitly designated as 'development_validation' during Stage 1.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import ORDERED_LABELS, LABEL2ID, ID2LABEL


def compute_evaluation_metrics(
    y_true: Union[List[str], np.ndarray, List[int]],
    y_pred: Union[List[str], np.ndarray, List[int]],
    model_name: str,
    split_name: str = "dev"
) -> Dict[str, Any]:
    """
    Computes complete classification metrics for a given model predictions.

    Args:
        y_true: Ground truth labels (strings or integer IDs).
        y_pred: Predicted labels (strings or integer IDs).
        model_name: Identifier of the evaluated model.
        split_name: Name of evaluated split (must be 'dev' or 'validation' in Stage 1).

    Returns:
        Structured dictionary containing all computed metrics.
    """
    # Convert integer IDs to string labels if necessary
    if len(y_true) > 0 and isinstance(y_true[0], (int, np.integer)):
        y_true_str = [ID2LABEL[int(i)] for i in y_true]
    else:
        y_true_str = [str(i).strip().lower() for i in y_true]

    if len(y_pred) > 0 and isinstance(y_pred[0], (int, np.integer)):
        y_pred_str = [ID2LABEL[int(i)] for i in y_pred]
    else:
        y_pred_str = [str(i).strip().lower() for i in y_pred]

    acc = float(accuracy_score(y_true_str, y_pred_str))
    macro_f1 = float(f1_score(y_true_str, y_pred_str, labels=ORDERED_LABELS, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_str, y_pred_str, labels=ORDERED_LABELS, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_str,
        y_pred_str,
        labels=ORDERED_LABELS,
        average=None,
        zero_division=0
    )

    per_class = {}
    for idx, label in enumerate(ORDERED_LABELS):
        per_class[label] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx])
        }

    cm = confusion_matrix(y_true_str, y_pred_str, labels=ORDERED_LABELS)

    metrics_payload = {
        "evaluation_phase": "Stage_1_Model_Training_Validation",
        "evaluation_split": split_name,
        "is_final_test_evaluation": False,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "primary_metric": "macro_f1",
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels_order": ORDERED_LABELS,
        "sample_count": len(y_true_str)
    }

    return metrics_payload


def save_validation_results(
    metrics: Dict[str, Any],
    output_dir: Path,
    model_slug: str
) -> Tuple[Path, Path]:
    """
    Saves validation metrics to structured JSON and flat CSV summary files.

    Args:
        metrics: Dictionary returned by compute_evaluation_metrics.
        output_dir: Directory where results should be stored (e.g. results/validation/).
        model_slug: Clean identifier string for the model (e.g. 'classical_baseline').

    Returns:
        Tuple of (json_path, csv_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{model_slug}_val_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Flatten for CSV benchmark table
    row = {
        "model_name": metrics["model_name"],
        "evaluation_split": metrics["evaluation_split"],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "weighted_f1": metrics["weighted_f1"],
        "neg_f1": metrics["per_class"]["negative"]["f1"],
        "neu_f1": metrics["per_class"]["neutral"]["f1"],
        "pos_f1": metrics["per_class"]["positive"]["f1"],
        "sample_count": metrics["sample_count"],
        "timestamp": metrics["timestamp"]
    }

    csv_path = output_dir / f"{model_slug}_val_summary.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    return json_path, csv_path


def save_confusion_matrix_plot(
    cm: Union[List[List[int]], np.ndarray],
    model_name: str,
    output_path: Path,
    macro_f1: float
) -> Path:
    """
    Generates and saves a publication-quality confusion matrix plot.

    Args:
        cm: Confusion matrix array or list of lists.
        model_name: Model display name.
        output_path: File path to save the PNG image.
        macro_f1: Macro-F1 score to annotate on title.

    Returns:
        Path to the saved PNG plot.
    """
    cm_arr = np.array(cm)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ORDERED_LABELS,
        yticklabels=ORDERED_LABELS,
        cbar=True
    )
    plt.title(f"{model_name} (Dev Set)\nValidation Macro-F1: {macro_f1:.4f}", fontsize=12, pad=12)
    plt.xlabel("Predicted Sentiment", fontsize=11)
    plt.ylabel("Ground Truth Sentiment", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path
