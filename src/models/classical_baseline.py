"""
Model A — Classical Baseline Pipeline.
Architecture: Word TF-IDF + Character TF-IDF (char_wb) + LinearSVC.

Combines word n-grams (1, 2) and character n-grams (2, 5) using a scikit-learn
FeatureUnion and Pipeline. Fits strictly on train.csv and validates on dev.csv.
The entire fitted pipeline (vectorizers + classifier) is serialized as a single artifact.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.reproducibility import set_seed, get_runtime_metadata
from src.evaluation.metrics import (
    compute_evaluation_metrics,
    save_validation_results,
    save_confusion_matrix_plot
)


DEFAULT_CLASSICAL_CONFIG: Dict[str, Any] = {
    "model_family": "classical",
    "model_name": "classical_baseline",
    "word_ngram_range": [1, 2],
    "char_ngram_range": [2, 5],
    "word_max_features": 50000,
    "char_max_features": 50000,
    "word_min_df": 2,
    "char_min_df": 2,
    "sublinear_tf": True,
    "svm_c": 1.0,
    "svm_loss": "squared_hinge",
    "svm_max_iter": 2000,
    "random_seed": 42
}


def build_classical_pipeline(config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Constructs the reproducible scikit-learn FeatureUnion Pipeline.

    Args:
        config: Configuration dictionary overriding DEFAULT_CLASSICAL_CONFIG.

    Returns:
        Configured scikit-learn Pipeline.
    """
    cfg = dict(DEFAULT_CLASSICAL_CONFIG)
    if config:
        cfg.update(config)

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(cfg["word_ngram_range"]),
        max_features=cfg.get("word_max_features"),
        min_df=cfg.get("word_min_df", 2),
        sublinear_tf=cfg.get("sublinear_tf", True),
        dtype=np.float64
    )

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(cfg["char_ngram_range"]),
        max_features=cfg.get("char_max_features"),
        min_df=cfg.get("char_min_df", 2),
        sublinear_tf=cfg.get("sublinear_tf", True),
        dtype=np.float64
    )

    features = FeatureUnion([
        ("word_tfidf", word_vec),
        ("char_tfidf", char_vec)
    ])

    classifier = LinearSVC(
        C=cfg.get("svm_c", 1.0),
        loss=cfg.get("svm_loss", "squared_hinge"),
        max_iter=cfg.get("svm_max_iter", 2000),
        random_state=cfg.get("random_seed", 42)
    )

    pipeline = Pipeline([
        ("features", features),
        ("classifier", classifier)
    ])

    return pipeline


def train_classical_baseline(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None,
    val_dir: Optional[Path] = None
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains the classical baseline pipeline on train_df and evaluates on dev_df.

    Args:
        train_df: Training DataFrame containing 'text' and 'sentiment'.
        dev_df: Validation DataFrame containing 'text' and 'sentiment'.
        config: Configuration dictionary.
        models_dir: Directory to save serialized model artifact.
        val_dir: Directory to save validation metrics and plots.

    Returns:
        Tuple of (fitted_pipeline, validation_metrics_dict).
    """
    cfg = dict(DEFAULT_CLASSICAL_CONFIG)
    if config:
        cfg.update(config)

    # Set deterministic seeds
    seed_info = set_seed(cfg.get("random_seed", 42))

    if models_dir is None:
        models_dir = Path("results/models")
    if val_dir is None:
        val_dir = Path("results/validation")

    models_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("   MODEL A: WORD + CHAR TF-IDF + LINEARSVC (CLASSICAL BASELINE)   ")
    print("=" * 70)
    print(f"Training instances: {len(train_df):,} | Validation instances: {len(dev_df):,}")
    print(f"Word n-grams: {cfg['word_ngram_range']} | Char_wb n-grams: {cfg['char_ngram_range']}")
    print(f"SVM C: {cfg['svm_c']} | Loss: {cfg['svm_loss']} | Seed: {cfg['random_seed']}")

    # Build and fit pipeline
    pipeline = build_classical_pipeline(cfg)

    start_time = time.time()
    print("\nFitting feature extractors and LinearSVC on train.csv...")
    pipeline.fit(train_df["text"], train_df["sentiment"])
    fit_duration = time.time() - start_time
    print(f"Training completed in {fit_duration:.2f} seconds.")

    # Validate strictly on dev_df
    print("\nEvaluating on dev.csv...")
    dev_preds = pipeline.predict(dev_df["text"])

    metrics = compute_evaluation_metrics(
        y_true=dev_df["sentiment"].values,
        y_pred=dev_preds,
        model_name=cfg.get("model_name", "classical_baseline"),
        split_name="dev"
    )

    print(f"\n--- Validation Results (Dev Set) ---")
    print(f"Macro-F1 (Primary): {metrics['macro_f1']:.4f}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Weighted-F1:        {metrics['weighted_f1']:.4f}")
    for label, scores in metrics["per_class"].items():
        print(f"  Class [{label:8s}] -> F1: {scores['f1']:.4f} | Prec: {scores['precision']:.4f} | Rec: {scores['recall']:.4f}")

    # Save validation results
    model_slug = "classical_baseline"
    json_path, csv_path = save_validation_results(metrics, val_dir, model_slug)
    print(f"\nSaved validation metrics JSON -> {json_path}")
    print(f"Saved validation metrics CSV  -> {csv_path}")

    # Save confusion matrix plot
    cm_path = val_dir / f"{model_slug}_confusion_matrix.png"
    save_confusion_matrix_plot(
        cm=metrics["confusion_matrix"],
        model_name="Model A: Word+Char TF-IDF + LinearSVC",
        output_path=cm_path,
        macro_f1=metrics["macro_f1"]
    )
    print(f"Saved confusion matrix plot   -> {cm_path}")

    # Serialize fitted pipeline (both vectorizers and SVM together)
    model_artifact_path = models_dir / f"{model_slug}_pipeline.joblib"
    joblib.dump(pipeline, model_artifact_path)
    print(f"Serialized fitted pipeline    -> {model_artifact_path}")

    # Save metadata & configuration
    metadata = {
        "model_slug": model_slug,
        "config": cfg,
        "seed_info": seed_info,
        "fit_duration_seconds": round(fit_duration, 2),
        "runtime_meta": get_runtime_metadata(),
        "validation_metrics": metrics
    }
    meta_path = val_dir / f"{model_slug}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved execution metadata      -> {meta_path}")

    return pipeline, metrics
