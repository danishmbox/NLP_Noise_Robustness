"""
Robustness Evaluation Runner for NLP Noise Robustness Research.

Evaluates trained sentiment classification models on the held-out development set (dev.csv)
under controlled typographical noise and Hinglish transliteration variation at 5%, 10%, 15%, 20%
severities without retraining the model.

Supports:
1. Classical Baseline (.joblib scikit-learn pipeline)
2. Fine-tuned Transformer Classifiers (MuRIL, XLM-RoBERTa, etc. in Hugging Face checkpoint format)

Strictly protects test.csv from being loaded or evaluated.
Enforces shared pre-generated noisy datasets, canonical label mapping validation,
and generates structured CSV summaries, detail JSONs, and publication degradation plots.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (
    load_dev_data,
    guard_against_test_set_usage,
    ORDERED_LABELS,
    LABEL2ID,
    ID2LABEL
)
from src.evaluation.metrics import compute_evaluation_metrics
from src.utils.reproducibility import get_runtime_metadata


# Canonical label ordering expected project-wide
CANONICAL_ID2LABEL: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

REQUIRED_SUMMARY_FILES = [
    "classical_baseline_robustness_summary.csv",
    "muril_robustness_summary.csv",
    "xlm_roberta_robustness_summary.csv"
]


def validate_model_label_mapping(config_or_id2label: Any) -> Dict[int, str]:
    """
    Validates that the model's id2label mapping matches the canonical project mapping:
    0: 'negative', 1: 'neutral', 2: 'positive'.

    Handles both integer keys ({0: 'negative', ...}) and string keys ({'0': 'negative', ...})
    commonly found in Hugging Face config.json files.

    Args:
        config_or_id2label: Either a HF PretrainedConfig object, a dict, or config.json contents.

    Returns:
        Dict[int, str] containing the validated normalized mapping.

    Raises:
        ValueError: If id2label is missing or values do not match canonical project mapping.
    """
    if hasattr(config_or_id2label, "id2label"):
        raw_id2label = config_or_id2label.id2label
    elif isinstance(config_or_id2label, dict):
        raw_id2label = config_or_id2label.get("id2label", config_or_id2label)
    else:
        raise ValueError(f"Invalid label mapping source object: {type(config_or_id2label)}")

    if not raw_id2label:
        raise ValueError("Model configuration does not contain 'id2label'.")

    # Normalize string/int keys and lower-cased string values
    normalized: Dict[int, str] = {}
    for k, v in raw_id2label.items():
        try:
            int_k = int(k)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot cast label ID '{k}' to integer in id2label: {raw_id2label}")
        normalized[int_k] = str(v).strip().lower()

    if normalized != CANONICAL_ID2LABEL:
        raise ValueError(
            f"Model label mapping mismatch! Expected canonical mapping: {CANONICAL_ID2LABEL}, "
            f"but found: {normalized}."
        )

    return normalized


def resolve_checkpoint_path(
    model_slug: str,
    explicit_path: Optional[Union[str, Path]] = None,
    candidate_paths: Optional[List[Union[str, Path]]] = None
) -> Path:
    """
    Resolves the actual existing directory for a transformer checkpoint.
    Never assumes a path exists until verified with .is_dir() and 'config.json' check.

    Args:
        model_slug: Identifier ('muril', 'xlm_roberta', etc.).
        explicit_path: User-provided CLI path.
        candidate_paths: Optional explicit list of paths to check.

    Returns:
        Path to the verified existing checkpoint directory.

    Raises:
        FileNotFoundError: If no valid checkpoint directory containing 'config.json' is found.
    """
    paths_to_test: List[Path] = []
    if explicit_path:
        paths_to_test.append(Path(explicit_path))

    # Known candidate backup and local locations
    specific_backup_paths = {
        "muril": [
            Path("/content/drive/MyDrive/NLP_Noise_Robustness_Backup/MuRIL_2026-10-14"),
            Path("/content/drive/MyDrive/NLP_Noise_Robustness_Backup/MuRIL"),
            Path("results/models/muril_best_checkpoint"),
            Path("/content/NLP_Noise_Robustness/results/models/muril_best_checkpoint"),
            Path("/content/drive/MyDrive/NLP_Noise_Robustness/results/models/muril_best_checkpoint"),
            Path("/content/drive/MyDrive/models/muril_best_checkpoint")
        ],
        "xlm_roberta": [
            Path("/content/drive/MyDrive/NLP_Noise_Robustness_Backup/XLM-RoBERTa_2026-10-17"),
            Path("/content/drive/MyDrive/NLP_Noise_Robustness_Backup/XLM-RoBERTa"),
            Path("results/models/xlm_roberta_best_checkpoint"),
            Path("/content/NLP_Noise_Robustness/results/models/xlm_roberta_best_checkpoint"),
            Path("/content/drive/MyDrive/NLP_Noise_Robustness/results/models/xlm_roberta_best_checkpoint"),
            Path("/content/drive/MyDrive/models/xlm_roberta_best_checkpoint")
        ]
    }

    if candidate_paths:
        paths_to_test.extend([Path(p) for p in candidate_paths])
    elif model_slug in specific_backup_paths:
        paths_to_test.extend(specific_backup_paths[model_slug])

    paths_to_test.append(PROJECT_ROOT / "results" / "models" / f"{model_slug}_best_checkpoint")

    checked_paths: List[Path] = []
    for p in paths_to_test:
        checked_paths.append(p)
        if p.exists() and p.is_dir():
            if (p / "config.json").exists():
                return p.resolve()

    searched_str = "\n".join(f" - {p}" for p in checked_paths)
    raise FileNotFoundError(
        f"Could not locate a valid checkpoint directory containing 'config.json' for model '{model_slug}'.\n"
        f"Searched paths:\n{searched_str}\n\n"
        f"If running in Google Colab, verify that Google Drive is mounted:\n"
        f"  from google.colab import drive\n"
        f"  drive.mount('/content/drive')\n"
        f"And ensure the checkpoint directory exists at one of the searched paths."
    )


def load_and_validate_noisy_datasets(
    clean_df: pd.DataFrame,
    datasets_dir: Path,
    severities: List[float]
) -> Dict[str, Dict[float, pd.DataFrame]]:
    """
    Loads and strictly validates all 8 pre-generated shared noisy dev CSVs.
    Verifies row counts, column schemas, and index-by-index alignment of UID and labels.
    Never regenerates datasets, ensuring 100% identical evaluation benchmarks.

    Args:
        clean_df: Ground truth clean dev DataFrame.
        datasets_dir: Directory containing dev_typo_* and dev_transliteration_* CSVs.
        severities: Expected severity levels (e.g. [0.05, 0.10, 0.15, 0.20]).

    Returns:
        Dict: {noise_type: {severity: noisy_df}}

    Raises:
        FileNotFoundError: If any required shared dataset is missing.
        ValueError: If row counts, IDs, or labels fail alignment checks.
    """
    noise_types = ["typo", "transliteration"]
    loaded_datasets: Dict[str, Dict[float, pd.DataFrame]] = {
        "typo": {},
        "transliteration": {}
    }

    clean_len = len(clean_df)
    clean_uids = clean_df["uid"].astype(str).tolist()
    clean_sentiments = clean_df["sentiment"].tolist()
    clean_labels = clean_df["label"].tolist()

    for n_type in noise_types:
        for sev in severities:
            sev_pct_int = int(sev * 100)
            filename = f"dev_{n_type}_{sev_pct_int:02d}pct.csv"
            filepath = datasets_dir / filename

            if not filepath.exists():
                raise FileNotFoundError(
                    f"Required shared noisy dataset not found: {filepath}.\n"
                    f"Datasets must not be regenerated. Ensure the pre-generated files "
                    f"under results/robustness/datasets/ are intact."
                )

            noisy_df = pd.read_csv(filepath, dtype={"uid": str, "text": str, "sentiment": str})
            noisy_df["text"] = noisy_df["text"].fillna("")
            noisy_df["sentiment"] = noisy_df["sentiment"].str.strip().str.lower()
            if "label" not in noisy_df.columns:
                noisy_df["label"] = noisy_df["sentiment"].map(LABEL2ID)

            # 1. Row count validation
            if len(noisy_df) != clean_len:
                raise ValueError(
                    f"Row count mismatch in '{filename}': {len(noisy_df)} vs clean {clean_len}."
                )

            # 2. Schema validation
            for col in ["uid", "text", "sentiment", "label"]:
                if col not in noisy_df.columns:
                    raise ValueError(f"Missing required column '{col}' in '{filename}'.")

            # 3. UID alignment check
            noisy_uids = noisy_df["uid"].astype(str).tolist()
            if noisy_uids != clean_uids:
                raise ValueError(f"UID sequence mismatch between '{filename}' and clean dev set.")

            # 4. Sentiment label alignment check
            noisy_sentiments = noisy_df["sentiment"].tolist()
            if noisy_sentiments != clean_sentiments:
                raise ValueError(f"Sentiment label sequence mismatch between '{filename}' and clean dev set.")

            noisy_labels = noisy_df["label"].tolist()
            if noisy_labels != clean_labels:
                raise ValueError(f"Numeric label sequence mismatch between '{filename}' and clean dev set.")

            loaded_datasets[n_type][sev] = noisy_df

    return loaded_datasets


def predict_transformer(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    device: Any,
    batch_size: int = 32,
    max_length: int = 128,
    use_fp16: bool = True,
    id2label: Optional[Dict[int, str]] = None
) -> List[str]:
    """
    Inference-only batch prediction for transformer classifiers.
    Strictly uses torch.inference_mode() or torch.no_grad().
    Safe fallback if mixed precision is unsupported.

    Args:
        model: Pre-trained Hugging Face sequence classification model.
        tokenizer: Hugging Face tokenizer.
        texts: List of text strings to predict.
        device: PyTorch device ('cuda' or 'cpu').
        batch_size: Evaluation batch size.
        max_length: Maximum sequence length.
        use_fp16: Enable FP16 autocast on CUDA if supported.
        id2label: Validated mapping from integer class to sentiment string.

    Returns:
        List of predicted sentiment label strings ('negative', 'neutral', 'positive').
    """
    import torch

    if id2label is None:
        id2label = CANONICAL_ID2LABEL

    model.eval()
    all_preds: List[str] = []

    # Choose inference context manager
    inf_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
    enable_fp16 = bool(use_fp16 and str(device).startswith("cuda"))

    with inf_ctx:
        total = len(texts)
        for i in range(0, total, batch_size):
            batch_texts = [str(t) for t in texts[i:i + batch_size]]
            encodings = tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            try:
                if enable_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception as e:
                # Safe fallback if autocast fails
                if enable_fp16:
                    enable_fp16 = False
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    raise e

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            for p in preds:
                all_preds.append(id2label[int(p)])

    return all_preds


def run_robustness_evaluation(
    model_path: Optional[Union[str, Path]] = None,
    data_dir: Path = PROJECT_ROOT / "data" / "processed",
    output_dir: Path = PROJECT_ROOT / "results" / "robustness",
    model_slug: str = "classical_baseline",
    model_name: str = "Model A: Word+Char TF-IDF + LinearSVC",
    model_type: Optional[str] = None,
    batch_size: int = 32,
    use_fp16: bool = True,
    seed: int = 42,
    severities: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Executes the robustness evaluation protocol on dev.csv.
    Polymorphic for Classical (joblib) and Transformer (checkpoint dir) models.

    Args:
        model_path: Path to serialized artifact (.joblib) or checkpoint directory.
        data_dir: Processed datasets directory (data/processed).
        output_dir: Output directory (results/robustness).
        model_slug: Clean identifier string ('classical_baseline', 'muril', 'xlm_roberta').
        model_name: Human-readable model display name.
        model_type: 'classical' or 'transformer' (auto-detected if None).
        batch_size: Batch size for transformer evaluation.
        use_fp16: Use FP16 for transformer inference on CUDA.
        seed: Random seed.
        severities: List of severity levels (default: [0.05, 0.10, 0.15, 0.20]).

    Returns:
        Structured dictionary of evaluation results.
    """
    if severities is None:
        severities = [0.05, 0.10, 0.15, 0.20]

    # Stage guard: Enforce dev-only evaluation
    guard_against_test_set_usage("dev")

    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine model type
    if model_type is None:
        if model_path and str(model_path).endswith(".joblib"):
            model_type = "classical"
        elif model_slug in ["muril", "xlm_roberta", "mbert"]:
            model_type = "transformer"
        else:
            model_type = "classical"

    print("=" * 76)
    print("      STAGE 2: NLP NOISE ROBUSTNESS EVALUATION (DEV SET ONLY)       ")
    print("=" * 76)
    print(f"Model Name:        {model_name}")
    print(f"Model Slug:        {model_slug}")
    print(f"Model Type:        {model_type.upper()}")
    print(f"Data Directory:    {data_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Severities:        {[f'{int(s * 100)}%' for s in severities]}")
    print(f"Random Seed:       {seed}")
    print("-" * 76)

    # 2. Load model
    transformer_model = None
    transformer_tok = None
    classical_pipeline = None
    device = None
    validated_id2label = CANONICAL_ID2LABEL

    if model_type == "classical":
        if model_path is None:
            model_path = PROJECT_ROOT / "results" / "models" / f"{model_slug}_pipeline.joblib"
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Classical model artifact not found at: {model_path}")

        print(f"Loading classical pipeline from {model_path}...")
        classical_pipeline = joblib.load(model_path)
        print("Classical pipeline loaded successfully.")

    elif model_type == "transformer":
        # Resolve existing checkpoint directory
        resolved_ckpt = resolve_checkpoint_path(model_slug=model_slug, explicit_path=model_path)
        print(f"Resolved verified transformer checkpoint: {resolved_ckpt}")

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Execution Device:  {device} | Mixed Precision (FP16): {use_fp16 and torch.cuda.is_available()}")

        print(f"Loading tokenizer from {resolved_ckpt}...")
        transformer_tok = AutoTokenizer.from_pretrained(resolved_ckpt)

        print(f"Loading model from {resolved_ckpt}...")
        transformer_model = AutoModelForSequenceClassification.from_pretrained(resolved_ckpt)

        # Validate label mapping against canonical definition
        validated_id2label = validate_model_label_mapping(transformer_model.config)
        print(f"Validated label mapping: {validated_id2label}")

        transformer_model.to(device)
        transformer_model.eval()
        print("Transformer model loaded and set to eval() mode successfully.")

    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Expected 'classical' or 'transformer'.")

    # Predict helper function
    def run_prediction(texts: List[str]) -> List[str]:
        if model_type == "classical":
            return list(classical_pipeline.predict(texts))
        else:
            return predict_transformer(
                model=transformer_model,
                tokenizer=transformer_tok,
                texts=texts,
                device=device,
                batch_size=batch_size,
                max_length=128,
                use_fp16=use_fp16,
                id2label=validated_id2label
            )

    # 3. Load clean dev.csv
    print(f"\nLoading clean development set from {data_dir / 'dev.csv'}...")
    clean_dev_df = load_dev_data(data_dir)
    print(f"Loaded {len(clean_dev_df):,} validation samples.")

    # 4. Load and validate shared pre-generated noisy datasets
    print("Loading and validating existing shared noisy dev datasets...")
    noisy_datasets = load_and_validate_noisy_datasets(clean_dev_df, datasets_dir, severities)
    print("All 8 shared noisy datasets successfully verified (100% row & label aligned).")

    # 5. Clean dev evaluation (baseline reference)
    print("\n--- Evaluating Clean Development Set ---")
    clean_preds = run_prediction(clean_dev_df["text"].tolist())
    clean_metrics = compute_evaluation_metrics(
        y_true=clean_dev_df["sentiment"].values,
        y_pred=clean_preds,
        model_name=model_name,
        split_name="dev"
    )
    clean_macro_f1 = clean_metrics["macro_f1"]
    clean_acc = clean_metrics["accuracy"]
    clean_weighted_f1 = clean_metrics["weighted_f1"]
    print(f"Clean Dev Macro-F1: {clean_macro_f1:.4f} | Accuracy: {clean_acc:.4f} | Weighted-F1: {clean_weighted_f1:.4f}")

    # 6. Perturbation & Evaluation Loop over Pre-generated Sets
    summary_rows: List[Dict[str, Any]] = []

    # Clean baseline row (Severity = 0%)
    summary_rows.append({
        "model_name": model_name,
        "model_slug": model_slug,
        "noise_type": "clean",
        "severity": 0.0,
        "severity_pct": "0%",
        "macro_f1": clean_macro_f1,
        "delta_macro_f1": 0.0,
        "retention_pct": 100.0,
        "accuracy": clean_acc,
        "delta_accuracy": 0.0,
        "weighted_f1": clean_weighted_f1,
        "neg_f1": clean_metrics["per_class"]["negative"]["f1"],
        "neu_f1": clean_metrics["per_class"]["neutral"]["f1"],
        "pos_f1": clean_metrics["per_class"]["positive"]["f1"],
        "sample_count": len(clean_dev_df),
        "seed": seed
    })

    detailed_experiments: Dict[str, Dict[str, Any]] = {
        "clean": clean_metrics
    }

    noise_types = ["typo", "transliteration"]
    for n_type in noise_types:
        print(f"\n>>> Evaluating on Existing Benchmark: {n_type.upper()} NOISE <<<")
        detailed_experiments[n_type] = {}

        for sev in severities:
            sev_pct_str = f"{int(sev * 100)}%"
            noisy_df = noisy_datasets[n_type][sev]

            noisy_preds = run_prediction(noisy_df["text"].tolist())
            noisy_metrics = compute_evaluation_metrics(
                y_true=noisy_df["sentiment"].values,
                y_pred=noisy_preds,
                model_name=model_name,
                split_name="dev"
            )

            macro_f1 = noisy_metrics["macro_f1"]
            acc = noisy_metrics["accuracy"]
            weighted_f1 = noisy_metrics["weighted_f1"]

            delta_macro_f1 = round(clean_macro_f1 - macro_f1, 4)
            delta_acc = round(clean_acc - acc, 4)
            retention_pct = round((macro_f1 / clean_macro_f1) * 100, 2) if clean_macro_f1 > 0 else 0.0

            print(
                f"  [{n_type:15s} | Sev: {sev_pct_str:4s}] "
                f"Macro-F1: {macro_f1:.4f} (Delta: -{delta_macro_f1:+.4f}, Ret: {retention_pct:5.1f}%) | "
                f"Acc: {acc:.4f} (Delta: -{delta_acc:+.4f})"
            )

            row = {
                "model_name": model_name,
                "model_slug": model_slug,
                "noise_type": n_type,
                "severity": sev,
                "severity_pct": sev_pct_str,
                "macro_f1": macro_f1,
                "delta_macro_f1": delta_macro_f1,
                "retention_pct": retention_pct,
                "accuracy": acc,
                "delta_accuracy": delta_acc,
                "weighted_f1": weighted_f1,
                "neg_f1": noisy_metrics["per_class"]["negative"]["f1"],
                "neu_f1": noisy_metrics["per_class"]["neutral"]["f1"],
                "pos_f1": noisy_metrics["per_class"]["positive"]["f1"],
                "sample_count": len(noisy_df),
                "seed": seed
            }
            summary_rows.append(row)
            detailed_experiments[n_type][sev_pct_str] = noisy_metrics

    # 7. Export Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / f"{model_slug}_robustness_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[+] Saved robustness summary CSV -> {summary_csv_path}")

    # 8. Export Detailed JSON
    full_payload = {
        "metadata": {
            "model_name": model_name,
            "model_slug": model_slug,
            "model_type": model_type,
            "model_path": str(model_path),
            "seed": seed,
            "severities": severities,
            "runtime": get_runtime_metadata(),
            "clean_baseline": {
                "macro_f1": clean_macro_f1,
                "accuracy": clean_acc,
                "weighted_f1": clean_weighted_f1
            }
        },
        "experiments": detailed_experiments
    }
    json_path = output_dir / f"{model_slug}_robustness_details.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    print(f"[+] Saved robustness details JSON -> {json_path}")

    # 9. Generate Degradation Plots
    plot_degradation_curves(summary_df, model_name, output_dir, model_slug)

    print("\n" + "=" * 76)
    print("                      ROBUSTNESS EVALUATION COMPLETE                       ")
    print("=" * 76)

    return full_payload


def plot_degradation_curves(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    model_slug: str
) -> None:
    """
    Generates publication-quality degradation plots for Macro-F1 and Accuracy.
    """
    clean_row = df[df["noise_type"] == "clean"].iloc[0]
    clean_f1 = clean_row["macro_f1"]
    clean_acc = clean_row["accuracy"]

    typo_df = df[df["noise_type"] == "typo"]
    translit_df = df[df["noise_type"] == "transliteration"]

    # X-axis points: 0%, 5%, 10%, 15%, 20%
    x_vals = [0, 5, 10, 15, 20]
    typo_f1 = [clean_f1] + typo_df["macro_f1"].tolist()
    translit_f1 = [clean_f1] + translit_df["macro_f1"].tolist()

    typo_acc = [clean_acc] + typo_df["accuracy"].tolist()
    translit_acc = [clean_acc] + translit_df["accuracy"].tolist()

    sns.set_theme(style="whitegrid", font="sans-serif")

    # --- Plot 1: Macro-F1 Degradation ---
    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, typo_f1, marker="o", color="#d95f02", linewidth=2.2, label="Typographical Noise (QWERTY)")
    plt.plot(x_vals, translit_f1, marker="s", color="#7570b3", linewidth=2.2, label="Hinglish Transliteration Noise")
    plt.axhline(clean_f1, color="#1b9e77", linestyle="--", linewidth=1.5, label=f"Clean Baseline ({clean_f1:.4f})")

    for x, y in zip(x_vals[1:], typo_f1[1:]):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    for x, y in zip(x_vals[1:], translit_f1[1:]):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)

    plt.title(f"{model_name}\nMacro-F1 Degradation Under Linguistic Noise", fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Perturbation Severity (%)", fontsize=11)
    plt.ylabel("Macro-F1 Score", fontsize=11)
    plt.xticks(x_vals, [f"{x}%" for x in x_vals])
    plt.ylim(min(min(typo_f1), min(translit_f1)) - 0.05, clean_f1 + 0.04)
    plt.legend(frameon=True, loc="lower left", fontsize=9)
    plt.tight_layout()
    f1_plot_path = output_dir / f"{model_slug}_macro_f1_degradation.png"
    plt.savefig(f1_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved Macro-F1 degradation plot -> {f1_plot_path}")

    # --- Plot 2: Accuracy Degradation ---
    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, typo_acc, marker="o", color="#d95f02", linewidth=2.2, label="Typographical Noise (QWERTY)")
    plt.plot(x_vals, translit_acc, marker="s", color="#7570b3", linewidth=2.2, label="Hinglish Transliteration Noise")
    plt.axhline(clean_acc, color="#1b9e77", linestyle="--", linewidth=1.5, label=f"Clean Baseline ({clean_acc:.4f})")

    for x, y in zip(x_vals[1:], typo_acc[1:]):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    for x, y in zip(x_vals[1:], translit_acc[1:]):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)

    plt.title(f"{model_name}\nAccuracy Degradation Under Linguistic Noise", fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Perturbation Severity (%)", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.xticks(x_vals, [f"{x}%" for x in x_vals])
    plt.ylim(min(min(typo_acc), min(translit_acc)) - 0.05, clean_acc + 0.04)
    plt.legend(frameon=True, loc="lower left", fontsize=9)
    plt.tight_layout()
    acc_plot_path = output_dir / f"{model_slug}_accuracy_degradation.png"
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved Accuracy degradation plot -> {acc_plot_path}")

    # --- Plot 3: 2-Panel Combined Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Macro-F1
    axes[0].plot(x_vals, typo_f1, marker="o", color="#d95f02", linewidth=2.2, label="Typographical")
    axes[0].plot(x_vals, translit_f1, marker="s", color="#7570b3", linewidth=2.2, label="Transliteration")
    axes[0].axhline(clean_f1, color="#1b9e77", linestyle="--", linewidth=1.5, label=f"Clean ({clean_f1:.4f})")
    axes[0].set_title("A. Macro-F1 vs Severity", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Severity (%)", fontsize=10)
    axes[0].set_ylabel("Macro-F1", fontsize=10)
    axes[0].set_xticks(x_vals)
    axes[0].set_xticklabels([f"{x}%" for x in x_vals])
    axes[0].legend(loc="lower left", fontsize=9)

    # Panel B: Accuracy
    axes[1].plot(x_vals, typo_acc, marker="o", color="#d95f02", linewidth=2.2, label="Typographical")
    axes[1].plot(x_vals, translit_acc, marker="s", color="#7570b3", linewidth=2.2, label="Transliteration")
    axes[1].axhline(clean_acc, color="#1b9e77", linestyle="--", linewidth=1.5, label=f"Clean ({clean_acc:.4f})")
    axes[1].set_title("B. Accuracy vs Severity", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Severity (%)", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].set_xticks(x_vals)
    axes[1].set_xticklabels([f"{x}%" for x in x_vals])
    axes[1].legend(loc="lower left", fontsize=9)

    fig.suptitle(f"{model_name} - Robustness Profile (Dev Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    comb_plot_path = output_dir / f"{model_slug}_combined_robustness.png"
    plt.savefig(comb_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved Combined robustness plot  -> {comb_plot_path}")


def aggregate_tier1_robustness(
    robustness_dir: Path,
    output_file: Optional[Path] = None
) -> pd.DataFrame:
    """
    Aggregates strictly the three Tier 1 model robustness summary files:
    1. classical_baseline_robustness_summary.csv
    2. muril_robustness_summary.csv
    3. xlm_roberta_robustness_summary.csv

    Fails clearly if any of the three required files is missing.
    Never aggregates an existing combined benchmark into itself.

    Args:
        robustness_dir: Path to directory containing robustness CSV summaries.
        output_file: Path to save the combined benchmark CSV.

    Returns:
        Combined pandas DataFrame.

    Raises:
        FileNotFoundError: If any of the 3 required summaries is missing.
    """
    missing_files = []
    dfs = []

    for fname in REQUIRED_SUMMARY_FILES:
        fpath = robustness_dir / fname
        if not fpath.exists():
            missing_files.append(fname)
        else:
            dfs.append(pd.read_csv(fpath))

    if missing_files:
        present_files = [f for f in REQUIRED_SUMMARY_FILES if f not in missing_files]
        raise FileNotFoundError(
            f"Cannot aggregate Tier 1 robustness benchmark. Missing required model summary files: {missing_files}.\n"
            f"Present files: {present_files}.\n"
            f"Please run the robustness evaluation for all three models before aggregating."
        )

    combined_df = pd.concat(dfs, ignore_index=True)

    if output_file is None:
        output_file = robustness_dir / "tier1_cross_model_robustness_benchmark.csv"

    combined_df.to_csv(output_file, index=False)
    print(f"\n[+] Saved aggregated Tier 1 robustness benchmark -> {output_file}")

    # Generate Cross-Model Comparison Degradation Plot
    plot_cross_model_comparison(combined_df, robustness_dir)

    return combined_df


def plot_cross_model_comparison(combined_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generates a 2-panel comparative figure displaying Macro-F1 across all 3 models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="whitegrid", font="sans-serif")

    model_colors = {
        "classical_baseline": "#1b9e77",
        "muril": "#d95f02",
        "xlm_roberta": "#7570b3"
    }

    noise_types = ["typo", "transliteration"]
    titles = [
        "A. Typographical Noise (QWERTY)",
        "B. Hinglish Transliteration Noise"
    ]

    for idx, (ntype, title) in enumerate(zip(noise_types, titles)):
        ax = axes[idx]
        for slug, color in model_colors.items():
            model_data = combined_df[combined_df["model_slug"] == slug]
            if model_data.empty:
                continue

            clean_row = model_data[model_data["noise_type"] == "clean"]
            if clean_row.empty:
                continue
            clean_f1 = clean_row.iloc[0]["macro_f1"]
            model_name = clean_row.iloc[0]["model_name"]

            noisy_data = model_data[model_data["noise_type"] == ntype].sort_values(by="severity")
            x_vals = [0] + (noisy_data["severity"] * 100).astype(int).tolist()
            y_vals = [clean_f1] + noisy_data["macro_f1"].tolist()

            ax.plot(x_vals, y_vals, marker="o", linewidth=2.0, color=color, label=f"{model_name}")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Perturbation Severity (%)", fontsize=10)
        ax.set_ylabel("Macro-F1 Score", fontsize=10)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xticklabels(["0%", "5%", "10%", "15%", "20%"])
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle("Tier 1 Models: Cross-Architecture Robustness Comparison (Dev Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    comparison_plot_path = output_dir / "tier1_cross_model_macro_f1_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved cross-model comparison plot -> {comparison_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model robustness to linguistic noise.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to serialized model artifact (.joblib) or transformer checkpoint directory"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "robustness",
        help="Path to output robustness directory"
    )
    parser.add_argument(
        "--model_slug",
        type=str,
        default="classical_baseline",
        help="Clean identifier for the model ('classical_baseline', 'muril', 'xlm_roberta')"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Model A: Word+Char TF-IDF + LinearSVC",
        help="Display name for plots and tables"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["classical", "transformer"],
        help="Type of model ('classical' or 'transformer')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for transformer evaluation"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=True,
        help="Use FP16 mixed precision on CUDA"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise generation"
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate the 3 Tier 1 model robustness summaries into a combined benchmark"
    )

    args = parser.parse_args()

    if args.aggregate:
        aggregate_tier1_robustness(robustness_dir=args.output_dir)
    else:
        run_robustness_evaluation(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_slug=args.model_slug,
            model_name=args.model_name,
            model_type=args.model_type,
            batch_size=args.batch_size,
            use_fp16=args.use_fp16,
            seed=args.seed
        )
