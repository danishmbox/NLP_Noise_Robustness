"""
Data loading and dataset validation module for NLP Noise Robustness research.

Enforces strict split rules, schema validation, and shared label-to-id mappings.
Under Stage 1 protocol, loading or evaluating on test.csv is strictly guarded against.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Shared explicit label-to-id mapping used across ALL models
LABEL2ID: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

ID2LABEL: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

ORDERED_LABELS: List[str] = ["negative", "neutral", "positive"]
REQUIRED_COLUMNS: List[str] = ["uid", "text", "sentiment"]


def get_label_mappings() -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """
    Returns the shared label mappings and ordered class names.
    """
    return LABEL2ID, ID2LABEL, ORDERED_LABELS


def validate_dataset_schema(df: pd.DataFrame, split_name: str) -> None:
    """
    Validates dataset columns, non-empty state, and sentiment label values.

    Args:
        df: Pandas DataFrame to validate.
        split_name: Name of split ('train' or 'dev').

    Raises:
        ValueError: If schema or labels fail validation.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset split '{split_name}' is missing required columns: {missing_cols}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    if len(df) == 0:
        raise ValueError(f"Dataset split '{split_name}' contains zero rows.")

    unique_labels = set(df["sentiment"].dropna().unique())
    invalid_labels = unique_labels - set(LABEL2ID.keys())
    if invalid_labels:
        raise ValueError(
            f"Dataset split '{split_name}' contains unexpected sentiment labels: {invalid_labels}. "
            f"Allowed labels: {list(LABEL2ID.keys())}"
        )


def load_train_and_dev_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and validates train.csv and dev.csv from data/processed/.

    Args:
        data_dir: Path to directory containing processed CSVs (e.g. data/processed/).

    Returns:
        Tuple of (train_df, dev_df).

    Raises:
        FileNotFoundError: If train.csv or dev.csv is missing.
        ValueError: If validation fails.
    """
    train_path = data_dir / "train.csv"
    dev_path = data_dir / "dev.csv"

    for path, split in [(train_path, "train"), (dev_path, "dev")]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {path}. "
                f"Ensure preprocessing has been completed before training."
            )

    train_df = pd.read_csv(train_path, dtype={"uid": str, "text": str, "sentiment": str})
    dev_df = pd.read_csv(dev_path, dtype={"uid": str, "text": str, "sentiment": str})

    # Validate before and after null sanitization
    validate_dataset_schema(train_df, "train")
    validate_dataset_schema(dev_df, "dev")

    # Safe null handling for text without altering on-disk files
    train_df["text"] = train_df["text"].fillna("")
    dev_df["text"] = dev_df["text"].fillna("")

    # Standardize label strings
    train_df["sentiment"] = train_df["sentiment"].str.strip().str.lower()
    dev_df["sentiment"] = dev_df["sentiment"].str.strip().str.lower()

    # Re-validate
    validate_dataset_schema(train_df, "train")
    validate_dataset_schema(dev_df, "dev")

    # Add numeric label column
    train_df["label"] = train_df["sentiment"].map(LABEL2ID)
    dev_df["label"] = dev_df["sentiment"].map(LABEL2ID)

    return train_df, dev_df


def load_dev_data(data_dir: Path) -> pd.DataFrame:
    """
    Loads and validates ONLY dev.csv from data/processed/.
    Strictly avoids touching or loading train.csv or test.csv.

    Args:
        data_dir: Path to directory containing processed CSVs (e.g. data/processed/).

    Returns:
        Validated dev DataFrame with sanitized text and numeric 'label' column.

    Raises:
        FileNotFoundError: If dev.csv is missing.
        ValueError: If validation fails.
    """
    guard_against_test_set_usage("dev")
    dev_path = data_dir / "dev.csv"
    if not dev_path.exists():
        raise FileNotFoundError(
            f"Required development dataset file not found: {dev_path}."
        )

    dev_df = pd.read_csv(dev_path, dtype={"uid": str, "text": str, "sentiment": str})
    validate_dataset_schema(dev_df, "dev")
    dev_df["text"] = dev_df["text"].fillna("")
    dev_df["sentiment"] = dev_df["sentiment"].str.strip().str.lower()
    validate_dataset_schema(dev_df, "dev")
    dev_df["label"] = dev_df["sentiment"].map(LABEL2ID)
    return dev_df


def guard_against_test_set_usage(split_requested: str) -> None:
    """
    Enforces the Stage 1 rule strictly prohibiting test.csv access during training.

    Raises:
        PermissionError: If 'test' split is requested in Stage 1.
    """
    if "test" in split_requested.lower():
        raise PermissionError(
            "Access to the held-out test split ('test.csv') is strictly PROHIBITED during "
            "Stage 1 model training and checkpoint selection. Only train.csv and dev.csv "
            "may be used for development."
        )
