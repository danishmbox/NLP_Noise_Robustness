"""
Data Preprocessing Pipeline for SemEval-2020 Task 9 (SentiMix Hindi-English).

This module handles:
1. Parsing the raw CoNLL formatted files for train, dev, and test splits.
2. Reconstructing raw tweet text from token rows while preserving original tokens,
   spelling, code-mixing, transliteration, punctuation, and emojis.
3. Aligning official ground truth labels for the test split using tweet UIDs.
4. Exporting standardized CSV files (uid, text, sentiment) to data/processed/.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_conll_file(file_path: Path) -> pd.DataFrame:
    """
    Parses a SentiMix CoNLL-formatted text file into a DataFrame.

    Each instance begins with a header:
        meta    <uid>    <sentiment>   (for train and dev)
        meta    <uid>                  (for unlabelled test)
    followed by token-level rows:
        <token> <language_tag>
    separated by empty lines.

    Args:
        file_path (Path): Path to the CoNLL-formatted file.

    Returns:
        pd.DataFrame: DataFrame containing columns ['uid', 'text', 'sentiment'].
    """
    records: List[Dict[str, Optional[str]]] = []
    current_uid: Optional[str] = None
    current_sentiment: Optional[str] = None
    current_tokens: List[str] = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_clean = line.rstrip("\r\n")
            if not line_clean:
                continue

            if line_clean.startswith("meta\t"):
                if current_uid is not None:
                    records.append({
                        "uid": current_uid,
                        "text": " ".join(current_tokens),
                        "sentiment": current_sentiment
                    })
                parts = line_clean.split("\t")
                current_uid = parts[1].strip()
                current_sentiment = parts[2].strip() if len(parts) > 2 else None
                current_tokens = []
            else:
                parts = line_clean.split("\t")
                token = parts[0]
                # Retain all non-empty tokens without altering case, punctuation, or spelling
                if token:
                    current_tokens.append(token)

        # Append last instance
        if current_uid is not None:
            records.append({
                "uid": current_uid,
                "text": " ".join(current_tokens),
                "sentiment": current_sentiment
            })

    df = pd.DataFrame(records)
    df["uid"] = df["uid"].astype(str).str.strip()
    return df


def load_test_labels(labels_path: Path) -> pd.DataFrame:
    """
    Loads official gold test labels from CSV/TSV format.

    Args:
        labels_path (Path): Path to the test labels file.

    Returns:
        pd.DataFrame: DataFrame with ['uid', 'sentiment'].
    """
    df = pd.read_csv(labels_path, dtype={"Uid": str, "Sentiment": str})
    df = df.rename(columns={"Uid": "uid", "Sentiment": "sentiment"})
    df["uid"] = df["uid"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    return df


def run_preprocessing(raw_dir: Path, processed_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Executes full preprocessing and saves processed CSV files.

    Args:
        raw_dir (Path): Directory containing raw dataset files.
        processed_dir (Path): Output directory for processed CSV files.

    Returns:
        Dict[str, pd.DataFrame]: Preprocessed DataFrames for 'train', 'dev', and 'test'.
    """
    train_path = raw_dir / "train_14k_split_conll.txt"
    dev_path = raw_dir / "dev_3k_split_conll.txt"
    test_path = raw_dir / "Hindi_test_unalbelled_conll_updated.txt"
    test_labels_path = raw_dir / "test_labels_hinglish.txt"

    print(f"Reading training set from: {train_path.name}...")
    train_df = parse_conll_file(train_path)

    print(f"Reading dev set from: {dev_path.name}...")
    dev_df = parse_conll_file(dev_path)

    print(f"Reading test set from: {test_path.name}...")
    test_df = parse_conll_file(test_path)

    print(f"Aligning gold test labels from: {test_labels_path.name}...")
    test_labels = load_test_labels(test_labels_path)
    test_df = test_df.drop(columns=["sentiment"]).merge(test_labels, on="uid", how="left")

    processed_dir.mkdir(parents=True, exist_ok=True)

    output_files = {
        "train": (train_df, processed_dir / "train.csv"),
        "dev": (dev_df, processed_dir / "dev.csv"),
        "test": (test_df, processed_dir / "test.csv"),
    }

    for split_name, (df, out_path) in output_files.items():
        # Ensure exact column structure: uid, text, sentiment
        df = df[["uid", "text", "sentiment"]]
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Successfully saved {split_name} ({len(df):,} rows) -> {out_path}")

    return {
        "train": train_df[["uid", "text", "sentiment"]],
        "dev": dev_df[["uid", "text", "sentiment"]],
        "test": test_df[["uid", "text", "sentiment"]],
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    print("=" * 70)
    print("SentiMix Hindi-English Preprocessing")
    print("=" * 70)

    dfs = run_preprocessing(raw_dir, processed_dir)

    print("\n" + "=" * 70)
    print("DATASET VALIDATION & STATISTICS SUMMARY")
    print("=" * 70)

    for split, df in dfs.items():
        print(f"\n--- Split: {split.upper()} ---")
        print(f"Total Rows: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        print("Missing Values:")
        for col, count in df.isnull().sum().items():
            print(f"  - {col}: {count}")
        print(f"Duplicate Exact Rows: {df.duplicated().sum()}")
        print(f"Duplicate UIDs: {df['uid'].duplicated().sum()}")
        print(f"Duplicate Texts: {df['text'].duplicated().sum()}")
        print("Class Distribution:")
        val_counts = df["sentiment"].value_counts()
        for label, count in val_counts.items():
            pct = (count / len(df)) * 100
            print(f"  - {label}: {count:,} ({pct:.2f}%)")

    print("\n" + "=" * 70)
    print("FIRST 5 SAMPLE ROWS FROM EACH PROCESSED DATASET")
    print("=" * 70)

    for split, df in dfs.items():
        print(f"\n=== Top 5 Rows: {split.upper()} ===")
        for i, row in df.head(5).iterrows():
            print(f"[{i + 1}] UID: {row['uid']} | Sentiment: {row['sentiment']}")
            print(f"    Text: {row['text']}\n")


if __name__ == "__main__":
    main()
