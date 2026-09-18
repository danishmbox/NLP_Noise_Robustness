"""
Exploratory Data Analysis (EDA) for SemEval-2020 Task 9 (SentiMix Hindi-English).

Analyzes:
1. Sample counts across splits (Train, Dev, Test).
2. Class distributions and percentages.
3. Word and character length statistics (min, max, mean, median, percentiles).
4. Text length distributions overall and per class.
5. Most frequent tokens overall and per sentiment class.
6. Vocabulary size and vocabulary overlap across splits.
7. Distinctive characteristics of Hindi-English code-mixed social media text.
8. Representative examples from each sentiment category.

Generates visualization figures saved to results/eda/:
- class_distribution.png
- text_length_distribution.png
- top_words_frequency.png
"""

import os
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_datasets(processed_dir: Path) -> Dict[str, pd.DataFrame]:
    """Loads processed CSV splits into DataFrames."""
    splits = {}
    for name in ["train", "dev", "test"]:
        path = processed_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required dataset: {path}")
        df = pd.read_csv(path, dtype={"uid": str, "text": str, "sentiment": str})
        df["text"] = df["text"].fillna("")
        df["word_count"] = df["text"].apply(lambda t: len(t.split()))
        df["char_count"] = df["text"].apply(len)
        splits[name] = df
    return splits


def compute_length_stats(series: pd.Series) -> Dict[str, float]:
    """Computes summary statistics for a length series."""
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
    }


def analyze_code_mixing_features(df: pd.DataFrame) -> Dict[str, float]:
    """Computes Hinglish / code-mixed specific characteristics."""
    total = len(df)
    has_mention = df["text"].str.contains(r"@\w+|@\s+\w+", regex=True).sum()
    has_hashtag = df["text"].str.contains(r"#\w+", regex=True).sum()
    has_url = df["text"].str.contains(r"https?://|https?\s+//|t\.co|tco", regex=True, case=False).sum()
    # Emojis or Unicode outside basic ASCII
    has_non_ascii = df["text"].apply(lambda t: any(ord(c) > 127 for c in t)).sum()
    # Repeated characters (elongations like 'haaaan', 'plzzzz', 'soooo')
    has_elongation = df["text"].apply(lambda t: bool(re.search(r"([a-zA-Z])\1{2,}", t))).sum()

    return {
        "mention_pct": (has_mention / total) * 100,
        "hashtag_pct": (has_hashtag / total) * 100,
        "url_pct": (has_url / total) * 100,
        "non_ascii_or_emoji_pct": (has_non_ascii / total) * 100,
        "elongation_pct": (has_elongation / total) * 100,
    }


def get_token_frequencies(texts: pd.Series, top_n: int = 25) -> List[Tuple[str, int]]:
    """Tokenizes space-separated texts and returns top frequent tokens."""
    counter = Counter()
    for text in texts:
        tokens = [t.lower() for t in text.split() if t.strip()]
        counter.update(tokens)
    return counter.most_common(top_n)


def generate_plots(splits: Dict[str, pd.DataFrame], output_dir: Path):
    """Generates and saves informative EDA visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#3498db"}

    # 1. Class Distribution Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for idx, (name, df) in enumerate(splits.items()):
        ax = axes[idx]
        counts = df["sentiment"].value_counts()[["positive", "negative", "neutral"]]
        percentages = (counts / len(df)) * 100
        bars = ax.bar(
            counts.index,
            counts.values,
            color=[palette[k] for k in counts.index],
            edgecolor="black",
            alpha=0.85
        )
        ax.set_title(f"{name.upper()} Split (N={len(df):,})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sentiment Class", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Number of Samples", fontsize=11)

        # Value and percentage labels on bars
        for bar, count, pct in zip(bars, counts.values, percentages.values):
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 60,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="semibold"
            )
        ax.set_ylim(0, max(df["sentiment"].value_counts().max() for df in splits.values()) + 700)

    plt.suptitle("Class Distribution across SentiMix Splits", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Text Length Distribution Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Word Count Histogram (Train, Dev, Test)
    ax_wc = axes[0, 0]
    for name, df in splits.items():
        sns.kdeplot(df["word_count"], ax=ax_wc, label=f"{name.capitalize()} (median: {df['word_count'].median():.0f})", linewidth=2)
    ax_wc.set_title("Word Count Distribution by Split (KDE)", fontsize=13, fontweight="bold")
    ax_wc.set_xlabel("Word Count (Tokens per Tweet)", fontsize=11)
    ax_wc.set_ylabel("Density", fontsize=11)
    ax_wc.set_xlim(0, 70)
    ax_wc.legend(frameon=True)

    # (b) Character Count Histogram (Train, Dev, Test)
    ax_cc = axes[0, 1]
    for name, df in splits.items():
        sns.kdeplot(df["char_count"], ax=ax_cc, label=f"{name.capitalize()} (median: {df['char_count'].median():.0f})", linewidth=2)
    ax_cc.set_title("Character Count Distribution by Split (KDE)", fontsize=13, fontweight="bold")
    ax_cc.set_xlabel("Character Count per Tweet", fontsize=11)
    ax_cc.set_ylabel("Density", fontsize=11)
    ax_cc.set_xlim(0, 350)
    ax_cc.legend(frameon=True)

    # (c) Word Count Boxplot by Sentiment (Combined Dataset)
    ax_box_w = axes[1, 0]
    train_df = splits["train"]
    sns.boxplot(
        data=train_df,
        x="sentiment",
        y="word_count",
        hue="sentiment",
        legend=False,
        order=["positive", "negative", "neutral"],
        palette=palette,
        ax=ax_box_w,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    ax_box_w.set_title("Training Set: Word Count by Sentiment Class", fontsize=13, fontweight="bold")
    ax_box_w.set_xlabel("Sentiment Class", fontsize=11)
    ax_box_w.set_ylabel("Word Count", fontsize=11)
    ax_box_w.set_ylim(0, 75)

    # (d) Character Count Boxplot by Sentiment
    ax_box_c = axes[1, 1]
    sns.boxplot(
        data=train_df,
        x="sentiment",
        y="char_count",
        hue="sentiment",
        legend=False,
        order=["positive", "negative", "neutral"],
        palette=palette,
        ax=ax_box_c,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    ax_box_c.set_title("Training Set: Character Count by Sentiment Class", fontsize=13, fontweight="bold")
    ax_box_c.set_xlabel("Sentiment Class", fontsize=11)
    ax_box_c.set_ylabel("Character Count", fontsize=11)
    ax_box_c.set_ylim(0, 380)

    plt.suptitle("Text Length Distribution & Boxplots", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_dir / "text_length_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Top Word Frequencies Plot
    train_df = splits["train"]
    # Filter common punctuation tokens for informative token display
    punct_set = {"@", "#", ".", ",", "!", "?", "-", ":", "...", "/", "//", "’", "'", "\"", ";", ")", "(", "“", "”", "…"}

    fig, axes = plt.subplots(1, 4, figsize=(22, 7))

    # Overall top 15 words
    overall_tokens = [t.lower() for text in train_df["text"] for t in text.split() if t.lower() not in punct_set]
    top_overall = Counter(overall_tokens).most_common(15)
    words, freqs = zip(*top_overall)
    sns.barplot(x=list(freqs), y=list(words), ax=axes[0], color="#5c6bc0", edgecolor="black")
    axes[0].set_title("Top 15 Tokens (Overall Train)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Frequency", fontsize=10)

    # Top words per sentiment class
    class_colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#3498db"}
    for idx, s_class in enumerate(["positive", "negative", "neutral"], 1):
        class_texts = train_df[train_df["sentiment"] == s_class]["text"]
        class_tokens = [t.lower() for text in class_texts for t in text.split() if t.lower() not in punct_set]
        top_cls = Counter(class_tokens).most_common(15)
        w, f = zip(*top_cls)
        sns.barplot(x=list(f), y=list(w), ax=axes[idx], color=class_colors[s_class], edgecolor="black")
        axes[idx].set_title(f"Top 15 Tokens ({s_class.capitalize()})", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Frequency", fontsize=10)

    plt.suptitle("Most Frequent Words in SentiMix Training Data", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "top_words_frequency.png", dpi=300, bbox_inches="tight")
    plt.close()


def print_eda_report(splits: Dict[str, pd.DataFrame]):
    """Prints comprehensive terminal EDA report."""
    train_df = splits["train"]
    dev_df = splits["dev"]
    test_df = splits["test"]
    all_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    print("=" * 80)
    print("      EXPLORATORY DATA ANALYSIS (EDA) REPORT - SENTIMIX HINDI-ENGLISH      ")
    print("=" * 80)

    # 1. Samples
    print("\n[1] SAMPLE COUNTS ACROSS SPLITS")
    print("-" * 80)
    print(f"  - Train Split:      {len(train_df):>6,} samples ({len(train_df) / len(all_df) * 100:.1f}%)")
    print(f"  - Dev/Val Split:    {len(dev_df):>6,} samples ({len(dev_df) / len(all_df) * 100:.1f}%)")
    print(f"  - Test Split:       {len(test_df):>6,} samples ({len(test_df) / len(all_df) * 100:.1f}%)")
    print(f"  - TOTAL CORPUS:     {len(all_df):>6,} samples (100.0%)")

    # 2 & 3. Class Distribution & Percentages
    print("\n[2 & 3] CLASS DISTRIBUTION AND PERCENTAGES")
    print("-" * 80)
    header = f"{'Split':<10} | {'Neutral':<18} | {'Positive':<18} | {'Negative':<18} | {'Total':<8}"
    print(header)
    print("-" * len(header))
    for name, df in splits.items():
        vc = df["sentiment"].value_counts()
        neu, pos, neg = vc.get("neutral", 0), vc.get("positive", 0), vc.get("negative", 0)
        tot = len(df)
        print(
            f"{name.capitalize():<10} | "
            f"{neu:,} ({neu / tot * 100:5.2f}%) | "
            f"{pos:,} ({pos / tot * 100:5.2f}%) | "
            f"{neg:,} ({neg / tot * 100:5.2f}%) | "
            f"{tot:,}"
        )

    # 4 & 5. Text Length Statistics
    print("\n[4 & 5] TEXT LENGTH STATISTICS (WORD COUNT & CHAR COUNT)")
    print("-" * 80)
    for name, df in splits.items():
        w_stats = compute_length_stats(df["word_count"])
        c_stats = compute_length_stats(df["char_count"])
        print(f"\n--- {name.upper()} SPLIT (N={len(df):,}) ---")
        print(
            f"  Word Count:   Min={w_stats['min']:.0f} | Max={w_stats['max']:.0f} | "
            f"Mean={w_stats['mean']:.2f} | Median={w_stats['median']:.0f} | "
            f"Std={w_stats['std']:.2f} | IQR=[{w_stats['p25']:.0f} - {w_stats['p75']:.0f}] | 95th={w_stats['p95']:.0f}"
        )
        print(
            f"  Char Count:   Min={c_stats['min']:.0f} | Max={c_stats['max']:.0f} | "
            f"Mean={c_stats['mean']:.2f} | Median={c_stats['median']:.0f} | "
            f"Std={c_stats['std']:.2f} | IQR=[{c_stats['p25']:.0f} - {c_stats['p75']:.0f}] | 95th={c_stats['p95']:.0f}"
        )

    print("\n  Length by Sentiment Class (Train Set):")
    for s_class in ["neutral", "positive", "negative"]:
        sub_df = train_df[train_df["sentiment"] == s_class]
        ws = compute_length_stats(sub_df["word_count"])
        cs = compute_length_stats(sub_df["char_count"])
        print(
            f"    - {s_class.capitalize():<8}: Words [Mean={ws['mean']:5.2f}, Med={ws['median']:.0f}] | "
            f"Chars [Mean={cs['mean']:6.2f}, Med={cs['median']:.0f}]"
        )

    # 6 & 7. Unique Tokens and Vocabulary Size
    print("\n[6 & 7] VOCABULARY SIZE & MOST FREQUENT TOKENS")
    print("-" * 80)
    train_tokens = [t.lower() for text in train_df["text"] for t in text.split()]
    dev_tokens = [t.lower() for text in dev_df["text"] for t in text.split()]
    test_tokens = [t.lower() for text in test_df["text"] for t in text.split()]

    train_vocab = set(train_tokens)
    dev_vocab = set(dev_tokens)
    test_vocab = set(test_tokens)
    total_vocab = train_vocab | dev_vocab | test_vocab

    print(f"  - Total Tokens in Train:      {len(train_tokens):,}")
    print(f"  - Unique Vocabulary (Train):  {len(train_vocab):,} unique words/tokens")
    print(f"  - Unique Vocabulary (Dev):    {len(dev_vocab):,} unique words/tokens")
    print(f"  - Unique Vocabulary (Test):   {len(test_vocab):,} unique words/tokens")
    print(f"  - Total Corpus Vocabulary:    {len(total_vocab):,} unique words/tokens")

    dev_oov = len(dev_vocab - train_vocab)
    test_oov = len(test_vocab - train_vocab)
    print(f"  - Dev Out-Of-Vocabulary:      {dev_oov:,} words ({dev_oov / len(dev_vocab) * 100:.2f}% of dev vocab)")
    print(f"  - Test Out-Of-Vocabulary:     {test_oov:,} words ({test_oov / len(test_vocab) * 100:.2f}% of test vocab)")

    punct_set = {"@", "#", ".", ",", "!", "?", "-", ":", "...", "/", "//", "’", "'", "\"", ";", ")", "(", "“", "”", "…"}
    content_tokens = [t for t in train_tokens if t not in punct_set]
    top_20 = Counter(content_tokens).most_common(20)
    print("\n  Top 20 Most Frequent Words in Training Set:")
    top_str = ", ".join([f"{w} ({c:,})" for w, c in top_20])
    print(f"    {top_str}")

    # 8. Basic Characteristics of Code-Mixed Text
    print("\n[8] BASIC CHARACTERISTICS OF HINDI-ENGLISH CODE-MIXED TEXT")
    print("-" * 80)
    cm_stats = analyze_code_mixing_features(train_df)
    print(f"  - Tweets containing User Mentions (@):            {cm_stats['mention_pct']:5.2f}%")
    print(f"  - Tweets containing Hashtags (#):                 {cm_stats['hashtag_pct']:5.2f}%")
    print(f"  - Tweets containing URLs / Links:                 {cm_stats['url_pct']:5.2f}%")
    print(f"  - Tweets containing Emojis or Non-ASCII symbols:  {cm_stats['non_ascii_or_emoji_pct']:5.2f}%")
    print(f"  - Tweets containing Word Elongation (e.g. 'haa'): {cm_stats['elongation_pct']:5.2f}%")
    print("\n  Key Linguistic Observations:")
    print("    1. Intricate Code-Mixing: Sentences frequently interleave English lexical items with")
    print("       Romanized Hindi function words (e.g., 'hai', 'ki', 'ke', 'bhi', 'ka', 'mein', 'aur', 'ko').")
    print("    2. High Orthographic & Transliteration Variance: The same Hindi word is spelled in multiple ways")
    print("       (e.g., 'nahi' / 'nhi' / 'nahin', 'bhi' / 'bi', 'hoga' / 'hga', 'kya' / 'kyaa').")
    print("    3. Informal Social Shorthands: Heavy use of acronyms and abbreviations (e.g., 'rt', 'u', 'pls', 'bjp').")
    print("    4. Substantial OOV Rate: ~43.7% of test unique tokens are unseen in train due to user handles, slang, and typos.")

    # 9. Representative Examples
    print("\n[9] REPRESENTATIVE EXAMPLES FROM EACH SENTIMENT CLASS")
    print("-" * 80)
    for s_class in ["positive", "negative", "neutral"]:
        print(f"\n>>> Class: {s_class.upper()} (Train Split) <<<")
        samples = train_df[train_df["sentiment"] == s_class].head(3)
        for i, (_, row) in enumerate(samples.iterrows(), 1):
            print(f"  ({i}) UID: {row['uid']}")
            print(f"      Text: {row['text']}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "results" / "eda"

    splits = load_datasets(processed_dir)

    print("Generating EDA visualization plots...")
    generate_plots(splits, results_dir)
    print(f"Plots saved to: {results_dir}")

    print_eda_report(splits)


if __name__ == "__main__":
    main()
