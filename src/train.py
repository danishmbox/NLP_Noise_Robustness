"""
Unified Training Runner for Tier 1 Model Pipelines.

Supports independent, decoupled execution of:
- Model A: classical (Word+Char TF-IDF + LinearSVC)
- Model B: muril (google/muril-base-cased)
- Model C: xlm_roberta (xlm-roberta-base)
- Model D: mbert (bert-base-multilingual-cased)

Usage Examples:
    python src/train.py --model classical
    python src/train.py --model muril
    python src/train.py --model xlm_roberta
    python src/train.py --model mbert
    python src/train.py --config configs/classical_baseline.json
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure utf-8 encoding on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path for direct execution (e.g. python src/train.py)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_train_and_dev_data
from src.models.classical_baseline import train_classical_baseline
from src.models.transformer_classifier import train_transformer_classifier


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NLP Noise Robustness: Tier 1 Model Training Pipeline Runner"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["classical", "classical_baseline", "muril", "xlm_roberta", "mbert"],
        help="Select which Tier 1 model to train independently."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to an optional JSON configuration file overriding defaults."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing train.csv and dev.csv (default: data/processed)."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="results/models",
        help="Directory to save model checkpoints and artifacts."
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="results/validation",
        help="Directory to save validation metrics and plots."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed across all libraries."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs (transformers)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size (transformers)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (transformers)."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Override maximum sequence length (transformers)."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.model and not args.config:
        print("Error: You must specify either --model [classical|muril|xlm_roberta|mbert] or --config <path>.")
        sys.exit(1)

    # Load custom config if provided
    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    # CLI arguments override config
    model_choice = args.model or config.get("model_name")
    if args.seed is not None:
        config["random_seed"] = args.seed
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.max_length is not None:
        config["max_length"] = args.max_length

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    val_dir = Path(args.val_dir)

    print("Loading verified train and dev splits...")
    train_df, dev_df = load_train_and_dev_data(data_dir)
    print(f"Loaded train: {len(train_df):,} samples | dev: {len(dev_df):,} samples")

    if model_choice in ["classical", "classical_baseline"]:
        train_classical_baseline(
            train_df=train_df,
            dev_df=dev_df,
            config=config,
            models_dir=models_dir,
            val_dir=val_dir
        )
    elif model_choice in ["muril", "xlm_roberta", "mbert"]:
        train_transformer_classifier(
            model_name=model_choice,
            train_df=train_df,
            dev_df=dev_df,
            config=config,
            models_dir=models_dir,
            val_dir=val_dir
        )
    else:
        print(f"Error: Unknown model choice '{model_choice}'. Choose from [classical, muril, xlm_roberta, mbert].")
        sys.exit(1)


if __name__ == "__main__":
    main()
