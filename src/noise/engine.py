"""
Unified Noise Engine for NLP Robustness Experiments.

Coordinates deterministic perturbation of text corpora across:
- Typographical noise (QWERTY adjacency, deletion, insertion, transposition)
- Hinglish transliteration variation (lexical and phonetic digraph alternations)
at specified severity levels (e.g. 5%, 10%, 15%, 20%).

Ensures identical perturbed benchmarks across all models, strictly preserving
ground-truth labels, UIDs, and dataset schema.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from src.noise.typo_generator import TypoGenerator
from src.noise.transliteration import TransliterationGenerator


class NoiseEngine:
    """
    Orchestrates dataset-level perturbation with seed-isolated generators.
    """

    SUPPORTED_NOISE_TYPES = ["typo", "transliteration"]
    DEFAULT_SEVERITIES = [0.05, 0.10, 0.15, 0.20]

    def __init__(self, seed: int = 42):
        """
        Initializes the noise engine with a fixed random seed.

        Args:
            seed: Integer random seed (default: 42).
        """
        self.seed = seed
        self.typo_gen = TypoGenerator(seed=seed)
        self.translit_gen = TransliterationGenerator(seed=seed)

    def perturb_text(self, text: str, noise_type: str, severity: float) -> str:
        """
        Perturbs a single text string according to the requested noise type and severity.

        Args:
            text: Input text string.
            noise_type: 'typo' or 'transliteration'.
            severity: Perturbation rate (0.0 to 1.0).

        Returns:
            Perturbed text string.
        """
        if severity <= 0.0 or not text:
            return text

        if noise_type == "typo":
            return self.typo_gen.perturb_text(text, severity)
        elif noise_type == "transliteration":
            return self.translit_gen.perturb_text(text, severity)
        else:
            raise ValueError(
                f"Unsupported noise_type '{noise_type}'. "
                f"Supported types: {self.SUPPORTED_NOISE_TYPES}"
            )

    def perturb_dataframe(
        self,
        df: pd.DataFrame,
        noise_type: str,
        severity: float,
        text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Generates a perturbed copy of a DataFrame.
        Preserves all other columns (uid, sentiment, label) untouched.

        Args:
            df: Input DataFrame (e.g. dev_df).
            noise_type: 'typo' or 'transliteration'.
            severity: Severity level (e.g. 0.05, 0.10, 0.15, 0.20).
            text_column: Column name containing text to perturb.

        Returns:
            New DataFrame with perturbed text column.
        """
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame.")

        # Reset generator seed before each full dataset pass for reproducible ordering
        if noise_type == "typo":
            self.typo_gen.reset_seed(self.seed)
        elif noise_type == "transliteration":
            self.translit_gen.reset_seed(self.seed)
        else:
            raise ValueError(
                f"Unsupported noise_type '{noise_type}'. "
                f"Supported types: {self.SUPPORTED_NOISE_TYPES}"
            )

        perturbed_df = df.copy(deep=True)
        perturbed_df[text_column] = [
            self.perturb_text(t, noise_type, severity)
            for t in df[text_column]
        ]

        return perturbed_df

    def generate_all_perturbed_sets(
        self,
        df: pd.DataFrame,
        severities: Optional[List[float]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Dict[float, pd.DataFrame]]:
        """
        Generates perturbed DataFrames across all noise types and severities.
        Optionally persists them to disk to guarantee identical testbeds across models.

        Args:
            df: Input validation DataFrame (e.g. dev_df).
            severities: List of severity rates. Defaults to [0.05, 0.10, 0.15, 0.20].
            output_dir: Optional directory to save perturbed CSV datasets.

        Returns:
            Nested dict: {noise_type: {severity: perturbed_df}}
        """
        if severities is None:
            severities = self.DEFAULT_SEVERITIES

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict[float, pd.DataFrame]] = {}

        for n_type in self.SUPPORTED_NOISE_TYPES:
            results[n_type] = {}
            for sev in severities:
                noisy_df = self.perturb_dataframe(df, noise_type=n_type, severity=sev)
                results[n_type][sev] = noisy_df

                if output_dir:
                    sev_str = f"{int(sev * 100):02d}pct"
                    csv_path = output_dir / f"dev_{n_type}_{sev_str}.csv"
                    noisy_df.to_csv(csv_path, index=False)

        return results
