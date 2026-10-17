"""
Unit test suite for Stage 2 Noise Robustness module.

Verifies:
1. Typographical noise generation (QWERTY adjacency, deletion, insertion, transposition).
2. Hinglish transliteration generation (lexicon variation, phonetic digraphs, casing).
3. Deterministic reproducibility across seeds.
4. Schema and ground-truth label preservation in perturbed datasets.
5. Strict test set protection (guard_against_test_set_usage).
6. Classical baseline pipeline loading and prediction sanity.
7. Stage 2B: Label mapping validation from model configs (int and string keys).
8. Stage 2B: Shared noisy datasets row alignment and UID validation.
9. Stage 2B: Checkpoint resolution failure handling (no guessing).
10. Stage 2B: Strict 3-file aggregation validation.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import joblib

from src.noise.typo_generator import TypoGenerator
from src.noise.transliteration import TransliterationGenerator
from src.noise.engine import NoiseEngine
from src.data.loader import (
    guard_against_test_set_usage,
    load_dev_data,
    ORDERED_LABELS,
    LABEL2ID
)
from src.evaluate_robustness import (
    validate_model_label_mapping,
    resolve_checkpoint_path,
    load_and_validate_noisy_datasets,
    aggregate_tier1_robustness,
    CANONICAL_ID2LABEL
)


class TestNoiseRobustness(unittest.TestCase):
    """
    Test suite for noise generation, reproducibility, and robustness pipeline guards.
    """

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.sample_texts = [
            "ye movie bahut achi hai aur actors ne kamaal kiya @director",
            "bakwaas film thi time waste ho gaya bilkul http://example.com/review",
            "theek thaak average movie hai kuch khaas nahi"
        ]
        self.sample_df = pd.DataFrame({
            "uid": ["101", "102", "103"],
            "text": self.sample_texts,
            "sentiment": ["positive", "negative", "neutral"],
            "label": [2, 0, 1]
        })

    def test_typo_generator_severity_zero(self):
        """Severity 0.0 should leave text completely unchanged."""
        gen = TypoGenerator(seed=42)
        for text in self.sample_texts:
            self.assertEqual(gen.perturb_text(text, severity=0.0), text)

    def test_typo_generator_perturbation_and_preservation(self):
        """Typographical noise should perturb letters but preserve @mentions and URLs."""
        gen = TypoGenerator(seed=42)
        text = "dekho bhai kamaal ki acting thi @director http://review.in"
        perturbed = gen.perturb_text(text, severity=0.20)
        self.assertNotEqual(text, perturbed)
        # @mentions and URLs should remain intact
        self.assertIn("@director", perturbed)
        self.assertIn("http://review.in", perturbed)

    def test_transliteration_generator_severity_zero(self):
        """Severity 0.0 should leave text completely unchanged."""
        gen = TransliterationGenerator(seed=42)
        for text in self.sample_texts:
            self.assertEqual(gen.perturb_text(text, severity=0.0), text)

    def test_transliteration_lexicon_and_case(self):
        """Transliteration should perturb known Hinglish tokens and preserve casing."""
        gen = TransliterationGenerator(seed=42)
        text = "NAHI yeh theek nahi hai dost"
        perturbed = gen.perturb_text(text, severity=1.0)
        self.assertNotEqual(text, perturbed)
        tokens = perturbed.split()
        first_token = tokens[0]
        self.assertTrue(first_token.isupper(), f"Expected uppercase token, got {first_token}")

    def test_reproducibility_identical_seeds(self):
        """Identical seeds must produce byte-for-byte identical output."""
        engine1 = NoiseEngine(seed=42)
        engine2 = NoiseEngine(seed=42)

        df1 = engine1.perturb_dataframe(self.sample_df, noise_type="typo", severity=0.15)
        df2 = engine2.perturb_dataframe(self.sample_df, noise_type="typo", severity=0.15)
        pd.testing.assert_frame_equal(df1, df2)

        df3 = engine1.perturb_dataframe(self.sample_df, noise_type="transliteration", severity=0.15)
        df4 = engine2.perturb_dataframe(self.sample_df, noise_type="transliteration", severity=0.15)
        pd.testing.assert_frame_equal(df3, df4)

    def test_reproducibility_different_seeds(self):
        """Different seeds should produce divergent outputs."""
        engine1 = NoiseEngine(seed=42)
        engine2 = NoiseEngine(seed=999)

        text = "ye movie bahut achi hai aur actors ne kamaal kiya"
        pert1 = engine1.perturb_text(text, noise_type="typo", severity=0.20)
        pert2 = engine2.perturb_text(text, noise_type="typo", severity=0.20)
        self.assertNotEqual(pert1, pert2)

    def test_label_and_schema_preservation(self):
        """Perturbing a DataFrame must never alter uids, sentiments, labels, or row order."""
        engine = NoiseEngine(seed=42)
        for n_type in ["typo", "transliteration"]:
            for sev in [0.05, 0.10, 0.15, 0.20]:
                noisy_df = engine.perturb_dataframe(self.sample_df, noise_type=n_type, severity=sev)
                self.assertEqual(len(noisy_df), len(self.sample_df))
                self.assertListEqual(list(noisy_df.columns), list(self.sample_df.columns))
                pd.testing.assert_series_equal(noisy_df["uid"], self.sample_df["uid"])
                pd.testing.assert_series_equal(noisy_df["sentiment"], self.sample_df["sentiment"])
                pd.testing.assert_series_equal(noisy_df["label"], self.sample_df["label"])

    def test_test_set_guard_enforcement(self):
        """Accessing or requesting 'test' split must raise PermissionError."""
        with self.assertRaises(PermissionError):
            guard_against_test_set_usage("test")

        with self.assertRaises(PermissionError):
            guard_against_test_set_usage("test.csv")

        with self.assertRaises(PermissionError):
            guard_against_test_set_usage("held_out_test")

        try:
            guard_against_test_set_usage("dev")
            guard_against_test_set_usage("train")
            guard_against_test_set_usage("validation")
        except PermissionError:
            self.fail("guard_against_test_set_usage raised PermissionError unexpectedly for valid dev/train splits.")

    def test_classical_pipeline_artifact_loading_and_prediction(self):
        """The trained classical pipeline artifact must load and predict valid labels."""
        model_path = self.project_root / "results" / "models" / "classical_baseline_pipeline.joblib"
        self.assertTrue(model_path.exists(), f"Model artifact not found at {model_path}")

        pipeline = joblib.load(model_path)
        self.assertTrue(hasattr(pipeline, "predict"))

        preds = pipeline.predict(self.sample_texts)
        self.assertEqual(len(preds), len(self.sample_texts))
        for p in preds:
            self.assertIn(p, ORDERED_LABELS)

    # --- Stage 2B Tests ---

    def test_validate_model_label_mapping_valid_int_keys(self):
        """Valid integer-keyed id2label must match CANONICAL_ID2LABEL."""
        valid_int_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        res = validate_model_label_mapping(valid_int_mapping)
        self.assertEqual(res, CANONICAL_ID2LABEL)

    def test_validate_model_label_mapping_valid_str_keys(self):
        """Valid JSON string-keyed id2label must be correctly parsed and normalized."""
        valid_str_mapping = {"0": "negative", "1": "neutral", "2": "positive"}
        res = validate_model_label_mapping(valid_str_mapping)
        self.assertEqual(res, CANONICAL_ID2LABEL)

        # Whitespace and upper case tolerance
        messy_str_mapping = {"0": " Negative ", "1": " NEUTRAL", "2": "positive "}
        res2 = validate_model_label_mapping(messy_str_mapping)
        self.assertEqual(res2, CANONICAL_ID2LABEL)

    def test_validate_model_label_mapping_invalid_mapping(self):
        """Mismatched labels or wrong ordering must raise ValueError."""
        # Wrong order
        reversed_mapping = {"0": "positive", "1": "neutral", "2": "negative"}
        with self.assertRaises(ValueError):
            validate_model_label_mapping(reversed_mapping)

        # Missing class
        missing_class = {"0": "negative", "1": "neutral"}
        with self.assertRaises(ValueError):
            validate_model_label_mapping(missing_class)

        # Invalid label name
        wrong_name = {"0": "neg", "1": "neu", "2": "pos"}
        with self.assertRaises(ValueError):
            validate_model_label_mapping(wrong_name)

    def test_load_and_validate_noisy_datasets_existing_all_pass(self):
        """All 8 shared pre-generated noisy datasets in results/robustness/datasets must pass validation."""
        data_dir = self.project_root / "data" / "processed"
        datasets_dir = self.project_root / "results" / "robustness" / "datasets"

        clean_dev = load_dev_data(data_dir)
        noisy_dict = load_and_validate_noisy_datasets(clean_dev, datasets_dir, severities=[0.05, 0.10, 0.15, 0.20])

        self.assertIn("typo", noisy_dict)
        self.assertIn("transliteration", noisy_dict)
        for ntype in ["typo", "transliteration"]:
            for sev in [0.05, 0.10, 0.15, 0.20]:
                self.assertIn(sev, noisy_dict[ntype])
                self.assertEqual(len(noisy_dict[ntype][sev]), len(clean_dev))

    def test_load_and_validate_noisy_datasets_detects_mismatch(self):
        """Validation must detect row count or UID tampering in a noisy dataset."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create a tampered noisy dataset with a missing row
            tampered_df = self.sample_df.iloc[:2].copy()
            tampered_df.to_csv(temp_dir / "dev_typo_05pct.csv", index=False)

            with self.assertRaises((ValueError, FileNotFoundError)):
                load_and_validate_noisy_datasets(self.sample_df, temp_dir, severities=[0.05])
        finally:
            shutil.rmtree(temp_dir)

    def test_resolve_checkpoint_path_missing_raises_error(self):
        """Resolving a non-existent checkpoint path must raise FileNotFoundError with search info."""
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_checkpoint_path(
                model_slug="nonexistent_model",
                explicit_path="/invalid/path/that/does/not/exist"
            )
        self.assertIn("Could not locate a valid checkpoint directory", str(ctx.exception))

    def test_aggregate_tier1_robustness_missing_file_raises_error(self):
        """Aggregator must raise FileNotFoundError if any of the 3 required summary files is missing."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Only put one of the required files
            (temp_dir / "classical_baseline_robustness_summary.csv").write_text("model_name\nclassical\n")

            with self.assertRaises(FileNotFoundError) as ctx:
                aggregate_tier1_robustness(robustness_dir=temp_dir)
            self.assertIn("Missing required model summary files", str(ctx.exception))
            self.assertIn("muril_robustness_summary.csv", str(ctx.exception))
            self.assertIn("xlm_roberta_robustness_summary.csv", str(ctx.exception))
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
