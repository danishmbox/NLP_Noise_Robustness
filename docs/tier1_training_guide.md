# Tier 1 Model Training & Validation Guide
**Project:** NLP Noise Robustness in Hindi-English Code-Mixed Sentiment Analysis  
**Repository:** [https://github.com/danishmbox/NLP_Noise_Robustness.git](https://github.com/danishmbox/NLP_Noise_Robustness.git)  
**Stage:** Stage 1 — Tier 1 Model Training Implementation

---

## 1. Project Organization & Modules

The Tier 1 training pipeline is modularized into reusable components adhering to strict separation of concerns:

```text
NLP_Noise_Robustness/
├── configs/
│   ├── classical_baseline.json     # Hyperparameters for Model A (TF-IDF + LinearSVC)
│   ├── muril.json                  # Hyperparameters for Model B (MuRIL)
│   ├── xlm_roberta.json            # Hyperparameters for Model C (XLM-RoBERTa)
│   └── mbert.json                  # Hyperparameters for Model D (mBERT)
├── data/
│   ├── raw/                        # Original CoNLL datasets
│   └── processed/                  # train.csv, dev.csv, test.csv
├── docs/
│   └── tier1_training_guide.md     # This comprehensive execution guide
├── notebooks/
│   ├── 01_EDA.ipynb                # Stage 0 Exploratory Data Analysis
│   └── 02_Train_Tier1_Transformers_Colab.ipynb # Google Colab GPU Training Notebook
├── reports/
│   └── model_feasibility_audit.md  # Stage 0 Feasibility Audit Report
├── results/
│   ├── models/                     # Serialized model pipelines (.joblib) & checkpoints
│   └── validation/                 # Structured dev metrics (JSON/CSV), logs, & plots
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py               # Dataset validation, schema checks, test guard, & label mappings
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              # Macro-F1, accuracy, per-class F1, confusion matrices, JSON/CSV export
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical_baseline.py   # Model A implementation (FeatureUnion + LinearSVC)
│   │   └── transformer_classifier.py # Models B, C, D implementation (MuRIL, XLM-R, mBERT)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── reproducibility.py      # Deterministic seed setting, metadata capture, caveat documentation
│   ├── eda.py
│   ├── preprocess.py
│   └── train.py                    # Unified CLI entry point for independent model runs
```

---

## 2. Shared Label Mapping & Evaluation Protocol

### 2.1 Explicit Shared Label Mapping
All models strictly use this canonical mapping defined in `src/data/loader.py`:
- `0`: `"negative"`
- `1`: `"neutral"`
- `2`: `"positive"`

### 2.2 Primary Evaluation Metric
- **Macro-F1 (Unweighted mean of per-class F1 scores):**
  $$\text{Macro-F1} = \frac{F1_{\text{negative}} + F1_{\text{neutral}} + F1_{\text{positive}}}{3}$$
- Additional reported metrics: Accuracy, Weighted-F1, Per-class Precision / Recall / F1, and Confusion Matrices.

### 2.3 Strict Split Guarding (Test Set Protection)
- **Train Split (`train.csv`):** 14,000 samples — Used strictly for fitting models.
- **Validation Split (`dev.csv`):** 3,000 samples — Used strictly for checkpoint selection, validation logging, and early stopping.
- **Test Split (`test.csv`):** 3,000 samples — **Strictly guarded against in Stage 1**. Calling or loading the test set during this stage is blocked programmatically by `guard_against_test_set_usage()` in `src/data/loader.py`.

---

## 3. Environment & Dependency Specifications

### 3.1 Local CPU Environment (Model A — Classical Baseline)
Model A executes fully in the existing local environment in under **1 minute**:
- `scikit-learn >= 1.3.0`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `joblib >= 1.3.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`

### 3.2 Google Colab GPU Environment (Models B, C, D — Transformers)
Transformers require PyTorch and Hugging Face acceleration. Execute them via Google Colab with a free Tesla T4 GPU:
- `torch >= 2.0.0`
- `transformers >= 4.35.0`
- `accelerate >= 0.24.0`
- `datasets >= 2.14.0`
- `evaluate >= 0.4.0`

---

## 4. Execution Commands (Run Each Model Independently)

Every pipeline is completely decoupled. Running one model will **never** trigger the training of another.

### 4.1 Running Model A: Classical Baseline (Local CPU)
```bash
# Using CLI model flag (both "classical" and "classical_baseline" supported)
python src/train.py --model classical
# (or python src/train.py --model classical_baseline)

# Or using the config file directly
python src/train.py --config configs/classical_baseline.json
```
**Artifacts Generated:**
- Model artifact: `results/models/classical_baseline_pipeline.joblib`
- Metrics JSON: `results/validation/classical_baseline_val_metrics.json`
- Metrics CSV: `results/validation/classical_baseline_val_summary.csv`
- Confusion matrix: `results/validation/classical_baseline_confusion_matrix.png`
- Metadata: `results/validation/classical_baseline_metadata.json`

---

### 4.2 Running Model B: MuRIL (`google/muril-base-cased`) (GPU / Colab)
```bash
python src/train.py --model muril

# Optional overrides:
python src/train.py --model muril --epochs 3 --batch_size 16 --lr 2e-5 --max_length 128
```
**Artifacts Generated:**
- Best checkpoint: `results/models/muril_best_checkpoint/` (model weights + tokenizer)
- Metrics JSON: `results/validation/muril_val_metrics.json`
- Metrics CSV: `results/validation/muril_val_summary.csv`
- Epoch history: `results/validation/muril_epoch_history.json`
- Confusion matrix: `results/validation/muril_confusion_matrix.png`
- Metadata: `results/validation/muril_metadata.json`

---

### 4.3 Running Model C: XLM-RoBERTa (`xlm-roberta-base`) (GPU / Colab)
```bash
python src/train.py --model xlm_roberta

# Optional overrides:
python src/train.py --model xlm_roberta --epochs 3 --batch_size 16 --lr 2e-5
```
**Artifacts Generated:**
- Best checkpoint: `results/models/xlm_roberta_best_checkpoint/`
- Metrics JSON: `results/validation/xlm_roberta_val_metrics.json`
- Metrics CSV: `results/validation/xlm_roberta_val_summary.csv`
- Epoch history: `results/validation/xlm_roberta_epoch_history.json`
- Confusion matrix: `results/validation/xlm_roberta_confusion_matrix.png`

---

### 4.4 Running Model D: mBERT (`bert-base-multilingual-cased`) (GPU / Colab)
```bash
python src/train.py --model mbert

# Optional overrides:
python src/train.py --model mbert --epochs 3 --batch_size 16 --lr 2e-5
```
**Artifacts Generated:**
- Best checkpoint: `results/models/mbert_best_checkpoint/`
- Metrics JSON: `results/validation/mbert_val_metrics.json`
- Metrics CSV: `results/validation/mbert_val_summary.csv`
- Epoch history: `results/validation/mbert_epoch_history.json`
- Confusion matrix: `results/validation/mbert_confusion_matrix.png`

---

## 5. Google Colab GPU Workflow

To train the transformer models using free GPU acceleration:

1. **Open Google Colab:** Go to [https://colab.research.google.com](https://colab.research.google.com) and set the runtime to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
2. **Open the Notebook:** Upload [notebooks/02_Train_Tier1_Transformers_Colab.ipynb](file:///C:/Users/Danish%20ahmad/.gemini/antigravity-ide/scratch/NLP_Noise_Robustness/notebooks/02_Train_Tier1_Transformers_Colab.ipynb) or clone the repository directly:
   ```bash
   !git clone https://github.com/danishmbox/NLP_Noise_Robustness.git
   %cd NLP_Noise_Robustness
   ```
3. **Install Dependencies:**
   ```bash
   !pip install -q transformers accelerate datasets evaluate
   ```
4. **Execute Individual Models:** Run cells or terminal commands:
   ```bash
   !python src/train.py --model muril
   !python src/train.py --model xlm_roberta
   !python src/train.py --model mbert
   ```
5. **Download / Sync Artifacts:** Copy the `results/` folder back to your local workspace or Google Drive.

---

## 6. Checkpoint Resumption & OOM Troubleshooting

### 6.1 Resuming from a Saved Checkpoint
To resume or load a trained transformer checkpoint for feature extraction or error analysis:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint_path = "results/models/muril_best_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
```

### 6.2 GPU Out-of-Memory (OOM) Remediation
If an Out-of-Memory error occurs on smaller GPUs:
1. **Reduce batch size:** Decrease `--batch_size` from `16` to `8` or `4`.
2. **Increase gradient accumulation:** Edit the corresponding config (e.g. `configs/muril.json`) to set `"gradient_accumulation_steps": 2` or `4`.
3. **Verify FP16:** Confirm `"fp16": true` in the configuration.
