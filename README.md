# BBT: Multivariate Time Series Transformer Model for fMRI Biomarker Discovery


A minimal, modular, and interpretable foundation model for fMRI-based brain disorder classification using a multivariate time series model with rotary embeddings and grouped-query attention.

Trained on 414-region Glasser+Tian parcellated time series, BBTransformer enables transfer learning, fine-tuning, and biomarker discovery across ADHD, ASD, UK Biobank ICD conditions, and more.

<br>




<img width="1408" height="768" alt="ukbb_diagram" src="https://github.com/user-attachments/assets/f343fb1e-9a25-49c6-ae17-2f8f0cb226df" />



<br>

<br>

## Model Architecture

BBTransformer processes **150 × 414** fMRI time series through three integrated streams [2]:

1.  **Primary Temporal Stream**:
    -   7 transformer layers
    -   512-dim embeddings, 16 query heads, 4 KV heads (GQA)
    -   Rotary position embeddings (RoPE), RMSNorm, SwiGLU
2.  **Local Patch Stream**:
    -   Patch size 3 with cross-attention fusion
    -   Captures short-scale temporal dynamics
3.  **Temporal Attention Pooling**:
    -   Learned weighting of timepoints for diagnostic decisions
    -   Integrates confounder embeddings (Age, Sex)

Final output: Single probability via sigmoid for binary classification.

<br>

## Quick Start (Recommended)

The simplest way to run a full analysis pipeline (training + evaluation + biomarker discovery) is via `run_analysis`:

```python
from bbtransformer import run_analysis

results = run_analysis(
    target_column="ADHD",                    # Binary column in CSV
    data_path="dataset/fmri_ADHD.npz",       # fMRI features (.npz)
    pheno_path="dataset/pheno_ADHD.csv",     # Phenotype data (.csv)
    use_pretrained=False,                    # Set True for transfer learning
    compute_importance=True,                 # Rank brain regions
    save_plots=True                          # Auto-save evaluation figures
)

print(f"F1 Score: {results['metrics']['f1']:.4f}")
print(f"Top ROI:  {results['importance_scores'].argmax()}")
```

> ⚠️ **Critical Data Requirement**: Your phenotype CSV **must** contain columns matching `[target_column]`, `Age`, and `Sex`. Missing columns will raise a `ValueError` during validation.

### Output Files
When `save_plots=True` and `save_json=True`, the following are generated in `results/`:
-   `{target}_results.png`: ROC Curve, Precision-Recall, Confusion Matrix, Calibration Curve
-   `{target}_results.json`: Metrics summary + top 10 important ROIs with anatomical names
-   `importance_{target}.csv`: Ranked list of all 414 brain regions

<br>

## Advanced: Component-Level API

For custom workflows, all components are individually accessible:

```python
import bbtransformer as bbt

# 1. Load data with stratified splits & age normalization
train_loader, val_loader, test_loader, metadata = bbt.prepare_fmri_data(
    data_path='dataset/fmri_ADHD.npz',
    pheno_path='dataset/pheno_ADHD.csv',
    target_column='ADHD'
)

# 2. Create model with custom architecture
model = bbt.create_bbtransformer({
    'feature_dim': 414,
    'embed_dim': 256,      # Override default 512
    'num_heads': 8,        # Override default 16
    'num_layers': 4        # Override default 7
})

# 3. Train with Ranger21 + early stopping
trained_model = bbt.train_model(
    model, train_loader, val_loader,
    early_stop_metric="f1",
    use_focal_loss=True   # For imbalanced datasets
)

# 4. Single-subject diagnosis with confidence
predictor = bbt.Diagnostic(trained_model, metadata)
result = predictor.predict_single(fmri, age=56.3, ext=1)
print(result['interpretation'])
```

<br>

## Key Features

### Architecture
-   **Rotary Positional Embeddings** for temporal awareness
-   **Grouped-Query Attention (GQA)** for memory efficiency
-   **Multi-scale fusion** via patch embedding + cross-attention
-   **Stochastic Depth** (DropPath) for regularization

### Training
-   **Ranger21 optimizer** with internal LR scheduling
-   **Adaptive Focal Loss** for class imbalance
-   **Early stopping** on F1 or validation loss
-   **Mixed precision** training via `torch.amp`

### Interpretation
-   **Permutation importance** for brain region ranking
-   **Single-subject diagnosis** with confidence scoring (`Diagnostic` class)
-   **Attention visualization** (when `return_attn_weights=True`)

<br>

## Installation

```bash
pip install git+https://github.com/cimt-unia/BBTransformer.git
```



<br>


### Datasets

| Dataset        | Conditions                              | Atlas              | Preprocessing      | Role               |
| :------------- | :-------------------------------------- | :----------------- | :----------------- | :----------------- |
| **ABIDE**      | ASD vs Controls                         | Glasser+Tian (414) | C-PAC              | Foundation Model   |
| **UK Biobank** | ICD F32, G20, G40, etc. (10 conditions) | Glasser+Tian (414) | Official UKB Pipeline | Transfer Learning  |
| **ADHD-200**   | ADHD vs Controls                        | Glasser+Tian (414) | Athena (AFNI/FSL)  | External Validation |
| **UCLA LA5c**  | Schizophrenia, Bipolar, ADHD vs Controls | Glasser+Tian (414) | fMRIPrep v0.4.4    | External Validation |

> ⚠️ **Note**: All datasets are standardized to **150 timepoints @ 2.0s TR** across 414 ROIs via cubic spline interpolation and per-subject z-scoring before model input

<br>

## ⚙️ Default Configuration

Best hyperparameters from clinically validated Optuna tuning [2]:

```python
DEFAULT_HP = {
    'feature_dim': 414,
    'embed_dim': 512,
    'num_heads': 16,
    'num_layers': 7,
    'n_kv_heads': 4,
    'dropout_input': 0.17,
    'dropout_classifier': 0.04,
    'patch_size': 3,
    'stochastic_depth_rate': 0.10,
    'embed_dim_age': 32,
    'embed_dim_ext': 16,
}
```

<br>

## 📁 Project Structure

```
BBT/
├── bbtransformer/          # Core package (installable)
│   ├── __init__.py         # Public API exports
│   ├── model.py            # BBTransformer, create_bbtransformer
│   ├── utils.py            # Weight I/O, ROI metadata loading
│   └── trainer/            # exe, train, eval, rank, pred, tune, loader
├── notebooks/              # Analysis & tutorials
├── preprocessing/          # Dataset-specific curation
└── weights/                # Pretrained & fine-tuned model weights
```

