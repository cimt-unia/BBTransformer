# BBTransformer: In-Depth Fine-Tuning Tutorial

This guide provides a comprehensive walkthrough for fine-tuning a pretrained BBTransformer model on custom clinical cohorts using the `run_analysis` pipeline.

## 1. The Complete Fine-Tuning Script

Save this as `run_analysis.py`. This script uses robust path resolution and is configured for transfer learning [1].

```python
import logging
from pathlib import Path
from bbtransformer.trainer.exe import run_analysis

# Enable detailed progress logs
logging.basicConfig(level=logging.INFO)

# --- ROBUST PATH DEFINITION ---
# Use __file__ to ensure paths resolve relative to THIS script, not your terminal location
BASE_DIR = Path(__file__).parent / "data" 
WEIGHTS_DIR = Path(__file__).parent / "weights"

DATA_PATH = BASE_DIR / "fmri_Depression_vs_SubstanceUse.npz"
PHENO_PATH = BASE_DIR / "pheno_Depression_vs_SubstanceUse.csv"
PRE_WEIGHTS_NAME = "neuroX.pth"  # ✅ Filename only (not full path)

# --- RUN THE FULL PIPELINE ---
results = run_analysis(
    # 1. DATA IDENTIFICATION
    target_column="target_label",      # Binary column (0/1) in CSV [2]
    data_path=str(DATA_PATH),          # fMRI features (.npz) [1]
    pheno_path=str(PHENO_PATH),        # Phenotype + Age + Sex (.csv) [1]

    # 2. PRETRAINED WEIGHTS (TRANSFER LEARNING)
    project_root=str(Path(__file__).parent),  # Base dir for ./weights/ and ./results/
    use_pretrained=True,               # Enable weight loading before training
    pretrained_weight_file=PRE_WEIGHTS_NAME, # Filename inside project_root/weights/ [4]

    # 3. FINE-TUNING HYPERPARAMETERS
    training_config={
        "epochs": 500,                  # Reduced from default 5000 for fine-tuning [5]
        "lr": 1e-5,                     # Lower LR prevents catastrophic forgetting [5]
        "weight_decay": 1.14e-06,       # L2 regularization strength
        "patience": 50,                 # Early stopping window [5]
    },
    early_stop_metric="loss",          # Monitor "loss" (recommended for FT) or "f1" [5]
    use_focal_loss=False,              # Set True if class imbalance > 70/30 [5]

    # 4. BIOMARKER DISCOVERY (PERMUTATION IMPORTANCE)
    compute_importance=True,           # Rank all 414 brain regions [7]
    importance_metric="loss",          # "loss" is more stable than "f1" for ranking [7]
    importance_n_repeats=30,           # 30=research-grade; 5=quick sanity check [7]

    # 5. OUTPUT & REPRODUCIBILITY
    save_plots=True,                   # 6-panel evaluation figure [9]
    save_json=True,                    # Metrics + top ROIs summary [1]
    random_seed=42                     # Locks PyTorch/NumPy/CUDA seeds [8]
)

print(f"✅ Fine-tuning complete! F1: {results['metrics']['f1']:.4f}")
```

## 2. Parameter Deep Dive

### Data Requirements (Strict Validation)
The `_validate_inputs` function enforces these rules before training begins [2]:
-   **CSV Columns:** Must contain exactly `[target_column]`, `Age`, and `Sex`. Case-sensitive.
-   **NPZ Arrays:** Must contain `'data'` (shape: `subjects × timepoints × 414`) and `'subject_ids'`.
-   **Subject Alignment:** IDs in NPZ must exist in CSV index/column. Mismatches raise `ValueError` [3].

### Weight Loading Mechanics
Understanding how `pretrained_weight_file` is resolved prevents the most common failure mode [4]:

| Scenario | What to Pass | Internal Resolution |
| :--- | :--- | :--- |
| Weights in `project_root/weights/` | `"neuroX.pth"` | `project_root/weights/neuroX.pth` ✅ |
| Weights elsewhere (absolute) | `"/mnt/data/neuroX.pth"` | Used directly via `os.path.isabs()` ✅ |
| ❌ Relative path with subdir | `"./weights/neuroX.pth"` | `project_root/weights/./weights/neuroX.pth` ❌ |

> ⚠️ **Critical:** When weights are in the default `project_root/weights/` folder, pass **only the filename**. Passing `"./weights/neuroX.pth"` causes double-nesting and `FileNotFoundError` [1][4].

### Training Configuration for Fine-Tuning vs. From-Scratch

| Parameter | Fine-Tuning | From Scratch | Rationale |
| :--- | :--- | :--- | :--- |
| `epochs` | 300–800 | 5000 | Pretrained weights need fewer cycles to adapt [5] |
| `lr` | 1e-5 – 5e-6 | 2.3e-5 | Lower LR preserves pretrained representations [5] |
| `patience` | 30–50 | 90 | Convergence is faster; stop earlier to avoid overfitting [5] |
| `early_stop_metric` | `"loss"` | `"f1"` | Loss is smoother during FT; F1 can be noisy early on [5] |
| `use_focal_loss` | Only if imbalanced | Default False | Adaptive focal loss helps when prevalence <30% or >70% [5] |

### Permutation Importance Tradeoffs
`importance_n_repeats` controls the accuracy/speed tradeoff [7]:

-   **5 repeats:** ~20 minutes. Good for debugging and quick checks. High variance.
-   **30 repeats:** ~2 hours. Research-grade stability. Recommended for publication.
-   **Metric choice:** `"loss"` measures continuous degradation and is generally more stable than `"f1"`, which can be noisy due to threshold effects at 0.5.

## 3. Output Files Explained

After completion, inspect these artifacts in `project_root/results/`:

| File | Contents | Use Case |
| :--- | :--- | :--- |
| `{target}_results.png` | ROC, PR curve, Confusion Matrix, Calibration, Probability Distribution, Brier Score [9] | Visual quality assessment |
| `{target}_results.json` | All metrics + top 10 ROIs with anatomical names + metadata | Programmatic access, reporting |
| `importance_{target}.csv` | All 414 regions ranked by importance score | Biomarker identification |
| `importance_{target}_plot.png` | Horizontal bar chart of top 5 regions | Presentations, quick inspection |
| `weights/weights_{target}.pth` | Fine-tuned model state dict (safe format) | Downstream inference, further FT [4] |

## 4. Troubleshooting Common Failures

| Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `ValueError: Missing columns in phenotype` | CSV missing `Age`, `Sex`, or target column [2] | Verify exact column names (case-sensitive) |
| `FileNotFoundError: Pretrained weights` | Double-nested path or wrong filename [4] | Pass only filename if in `project_root/weights/`; verify file exists |
| `RuntimeError: Failed to load pretrained weights` | Architecture mismatch between weights and model | Ensure `model_config` matches architecture used during pretraining |
| Model predicts only one class | Severe class imbalance without focal loss [5] | Set `use_focal_loss=True`; verify label distribution |
| Importance plot is noisy/unstable | Too few permutation repeats [7] | Increase `importance_n_repeats` to ≥20 |
| `Pathlib` resolves wrong directory | Using absolute string in `/` join | Use `Path(__file__).parent / "relative"` instead of `BASE_DIR / "/absolute"` |

## Sources
[1] `bbtransformer/trainer/exe.py` – `run_analysis` function signature, path resolution, and output saving
[2] `bbtransformer/trainer/exe.py` – `_validate_inputs` enforcing target_column, Age, Sex requirements
[3] `bbtransformer/trainer/loader.py` – `prepare_fmri_data` NPZ format validation and subject alignment
[4] `bbtransformer/utils.py` – `load_model_weights` absolute vs. relative path resolution logic
[5] `bbtransformer/trainer/train.py` – Ranger21 optimizer, AdaptiveFocalLoss, early stopping implementation
[7] `bbtransformer/trainer/rank.py` – `calculate_permutation_importance` supporting "f1" and "loss" metrics
[8] `bbtransformer/trainer/exe.py` – `_configure_seeds` setting torch, numpy, and cuda seeds
[9] `bbtransformer/trainer/eval.py` – `plot_results` generating 6-panel evaluation figure
