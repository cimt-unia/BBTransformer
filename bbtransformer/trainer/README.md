# BBTransformer `run_analysis` Tutorial

## 1. Prerequisites and Data Requirements

Before executing the analysis, ensure the input data adheres to the following structural constraints. The pipeline validates these inputs strictly and will raise errors if requirements are not met.

-   **Phenotype File (`.csv`):** Must contain at minimum three columns:
    -   `target_column`: Binary labels (0 or 1) matching the name provided in the function call.
    -   `Age`: Numerical age values.
    -   `Sex`: Biological sex or gender encoding as expected by the model.
-   **fMRI Data File (`.npz`):** Must contain keys `data` (feature matrix) and `subject_ids`.
-   **Pretrained Weights:** If fine-tuning, the specified `.pth` file must exist within `{project_root}/weights/`.

## 2. Execution Script

The following script demonstrates the complete parameter set for `run_analysis`. It utilizes `pathlib.Path` for robust path management and is configured for fine-tuning a pretrained base model.

```python
# Fine-tune BBTransformer on depressive episode cohort using pathlib for path management
import logging
from pathlib import Path

from bbtransformer.trainer.exe import run_analysis

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path("")
DATA_PATH = BASE_DIR / "fmri_ICD_F32_Depressive_Episode_vs_Psychopathology_Substance_Use.npz"
PHENO_PATH = BASE_DIR / "pheno_ICD_F32_Depressive_Episode_vs_Psychopathology_Substance_Use.csv"

results = run_analysis(
    # --- DATA IDENTIFICATION ---
    target_column="target_label",
    data_path=str(DATA_PATH),
    pheno_path=str(PHENO_PATH),
    base_dir=None,

    # --- PROJECT STRUCTURE ---
    project_root=".",
    use_pretrained=True,
    pretrained_weight_file="ukbb_base_model.pth",

    # --- TRAINING HYPERPARAMETERS ---
    training_config={
        "epochs": 500,
        "lr": 1e-5,
        "weight_decay": 1.14e-06,
        "patience": 50,
    },
    early_stop_metric="f1",
    use_focal_loss=False,

    # --- MODEL ARCHITECTURE ---
    model_config=None,

    # --- IMPORTANCE ANALYSIS ---
    compute_importance=True,
    importance_metric="loss",
    importance_n_repeats=30,

    # --- REPRODUCIBILITY AND HARDWARE ---
    random_seed=42,
    device=None,

    # --- OUTPUT CONTROL ---
    save_plots=True,
    save_json=True,
)
```

## 3. Comprehensive Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `target_column` | `str` | *Required* | Name of the binary label column in the phenotype CSV. |
| `data_path` | `str \| None` | `None` | Absolute path to the fMRI `.npz` file. Required if `base_dir` is `None`. |
| `pheno_path` | `str \| None` | `None` | Absolute path to the phenotype `.csv` file. Required if `base_dir` is `None`. |
| `base_dir` | `str \| None` | `None` | Fallback directory. Auto-resolves paths as `fmri_{target}.npz` and `pheno_{target}.csv`. |
| `project_root` | `str` | `"."` | Root directory for creating `weights/` and `results/` subdirectories. |
| `pretrained_weight_file` | `str \| None` | `None` | Filename of weights inside `{project_root}/weights/`. Mandatory when `use_pretrained=True`. |
| `use_pretrained` | `bool` | `True` | Loads existing weights before training. Set `False` to train from scratch. |
| `training_config` | `Dict \| None` | `None` | Overrides default training hyperparameters (`epochs`, `lr`, `weight_decay`, `patience`). |
| `model_config` | `Dict \| None` | `None` | Overrides default architecture hyperparameters (`embed_dim`, `num_heads`, etc.). |
| `compute_importance` | `bool` | `True` | Enables permutation-based feature importance ranking post-training. |
| `random_seed` | `int` | `42` | Seeds Python, NumPy, PyTorch CPU, and CUDA generators for reproducibility. |
| `device` | `str \| None` | `None` | Hardware target (`"cuda"` or `"cpu"`). Auto-detects GPU availability when `None`. |
| `save_plots` | `bool` | `False` | Generates and saves ROC, PR, confusion matrix, and importance plots to `results/`. |
| `save_json` | `bool` | `True` | Writes metrics, metadata, and top-10 ROIs to a JSON file in `results/`. |
| `early_stop_metric` | `str` | `"f1"` | Validation metric monitored for early stopping (`"f1"` or `"loss"`). |
| `use_focal_loss` | `bool` | `False` | Activates adaptive focal loss for handling class imbalance. |
| `importance_metric` | `str` | `"loss"` | Metric used to evaluate permutation importance (`"f1"` or `"loss"`). |
| `importance_n_repeats` | `int` | `30` | Number of permutation shuffles per ROI. Higher values increase stability but also compute time. |

## 4. Output Artifacts

Upon successful completion, the pipeline generates the following artifacts within `{project_root}/results/`:

-   **`{target}_results.json`:** Structured summary containing all evaluation metrics, dataset metadata, and the top-10 most important ROIs with scores.
-   **`{target}_results.png`:** Composite visualization including ROC curve, Precision-Recall curve, confusion matrix, and prediction probability distribution (generated only if `save_plots=True`).
-   **`importance_{target}.csv`:** Full ranked list of all brain regions sorted by permutation importance score.
-   **`importance_{target}_plot.png`:** Bar chart of the top-5 most important ROIs (generated only if `save_plots=True` and `compute_importance=True`).
-   **`{project_root}/weights/weights_{target}.pth`:** Saved checkpoint of the final trained model.

## 5. Technical Verification and Constraints

-   **Path Handling:** While `pathlib.Path` is recommended for constructing paths in user scripts, `run_analysis` internally expects string arguments for `data_path` and `pheno_path`. Always cast `Path` objects using `str()` before passing them to the function to avoid type-related failures.
-   **Path Resolution Logic:** The function requires either explicit `data_path`/`pheno_path` OR a valid `base_dir`. Providing neither raises `ValueError`. Explicit paths take precedence over `base_dir`.
-   **Mandatory Phenotype Columns:** The validator enforces the presence of `target_column`, `Age`, and `Sex`. Missing columns cause immediate failure before any data loading occurs.
-   **Pretrained Weight Dependency:** When `use_pretrained=True`, omitting `pretrained_weight_file` raises `ValueError`. A missing weight file at the resolved path raises `FileNotFoundError`.
-   **Training Config Merging:** User-provided `training_config` dictionaries perform shallow merges with defaults. Unspecified keys retain their default values defined in `DEFAULT_TRAIN_CFG`.
-   **Importance Computation Cost:** Permutation importance scales linearly with `importance_n_repeats` and the number of ROIs. For large cohorts or high repeat counts, expect significant additional runtime after training completes.
-   **Device Handling:** When `device=None`, CUDA availability is checked at runtime. If CUDA is unavailable, execution silently falls back to CPU without warning beyond the initial log message.
