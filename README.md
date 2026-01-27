# Brain Biomarker Transformer for fMRI Classification

A minimal, modular, and interpretable foundation model for fMRI-based brain disorder classification using a multi-scale transformer architecture with rotary embeddings and grouped-query attention.

Trained on 414-region Glasser+Tian parcellated time series, BBTransformer enables transfer learning, fine-tuning, and biomarker discovery across ADHD, ASD, UK Biobank ICD conditions, and more.

<br>


### **A Multi-Scale Spatiotemporal Decoder**

BBTransformer processes the 150 × 414 input through three integrated streams. The primary temporal stream consists of six transformer layers with 512-dimensional embeddings, eight query heads, four key-value heads implementing grouped-query attention, rotary position embeddings for relative timing, root mean square layer normalization, and SwiGLU activation functions. The local temporal stream applies patch embeddings with a patch size of two, yielding 75 coarse-grained tokens that are fused with the primary stream via cross-attention mechanisms beginning at the fourth layer. Finally, a learned temporal attention pooling mechanism dynamically weights each of the 150 timepoints according to its contribution to the final diagnostic decision.


<img width="1408" height="768" alt="ukbb_diagram" src="https://github.com/user-attachments/assets/f343fb1e-9a25-49c6-ae17-2f8f0cb226df" />



<br>



<br>

## Model Architecture

BBTransformer processes **150 × 414** fMRI time series through three integrated streams:

1. **Primary Temporal Stream**:  
   - 6 transformer layers  
   - 512-dim embeddings, 8 query heads, 4 KV heads (GQA)  
   - Rotary position embeddings, RMSNorm, SwiGLU  

2. **Local Patch Stream**:  
   - Patch size = 2 → 75 tokens  
   - Cross-attention fusion at layer 4  

3. **Temporal Attention Pooling**:  
   - Learned weighting of timepoints for diagnostic decisions  

Final output: single probability via sigmoid for binary classification.

---

## Quick Start

### 1. Install as a Package
```bash
git clone https://github.com/yourname/BBT.git
cd BBT
pip install -e .
```



### 2. Prepare Data (Dataset-Specific)
Run preprocessing notebooks to generate `.npz` + `.csv`:
```bash
jupyter notebook preprocessing/ADHD200/prepare_adhd200_gt.ipynb
```

### 3. Train or Fine-Tune
```python
import bbtransformer as bbt

# Load data
train_loader, val_loader, test_loader, metadata = bbt.prepare_fmri_data(
    data_path='dataset/fmri_ADHD.npz',
    pheno_path='dataset/pheno_ADHD.csv',
    target_column='ADHD'
)

# Create model
model = bbt.create_bbtransformer({
    'feature_dim': 414,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 6,
    # ... other hyperparameters
})

# Train
trained_model = bbt.train_model(model, train_loader, val_loader)

# Save
bbt.save_model_weights(trained_model, target_name='ADHD')
```

### 4. Interpret Results
```python
# Permutation importance
roi_names = bbt.load_roi_names()  # ✅ No path needed!
importance = bbt.calculate_permutation_importance(trained_model, val_loader, 414)
bbt.plot_importance(importance, roi_names, top_n=30)

# Single-subject prediction
predictor = bbt.Diagnostic(trained_model, metadata)
result = predictor.predict_single(fmri, age=56.3, ext=1)
print(result['interpretation'])
```

---

## Key Features

### Architecture
- **Rotary Positional Embeddings** for temporal awareness
- **Grouped-Query Attention (GQA)** for efficiency
- **Multi-scale fusion** with patch embedding
- **Temporal attention pooling** for sequence summarization

### Training
- **Ranger21 optimizer** with cosine annealing
- **Flexible loss**: BCE, Focal Loss, Adaptive Focal Loss
- **Early stopping** on loss, AUC, or F1

### Interpretation
- **Permutation importance** for brain region ranking
- **Single-subject diagnosis** with confidence scoring (`Diagnostic` class)
- **Attention visualization** (when `return_attn_weights=True`)

---

## Supported Datasets

| Dataset | Conditions | Atlas |
|--------|------------|-------|
| **ADHD-200** | ADHD vs Controls | Glasser+Tian (414 ROIs) |
| **ABIDE** | ASD vs Controls | Glasser+Tian (414 ROIs) |
| **UK Biobank** | ICD F32, G20, G40, etc. | Glasser+Tian (414 ROIs) |
| **NAKO** | (In progress) | Glasser+Tian (414 ROIs) |

---

## 📈 Analysis Projects

- `notebooks/analysis/ADHD_adhd200_classification/`
- `notebooks/analysis/ASD_abide_classification/`
- `notebooks/analysis/PDD_ukbb_classification/`
- `notebooks/analysis/Sex_ukbb_classification/`

## 📘 Tutorials

- `notebooks/tutorials/hyperparameter_tuning.ipynb`
- `notebooks/tutorials/transfer_learning.ipynb`
- `preprocessing/UKBB/balance_cohorts/Cohort_creation.ipynb`

---

## ⚙️ Configuration

Best hyperparameters from Optuna tuning:
```python
BEST_HP = {
    'feature_dim': 414,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 6,
    'dropout_input': 0.27,
    'dropout_classifier': 0.03,
    'patch_size': 3,
    'embed_dim_age': 32,
    'embed_dim_ext': 16,
    # ...
}
```

---

## 📦 Requirements

- Python 3.8–3.12 (**PyTorch does not support 3.13 yet**)
- PyTorch ≥ 2.0
- nilearn (for preprocessing)
- pytorch_optimizer, optuna ≥ 3.0
- scikit-learn, pandas, numpy, matplotlib, seaborn, tqdm, jupyter

Install via:
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
BBT/
├── bbtransformer/          # Core package (installable)
│   ├── __init__.py
│   ├── model.py            # BBTransformer, create_bbtransformer
│   ├── utils.py            # load_roi_names, weight utilities
│   └── trainer/            # train, eval, rank, pred, tune
├── notebooks/              # Analysis & tutorials
├── preprocessing/          # Dataset-specific curation
└── weights/                # (User-provided) pretrained weights
```

All functions are accessible via:
```python
from bbtransformer import prepare_fmri_data, create_bbtransformer, Diagnostic, ...
# or
import bbtransformer as bbt
```



