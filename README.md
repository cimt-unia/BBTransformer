# BBTransformer: Brain Biomarker Transformer for fMRI Classification



A minimal, modular framework for interpretable fMRI-based brain disorder classification using transformer architectures with rotary embeddings and grouped-query attention.


### **A Multi-Scale Spatiotemporal Decoder**

BBTransformer processes the 150 × 414 input through three integrated streams. The primary temporal stream consists of six transformer layers with 512-dimensional embeddings, eight query heads, four key-value heads implementing grouped-query attention, rotary position embeddings for relative timing, root mean square layer normalization, and SwiGLU activation functions. The local temporal stream applies patch embeddings with a patch size of two, yielding 75 coarse-grained tokens that are fused with the primary stream via cross-attention mechanisms beginning at the fourth layer. Finally, a learned temporal attention pooling mechanism dynamically weights each of the 150 timepoints according to its contribution to the final diagnostic decision.


<img width="1408" height="768" alt="ukbb_diagram" src="https://github.com/user-attachments/assets/f343fb1e-9a25-49c6-ae17-2f8f0cb226df" />

### Model Architecture

The model accepts as input a 150 × 414 matrix representing the time series for all regions. This input is processed through three parallel streams. The primary stream consists of six transformer encoder layers, each with 512-dimensional embeddings, eight query attention heads, and four key-value heads in a grouped-query attention configuration. Rotary position embeddings with dimensionality 64 encode relative temporal positions. Each layer uses root mean square layer normalization before attention and feedforward operations. The feedforward network within each layer employs SwiGLU activation with an expansion factor of 8/3.

The patch embedding stream applies a patch size of two along the temporal dimension, yielding 75 tokens. These patches are projected to 512 dimensions and processed through two transformer layers with the same architecture as the primary stream. Patch representations are integrated with the primary stream through cross-attention in the fourth layer.

The temporal attention pooling module computes attention weights across the 150 timepoints using a two-layer multilayer perceptron with hidden dimension 128 and GELU activation. These weights are applied to the primary stream outputs to produce a single 512-dimensional vector, which is passed through a final linear layer and sigmoid activation for binary classification.

## Project Structure
```
root/
├── 📄 data.py              # Data loading & cohort creation
├── 📄 eval.py              # Model evaluation & plotting  
├── 📄 interpret.py         # Permutation importance & interpretation
├── 📄 model.py             # BBTransformer architecture
├── 📄 train.py             # Training with Ranger21 optimizer
├── 📄 tuning.py            # Hyperparameter optimization
├── 📄 utils.py             # Weight management utilities
├── 📁 config/              # Model configurations
├── 📁 notebooks/           # Jupyter notebooks
│   ├── 📁 analysis/        # Disease-specific studies
│   └── 📁 tutorials/       # Step-by-step guides
├── 📁 preprocessing/       # Dataset-specific preparation
└── 📁 weights/             # Trained model checkpoints
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Run dataset-specific preprocessing:
```bash
# ADHD-200 example
jupyter notebook preprocessing/ADHD200/prepare_adhd200_gt.ipynb
```

### 3. Train Model
Use pre-tuned configuration:
```python
from model import create_bbtransformer
from train import train_model
from data import prepare_fmri_data

# Load data
train_loader, val_loader, test_loader, metadata = prepare_fmri_data(
    data_path='dataset/fmri_ADHD.npz',
    pheno_path='dataset/pheno_ADHD.csv',
    target_column='ADHD'
)

# Create and train model
model = create_bbtransformer(config)
trained_model = train_model(model, train_loader, val_loader)
```

### 4. Analyze Results
Explore analysis notebooks:
- `notebooks/analysis/ADHD_adhd200_classification/ADHD_notebook.ipynb`
- `notebooks/analysis/ASD_abide_classification/ASD_notebook.ipynb`

## Key Features

### Architecture
- **Rotary Positional Embeddings** for temporal awareness
- **Grouped-Query Attention** for efficiency
- **Multi-scale fusion** with patch embedding
- **Temporal attention pooling** for sequence summarization

### Training
- **Ranger21 optimizer** with adaptive learning rates
- **Flexible loss functions**: BCE, Focal Loss, Adaptive Focal Loss
- **Early stopping** with multiple metrics (loss, AUC, F1)

### Interpretation
- **Permutation importance** for brain region ranking
- **Single-subject prediction** with confidence scoring
- **Attention map visualization** (when enabled)

## 📈 Supported Datasets

| Dataset | Conditions | Atlas |
|---------|------------|-------|
| **ADHD-200** | ADHD vs Controls | Glasser+Tian (414 ROIs) |
| **ABIDE** | ASD vs Controls | Glasser+Tian (414 ROIs) |
| **UK Biobank** | Multiple ICD conditions | Glasser+Tian (414 ROIs) |


## Analysis Projects

- **ADHD Classification** (`ADHD_adhd200_classification/`)
- **ASD Classification** (`ASD_abide_classification/`)  
- **PDD Classification** (`PDD_ukbb_classification/`)
- **Biological Sex Prediction** (`Sex_ukbb_classification/`)

## Tutorials

- `finetuning_transfer.ipynb` - Transfer learning workflow
- `hyperparameter_tuning.ipynb` - Optuna optimization guide
- `ukbb/Cohort_creation.ipynb` - UKB-specific preprocessing

## Configuration

Best hyperparameters from Optuna tuning are stored in:
```
config/best_bbtransformer_config.json
```

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- pytorch_optimizer
- Optuna
- scikit-learn
- pandas, numpy
- nilearn (for preprocessing)

