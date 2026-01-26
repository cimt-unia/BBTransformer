# BBTransformer: Brain Biomarker Transformer for fMRI Classification

A minimal, modular framework for interpretable fMRI-based brain disorder classification using transformer architectures with rotary embeddings and grouped-query attention.

## 📁 Project Structure
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

## 🚀 Quick Start

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

## 📊 Key Features

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


## 🎯 Analysis Projects

- **ADHD Classification** (`ADHD_adhd200_classification/`)
- **ASD Classification** (`ASD_abide_classification/`)  
- **PDD Classification** (`PDD_ukbb_classification/`)
- **Biological Sex Prediction** (`Sex_ukbb_classification/`)

## 📝 Tutorials

- `finetuning_transfer.ipynb` - Transfer learning workflow
- `hyperparameter_tuning.ipynb` - Optuna optimization guide
- `ukbb/Cohort_creation.ipynb` - UKB-specific preprocessing

## 🔧 Configuration

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

