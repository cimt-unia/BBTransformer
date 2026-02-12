# bbtransformer/__init__.py

# Model
from .model import BBTransformer, create_bbtransformer

# Utilities
from .utils import save_model_weights, load_model_weights, load_roi_names

# Trainer components
from .trainer.loader import prepare_fmri_data
from .trainer.train import train_model
from .trainer.exe import run_analysis
from .trainer.eval import evaluate_model, plot_results
from .trainer.rank import (
    calculate_permutation_importance,
    save_top_importance_to_csv,
    plot_importance
)
from .trainer.tune import tune_hyperparameters
from .trainer.pred import Diagnostic

