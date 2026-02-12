# bbtransformer\trainer\tune.py

# Import Essentials
import torch
import numpy as np
import optuna
from typing import Optional, Dict, Any, List
from ..model import create_bbtransformer
from .train import train_model  
from sklearn.metrics import f1_score, roc_auc_score


# ======================
# HYPERPARAMETER TUNING 
# ======================

def optuna_objective(trial, train_loader, val_loader, feature_dim, search_config=None):
    if search_config is None:
        search_config = {}

    # Helper to safely get config with defaults
    def get(key, default):
        return search_config.get(key, default)

    # --- Hyperparameter suggestions ---
    config = {
        'feature_dim': feature_dim,
        'num_classes': 1,
        'embed_dim': trial.suggest_categorical('embed_dim', get('embed_dim', [128, 256, 512])),
        'num_heads': trial.suggest_categorical('num_heads', get('num_heads', [4, 8, 16])),
        'num_layers': trial.suggest_int('num_layers', 3, 8),
        'dropout_input': trial.suggest_float('dropout_input', 0.05, 0.3),
        'dropout_attn': trial.suggest_float('dropout_attn', 0.05, 0.3),
        'dropout_ffn': trial.suggest_float('dropout_ffn', 0.1, 0.4),
        'dropout_classifier': trial.suggest_float(
            'dropout_classifier',
            *get('dropout_classifier', (0.0, 0.3))
        ),
        'dropout_temporal': trial.suggest_float('dropout_temporal', 0.0, 0.2),
        'embed_dim_age': trial.suggest_categorical('embed_dim_age', [8, 16, 32]),
        'embed_dim_ext': trial.suggest_categorical('embed_dim_ext', [8, 16, 32]),
        'patch_size': 3,
        'patch_embed_ratio': trial.suggest_float('patch_embed_ratio', 0.5, 1.0, step=0.25),
        'temp_attn_hidden': trial.suggest_categorical('temp_attn_hidden', [32, 64, 128]),
        'n_kv_heads': trial.suggest_categorical('n_kv_heads', [None, 2, 4, 8]),
        'return_attn_weights': False,
    }

    # LR and weight decay 
    lr = trial.suggest_float('lr', *get('lr_range', (1e-5, 1e-3)), log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # --- Reproducibility ---
    seed = 42 + trial.number
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # --- Model ---
    model = create_bbtransformer(config)

    # Updated training call 
    try:
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=get('epochs', 100),
            lr=lr,
            weight_decay=weight_decay,
            patience=get('patience', 20),
            use_focal_loss=False,      # ← balanced data
            early_stop_metric="f1"
        )
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.TrialPruned()

    # --- Final evaluation: F1 (for Optuna) + AUC (for diagnostics) ---
    device = next(trained_model.parameters()).device
    trained_model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for fmri, age, ext, labels in val_loader:
            fmri = fmri.to(device, dtype=torch.float32)
            age = age.to(device, dtype=torch.float32)
            ext = ext.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float32)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = trained_model(fmri, age, ext)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.ravel())
            all_targets.extend(labels.cpu().numpy().ravel())

    # Compute F1 (primary metric for Optuna)
    try:
        preds = (np.array(all_probs) > 0.5).astype(int)
        val_f1 = f1_score(all_targets, preds, zero_division=0)
    except:
        val_f1 = 0.0

    # Compute AUC (for logging only)
    try:
        val_auc = roc_auc_score(all_targets, all_probs)
    except:
        val_auc = 0.5

    # Log both for diagnostics
    print(f"Trial {trial.number} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

    trial.report(-val_f1, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return -val_f1


def tune_hyperparameters(train_loader, val_loader, feature_dim, n_trials=50, search_config=None):
    """
    Perform hyperparameter tuning using Optuna (2025 best practices)
    - Uses MedianPruner for early stopping
    - Reports intermediate metrics
    - Handles GPU memory cleanup
    
    Args:
        search_config (dict, optional): Override default search space.
            See optuna_objective for allowed keys.
    """
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,   # don't prune first 10 trials
        n_warmup_steps=5,      # don't prune before epoch 5
        interval_steps=2       # check every 2 epochs
    )

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        study_name="bbtransformer_tuning_2025"
    )

    def objective(trial):
        return optuna_objective(trial, train_loader, val_loader, feature_dim, search_config)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )

    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*60)
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    complete_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"  Trials completed: {complete_trials}")
    print(f"  Trials pruned:    {pruned_trials}")
    print(f"Best F1 Score: {-study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*60)

    return study.best_params, study

