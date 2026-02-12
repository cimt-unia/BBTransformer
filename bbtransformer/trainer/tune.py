# bbtransformer\trainer\tune.py

# Import Essentials
import torch
import numpy as np
import optuna
from typing import Optional, Dict, Any, List
from datetime import datetime
import os
from ..model import create_bbtransformer
from .train import train_model  
from .eval import evaluate_model  


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
        'patch_size': 3,  # ← Keep fixed per reference implementation
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
            epochs=get('epochs', 5000),      # ← Default to 5000
            lr=lr,
            weight_decay=weight_decay,
            patience=get('patience', 80),     # ← Default to 80
            use_focal_loss=False,
            early_stop_metric="f1"
        )
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.TrialPruned()

    # --- Full evaluation using evaluate_model (per bbt_transformer.txt) ---
    metrics, _, _ = evaluate_model(trained_model, val_loader)
    
    def to_float(x):
        if isinstance(x, (np.ndarray, np.generic)):
            return float(x.item() if x.ndim == 0 else x[0])
        elif torch.is_tensor(x):
            return float(x.item())
        else:
            return float(x)
    
    f1 = to_float(metrics.get('f1', 0.0))
    roc_auc = to_float(metrics.get('roc_auc', 0.0))
    accuracy = to_float(metrics.get('accuracy', 0.0))
    precision = to_float(metrics.get('precision', 0.0))
    recall = to_float(metrics.get('recall', 0.0))
    
    metric_vals = [f1, roc_auc, accuracy, precision, recall]
    THRESHOLD = 0.60
    valid = all(m >= THRESHOLD for m in metric_vals)
    
    composite = (
        0.35 * f1 +
        0.25 * roc_auc +
        0.15 * accuracy +
        0.15 * precision +
        0.10 * recall
    )
    
    # Save metadata
    clean_metrics = {'f1': f1, 'roc_auc': roc_auc, 'accuracy': accuracy, 'precision': precision, 'recall': recall}
    trial.set_user_attr("metrics", clean_metrics)
    trial.set_user_attr("composite", composite)
    
    if valid:
        return -composite
    else:
        return 1.0 - min(metric_vals)


def tune_hyperparameters(train_loader, val_loader, feature_dim, n_trials=50, search_config=None, target_name="disorder"):
    """
    Perform hyperparameter tuning with composite scoring and thresholding.
    Designed for clinical deployment readiness.
    """
    pruner = optuna.pruners.MedianPruner(  # ← Compatible with final-value reporting
        n_startup_trials=10,
        n_warmup_steps=0
    )

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        study_name=f"bbtransformer_tuning_{target_name}"
    )

    def objective(trial):
        return optuna_objective(trial, train_loader, val_loader, feature_dim, search_config)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )

    # Analyze results
    THRESHOLD = 0.60
    valid_trials = [
        t for t in study.trials 
        if t.state == optuna.trial.TrialState.COMPLETE and
        all(t.user_attrs.get("metrics", {}).get(m, 0) >= THRESHOLD 
            for m in ['f1', 'roc_auc', 'accuracy', 'precision', 'recall'])
    ]
    
    if valid_trials:
        best = min(valid_trials, key=lambda t: t.value)
        status = "✅ VALID"
    else:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print("❌ No trials completed successfully!\n")
            return None
        best = min(completed, key=lambda t: t.value if t.value < 1.0 else float('inf'))
        status = "⚠️  BEST EFFORT"
    
    best_params = best.params.copy()
    best_metrics = best.user_attrs.get("metrics", {})
    composite = best.user_attrs.get("composite", 0.0)
    
    # Save results
    os.makedirs('weights', exist_ok=True)
    results_path = f'weights/best_params_{target_name}.json'
    
    results = {
        'best_params': best_params,
        'metrics': best_metrics,
        'composite_score': composite,
        'search_method': 'Composite scoring for clinical deployment (2026 SOTA)',
        'threshold_met': bool(valid_trials),
        'total_trials': len(study.trials),
        'valid_trials': len(valid_trials),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"{status} TUNING COMPLETE: {target_name}")
    print(f"{'='*80}")
    print(f"📊 Trials: {len(study.trials)} total | {len(valid_trials)} valid")
    print(f"🎯 Composite Score: {composite:.4f}")
    
    print(f"\n📈 Metrics:")
    for metric in ['f1', 'roc_auc', 'accuracy', 'precision', 'recall']:
        val = best_metrics.get(metric, 0)
        check = '✅' if val >= THRESHOLD else '❌'
        print(f"   {metric:12s}: {val:.4f} {check}")
    
    print(f"\n🏆 Best Hyperparameters:")
    for k, v in best_params.items():
        if k == 'lr':
            print(f"   {k:24s}: {v:.6f}")
        elif k == 'weight_decay':
            print(f"   {k:24s}: {v:.2e}")
        else:
            print(f"   {k:24s}: {v:.4f}")
    
    print(f"\n💾 Saved to: {results_path}")
    print(f"{'='*80}\n")
    
    return best_params, study
