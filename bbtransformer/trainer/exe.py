# bbtransformer\trainer\exe.py


import os
import gc
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np
import pandas as pd

# Import from your bbtransformer library
from bbtransformer import (
    prepare_fmri_data,
    load_roi_names,
    create_bbtransformer,
    train_model,
    evaluate_model,
    plot_results,
    calculate_permutation_importance,
    save_top_importance_to_csv,
    plot_importance,
    load_model_weights,
    save_model_weights
)


def run_analysis(
    target_column: str,
    # --- Flexible path inputs (preferred) ---
    data_path: Optional[str] = None,
    pheno_path: Optional[str] = None,
    # --- Legacy fallback (for CHRT-style) ---
    base_dir: Optional[str] = None,
    # --- Weights & config ---
    weights_dir: str = '/mnt/movement/users/jaizor/xtra/notebooks/BBT/_weights',
    pretrained_weight_file: Optional[str] = None,
    use_pretrained: bool = True,
    compute_importance: bool = True,
    random_seed: int = 42,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Runs the full BBTransformer classification pipeline for a single disorder.

    Parameters
    ----------
    target_column : str
        Binary target column name in the phenotype file (e.g., 'ASD', 'ADHD').
    data_path : str, optional
        Full path to fMRI .npz file (e.g., '/.../fmri_ASD.npz').
    pheno_path : str, optional
        Full path to phenotype .csv file (e.g., '/.../pheno_ASD.csv').
    base_dir : str, optional
        Legacy: used only if data_path/pheno_path are not provided.
        Assumes files named `fmri_{target}.npz` and `pheno_{target}.csv`.
    weights_dir : str
        Directory containing pretrained .pth files.
    pretrained_weight_file : str or None
        Filename of pretrained weights (required if use_pretrained=True).
    use_pretrained : bool
        Whether to load pretrained weights.
    compute_importance : bool
        Whether to compute permutation importance.
    random_seed : int
        Random seed for reproducibility.
    device : str
        Device for model loading ('cpu' or 'cuda').

    Returns
    -------
    dict : Contains metrics, model, metadata, and (optionally) importance scores.
    """
    # Ensure output directories exist
    Path('weights').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    # 🔑 PATH RESOLUTION: Prefer explicit paths, fall back to base_dir
    if data_path is not None and pheno_path is not None:
        DATA_PATH = data_path
        PHENO_PATH = pheno_path
    elif base_dir is not None:
        DATA_PATH = os.path.join(base_dir, f"fmri_{target_column}.npz")
        PHENO_PATH = os.path.join(base_dir, f"pheno_{target_column}.csv")
    else:
        raise ValueError(
            "Either (data_path and pheno_path) OR base_dir must be provided."
        )

    # Validate data files exist
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"fMRI data not found: {DATA_PATH}")
    if not os.path.exists(PHENO_PATH):
        raise FileNotFoundError(f"Phenotype file not found: {PHENO_PATH}")

    # Handle pretrained weights
    WEIGHT_PATH = None
    if use_pretrained:
        if pretrained_weight_file is None:
            raise ValueError("pretrained_weight_file must be provided when use_pretrained=True")
        WEIGHT_PATH = os.path.join(weights_dir, pretrained_weight_file)
        if not os.path.exists(WEIGHT_PATH):
            raise FileNotFoundError(f"Pretrained weights not found: {WEIGHT_PATH}")

    # ===============================================
    # STEP 1: Load Data
    # ===============================================
    print("=" * 60)
    print(f"STEP 1: Loading Data for Target = '{target_column}'")
    print(f"  fMRI: {DATA_PATH}")
    print(f"  Phenotype: {PHENO_PATH}")
    print("=" * 60)

    pheno = pd.read_csv(PHENO_PATH)
    print(f"✓ Loaded phenotype: {pheno.shape}")

    features = np.load(DATA_PATH)
    data = features['data']
    subject_ids = features['subject_ids']

    roi_names = load_roi_names()
    print(f"✓ Loaded fMRI: {data.shape}")
    print(f"  Subjects: {len(subject_ids)}")
    print(f"  Timepoints: {data.shape[1]}")
    print(f"  Brain regions: {data.shape[2]}")
    print(f"  ROI labels: {len(roi_names)}")

    required_columns = [target_column, 'Age', 'Sex']
    missing = [col for col in required_columns if col not in pheno.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in phenotype: {missing}")
    print("✓ All required columns present")

    # ===============================================
    # STEP 2: Prepare Data Loaders
    # ===============================================
    print("\n" + "=" * 60)
    print("STEP 2: Preparing Data Loaders")
    print("=" * 60)

    train_loader, val_loader, test_loader, metadata = prepare_fmri_data(
        data_path=DATA_PATH,
        pheno_path=PHENO_PATH,
        target_column=target_column,
        age_column='Age',
        ext_column='Sex',
        batch_size=64,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_seed=random_seed
    )

    print("\n Dataset Meta")
    for key, value in metadata.items():
        if key not in ['age_mean', 'age_std']:
            print(f"  {key}: {value}")

    # ===============================================
    # STEP 3: Initialize Model
    # ===============================================
    print("\n" + "=" * 60)
    print("STEP 3: Initializing BBTransformer")
    print("=" * 60)

    torch.cuda.empty_cache()
    gc.collect()

    BEST_HP = {
        'feature_dim': metadata['feature_dim'],
        'num_classes': 1,
        'embed_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'dropout_input': 0.271037581013532,
        'dropout_attn': 0.14600627822960482,
        'dropout_ffn': 0.27515967497010174,
        'dropout_classifier': 0.029054607960590395,
        'dropout_temporal': 0.1670895241114272,
        'embed_dim_age': 32,
        'embed_dim_ext': 16,
        'patch_size': 3,
        'patch_embed_ratio': 0.5,
        'temp_attn_hidden': 128,
        'n_kv_heads': 4,
        'return_attn_weights': False
    }

    model = create_bbtransformer(BEST_HP)
    print(f"✓ Model created on {device} with {model.count_parameters():,} parameters")

    # ===============================================
    # STEP 3.5: Load Pretrained Weights (Optional)
    # ===============================================
    if use_pretrained:
        print("\n" + "=" * 60)
        print("STEP 3.5: Loading Pretrained Weights")
        print(f"  From: {WEIGHT_PATH}")
        print("=" * 60)
        success = load_model_weights(model, device=device, weight_paths=[WEIGHT_PATH])
        if not success:
            raise RuntimeError("❌ Failed to load pretrained weights.")
        print("✓ Pretrained weights loaded successfully.")
    else:
        print("\n[INFO] Training from scratch (no pretrained weights).")



    # ===============================================
    # STEP 4: Train Model
    # ===============================================
    print("\n" + "=" * 60)
    print("STEP 4: Training")
    print("=" * 60)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10000,
        lr=3e-4,  
        weight_decay=2.3798658050870825e-05,
        patience=90,
        use_focal_loss=False  # Balanced data 
    )

    # ===============================================
    # STEP 4.5: Save Final Weights
    # ===============================================
    save_model_weights(trained_model, target_name=target_column, safe_format=True)

    # ===============================================
    # STEP 5: Evaluate
    # ===============================================
    print("\n" + "=" * 60)
    print("STEP 5: Test Set Evaluation")
    print("=" * 60)

    metrics, probs, targets = evaluate_model(trained_model, test_loader)

    print("\n Performance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\n📊 Confusion Matrix:")
    print(metrics['confusion_matrix'])

    plot_results(metrics, probs, targets, save_path=f'results/{target_column}_results.png')

    # ===============================================
    # STEP 6: Permutation Importance (Conditional)
    # ===============================================
    importance_scores = None
    if compute_importance:
        print("\n" + "=" * 60)
        print("STEP 6: Permutation Importance (CSV)")
        print("=" * 60)

        importance_scores = calculate_permutation_importance(
            trained_model,
            val_loader,
            feature_dim=metadata['feature_dim'],
            n_repeats=30,
            seed=random_seed,
            target_name=target_column
        )

        save_top_importance_to_csv(
            importance_scores,
            roi_names=roi_names,
            target_name=target_column,
            top_n=10,
            save_path=f'results/importance_{target_column}.csv'
        )

        plot_importance(
            importance_scores,
            roi_names=roi_names,
            top_n=10,
            save_path=f'results/importance_{target_column}_plot.png'
        )

        print("✅ Permutation importance saved.")
    else:
        print("\n[INFO] Skipping permutation importance (compute_importance=False).")

    print("\n" + "=" * 60)
    print("✅ TRAINING & EVALUATION COMPLETE")
    print(f"→ Target: {target_column}")
    print("=" * 60)

    return {
        'metrics': metrics,
        'model': trained_model,
        'metadata': metadata,
        'importance_scores': importance_scores,
        'target': target_column
    }
