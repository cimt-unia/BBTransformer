# bbtransformer\trainer\rank.py

# Import Essentials
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.amp import autocast 
from .eval import evaluate_model
from ..utils import load_roi_names  # kept for backward compatibility


def calculate_permutation_importance(
    model, 
    data_loader, 
    feature_dim, 
    n_repeats=10, 
    seed=42,
    target_name="target",
    metric="f1"  
):
    """
    Calculate permutation importance for brain regions
    
    Args:
        model: Trained BBTransformer
        data_loader: DataLoader with validation data
        feature_dim: Number of brain regions (e.g., 414)
        n_repeats: Number of permutation repeats
        seed: Random seed for reproducibility
        target_name: Name of target variable (used in CSV column name)
        metric: Importance metric ("f1" or "loss")
    
    Returns:
        importance_scores: Array of importance scores (length = feature_dim)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = next(model.parameters()).device
    
    if metric == "f1":
        baseline_metrics, _, _ = evaluate_model(model, data_loader)
        baseline_score = baseline_metrics['f1']
    elif metric == "loss":
        baseline_score = _compute_validation_loss(model, data_loader, device)
    else:
        raise ValueError("metric must be 'f1' or 'loss'")
    
    importance_scores = np.zeros(feature_dim)
    print(f"Calculating permutation importance ({metric}) for {feature_dim} brain regions...")

    for region_idx in tqdm(range(feature_dim), desc="Permuting regions"):
        region_scores = []
        for _ in range(n_repeats):
            if metric == "f1":
                all_preds, all_probs, all_targets = [], [], []
                for fmri, age, ext, labels in data_loader:
                    fmri = fmri.to(device)
                    age = age.to(device)
                    ext = ext.to(device)
                    
                    perm = torch.randperm(fmri.size(0), device=device)
                    fmri_perm = fmri.clone()
                    fmri_perm[:, :, region_idx] = fmri[perm, :, region_idx]
                    
                    with torch.no_grad(), autocast(device_type=device.type):
                        logits = model(fmri_perm, age, ext)
                        probs = torch.sigmoid(logits).cpu().numpy()
                    
                    preds = (probs > 0.5).astype(int)
                    all_probs.extend(probs)
                    all_preds.extend(preds)
                    all_targets.extend(labels.cpu().numpy().astype(int))
                
                f1 = f1_score(all_targets, all_preds, zero_division=0)
                region_scores.append(f1)
            
            elif metric == "loss":
                total_loss = 0.0
                n_samples = 0
                for fmri, age, ext, labels in data_loader:
                    fmri = fmri.to(device)
                    age = age.to(device)
                    ext = ext.to(device)
                    labels = labels.to(device).float()
                    
                    perm = torch.randperm(fmri.size(0), device=device)
                    fmri_perm = fmri.clone()
                    fmri_perm[:, :, region_idx] = fmri[perm, :, region_idx]
                    
                    with torch.no_grad(), autocast(device_type=device.type):
                        logits = model(fmri_perm, age, ext)
                        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
                        total_loss += loss.item()
                        n_samples += labels.size(0)
                
                avg_loss = total_loss / n_samples if n_samples > 0 else float('inf')
                region_scores.append(avg_loss)
        
        if metric == "f1":
            importance_scores[region_idx] = baseline_score - np.mean(region_scores)
        elif metric == "loss":
            importance_scores[region_idx] = np.mean(region_scores) - baseline_score  # higher = more important

    return importance_scores


def _compute_validation_loss(model, data_loader, device):
    """Helper: compute average BCE loss on validation set."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for fmri, age, ext, labels in data_loader:
            fmri = fmri.to(device)
            age = age.to(device)
            ext = ext.to(device)
            labels = labels.to(device).float()
            
            with autocast(device_type=device.type):
                logits = model(fmri, age, ext)
                loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
                total_loss += loss.item()
                n_samples += labels.size(0)
    return total_loss / n_samples if n_samples > 0 else float('inf')


def save_top_importance_to_csv(
    importance_scores,
    roi_metadata=None,  # ← Accept full metadata DataFrame
    target_name="target",
    top_n=30,
    save_path=None
):
    """
    Save top permutation importance scores to CSV with full ROI metadata.
    
    Args:
        importance_scores: Array of importance scores (length = 414)
        roi_metadata: Full ROI metadata DataFrame (from load_roi_metadata())
        target_name: Name of target variable
        top_n: Number of top regions to save
        save_path: Output CSV path
    
    Returns:
        pd.DataFrame: Full importance table (all 414 regions, sorted)
    """
    assert len(importance_scores) == 414, "Expected 414 regions"
    
    if roi_metadata is not None:
        # Use full metadata (preferred)
        df = roi_metadata.copy()
        df[f'importance_{target_name}'] = importance_scores
    else:
        # Fallback to names only (backward compatibility)
        roi_names = load_roi_names()
        df = pd.DataFrame({
            'roi_index': np.arange(414),
            'roi_name': roi_names,
            f'importance_{target_name}': importance_scores
        })
    
    # Sort by importance (descending)
    df = df.sort_values(f'importance_{target_name}', ascending=False).reset_index(drop=True)
    
    # Save top N to CSV
    if save_path is None:
        save_path = f'importance_{target_name}.csv'
    
    df.head(top_n).to_csv(save_path, index=False)
    print(f"Saved top {top_n} features to: {save_path}")
    
    return df


def plot_importance(
    importance_scores,
    roi_names=None,
    top_n=20,
    save_path='brain_importance.png'
):
    assert len(importance_scores) == 414, "Expected 414 regions"
    top_indices = np.argsort(importance_scores)[-top_n:][::-1]
    top_scores = importance_scores[top_indices]
    
    if roi_names is not None:
        assert len(roi_names) == 414
        top_labels = [roi_names[i] for i in top_indices]
    else:
        top_labels = [f'Region {i}' for i in top_indices]

    plt.figure(figsize=(12, max(6, top_n * 0.35)))
    plt.barh(range(top_n), top_scores, color='steelblue')
    plt.yticks(range(top_n), top_labels)
    plt.xlabel('Permutation Importance')
    plt.ylabel('Brain Region')
    plt.title(f'Top {top_n} Most Important Brain Regions')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {save_path}")
