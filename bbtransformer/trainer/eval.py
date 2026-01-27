# bbtransformer\trainer\eval.py

# Import Essentials
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List



# ======================
# EVALUATION 
# ======================

def evaluate_model(model, data_loader):
    """Evaluate model performance with comprehensive metrics"""
    device = next(model.parameters()).device
    model.eval()
    
    all_preds, all_probs, all_targets = [], [], []
    
    with torch.no_grad():
        for fmri, age, ext, labels in tqdm(data_loader, desc="Evaluation", leave=False):
            fmri = fmri.to(device)
            age = age.to(device)
            ext = ext.to(device)
            
            with autocast(device_type=device.type):
                logits = model(fmri, age, ext)
                probs = torch.sigmoid(logits).cpu().numpy()
            
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
    
    if all_preds:
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0),
            'f1': f1_score(all_targets, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_targets, all_probs),
            'confusion_matrix': confusion_matrix(all_targets, all_preds),
        }
    else:
        metrics = {k: 0 for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
        metrics['confusion_matrix'] = np.zeros((2, 2))
    
    return metrics, all_probs, all_targets



def plot_results(metrics, probs, targets, save_path='model_evaluation.png'):
    """Visualize model performance with multiple plots"""
    plt.figure(figsize=(18, 12))
    probs = np.array(probs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    # ROC Curve
    plt.subplot(2, 2, 1)
    fpr, tpr, _ = roc_curve(targets, probs)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Precision-Recall Curve
    plt.subplot(2, 2, 2)
    precision, recall, _ = precision_recall_curve(targets, probs)
    ap_score = average_precision_score(targets, probs)
    plt.plot(recall, precision, label=f'PR (AP = {ap_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # Confusion Matrix
    plt.subplot(2, 2, 3)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Probability Distribution
    plt.subplot(2, 2, 4)
    df = pd.DataFrame({'probability': probs, 'actual': targets})
    sns.histplot(data=df, x='probability', hue='actual', bins=20, alpha=0.6)
    plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Probability Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved evaluation plots to {save_path}")





