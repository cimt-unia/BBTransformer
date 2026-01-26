# interpret.py


# Import Essentials
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.amp import autocast 
from eval import evaluate_model

# ======================
# PERMUTATION IMPORTANCE
# ======================

def load_roi_names(csv_path='roi_labels.csv'):
    """Load ROI names from CSV in correct order (index = feature dim)."""
    df = pd.read_csv(csv_path)
    assert df['index'].iloc[0] == 0 and df['index'].iloc[-1] == 413, "CSV index mismatch"
    assert len(df) == 414, f"Expected 414 ROIs, got {len(df)}"
    print(f"✅ Loaded 414 ROI labels from {csv_path}")
    return df['roi_name'].values  # array of strings, index-aligned


def calculate_permutation_importance(
    model, 
    data_loader, 
    feature_dim, 
    n_repeats=10, 
    seed=42,
    target_name="target"
):
    """
    Calculate permutation importance for brain regions (memory-efficient)
    
    Args:
        model: Trained BBTransformer
        data_loader: DataLoader with validation data
        feature_dim: Number of brain regions (e.g., 414)
        n_repeats: Number of permutation repeats
        seed: Random seed for reproducibility
        target_name: Name of target variable (used in CSV column name)
    
    Returns:
        importance_scores: Array of importance scores (length = feature_dim)
    """
    import torch
    import numpy as np
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = next(model.parameters()).device
    baseline_metrics, _, _ = evaluate_model(model, data_loader)
    baseline_score = baseline_metrics['f1']
    
    importance_scores = np.zeros(feature_dim)
    print(f"Calculating permutation importance for {feature_dim} brain regions...")

    for region_idx in tqdm(range(feature_dim), desc="Permuting regions"):
        region_scores = []
        for _ in range(n_repeats):
            all_preds, all_probs, all_targets = [], [], []
            for fmri, age, ext, labels in data_loader:
                fmri = fmri.to(device)
                age = age.to(device)
                ext = ext.to(device)
                
                # Permute the region across subjects *within the current batch*
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
        
        importance_scores[region_idx] = baseline_score - np.mean(region_scores)

    return importance_scores


def save_top_importance_to_csv(
    importance_scores,
    roi_names=None,
    target_name="target",
    top_n=30,
    save_path=None
):
    """
    Save top permutation importance scores to CSV (no plots, no numpy)
    
    Args:
        importance_scores: Array of importance scores (length = 414)
        roi_names: Array of ROI names (length = 414)
        target_name: Name of target variable (e.g., 'Sex', 'ICD_F32_Depressive_Episode')
        top_n: Number of top regions to save
        save_path: Output CSV path (e.g., 'importance_Sex.csv')
    
    Returns:
        pd.DataFrame: Full importance table (all 414 regions, sorted)
    """
    assert len(importance_scores) == 414, "Expected 414 regions"
    
    # Create full DataFrame
    df = pd.DataFrame({
        'roi_index': np.arange(414),
        'roi_name': roi_names if roi_names is not None else [f'Region_{i}' for i in range(414)],
        f'importance_{target_name}': importance_scores
    })
    
    # Sort by importance (descending)
    df = df.sort_values(f'importance_{target_name}', ascending=False).reset_index(drop=True)
    
    # Save top N to CSV
    if save_path is None:
        save_path = f'importance_{target_name}.csv'
    
    df.head(top_n).to_csv(save_path, index=False)
    print(f"✅ Saved top {top_n} features to: {save_path}")
    
    return df


def plot_brain_importance(
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
    plt.xlabel('Permutation Importance (Δ F1 Score)')
    plt.ylabel('Brain Region')
    plt.title(f'Top {top_n} Most Important Brain Regions')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved plot to {save_path}")


# ======================
# PREDICTION & INTERPRETATION
# ======================
class BrainPredictor:
    """
    Single subject prediction with interpretation
    """
    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata
        self.device = next(model.parameters()).device
        self.model.eval()
        
        # Validate that age normalization stats are available
        if 'age_mean' not in metadata or 'age_std' not in metadata:
            raise ValueError(
                "Metadata must contain 'age_mean' and 'age_std' from prepare_fmri_data. "
                "Please ensure you're using the updated prepare_fmri_data function."
            )

    def predict_single(self, fmri, age, ext):
        """
        Make prediction for a single subject
        Args:
            fmri: (490, 414) numpy array — assumed already z-scored per subject
            age: float — raw age value (e.g., 56.3)
            ext: int (0 or 1)
        Returns:
            Dictionary with prediction and confidence
        """
        # Normalize age using TRAIN SET statistics
        age_norm = (age - self.metadata['age_mean']) / self.metadata['age_std']
        
        # Convert to tensors
        fmri_tensor = torch.from_numpy(fmri).unsqueeze(0).float().to(self.device)
        age_tensor = torch.tensor([age_norm], dtype=torch.float32).to(self.device)
        ext_tensor = torch.tensor([ext], dtype=torch.long).to(self.device)

        with torch.no_grad(), autocast(device_type=self.device.type):
            logits = self.model(fmri_tensor, age_tensor, ext_tensor)
            prob = torch.sigmoid(logits).item()

        prediction = 1 if prob > 0.5 else 0
        confidence = abs(prob - 0.5) * 2  # Scale to [0, 1]

        return {
            'prediction': prediction,
            'probability': prob,
            'confidence': confidence,
            'interpretation': self._interpret_prediction(prob, confidence)
        }

    def _interpret_prediction(self, prob, confidence):
        """Generate human-readable interpretation"""
        if confidence < 0.3:
            strength = "LOW CONFIDENCE"
            recommendation = "Prediction uncertain - recommend additional assessment"
        elif confidence < 0.6:
            strength = "MODERATE CONFIDENCE"
            recommendation = "Fair confidence in prediction"
        else:
            strength = "HIGH CONFIDENCE"
            recommendation = "Strong confidence in prediction"
        direction = "POSITIVE" if prob > 0.5 else "NEGATIVE"
        return {
            'direction': direction,
            'strength': strength,
            'recommendation': recommendation
        }

    def predict_batch(self, data_loader):
        """
        Make predictions for a batch of subjects
        Returns:
            Dictionary with predictions, probabilities, and metrics
        """
        all_preds = []
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for fmri, age, ext, labels in tqdm(data_loader, desc="Predicting"):
                fmri = fmri.to(self.device)
                age = age.to(self.device)
                ext = ext.to(self.device)
                with autocast(device_type=self.device.type):
                    logits = self.model(fmri, age, ext)
                    probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())
        return {
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probs),
            'targets': np.array(all_targets)
        }


