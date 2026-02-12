# bbtransformer\trainer\train.py

# Import Essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, f1_score
from pytorch_optimizer import Ranger21
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Any, List

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ======================
# TRAIN FUNCTION
# ======================
class AverageMeter:
    def __init__(self): 
        self.reset()
    
    def reset(self): 
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10000,
    lr=3e-4,
    weight_decay=1e-4,
    patience=100,
    T_0=50,
    eta_min=1e-6,
    loss_type=None,           # None → legacy BCE; or 'bce', 'focal', 'adaptive_focal'
    loss_params=None,         # dict: e.g., {'alpha': 0.75, 'gamma': 2.0, 'pos_weight': 2.0}
    early_stop_metric="f1"  # 'loss', 'auc', or 'f1'
):
    # Train BBTransformer model with flexible loss and early stopping

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # ✅ Use Ranger21
    total_steps = epochs * len(train_loader)
    optimizer = Ranger21(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=epochs,
        num_iterations=total_steps
    )

    # --- Loss function ---
    if loss_type is None:
        # Legacy behavior: plain BCE
        criterion = nn.BCEWithLogitsLoss()
    else:
        if loss_params is None:
            loss_params = {}
        if loss_type == "bce":
            pos_weight = loss_params.get("pos_weight", None)
            if pos_weight is not None:
                pos_weight = torch.tensor([pos_weight], device=device, dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif loss_type == "focal":
            alpha = loss_params.get("alpha", 0.25)
            gamma = loss_params.get("gamma", 2.0)
            criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        elif loss_type == "adaptive_focal":
            gamma = loss_params.get("gamma", 2.0)
            criterion = AdaptiveFocalLoss(gamma=gamma, reduction='mean')
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    scaler = GradScaler()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)

    # Early stopping tracking
    best_val_loss = float('inf')
    best_val_score = -float('inf')  # for AUC/F1
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = AverageMeter()
        
        for fmri, age, ext, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            fmri = fmri.to(device)
            age = age.to(device)
            ext = ext.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=device.type):
                logits = model(fmri, age, ext)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss.update(loss.item(), fmri.size(0))

        # Validation
        model.eval()
        val_loss = AverageMeter()
        val_logits_list = []
        val_labels_list = []
        
        with torch.no_grad():
            for fmri, age, ext, labels in val_loader:
                fmri = fmri.to(device)
                age = age.to(device)
                ext = ext.to(device)
                labels = labels.to(device)
                
                with autocast(device_type=device.type):
                    logits = model(fmri, age, ext)
                    loss = criterion(logits, labels)
                
                val_loss.update(loss.item(), fmri.size(0))
                
                # Only collect logits/labels if needed for AUC/F1
                if early_stop_metric in ("auc", "f1"):
                    val_logits_list.append(logits.cpu())
                    val_labels_list.append(labels.cpu())
        
        scheduler.step()

        # Determine early stopping score
        if early_stop_metric == "loss":
            current_score = val_loss.avg
            is_better = current_score < best_val_loss
            if is_better:
                best_val_loss = current_score
        else:
            # Compute AUC or F1
            val_logits = torch.cat(val_logits_list)
            val_labels = torch.cat(val_labels_list)
            val_probs = torch.sigmoid(val_logits).numpy()
            val_labels_np = val_labels.numpy()
            
            if early_stop_metric == "auc":
                try:
                    score = roc_auc_score(val_labels_np, val_probs)
                except ValueError:  # e.g., only one class in val
                    score = 0.5
            elif early_stop_metric == "f1":
                preds = (val_probs > 0.5).astype(int)
                score = f1_score(val_labels_np, preds, zero_division=0)
            else:
                raise ValueError(f"Unknown early_stop_metric: {early_stop_metric}")
            
            current_score = score
            is_better = current_score > best_val_score
            if is_better:
                best_val_score = current_score

        # Early stopping logic
        if is_better:
            best_model_state = {k: v.cpu() for k, v in model.state_dict().copy().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} "
                      f"({early_stop_metric}: {current_score:.4f})")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    return model



# ======================
#  LOSS 
# ======================

class FocalLoss(nn.Module):
    """Numerically stable Focal Loss for imbalanced binary classification."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(inputs)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        alpha_factor = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = torch.where(targets == 1, 1 - pred, pred)
        focal_weight = alpha_factor * (focal_weight ** self.gamma)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        focal_loss = focal_weight * bce
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss

class AdaptiveFocalLoss(nn.Module):
    """Adaptive Focal Loss with dynamic alpha."""
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_pos = targets.sum()
        n_total = targets.numel()
        alpha = 1.0 - (n_pos / n_total) if n_total > 0 else 0.5
        pred = torch.sigmoid(inputs)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        alpha_factor = torch.where(targets == 1, alpha, 1.0 - alpha)
        focal_weight = torch.where(targets == 1, 1 - pred, pred)
        focal_weight = alpha_factor * (focal_weight ** self.gamma)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        focal_loss = focal_weight * bce
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss




