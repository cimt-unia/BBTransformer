# data.py

# Import Essentials
import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Any, List
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d


# ======================
# DATA CREATION
# ======================


def resample_fMRI_to_standard(
    fMRI_data, 
    original_tr, 
    target_tr=2.0, 
    target_duration=300.0,  # 5 minutes
    interpolation_kind='cubic'
):
    """
    Resample fMRI data to a universal temporal standard.
    
    Args:
        fMRI_data: (N, T_original, R) array
        original_tr: float, original repetition time in seconds
        target_tr: float, target TR in seconds (default: 2.0s)
        target_duration: float, target duration in seconds (default: 300s = 5 min)
        interpolation_kind: str, interpolation method
    
    Returns:
        resampled_data: (N, target_n_timepoints, R) array
    """
    # Calculate target timepoints
    target_n = int(target_duration / target_tr)
    target_time = np.arange(target_n) * target_tr  # [0, 2, 4, ..., 298]
    
    # Original time axis for each subject
    original_time = np.arange(fMRI_data.shape[1]) * original_tr
    
    # Validate we have enough data
    if original_time[-1] < target_duration:
        print(f"⚠️  Warning: Data duration ({original_time[-1]:.1f}s) < target ({target_duration}s)")
        print("   Subjects with insufficient data will be handled based on your cohort selection.")
    
    # Resample each subject
    resampled_data = np.zeros((fMRI_data.shape[0], target_n, fMRI_data.shape[2]), dtype=np.float32)
    
    for i in range(fMRI_data.shape[0]):
        for r in range(fMRI_data.shape[2]):
            # Safe interpolation with extrapolation for edge cases
            f = interp1d(
                original_time, fMRI_data[i, :, r],
                kind=interpolation_kind,
                bounds_error=False,
                fill_value="extrapolate"
            )
            resampled_data[i, :, r] = f(target_time)
    
    return resampled_data


def create_balanced_cohort(
    fmri_path='ukbb_features_zscored.npz',
    pheno_path='ukbb_pheno.csv',
    target_column='ICD_F32_Depressive_Episode',
    control_column='neuro_healthy',
    balance=True,
    random_seed=42,
    # ✅ NEW RESAMPLING PARAMETERS
    resample_data=True,
    original_tr=0.735,      # UKB TR
    target_tr=2.0,          # Standard TR
    target_duration=300.0,  # Standard duration (5 minutes),
    standardize_after_resample=True  # ← Explicit control (recommended: True)
):
    """
    Create balanced cohort with optional temporal standardization.
    
    Resampling creates a universal temporal standard compatible with any fMRI dataset.
    If resampling is applied, data is **z-scored per subject per ROI afterward** 
    to match ABIDE preprocessing and ensure distributional compatibility.
    """
    # --- Load phenotype ---
    pheno = pd.read_csv(pheno_path, dtype={'eid': str}).set_index('eid')
    print(f"✓ Loaded phenotype: {pheno.shape}")

    # --- Load fMRI ---
    features = np.load(fmri_path)
    data = features['data'].astype(np.float32)
    subject_ids = features['subject_ids'].astype(str)
    print(f"✓ Loaded fMRI: {data.shape}")
    print(f"  Subjects: {len(subject_ids)}")
    print(f"  Timepoints: {data.shape[1]}")
    print(f"  Brain regions: {data.shape[2]}")

    # --- Validate EID alignment ---
    fmri_set = set(subject_ids)
    pheno_set = set(pheno.index)
    
    missing_in_pheno = fmri_set - pheno_set
    if missing_in_pheno:
        raise ValueError(f"{len(missing_in_pheno)} fMRI subject IDs missing in phenotype.")
    
    missing_in_fmri = pheno_set - fmri_set
    if missing_in_fmri:
        print(f"⚠️  {len(missing_in_fmri)} phenotype subjects not in fMRI (will be ignored).")

    # --- Align phenotype to fMRI order ---
    pheno = pheno.loc[subject_ids]

    # --- Validate columns ---
    for col in [target_column, control_column]:
        if col not in pheno.columns:
            raise KeyError(f"Column '{col}' not found in phenotype.")

    # --- Identify cases and controls ---
    case_mask = (pheno[target_column] == 1)
    control_mask = (pheno[control_column] == 1) & (pheno[target_column] == 0)

    case_eids = subject_ids[case_mask]
    control_eids = subject_ids[control_mask]

    n_cases, n_controls = len(case_eids), len(control_eids)
    print(f"Found {n_cases} cases, {n_controls} eligible controls")

    if n_cases == 0:
        raise ValueError(f"No cases found for '{target_column}'")
    if n_controls == 0:
        raise ValueError(f"No eligible controls found")

    # --- Balance cohorts ---
    if balance:
        np.random.seed(random_seed)
        if n_controls > n_cases:
            control_eids = np.random.choice(control_eids, size=n_cases, replace=False)
        elif n_cases > n_controls:
            case_eids = np.random.choice(case_eids, size=n_controls, replace=False)

    # --- Combine and shuffle ---
    final_eids = np.concatenate([case_eids, control_eids])
    labels = np.array([1] * len(case_eids) + [0] * len(control_eids), dtype=np.float32)

    np.random.seed(random_seed)
    shuffle_idx = np.random.permutation(len(final_eids))
    final_eids = final_eids[shuffle_idx]
    labels = labels[shuffle_idx]

    # --- Extract fMRI in correct order ---
    eid_to_idx = {eid: i for i, eid in enumerate(subject_ids)}
    final_indices = np.array([eid_to_idx[eid] for eid in final_eids])
    data_final = data[final_indices]

    # --- ✅ Apply temporal standardization + post-resample z-scoring ---
    if resample_data:
        print(f"🔄 Resampling to universal standard:")
        print(f"   Original: TR={original_tr}s, Duration={data.shape[1] * original_tr:.1f}s")
        print(f"   Target:   TR={target_tr}s, Duration={target_duration}s ({target_duration/60:.0f} min)")
        print(f"   Output:   {int(target_duration/target_tr)} timepoints")
        
        data_final = resample_fMRI_to_standard(
            data_final,
            original_tr=original_tr,
            target_tr=target_tr,
            target_duration=target_duration
        )
        print(f"✅ Resampled shape: {data_final.shape}")

        # ✅ CRITICAL: Standardize AFTER resampling — per subject, per ROI
        if standardize_after_resample:
            print("📊 Standardizing after resampling (per subject, per ROI)...")
            for i in range(data_final.shape[0]):
                roi_mean = data_final[i].mean(axis=0, keepdims=True)
                roi_std = data_final[i].std(axis=0, keepdims=True) + 1e-8
                data_final[i] = (data_final[i] - roi_mean) / roi_std
            print("✅ Standardization complete.")

    # --- Extract full phenotype WITH eid as INDEX ---
    pheno_final = pheno.loc[final_eids].copy()
    pheno_final['label'] = labels

    # --- Final validation ---
    assert len(data_final) == len(labels) == len(final_eids) == len(pheno_final)
    assert list(pheno_final.index) == list(final_eids)

    print(f"✓ Final dataset: {len(final_eids)} subjects "
          f"({len(case_eids)} cases, {len(control_eids)} controls)")

    return data_final, labels, final_eids, pheno_final


