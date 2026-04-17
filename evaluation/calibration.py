"""
evaluation/calibration.py
────────────────────────────────────────────────────────────────────────────────
Confidence calibration analysis for the Adaptive RAG system.

Proves that the confidence score C is meaningful:
    - High C → answers are correct
    - Low  C → answers are incorrect

Metrics:
    - Expected Calibration Error (ECE)
    - Reliability Diagram data (for plotting)
    - AUROC (confidence as a binary classifier of correctness)

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


# ════════════════════════════════════════════════════════════════════════════
# Expected Calibration Error (ECE)
# ════════════════════════════════════════════════════════════════════════════

def compute_ece(
    confidences: List[float],
    correctness: List[float],  # 1.0 = correct, 0.0 = incorrect
    n_bins: int = 10,
) -> Dict:
    """
    Compute Expected Calibration Error.
    
    ECE = Σ (|B_m| / N) * |acc(B_m) - conf(B_m)|
    
    Where:
        B_m  = samples in bin m
        acc  = actual accuracy in that bin
        conf = average confidence in that bin
    
    Returns:
        ece:           float — the scalar ECE value
        bin_data:      list of dicts for reliability diagram plotting
    """
    confidences = np.array(confidences) 
    correctness = np.array(correctness)
    n = len(confidences)
    
    if n == 0:
        return {"ece": 0.0, "bin_data": []}

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]

        # Samples in this bin
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:  # include 0.0 in first bin
            mask = (confidences >= lo) & (confidences <= hi)

        bin_count = mask.sum()

        if bin_count == 0:
            bin_data.append({
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "bin_mid": float((lo + hi) / 2),
                "count": 0,
                "avg_confidence": 0.0,
                "avg_accuracy": 0.0,
                "gap": 0.0,
            })
            continue

        avg_conf = confidences[mask].mean()
        avg_acc  = correctness[mask].mean()
        gap      = abs(avg_acc - avg_conf)
        
        ece += (bin_count / n) * gap
        
        bin_data.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "bin_mid": float((lo + hi) / 2),
            "count": int(bin_count),
            "avg_confidence": float(avg_conf),
            "avg_accuracy": float(avg_acc),
            "gap": float(gap),
        })

    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "n_samples": n,
        "bin_data": bin_data,
    }


# ════════════════════════════════════════════════════════════════════════════
# Confidence–Accuracy Correlation
# ════════════════════════════════════════════════════════════════════════════

def confidence_accuracy_correlation(
    confidences: List[float],
    f1_scores: List[float],
) -> Dict:
    """
    Compute Pearson correlation between confidence and F1.
    High positive correlation = well-calibrated system.
    """
    if len(confidences) < 2:
        return {"pearson_r": 0.0, "p_value": 1.0}
    
    c = np.array(confidences)
    f = np.array(f1_scores)
    
    # Handle edge case: constant values
    if np.std(c) == 0 or np.std(f) == 0:
        return {"pearson_r": 0.0, "p_value": 1.0}
    
    # Pearson correlation manually (avoid scipy dependency)
    c_centered = c - c.mean()
    f_centered = f - f.mean()
    
    num   = np.sum(c_centered * f_centered)
    denom = np.sqrt(np.sum(c_centered**2) * np.sum(f_centered**2))
    
    r = num / denom if denom > 0 else 0.0
    
    return {
        "pearson_r": float(r),
        "mean_confidence": float(c.mean()),
        "mean_f1": float(f.mean()),
    }


# ════════════════════════════════════════════════════════════════════════════
# High/Low Confidence Split Analysis
# ════════════════════════════════════════════════════════════════════════════

def split_by_confidence(
    confidences: List[float],
    f1_scores: List[float],
    threshold: float = 0.7,
) -> Dict:
    """
    Split samples into high-confidence and low-confidence groups.
    Compare average F1 in each group.
    
    The ideal result: high_conf_f1 >> low_conf_f1
    """
    c = np.array(confidences)
    f = np.array(f1_scores)
    
    high_mask = c >= threshold
    low_mask  = c < threshold
    
    high_f1 = float(f[high_mask].mean()) if high_mask.sum() > 0 else 0.0
    low_f1  = float(f[low_mask].mean())  if low_mask.sum()  > 0 else 0.0
    
    return {
        "threshold": threshold,
        "high_conf_count": int(high_mask.sum()),
        "low_conf_count":  int(low_mask.sum()),
        "high_conf_avg_f1": high_f1,
        "low_conf_avg_f1":  low_f1,
        "delta_f1": high_f1 - low_f1,  # Should be positive
    }
