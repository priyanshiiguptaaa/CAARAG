"""
confidence/calibrator.py
────────────────────────────────────────────────────────────────────────────────
Calibrates raw signals (Sr, Sl, Sc) into a final confidence score (C).

Calibrated Fusion (Journal Upgrade):
    z = w1*Sr + w2*Sl + w3*Sc
    C = 1 / (1 + e^-(a*z + b))

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

from utils.logger import logger


# ════════════════════════════════════════════════════════════════════════════
class Calibrator:
    """
    Handles weighted fusion and sigmoid calibration.
    
    Calibration parameters a and b are loaded from config.yaml.
    """
    
    def __init__(self, config: Dict):
        self.c_cfg  = config.get("confidence", {})
        self.a      = self.c_cfg.get("a", 10.0)
        self.b      = self.c_cfg.get("b", -5.0)
        
        # Weights
        self.w_sr   = self.c_cfg.get("weight_retrieval",   0.3)
        self.w_sl   = self.c_cfg.get("weight_llm",         0.4)
        self.w_sc   = self.c_cfg.get("weight_consistency", 0.3)
        
        # Normalize weights to sum (ensure they do)
        total       = self.w_sr + self.w_sl + self.w_sc
        if abs(total - 1.0) > 1e-5:
            logger.warning(f"Confidence weights sum to {total:.3f} and will be normalized.")
            self.w_sr /= total
            self.w_sl /= total
            self.w_sc /= total

    # ── Calibration Logic ────────────────────────────────────────────────
    
    def weighted_fusion(self, sr: float, sl: float, sc: float) -> float:
        """Combine the three signals into a raw z-score."""
        z = (self.w_sr * sr) + (self.w_sl * sl) + (self.w_sc * sc)
        return float(z)

    def calibrate(self, z: float) -> float:
        """
        Sigmoid Calibration Bridge.
        Maps the raw combined score z to a calibrated probability C (0.0 to 1.0).
        """
        # Sigmoid: 1 / (1 + exp(-(a*z + b)))
        c = 1.0 / (1.0 + np.exp(-(self.a * z + self.b)))
        return float(c)

    def get_calibrated_confidence(self, sr: float, sl: float, sc: float) -> Tuple[float, float]:
        """Returns the raw z and the calibrated C."""
        z = self.weighted_fusion(sr, sl, sc)
        c = self.calibrate(z)
        return z, c
