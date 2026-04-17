"""
adaptive/diagnoser.py
────────────────────────────────────────────────────────────────────────────────
Signal-aware diagnostic controller.
Decides the "Next Action" based on which signal is the bottleneck.

Diagnostic Branching Logic:
    If Sr < 0.5 → Action: Expand k (fetch more context)
    If Sl < 0.5 → Action: Reformulate Query (LLM confusion)
    If Sc < 0.5 → Action: Increase Sampling (LLM consistency/reasoning gap)
    Else        → Action: Stop (High confidence)

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

from typing import Dict, Tuple

from utils.logger import logger


# ════════════════════════════════════════════════════════════════════════════
class AdaptiveDiagnoser:
    """
    Diagnostic Logic Layer (Signal → Targeted Fix).
    
    This is what transforms the system from 'retry-based' to 'diagnostic-aware'.
    """

    def __init__(self, config: Dict):
        self.a_cfg = config.get("adaptive", {})
        self.tau_sr = self.a_cfg.get("threshold_sr", 0.5)
        self.tau_sl = self.a_cfg.get("threshold_sl", 0.5)
        self.tau_sc = self.a_cfg.get("threshold_sc", 0.5)

    # ── Diagnostic Decision ────────────────────────────────────────────────
    
    def diagnose(self, sr: float, sl: float, sc: float) -> str:
        """
        Takes the three signals and returns a string identifying the 'Next Action'.
        """
        # 1. Retrieval Diagnosis (Bottleneck: Doc Coverage)
        if sr < self.tau_sr:
            logger.info(f"Diagnosis: Poor Retrieval (Sr={sr:.3f}) → Triggering k-expansion.")
            return "expand_k"
        
        # 2. LLM Intent Diagnosis (Bottleneck: Query-Doc Alignment)
        if sl < self.tau_sl:
            logger.info(f"Diagnosis: LLM Self-Doubt (Sl={sl:.3f}) → Triggering Query Reformulation.")
            return "reformulate"
        
        # 3. Consistency Diagnosis (Bottleneck: Multi-Sample Ambiguity)
        if sc < self.tau_sc:
            logger.info(f"Diagnosis: Answer Inconsistency (Sc={sc:.3f}) → Triggering Higher Sampling.")
            return "resample"
        
        # 4. Stop Case
        logger.info(f"Diagnosis: All signals exceed thresholds. Stopping.")
        return "stop"
