"""
evaluation/adaptation_analysis.py
────────────────────────────────────────────────────────────────────────────────
Measures the effectiveness of the adaptive retrieval loop.

Proves: "Adaptation Helps" by comparing Round 1 vs Final Round metrics.

Analysis:
    1. Confidence Lift:      C_final - C_round1  (should be positive)
    2. Adaptation Rate:      % of queries that required >1 round
    3. Convergence Rate:     % of queries that crossed tau after adaptation
    4. Per-Round Statistics:  avg Sr, Sl, Sc, C per round number

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any


def analyse_adaptation(results: List[Dict[str, Any]], tau: float = 0.7) -> Dict:
    """
    Comprehensive adaptation effectiveness analysis.
    
    Args:
        results: List of per-query adaptive results (from adaptive_results.json)
        tau:     Confidence threshold used by the system
    
    Returns:
        Dict with all analysis results for paper tables/figures.
    """
    n = len(results)
    if n == 0:
        return {}
    
    # ── 1. Adaptation Rate ──────────────────────────────────────────────
    rounds_counts = [r["total_rounds"] for r in results]
    adapted       = [1 for c in rounds_counts if c > 1]
    adaptation_rate = len(adapted) / n
    
    # ── 2. Confidence Lift ──────────────────────────────────────────────
    # Compare Round 1 confidence vs final confidence
    lifts       = []
    r1_confs    = []
    final_confs = []
    
    for r in results:
        rounds = r.get("rounds", [])
        if not rounds:
            continue
        c1 = rounds[0]["C"]
        cf = rounds[-1]["C"]
        r1_confs.append(c1)
        final_confs.append(cf)
        lifts.append(cf - c1)
    
    avg_lift = float(np.mean(lifts)) if lifts else 0.0
    
    # ── 3. Convergence Rate ─────────────────────────────────────────────
    # Queries that started below tau but ended above tau
    rescued = 0
    started_below = 0
    for r in results:
        rounds = r.get("rounds", [])
        if not rounds:
            continue
        if rounds[0]["C"] < tau:
            started_below += 1
            if rounds[-1]["C"] >= tau:
                rescued += 1
    
    convergence_rate = rescued / started_below if started_below > 0 else 0.0
    
    # ── 4. Per-Round Signal Averages ─────────────────────────────────────
    max_round = max(rounds_counts)
    per_round = {}
    for rnd in range(1, max_round + 1):
        srs, sls, scs, cs = [], [], [], []
        for r in results:
            rounds = r.get("rounds", [])
            for rd in rounds:
                if rd["round"] == rnd:
                    srs.append(rd["Sr"])
                    sls.append(rd["Sl"])
                    scs.append(rd["Sc"])
                    cs.append(rd["C"])
        if cs:
            per_round[f"round_{rnd}"] = {
                "count":  len(cs),
                "avg_Sr": float(np.mean(srs)),
                "avg_Sl": float(np.mean(sls)),
                "avg_Sc": float(np.mean(scs)),
                "avg_C":  float(np.mean(cs)),
            }
    
    # ── 5. Distribution of Rounds ────────────────────────────────────────
    from collections import Counter
    round_dist = dict(Counter(rounds_counts))
    
    return {
        "total_queries": n,
        "tau": tau,
        
        # Adaptation Rate
        "adaptation_rate": adaptation_rate,
        "queries_with_adaptation": len(adapted),
        "queries_single_round": n - len(adapted),
        
        # Confidence Lift
        "avg_confidence_lift": avg_lift,
        "avg_round1_confidence": float(np.mean(r1_confs)) if r1_confs else 0.0,
        "avg_final_confidence": float(np.mean(final_confs)) if final_confs else 0.0,
        
        # Convergence
        "queries_started_below_tau": started_below,
        "queries_rescued": rescued,
        "convergence_rate": convergence_rate,
        
        # Per-round breakdown
        "avg_rounds": float(np.mean(rounds_counts)),
        "max_rounds": max_round,
        "round_distribution": round_dist,
        "per_round_signals": per_round,
    }
