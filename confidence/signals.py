"""
confidence/signals.py
────────────────────────────────────────────────────────────────────────────────
Compute the three core confidence signals:
    Sr (Retrieval), Sl (LLM), Sc (Consistency).

Sc Upgrade: Uses SBERT embeddings for semantic similarity.

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from utils.logger import logger


def compute_Sr(scores: List[float]) -> float:
    """
    Retrieval Confidence (Sr).
    Measures the strength and stability of the top-k retrieved documents.
    
    Formula: mean(scores) - std(scores)
    
    Standardizes on the metric that clear, uniform high scores are best.
    """
    if not scores:
        return 0.0
    
    arr  = np.array(scores)
    mean = np.mean(arr)
    std  = np.std(arr)
    
    # Penalize high variance (unstable retrieval)
    sr = max(0.0, mean - std)
    return float(sr)


def compute_Sl(samples: List[Dict[str, Any]]) -> float:
    """
    LLM Self-Confidence (Sl).
    Extracted directly from the structured generation output.
    
    Takes the confidence from the first (or best) sample.
    """
    if not samples:
        return 0.0
    
    # Use the first sample's reported confidence
    # (Step 3 parser already extracted this as a float)
    conf = samples[0].get("confidence", 0.0)
    return float(min(max(conf, 0.0), 1.0))


def compute_Sc(answers: List[str], model: SentenceTransformer) -> float:
    """
    Semantic Consistency Score (Sc).
    Computes the average pairwise cosine similarity between n answers.
    
    Upgrade: Uses SBERT embeddings for semantic overlap (not just tokens).
    """
    if len(answers) <= 1:
        return 1.0 if answers else 0.0
    
    # ── 1. Embed all answers ──────────────────────────────────────────────
    embeddings = model.encode(answers, convert_to_numpy=True)
    
    # ── 2. Pairwise Cosine Similarity ────────────────────────────────────
    # Normalize for dot product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)
    
    sim_matrix = np.dot(embeddings, embeddings.T)
    
    # ── 3. Average off-diagonal elements ─────────────────────────────────
    n = len(answers)
    total_sim = np.sum(sim_matrix) - n # subtract diagonal (1s)
    avg_sim   = total_sim / (n * (n - 1))
    
    return float(max(0.0, avg_sim))
