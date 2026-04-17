"""
evaluation/answer_metrics.py
────────────────────────────────────────────────────────────────────────────────
Journal-grade answer quality metrics for QA evaluation.

Metrics:
    - Exact Match (EM)   : Binary — did we get the exact answer?
    - Token F1           : Precision/Recall/F1 over answer tokens.
    - ROUGE-L            : Longest Common Subsequence overlap.

All metrics follow the SQuAD / HotpotQA evaluation conventions:
    - Lowercase + strip whitespace.
    - Remove articles (a, an, the) and punctuation.

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Tuple


# ════════════════════════════════════════════════════════════════════════════
# Text Normalisation (SQuAD-style)
# ════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Lowercase, strip, remove articles and punctuation."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def _get_tokens(text: str) -> List[str]:
    return _normalize(text).split()


# ════════════════════════════════════════════════════════════════════════════
# Exact Match
# ════════════════════════════════════════════════════════════════════════════

def exact_match(prediction: str, gold: str) -> float:
    """Binary: 1.0 if normalised strings match, else 0.0."""
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


# ════════════════════════════════════════════════════════════════════════════
# Token-level F1 (SQuAD style)
# ════════════════════════════════════════════════════════════════════════════

def token_f1(prediction: str, gold: str) -> Dict[str, float]:
    """
    Compute token-level Precision, Recall, and F1.
    Returns dict with keys: precision, recall, f1.
    """
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(gold)

    if not gold_tokens and not pred_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold_tokens or not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    f1        = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# ════════════════════════════════════════════════════════════════════════════
# ROUGE-L (Longest Common Subsequence)
# ════════════════════════════════════════════════════════════════════════════

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute LCS length with O(min(m,n)) space."""
    m, n = len(x), len(y)
    if m < n:
        x, y = y, x
        m, n = n, m

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(prediction: str, gold: str) -> Dict[str, float]:
    """
    ROUGE-L F1 based on longest common subsequence.
    Returns dict with keys: precision, recall, f1.
    """
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(gold)

    if not gold_tokens and not pred_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold_tokens or not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(pred_tokens, gold_tokens)

    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall    = lcs / len(gold_tokens) if gold_tokens else 0.0

    if precision + recall == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ════════════════════════════════════════════════════════════════════════════
# Batch evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate_batch(predictions: List[str], golds: List[str]) -> Dict[str, float]:
    """
    Compute aggregate metrics over a list of (prediction, gold) pairs.
    Returns:
        avg_em, avg_f1, avg_rouge_l_f1, avg_precision, avg_recall
    """
    n = len(predictions)
    assert n == len(golds), "Prediction and gold lists must be the same length."

    total_em       = 0.0
    total_f1       = 0.0
    total_rouge_f1 = 0.0
    total_prec     = 0.0
    total_rec      = 0.0

    per_query = []

    for pred, gold in zip(predictions, golds):
        em  = exact_match(pred, gold)
        f1d = token_f1(pred, gold)
        rld = rouge_l(pred, gold)

        total_em       += em
        total_f1       += f1d["f1"]
        total_rouge_f1 += rld["f1"]
        total_prec     += f1d["precision"]
        total_rec      += f1d["recall"]

        per_query.append({
            "em": em,
            "f1": f1d["f1"],
            "precision": f1d["precision"],
            "recall": f1d["recall"],
            "rouge_l_f1": rld["f1"],
        })

    return {
        "n": n,
        "avg_em":         total_em / n,
        "avg_f1":         total_f1 / n,
        "avg_precision":  total_prec / n,
        "avg_recall":     total_rec / n,
        "avg_rouge_l_f1": total_rouge_f1 / n,
        "per_query": per_query,
    }
