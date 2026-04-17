"""
retrieval/hybrid_retriever.py
────────────────────────────────────────────────────────────────────────────────
Hybrid retrieval: Reciprocal Rank Fusion of Dense (FAISS) + Sparse (BM25).

Two fusion strategies are implemented:

1. RRF (Reciprocal Rank Fusion) — rank-based, parameter-free, robust.
   score_rrf(d) = Σ 1 / (k + rank_i(d))         [k = 60 by convention]

2. Linear score fusion — normalise both score vectors to [0,1], then:
   score(d) = alpha * dense_norm + (1-alpha) * bm25_norm

   • alpha = 0 → pure BM25
   • alpha = 1 → pure dense
   • alpha ≈ 0.6–0.7 → typical optimum for multi-hop QA

👉  RRF is recommended in the paper because it avoids tuning alpha.

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np

from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25_retriever  import BM25Retriever
from utils.logger import logger


# ════════════════════════════════════════════════════════════════════════════
class HybridRetriever:
    """
    Hybrid retriever combining DenseRetriever + BM25Retriever.

    Args:
        dense  : A loaded DenseRetriever instance.
        bm25   : A loaded BM25Retriever  instance.
        alpha  : Weight for dense score in linear fusion  (0–1).
        rrf_k  : RRF constant (default 60).
        method : "rrf" (default) | "linear".
    """

    def __init__(
        self,
        dense  : DenseRetriever,
        bm25   : BM25Retriever,
        alpha  : float = 0.65,
        rrf_k  : int   = 60,
        method : str   = "rrf",
    ):
        self.dense  = dense
        self.bm25   = bm25
        self.alpha  = alpha
        self.rrf_k  = rrf_k
        self.method = method

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _normalise(scores: np.ndarray) -> np.ndarray:
        """Min-max normalise scores to [0, 1]."""
        mn, mx = scores.min(), scores.max()
        if mx == mn:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)

    @staticmethod
    def _rrf_scores(ranked_ids: List[int], n: int, k: int) -> np.ndarray:
        """
        Given an ordered list of document indices, return RRF score array
        of length `n` (total corpus size).
        """
        scores = np.zeros(n, dtype="float32")
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] += 1.0 / (k + rank + 1)
        return scores

    # ── Retrieve ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query      : str,
        top_k      : int = 5,
        dense_top  : int = 100,   # candidate pool from each retriever
        bm25_top   : int = 100,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve top-k chunks using hybrid fusion.

        Strategy
        ────────
        1. Get a wide candidate pool from both retrievers (dense_top + bm25_top).
        2. Fuse scores via RRF or linear combination.
        3. Return top-k unique chunks after fusion.

        Args:
            query     : Natural-language question.
            top_k     : Final number of chunks to return.
            dense_top : How many to fetch from dense retriever.
            bm25_top  : How many to fetch from BM25 retriever.

        Returns:
            results : List of chunk dicts (length=top_k).
            scores  : Corresponding fused scores.
        """
        corpus = self.dense.corpus  # ground truth for integer indices

        # ── Dense retrieval ───────────────────────────────────────────────
        dense_results, dense_scores = self.dense.retrieve(query, top_k=dense_top)

        # ── BM25 retrieval ────────────────────────────────────────────────
        bm25_results, bm25_scores  = self.bm25.retrieve(query, top_k=bm25_top)

        # ── Map chunk_id → corpus index ───────────────────────────────────
        cid_to_idx = {doc["chunk_id"]: i for i, doc in enumerate(corpus)}
        n          = len(corpus)

        if self.method == "rrf":
            # ── Reciprocal Rank Fusion ────────────────────────────────────
            dense_ids = [cid_to_idx[d["chunk_id"]] for d in dense_results if d["chunk_id"] in cid_to_idx]
            bm25_ids  = [cid_to_idx[d["chunk_id"]] for d in bm25_results  if d["chunk_id"] in cid_to_idx]

            rrf_dense = self._rrf_scores(dense_ids, n, self.rrf_k)
            rrf_bm25  = self._rrf_scores(bm25_ids,  n, self.rrf_k)
            fused     = rrf_dense + rrf_bm25

        else:
            # ── Linear score fusion ───────────────────────────────────────
            dense_score_map = {d["chunk_id"]: s for d, s in zip(dense_results, dense_scores)}
            bm25_score_map  = {d["chunk_id"]: s for d, s in zip(bm25_results,  bm25_scores)}

            all_cids = set(dense_score_map) | set(bm25_score_map)
            fused    = np.zeros(n, dtype="float32")
            for cid in all_cids:
                idx = cid_to_idx.get(cid)
                if idx is None:
                    continue
                d_s = dense_score_map.get(cid, 0.0)
                b_s = bm25_score_map.get(cid, 0.0)
                fused[idx] = self.alpha * d_s + (1 - self.alpha) * b_s

        # ── Rank and pick top-k ───────────────────────────────────────────
        top_idxs = np.argsort(fused)[::-1][:top_k]
        results  = [corpus[i]        for i in top_idxs if fused[i] > 0]
        scores   = [float(fused[i])  for i in top_idxs if fused[i] > 0]

        return results[:top_k], scores[:top_k]
