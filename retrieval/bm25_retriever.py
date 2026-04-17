"""
retrieval/bm25_retriever.py
────────────────────────────────────────────────────────────────────────────────
Sparse BM25 retriever using rank_bm25.

BM25 is a classic term-frequency / inverse-document-frequency retrieval model.
It complements dense retrieval especially for:
  • Exact keyword matches
  • Rare named entities
  • Multi-hop questions where specific terms matter

Classes:
    BM25Retriever — builds BM25 index, retrieves, saves/loads with pickle.

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm

from utils.logger import logger


# ════════════════════════════════════════════════════════════════════════════
class BM25Retriever:
    """
    Sparse BM25 retriever backed by rank_bm25.BM25Okapi.

    Args:
        k1 : BM25 term saturation parameter (default 1.5).
        b  : BM25 length normalisation parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1      = k1
        self.b       = b
        self.bm25    = None
        self.corpus  : List[Dict] = []
        self._tokenised: List[List[str]] = []

    # ── Tokeniser ─────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """
        Lightweight whitespace tokeniser with minimal cleaning.
        Preserves important tokens (numbers, hyphenated words, etc.).
        """
        text   = text.lower()
        tokens = re.findall(r"\b[\w'-]+\b", text)
        return tokens

    # ── Index Building ────────────────────────────────────────────────────

    def build_index(
        self,
        corpus    : List[Dict],
        save_path : Optional[str] = None,
    ) -> None:
        """
        Tokenise all texts and build a BM25Okapi index.

        Args:
            corpus    : List of chunk dicts (must have a "text" key).
            save_path : If given, pickle the BM25 index to this path.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Run: pip install rank_bm25")

        self.corpus = corpus
        logger.info(f"Tokenising {len(corpus):,} chunks for BM25 …")
        t0 = time.time()

        self._tokenised = [
            self._tokenise(doc["text"])
            for doc in tqdm(corpus, desc="Tokenising (BM25)", unit="chunk")
        ]

        logger.info("Training BM25Okapi …")
        self.bm25 = BM25Okapi(self._tokenised, k1=self.k1, b=self.b)

        elapsed = time.time() - t0
        logger.success(f"BM25 index built in {elapsed:.1f}s")

        if save_path:
            self.save(save_path)

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the BM25 object + corpus to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25"      : self.bm25,
            "corpus"    : self.corpus,
            "tokenised" : self._tokenised,
            "k1"        : self.k1,
            "b"         : self.b,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.success(f"BM25 index saved → {path}")

    def load(self, path: str) -> None:
        """Load a previously saved BM25 index from disk."""
        logger.info(f"Loading BM25 index from: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.bm25       = payload["bm25"]
        self.corpus     = payload["corpus"]
        self._tokenised = payload["tokenised"]
        self.k1         = payload["k1"]
        self.b          = payload["b"]
        logger.success(f"BM25 index loaded  |  chunks={len(self.corpus):,}")

    # ── Query ─────────────────────────────────────────────────────────────

    def get_scores(self, query: str) -> np.ndarray:
        """Return raw BM25 scores for all corpus chunks."""
        assert self.bm25 is not None, "BM25 index not built. Call build_index() first."
        tokens = self._tokenise(query)
        return self.bm25.get_scores(tokens).astype("float32")

    def retrieve(
        self,
        query  : str,
        top_k  : int = 5,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve top-k chunks by BM25 score.

        Returns:
            results : List of chunk dicts.
            scores  : Corresponding raw BM25 scores.
        """
        all_scores = self.get_scores(query)
        top_idxs   = np.argsort(all_scores)[::-1][:top_k]

        results = [self.corpus[i]       for i in top_idxs]
        scores  = [float(all_scores[i]) for i in top_idxs]

        return results, scores
