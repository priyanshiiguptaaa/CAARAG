"""
retrieval/dense_retriever.py
────────────────────────────────────────────────────────────────────────────────
Dense retriever: Sentence-BERT embeddings + FAISS IVF index.

Classes:
    DenseRetriever   — embeds corpus, builds / loads FAISS index, retrieves.

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.logger import logger


# ════════════════════════════════════════════════════════════════════════════
class DenseRetriever:
    """
    Dense retrieval via Sentence-BERT + FAISS.

    Pipeline
    ────────
    1. Encode all corpus chunks with SentenceTransformer.
    2. L2-normalise so dot-product ≡ cosine similarity.
    3. Build an IVFFlat index (scalable) or FlatIP (exact baseline).
    4. At query time: encode → normalise → search top-k.

    Args:
        model_name   : HuggingFace model id (e.g. "all-mpnet-base-v2").
        index_type   : "ivf" (default, scalable) or "flat" (exact baseline).
        nlist        : Number of IVF cells (clusters). Ignored for flat index.
        nprobe       : Cells visited at query time (higher = more accurate).
        batch_size   : Encoding batch size.
        device       : "cpu" or "cuda".
    """

    def __init__(
        self,
        model_name : str  = "sentence-transformers/all-mpnet-base-v2",
        index_type : str  = "ivf",
        nlist      : int  = 100,
        nprobe     : int  = 10,
        batch_size : int  = 64,
        device     : str  = "cpu",
    ):
        self.model_name = model_name
        self.index_type = index_type
        self.nlist      = nlist
        self.nprobe     = nprobe
        self.batch_size = batch_size
        self.device     = device

        self.model      : Optional[SentenceTransformer] = None
        self.index      : Optional[faiss.Index]         = None
        self.corpus     : List[Dict]                    = []
        self.embeddings : Optional[np.ndarray]          = None

    # ── Model loading ─────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the SentenceTransformer model."""
        if self.model is not None:
            return
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.success(f"Model loaded on device: {self.device}")

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode_corpus(
        self,
        corpus: List[Dict],
        embeddings_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode all chunks in `corpus` and return float32 embeddings.

        Saves to `embeddings_path` if provided.
        Loads from cache if file already exists.
        """
        self.corpus = corpus

        # ── Cache hit ────────────────────────────────────────────────────
        if embeddings_path and Path(embeddings_path).exists():
            logger.info(f"Loading cached embeddings from: {embeddings_path}")
            self.embeddings = np.load(embeddings_path).astype("float32")
            logger.success(f"Embeddings loaded  shape={self.embeddings.shape}")
            return self.embeddings

        # ── Encode ───────────────────────────────────────────────────────
        self.load_model()
        texts = [doc["text"] for doc in corpus]

        logger.info(
            f"Encoding {len(texts):,} chunks  "
            f"(batch_size={self.batch_size}, model={self.model_name}) …"
        )
        t0 = time.time()

        self.embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,   # we normalise manually below
        ).astype("float32")

        elapsed = time.time() - t0
        logger.success(
            f"Encoding done in {elapsed/60:.1f} min  |  "
            f"shape={self.embeddings.shape}"
        )

        # ── L2 Normalise (cosine similarity via dot product) ─────────────
        faiss.normalize_L2(self.embeddings)
        logger.info("L2 normalisation applied.")

        # ── Save ─────────────────────────────────────────────────────────
        if embeddings_path:
            Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, self.embeddings)
            logger.success(f"Embeddings saved → {embeddings_path}")

        return self.embeddings

    # ── Index building ────────────────────────────────────────────────────

    def build_index(
        self,
        embeddings  : np.ndarray,
        index_path  : Optional[str] = None,
    ) -> faiss.Index:
        """
        Build a FAISS index from pre-computed `embeddings`.

        Index types
        ───────────
        "flat" — IndexFlatIP  (exact cosine, slow on large corpus)
        "ivf"  — IndexIVFFlat (approximate, fast, scalable) ← recommended
        """
        dim = embeddings.shape[1]
        logger.info(
            f"Building FAISS index  type={self.index_type}  "
            f"dim={dim}  vectors={len(embeddings):,}"
        )

        if self.index_type == "flat":
            # ── Exact baseline ───────────────────────────────────────────
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            logger.info(f"FlatIP index  |  total={self.index.ntotal:,}")

        else:
            # ── IVF — scalable (journal-level) ───────────────────────────
            #  nlist clusters, nprobe cells searched at query time
            quantizer  = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )

            logger.info(f"Training IVF index  nlist={self.nlist} …")
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = self.nprobe
            logger.info(
                f"IVFFlat index  |  nlist={self.nlist}  "
                f"nprobe={self.nprobe}  total={self.index.ntotal:,}"
            )

        # ── Save ─────────────────────────────────────────────────────────
        if index_path:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path))
            logger.success(f"FAISS index saved → {index_path}")

        return self.index

    # ── Index loading ─────────────────────────────────────────────────────

    def load_index(self, index_path: str) -> None:
        """Load a previously saved FAISS index from disk."""
        logger.info(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(str(index_path))
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe
        logger.success(f"Index loaded  |  total={self.index.ntotal:,}")

    # ── Corpus metadata I/O ───────────────────────────────────────────────

    def save_corpus_map(self, path: str) -> None:
        """Save corpus list as a pickle (fast random access by integer id)."""
        with open(path, "wb") as f:
            pickle.dump(self.corpus, f)
        logger.success(f"Corpus id-map saved → {path}")

    def load_corpus_map(self, path: str) -> None:
        """Load corpus list from pickle."""
        with open(path, "rb") as f:
            self.corpus = pickle.load(f)
        logger.info(f"Corpus id-map loaded  |  chunks={len(self.corpus):,}")

    # ── Query ─────────────────────────────────────────────────────────────

    def encode_query(self, query: str) -> np.ndarray:
        """Encode + L2-normalise a single query string."""
        self.load_model()
        vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def retrieve(
        self,
        query : str,
        top_k : int = 5,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve top-k chunks most similar to `query`.

        Returns:
            results : List of chunk dicts from corpus
            scores  : Corresponding cosine similarity scores
        """
        assert self.index  is not None, "No FAISS index loaded. Call build_index() or load_index()."
        assert self.corpus is not None, "No corpus loaded. Call encode_corpus() or load_corpus_map()."

        query_vec  = self.encode_query(query)
        scores, idxs = self.index.search(query_vec, top_k)

        results = []
        valid_scores = []
        for idx, score in zip(idxs[0], scores[0]):
            if 0 <= idx < len(self.corpus):
                results.append(self.corpus[idx])
                valid_scores.append(float(score))

        return results, valid_scores
