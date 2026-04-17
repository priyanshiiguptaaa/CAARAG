"""
step2_indexing.py
────────────────────────────────────────────────────────────────────────────────
STEP 2 — Embedding + FAISS Indexing (Journal-Level Implementation)
────────────────────────────────────────────────────────────────────────────────

Pipeline:
    2.1  Load embedding model (all-mpnet-base-v2 — best quality for journal)
    2.2  Batch-encode all corpus chunks → float32 embeddings
    2.3  L2-normalise embeddings (cosine similarity = dot product)
    2.4  Build FAISS IVFFlat index (scalable, paper-worthy)
    2.5  Build BM25 sparse index (for hybrid retrieval)
    2.6  Save all artefacts  (faiss_index.bin, embeddings.npy, id_map.pkl, bm25.pkl)
    2.7  Evaluate retrieval with a small held-out query set from validation data
    2.8  Print & save retrieval evaluation stats

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from utils.logger          import logger
from utils.config_loader   import load_config, get_corpus_config, get_retrieval_config
from retrieval.dense_retriever  import DenseRetriever
from retrieval.bm25_retriever   import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever


# ════════════════════════════════════════════════════════════════════════════
# 2.8  Retrieval Evaluation (held-out validation queries)
# ════════════════════════════════════════════════════════════════════════════

def evaluate_retrieval(
    retriever,        # DenseRetriever or HybridRetriever
    val_samples: List[Dict],
    top_k: int = 5,
    max_eval: int = 200,
    retriever_name: str = "Dense",
) -> Dict:
    """
    Measure Recall@k and MRR@k on a small set of validation QA pairs.

    Recall@k  — fraction of questions where the gold answer text
                 appears in at least one of the top-k retrieved chunks.
    MRR@k     — Mean Reciprocal Rank  (how early the relevant doc appears).

    Args:
        retriever      : Any retriever with a .retrieve(query, top_k) method.
        val_samples    : List of validation samples (question + answer + context).
        top_k          : Number of chunks retrieved.
        max_eval       : Maximum questions to evaluate (for speed).
        retriever_name : Label for logs.

    Returns:
        dict with recall@k, mrr@k, avg_latency_ms.
    """
    logger.info(
        f"Evaluating {retriever_name}  |  "
        f"queries={min(max_eval, len(val_samples))}  top_k={top_k}"
    )

    hits    = 0
    rr_sum  = 0.0
    latencies = []

    for sample in val_samples[:max_eval]:
        query  = sample["question"]
        answer = sample["answer"].lower()

        t0 = time.time()
        results, _ = retriever.retrieve(query, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        # ── Check if answer appears in any retrieved chunk ────────────────
        found_rank = None
        for rank, chunk in enumerate(results):
            if answer in chunk["text"].lower():
                found_rank = rank + 1   # 1-indexed
                break

        if found_rank is not None:
            hits   += 1
            rr_sum += 1.0 / found_rank

    n       = min(max_eval, len(val_samples))
    recall  = hits / n if n > 0 else 0.0
    mrr     = rr_sum / n if n > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    results_dict = {
        "retriever"    : retriever_name,
        "top_k"        : top_k,
        "num_queries"  : n,
        f"recall@{top_k}" : round(recall, 4),
        f"mrr@{top_k}"    : round(mrr, 4),
        "avg_latency_ms"  : round(avg_lat, 2),
    }

    logger.success(
        f"{retriever_name}  "
        f"Recall@{top_k}={recall:.4f}  "
        f"MRR@{top_k}={mrr:.4f}  "
        f"Lat={avg_lat:.1f}ms"
    )
    return results_dict


# ════════════════════════════════════════════════════════════════════════════
# Qualitative sanity check
# ════════════════════════════════════════════════════════════════════════════

def qualitative_check(
    retriever,
    retriever_name: str,
    queries: List[str],
    top_k: int = 3,
) -> None:
    """Print top-k retrieved chunks for a handful of test queries."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Qualitative Check — {retriever_name}")
    logger.info(f"{'='*60}")

    for query in queries:
        logger.info(f"\n🔍 Query: {query}")
        results, scores = retriever.retrieve(query, top_k=top_k)

        for rank, (chunk, score) in enumerate(zip(results, scores), 1):
            logger.info(
                f"  [{rank}] (score={score:.4f})  source={chunk['source']}\n"
                f"      {chunk['text'][:180]} …"
            )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  STEP 2 — Embedding + FAISS Indexing  (Confidence-Aware RAG)")
    logger.info("=" * 70)

    # ── Load config ──────────────────────────────────────────────────────
    cfg      = load_config("configs/config.yaml")
    r_cfg    = get_retrieval_config(cfg)
    c_cfg    = get_corpus_config(cfg)
    ds_cfg   = cfg.get("dataset", {})

    model_name      = r_cfg.get("model",           "sentence-transformers/all-mpnet-base-v2")
    index_type      = r_cfg.get("index_type",       "ivf")
    top_k           = r_cfg.get("top_k",            5)
    nlist           = r_cfg.get("nlist",            100)
    nprobe          = r_cfg.get("nprobe",           10)
    batch_size      = r_cfg.get("batch_size",       64)
    device          = r_cfg.get("device",           "cpu")
    faiss_path      = r_cfg.get("index_path",       "data/faiss_index.bin")
    embeddings_path = r_cfg.get("embeddings_path",  "data/embeddings.npy")
    id_map_path     = r_cfg.get("id_map_path",      "data/id_map.pkl")
    bm25_path       = r_cfg.get("bm25_path",        "data/bm25.pkl")
    build_bm25      = r_cfg.get("build_bm25",       True)
    eval_queries    = r_cfg.get("max_eval_queries",  200)
    hybrid_method   = r_cfg.get("hybrid_method",    "rrf")
    alpha           = r_cfg.get("hybrid_alpha",     0.65)

    corpus_path     = c_cfg.get("output_path", "data/corpus.json")
    max_val         = ds_cfg.get("max_val_samples", 1000)

    # ── 2.2  Load corpus ─────────────────────────────────────────────────
    logger.info(f"Loading corpus from: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    logger.success(f"Corpus loaded  |  chunks={len(corpus):,}")

    # ── 2.1/2.2/2.3/2.4  Dense retriever ─────────────────────────────────
    dense = DenseRetriever(
        model_name = model_name,
        index_type = index_type,
        nlist      = nlist,
        nprobe     = nprobe,
        batch_size = batch_size,
        device     = device,
    )

    # Encode corpus (uses cache if embeddings.npy exists)
    embeddings = dense.encode_corpus(corpus, embeddings_path=embeddings_path)

    # Build FAISS index (save to disk)
    dense.build_index(embeddings, index_path=faiss_path)

    # Save corpus id-map for fast lookup by integer index
    dense.save_corpus_map(id_map_path)

    # ── 2.5  BM25 sparse index ────────────────────────────────────────────
    bm25 = None
    if build_bm25:
        bm25 = BM25Retriever()
        if Path(bm25_path).exists():
            bm25.load(bm25_path)
        else:
            bm25.build_index(corpus, save_path=bm25_path)

    # ── 2.7  Hybrid retriever ─────────────────────────────────────────────
    hybrid = None
    if bm25 is not None:
        hybrid = HybridRetriever(
            dense  = dense,
            bm25   = bm25,
            alpha  = alpha,
            method = hybrid_method,
        )

    # ── 2.8  Retrieval evaluation on validation set ───────────────────────
    logger.info("Loading HotpotQA validation data for evaluation …")
    try:
        from datasets import load_dataset
        dataset    = load_dataset("hotpot_qa", "fullwiki")
        val_data   = dataset["validation"]
        val_samples = [
            {
                "question": s["question"],
                "answer"  : s["answer"],
            }
            for s in list(val_data)[:max_val]
        ]
        logger.success(f"Val samples: {len(val_samples)}")

        eval_results = []

        # Evaluate dense retriever
        res_dense = evaluate_retrieval(
            dense, val_samples,
            top_k=top_k, max_eval=eval_queries,
            retriever_name="Dense (SBERT+FAISS)"
        )
        eval_results.append(res_dense)

        # Evaluate hybrid retriever
        if hybrid is not None:
            res_hybrid = evaluate_retrieval(
                hybrid, val_samples,
                top_k=top_k, max_eval=eval_queries,
                retriever_name=f"Hybrid (RRF, alpha={alpha})"
            )
            eval_results.append(res_hybrid)

        # Save eval results
        eval_path = "logs/step2_retrieval_eval.json"
        Path("logs").mkdir(exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        logger.success(f"Eval results saved → {eval_path}")

    except Exception as e:
        logger.warning(f"Retrieval evaluation skipped: {e}")

    # ── Qualitative sanity check ──────────────────────────────────────────
    sample_queries = [
        "Who is the president of France?",
        "What year did World War II end?",
        "Which company created the Python programming language?",
        "Where was Albert Einstein born?",
    ]

    qualitative_check(dense, "Dense SBERT+FAISS", sample_queries, top_k=3)

    if hybrid:
        qualitative_check(hybrid, "Hybrid (RRF)", sample_queries, top_k=3)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 2 COMPLETE — Index artefacts saved:")
    logger.info(f"    FAISS index    → {faiss_path}")
    logger.info(f"    Embeddings     → {embeddings_path}")
    logger.info(f"    Corpus id-map  → {id_map_path}")
    if build_bm25:
        logger.info(f"    BM25 index     → {bm25_path}")
    logger.info("  Ready for STEP 3 — Generation")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
