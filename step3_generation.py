"""
step3_generation.py
────────────────────────────────────────────────────────────────────────────────
STEP 3 — Confidence-Aware Local Question Answering (Generation)
────────────────────────────────────────────────────────────────────────────────

Pipeline:
    3.1  Initialise LLM (Mock/OpenAI/HF) and Hybrid Retriever.
    3.2  Load HotpotQA validation samples (small set).
    3.3  For each sample:
            • Retrieve top-K chunks (Hybrid/Dense).
            • Construct labeled context ([Doc 1], [Doc 2]...).
            • Generate multiple outputs (n=3) for consistency signaling.
            • Record Confidence Score + Explanation for Step 4 analysis.
    3.4  Save fully structured generation results to data/generation_results.json.

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Any

from utils.logger          import logger
from utils.config_loader   import load_config, get_retrieval_config
from retrieval.dense_retriever  import DenseRetriever
from retrieval.bm25_retriever   import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from generation.llm_wrapper     import get_llm_backend
from generation.generator       import RAGGenerator


# ════════════════════════════════════════════════════════════════════════════
# 3.1 Orchestrator Setup
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  STEP 3 — Confidence-Aware Question Answering (Generation) ")
    logger.info("=" * 70)

    # ── Load Config ──────────────────────────────────────────────────────
    cfg   = load_config("configs/config.yaml")
    r_cfg = get_retrieval_config(cfg)
    g_cfg = cfg.get("generation", {})
    ds_cfg= cfg.get("dataset", {})

    top_k       = r_cfg.get("top_k", 5)
    max_val     = ds_cfg.get("max_val_samples", 10) # process small set
    output_path = g_cfg.get("results_path", "data/generation_results.json")

    # ── Initialise Retriever ─────────────────────────────────────────────
    # Use the same setup as Step 2 (Dense + BM25 if enabled)
    dense = DenseRetriever(
        model_name = r_cfg.get("model", "all-mpnet-base-v2"),
        device     = r_cfg.get("device", "cpu"),
    )
    dense.load_index(r_cfg.get("index_path", "data/faiss_index.bin"))
    dense.load_corpus_map(r_cfg.get("id_map_path", "data/id_map.pkl"))

    bm25   = None
    ret_   = dense
    if r_cfg.get("build_bm25", True):
        bm25 = BM25Retriever()
        bm25.load(r_cfg.get("bm25_path", "data/bm25.pkl"))
        ret_ = HybridRetriever(
            dense  = dense,
            bm25   = bm25,
            method = r_cfg.get("hybrid_method", "rrf")
        )

    # ── Initialise LLM and Generator ─────────────────────────────────────
    llm    = get_llm_backend(cfg)
    gen    = RAGGenerator(llm=llm, config=cfg)

    # ── Load Validation Data ─────────────────────────────────────────────
    # Re-use HotpotQA samples for end-to-end evaluation
    logger.info("Loading validation data for RAG run …")
    try:
        from datasets import load_dataset
        dataset    = load_dataset("hotpot_qa", "fullwiki")
        val_data   = dataset["validation"]
        val_samples = list(val_data)[:max_val]
        logger.success(f"Samples loaded: {len(val_samples)}")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    # ── 3.3 Main Loop ────────────────────────────────────────────────────
    results = []
    logger.info(f"Generating answers for {len(val_samples)} queries …")

    t_start = time.time()
    for i, sample in enumerate(val_samples):
        query = sample["question"]
        gold  = sample["answer"]

        logger.info(f"[{i+1}/{len(val_samples)}] Processing: {query[:80]} …")

        # 1. Retrieval
        docs, scores = ret_.retrieve(query, top_k=top_k)

        # 2. Generation
        gen_output = gen.generate_answer(query, docs)

        # 3. Aggregate Data
        item = {
            "query"           : query,
            "gold_answer"     : gold,
            "retrieved_docs"  : docs,
            "retrieval_scores": [float(s) for s in scores],
            "generation"      : gen_output,
            "timestamp"       : time.time()
        }
        results.append(item)

    elapsed = time.time() - t_start
    logger.success(f"Generation loop complete in {elapsed:.1f}s")

    # ── 3.4 Save Intermediate Results ────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.success(f"Generation results saved → {output_path}")

    # ── Quick Analytics ──────────────────────────────────────────────────
    avg_conf = sum(r["generation"]["final_confidence"] for r in results) / len(results) if results else 0
    logger.info(f"Average LLM Confidence: {avg_conf:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info(" ✅ STEP 3 COMPLETE — generation_results.json is ready.")
    logger.info(" Proceed to STEP 4: Confidence Computation & Adaptive Re-Retrieval.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
