"""
step4_adaptive_rag.py
────────────────────────────────────────────────────────────────────────────────
STEP 4 — Signal-Aware Diagnostic Adaptive RAG (Final Pipeline)
────────────────────────────────────────────────────────────────────────────────

This script executes the knowledge-aware adaptive RAG loop.
It demonstrates the system's ability to self-correct by diagnosing failure
modes and triggering targeted adaptations (k-expansion, reformulation).

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
from adaptive.controller        import AdaptiveController


# ════════════════════════════════════════════════════════════════════════════
# 3.4 Visualization Logging (Sc (Confidence over Rounds))
# ════════════════════════════════════════════════════════════════════════════

def visualize_run(results: List[Dict[str, Any]]) -> None:
    """Print a ASCII summary of the confidence evolution for each query."""
    logger.info("\n" + "=" * 70)
    logger.info("  Signal-Aware Run Summary (Per-Query Evolution) ")
    logger.info("=" * 70)

    for i, res in enumerate(results, 1):
        q     = res["query"]
        ans   = res["final_answer"]
        conf  = res["final_confidence"]
        rounds= res["rounds"]
        
        logger.info(f"\n[{i}] Query: {q[:100]}…")
        
        # ASCII bar for rounds
        for r in rounds:
            c_val = r["C"]
            bar   = "█" * int(c_val * 20)
            logger.info(f"    R{r['round']} | {bar:<20} | C={c_val:.3f} (z={r['z']:.3f}) Signals: Sr={r['Sr']:.2f}, Sl={r['Sl']:.2f}, Sc={r['Sc']:.2f}")
        
        logger.info(f"    Final Answer: {ans[:200]}…")
        logger.info(f"    Final Confidence: {conf:.4f} ({'SUCCESS' if conf >= 0.7 else 'LOW_CONF'})")


# ════════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 80)
    logger.info("  STEP 4 — Signal-Aware Diagnostic Adaptive RAG (The Knowledge Loop) ")
    logger.info("=" * 80)

    # ── Load Config ──────────────────────────────────────────────────────
    cfg   = load_config("configs/config.yaml")
    r_cfg = get_retrieval_config(cfg)
    g_cfg = cfg.get("generation", {})
    a_cfg = cfg.get("adaptive", {})
    ds_cfg= cfg.get("dataset", {})

    output_path = a_cfg.get("results_path", "data/adaptive_results.json")
    max_val     = ds_cfg.get("max_val_samples", 5) # limit for fast pilot run

    # ── Initialise Retriever (Hybrid/IVF-FAISS) ─────────────────────────
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

    # ── Initialise Adaptive Controller (Diagnostic Loop) ─────────────────
    ctrl = AdaptiveController(retriever=ret_, generator=gen, config=cfg)

    # ── Load Validation Samples ──────────────────────────────────────────
    logger.info(f"Loading {max_val} validation samples for adaptive test …")
    try:
        from datasets import load_dataset
        dataset    = load_dataset("hotpot_qa", "fullwiki")
        val_data   = dataset["validation"]
        val_samples = list(val_data)[:max_val]
        logger.success(f"Samples loaded: {len(val_samples)}")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    # ── Step 4: Run the Dynamic Loop ─────────────────────────────────────
    results = []
    t_start = time.time()

    for i, sample in enumerate(val_samples):
        query = sample["question"]
        gold  = sample["answer"]
        
        logger.info(f"\n{'='*20} QUERY {i+1}/{len(val_samples)} {'='*20}")
        logger.info(f"Target Question: {query}")
        
        # RUN THE ADAPTIVE LOOP
        final_out = ctrl.run_query(query)
        final_out["gold_answer"] = gold
        
        results.append(final_out)

    elapsed = time.time() - t_start
    logger.success(f"Adaptive Loop Test complete in {elapsed/60:.1f} minutes.")

    # ── Save Results ──────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.success(f"Final results saved → {output_path}")

    # ── 4.10 Visualization ───────────────────────────────────────────────
    visualize_run(results)

    logger.info("\n" + "=" * 80)
    logger.info(" ✅ STEP 4 COMPLETE — Final Knowledge-Aware Pipeline is Operational.")
    logger.info(" Proceed to STEP 5: Final Evaluation, Ablation Analysis, and Results Visualization.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
