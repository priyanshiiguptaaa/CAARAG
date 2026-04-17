"""
adaptive/controller.py
────────────────────────────────────────────────────────────────────────────────
Iterative Adaptive RAG Controller.
Manages rounds, diagnostics, k-expansion, and query reformulation.

Loop Algorithm:
    While (Confidence < tau) and (Rounds < max_rounds):
        1. Retrieve top-k
        2. Generate samples
        3. Compute Sr, Sl, Sc -> C
        4. If C < tau: Diagnose -> Adapt (Expand k / Reformulate)

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import time
from typing import List, Dict, Any, Tuple

from utils.logger import logger
from confidence.signals import compute_Sr, compute_Sl, compute_Sc
from confidence.calibrator import Calibrator
from adaptive.diagnoser import AdaptiveDiagnoser


# ════════════════════════════════════════════════════════════════════════════
class AdaptiveController:
    """
    Main Orchestrator for the Signal-Aware Adaptive RAG loop.
    
    This class handles the end-to-end knowledge-gain lifecycle per query.
    """

    def __init__(self, retriever: Any, generator: Any, config: Dict):
        self.retriever = retriever
        self.generator = generator
        self.cfg       = config
        
        self.a_cfg     = config.get("adaptive", {})
        self.max_r     = self.a_cfg.get("max_rounds", 3)
        self.tau       = self.a_cfg.get("threshold_tau", 0.7)
        self.epsilon   = self.a_cfg.get("early_stop_epsilon", 0.01)
        self.delta_k   = self.a_cfg.get("delta_k", 5)
        
        self.calibrator = Calibrator(config)
        self.diagnoser  = AdaptiveDiagnoser(config)

    # ── 3.3 / 4.6 Targeted Adaptation ────────────────────────────────────
    
    def reformulate_query(self, query: str, context: str) -> str:
        """
        Use the LLM to rewrite the query based on existing context.
        (Strategy 2: Context-Aware Query Reformulation)
        """
        # Prompt to rewrite the search query
        prompt = (
            "The current retrieved context failed to answer the following question.\n\n"
            f"Question: {query}\n"
            f"Current Context: {context[:500]} ... (truncated)\n\n"
            "Rewrite a better, more specific search query to retrieve missing information from Wikipedia.\n"
            "Produce ONLY the new query string, nothing else.\n"
            "New Query:"
        )
        # Use existing LLM backend (internal generator has it)
        new_q = self.generator.llm.generate(prompt, temperature=0.0)
        return new_q.strip()

    # ── 4.7 End-to-End Adaptive Loop ─────────────────────────────────────
    
    def run_query(self, original_query: str) -> Dict[str, Any]:
        """
        Execute the adaptive knowledge-loop for a single query.
        """
        current_query = original_query
        k             = self.cfg.get("retrieval", {}).get("top_k", 5)
        
        rounds_data   = []
        prev_c        = -1.0
        
        for r in range(1, self.max_r + 1):
            logger.info(f"--- Round {r}  (k={k}) ---")
            
            # 1. Retrieval
            docs, scores = self.retriever.retrieve(current_query, top_k=k)
            
            # 2. Generation (multi-sample)
            gen_out      = self.generator.generate_answer(original_query, docs)
            samples      = gen_out["all_samples"]
            answers      = [s["answer"] for s in samples]
            
            # 3. Compute Signals
            Sr = compute_Sr(scores)
            Sl = compute_Sl(samples)
            # Semantic Sc (needs retriever's SBERT model)
            Sc = compute_Sc(answers, self.retriever.dense.model)
            
            # 4. Calibration
            z, c = self.calibrator.get_calibrated_confidence(Sr, Sl, Sc)
            
            logger.info(f"  Signals: Sr={Sr:.3f} Sl={Sl:.3f} Sc={Sc:.3f} | Confidence C={c:.3f}")
            
            # Record state
            round_info = {
                "round"     : r,
                "current_q" : current_query,
                "k"         : k,
                "Sr"        : Sr, "Sl": Sl, "Sc": Sc,
                "z"         : z,  "C": c,
                "final_ans" : gen_out["final_answer"]
            }
            rounds_data.append(round_info)
            
            # ── 4.5 Adaptive Decision ────────────────────────────────────
            # Success: Stop
            if c >= self.tau:
                logger.success(f"Final Confidence C={c:.3f} reached threshold ({self.tau}). Stopping Loop.")
                break
            
            # Improvement too slow: Early Stop
            if r > 1 and abs(c - prev_c) < self.epsilon:
                logger.warning(f"Confidence improvement {abs(c-prev_c):.4f} is too slow. Stopping Loop.")
                break
            
            # Max rounds reached: Stop
            if r == self.max_r:
                logger.warning("Max rounds reached. Terminating Adaptive Loop.")
                break
                
            # Diagnosis: Adapt
            next_action = self.diagnoser.diagnose(Sr, Sl, Sc)
            
            if next_action == "expand_k":
                k += self.delta_k
            elif next_action == "reformulate":
                current_query = self.reformulate_query(original_query, gen_out["context_used"])
                logger.info(f"  New Query: {current_query}")
            elif next_action == "resample":
                # Increase k as a fallback if Sc is low
                k += self.delta_k
            else:
                break
            
            prev_c = c

        return {
            "query"           : original_query,
            "final_answer"    : rounds_data[-1]["final_ans"],
            "final_confidence": rounds_data[-1]["C"],
            "total_rounds"    : len(rounds_data),
            "rounds"          : rounds_data
        }
