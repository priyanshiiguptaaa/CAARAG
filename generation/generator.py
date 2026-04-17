"""
generation/generator.py
────────────────────────────────────────────────────────────────────────────────
RAG logic: context building, strict prompting, multi-sampling, and parsing.

Classes:
    RAGGenerator — orchestrates the full generation step.

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Tuple

from utils.logger import logger


# ════════════════════════════════════════════════════════════════════════════
class RAGGenerator:
    """Orchestrates structured RAG generation with confidence signals."""

    def __init__(self, llm: Any, config: Dict):
        self.llm     = llm
        self.g_cfg   = config.get("generation", {})
        self.n       = self.g_cfg.get("n_samples", 3)
        self.temp    = self.g_cfg.get("temperature", 0.7)
        self.strict  = self.g_cfg.get("strict_mode", True)
        self.max_gen = self.g_cfg.get("max_new_tokens", 256)

    # ── 3.2 Context Builder ───────────────────────────────────────────────

    def build_context(self, docs: List[Dict]) -> str:
        """
        Combine retrieved chunks into a labeled context block.
        Format: [Doc 1] text \n [Doc 2] text ...
        """
        if not docs:
            return ""

        context_blocks = []
        for i, doc in enumerate(docs, 1):
            text = doc.get("text", "").strip()
            context_blocks.append(f"[Doc {i}] {text}")

        return "\n\n".join(context_blocks)

    # ── 3.3 / 3.4 Prompt Engineering ──────────────────────────────────────

    def get_prompt(self, query: str, context: str) -> str:
        """
        Consistent, structured RAG prompt.
        Forcing Confidence Score (0.0-1.0) and Explanation for Step 4 analysis.
        """
        # Modified: Allow model to use its knowledge when context is poor
        instruction = (
            "Use the provided context if it's relevant to answer the question.\n"
            "If the context is not relevant or doesn't contain the answer, "
            "use your general knowledge to provide an accurate answer.\n"
        )

        prompt = (
            "You are a highly accurate question-answering assistant.\n"
            f"{instruction}\n"
            "Produce your response in the following format:\n"
            "Confidence: [Score from 0.0 to 1.0]\n"
            "Explanation: [Short reason for your confidence]\n"
            "Answer: [Final Answer]\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Let's think step by step.\n"
        )
        return prompt

    # ── 3.5 / 3.6 Multi-Sample & Parsing ─────────────────────────────────

    def parse_output(self, text: str) -> Dict[str, Any]:
        """
        Extract Confidence, Explanation, and Answer from raw text using regex.
        """
        obj = {
            "confidence" : 0.0,
            "explanation": "Failed to parse",
            "answer"     : text.strip(),
            "raw"        : text
        }

        # Regex for Confidence
        conf_match = re.search(r"Confidence:\s*([\d\.]+)", text, re.IGNORECASE)
        if conf_match:
            try:
                obj["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        # Regex for Explanation
        expl_match = re.search(r"Explanation:\s*(.*?)(?=\nAnswer:|$)", text, re.IGNORECASE | re.DOTALL)
        if expl_match:
            obj["explanation"] = expl_match.group(1).strip()

        # Regex for Answer
        ans_match = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if ans_match:
            obj["answer"] = ans_match.group(1).strip()

        return obj

    def generate_answer(self, query: str, docs: List[Dict]) -> Dict[str, Any]:
        """
        Full orchestration of building context → multi-sampling → parsing.

        Returns one consolidated result containing all n outputs.
        """
        context = self.build_context(docs)
        prompt  = self.get_prompt(query, context)

        logger.debug(f"Prompt (length={len(prompt)}) build for query: {query}")

        outputs = []
        for i in range(self.n):
            logger.debug(f"  Generating sample {i+1}/{self.n} …")
            raw_text = self.llm.generate(
                prompt,
                temperature=self.temp if self.n > 1 else 0.0,
                max_new_tokens=self.max_gen
            )
            parsed = self.parse_output(raw_text)
            outputs.append(parsed)

        # ── 3.7 Selection Strategy ─────────────────────────────────────────
        # For now, pick the first or highest confidence.
        # Step 4 will do deeper consistency analysis.
        final_sample = max(outputs, key=lambda x: x["confidence"])

        return {
            "query"           : query,
            "final_answer"    : final_sample["answer"],
            "final_confidence": final_sample["confidence"],
            "all_samples"     : outputs,
            "context_used"    : context
        }
