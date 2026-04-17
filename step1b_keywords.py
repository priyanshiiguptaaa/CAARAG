"""
step1b_keywords.py
────────────────────────────────────────────────────────────────────────────────
OPTIONAL POST-PROCESSING STEP 1B — Add Keywords to Existing Corpus
────────────────────────────────────────────────────────────────────────────────
Run AFTER step1_corpus_preparation.py when you have time.

KeyBERT on 361k chunks takes ~2-4 hours on CPU.
For GPU it takes ~30 minutes.

Usage:
    python step1b_keywords.py                        # uses configs/config.yaml
    python step1b_keywords.py --top_n 3              # override keyword count
    python step1b_keywords.py --batch_size 256       # smaller GPU batch

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from keybert import KeyBERT
from tqdm import tqdm

from utils.logger import logger
from utils.config_loader import load_config, get_corpus_config


def parse_args():
    parser = argparse.ArgumentParser(description="Add KeyBERT keywords to corpus.json")
    parser.add_argument("--corpus",     type=str, default=None, help="Path to corpus.json")
    parser.add_argument("--output",     type=str, default=None, help="Output path (overwrites if same as input)")
    parser.add_argument("--top_n",      type=int, default=None, help="Keywords per chunk")
    parser.add_argument("--batch_size", type=int, default=512,  help="KeyBERT batch size")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config("configs/config.yaml")
    c_cfg  = get_corpus_config(cfg)

    corpus_path = args.corpus or c_cfg.get("output_path", "data/corpus.json")
    output_path = args.output or corpus_path   # overwrite by default
    top_n       = args.top_n  or c_cfg.get("keyword_top_n", 5)
    batch_size  = args.batch_size

    logger.info(f"Loading corpus from: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        chunked_docs = json.load(f)

    logger.info(f"Loaded {len(chunked_docs):,} chunks  |  top_n={top_n}, batch_size={batch_size}")

    kw_model = KeyBERT()
    texts    = [d["text"] for d in chunked_docs]

    all_keywords = []
    for i in tqdm(range(0, len(texts), batch_size), desc="KeyBERT", unit="batch"):
        batch  = texts[i : i + batch_size]
        result = kw_model.extract_keywords(
            batch,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
        )
        for kw_scores in result:
            all_keywords.append([kw for kw, _ in kw_scores])

    for doc, kws in zip(chunked_docs, all_keywords):
        doc["keywords"] = kws
        if "length" not in doc:
            doc["length"] = len(doc["text"].split())

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)

    logger.success(f"Keywords added → saved to {output_path}")


if __name__ == "__main__":
    main()
