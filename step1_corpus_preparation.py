"""
step1_corpus_preparation.py
────────────────────────────────────────────────────────────────────────────────
STEP 1 — Dataset & Corpus Preparation for Confidence-Aware Adaptive RAG
────────────────────────────────────────────────────────────────────────────────

Pipeline:
    1.1  Load HotpotQA (fullwiki split) from HuggingFace
    1.2  Extract unique Wikipedia documents from the `context` field
    1.3  Chunk each document into ~150-word windows
    1.4  Light-touch text preprocessing (lowercase + whitespace normalisation)
    1.5  Optional keyword extraction per chunk (KeyBERT) — journal bonus
    1.6  Save corpus.json + corpus_stats.json for downstream steps

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

# ── Third-party imports ──────────────────────────────────────────────────────
try:
    from datasets import load_dataset, Dataset
except ImportError:
    raise ImportError("Run: pip install datasets")

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Run: pip install tqdm")

# ── Internal imports ─────────────────────────────────────────────────────────
from utils.logger import logger
from utils.config_loader import load_config, get_corpus_config


# ════════════════════════════════════════════════════════════════════════════
# 1.1  Dataset Loader
# ════════════════════════════════════════════════════════════════════════════

def load_hotpotqa(
    max_train: Optional[int] = None,
    max_val: Optional[int] = 1000
) -> tuple[Dataset, Dataset]:
    """
    Load HotpotQA (fullwiki config) from HuggingFace datasets hub.

    Returns:
        (train_data, val_data) — HuggingFace Dataset objects.
    """
    logger.info("Loading HotpotQA (fullwiki) from HuggingFace …")
    t0 = time.time()

    dataset = load_dataset("hotpot_qa", "fullwiki")

    train_data = dataset["train"]
    val_data   = dataset["validation"]

    # Optional sub-sampling (useful during development)
    if max_train is not None:
        train_data = train_data.select(range(min(max_train, len(train_data))))
        logger.info(f"  → Training split limited to {max_train:,} samples")

    if max_val is not None:
        val_data = val_data.select(range(min(max_val, len(val_data))))
        logger.info(f"  → Validation split limited to {max_val:,} samples")

    elapsed = time.time() - t0
    logger.success(
        f"Dataset loaded in {elapsed:.1f}s  |  "
        f"train={len(train_data):,}  val={len(val_data):,}"
    )

    # ── Print a single sample for inspection ────────────────────────────
    sample = train_data[0]
    logger.debug(f"Sample keys : {list(sample.keys())}")
    logger.debug(f"Question    : {sample['question']}")
    logger.debug(f"Answer      : {sample['answer']}")
    logger.debug(f"Context docs: {len(sample['context']['title'])}")

    return train_data, val_data


# ════════════════════════════════════════════════════════════════════════════
# 1.2  Extract Unique Documents
# ════════════════════════════════════════════════════════════════════════════

def extract_unique_documents(dataset: Dataset) -> List[Dict[str, str]]:
    """
    Iterate over every sample and collect unique Wikipedia passages from the
    `context` field.

    HotpotQA context format:
        {
            "title":     ["Title1", "Title2", ...],
            "sentences": [["sent1", "sent2"], ["sent1", ...], ...]
        }

    Returns:
        List of dicts  →  {"title": str, "text": str}
    """
    logger.info("Extracting unique Wikipedia documents …")

    unique_docs: Dict[str, str] = {}   # title → concatenated text
    duplicate_count = 0

    for sample in tqdm(dataset, desc="Extracting docs", unit="sample"):
        titles    = sample["context"]["title"]
        sentences = sample["context"]["sentences"]

        for title, sent_list in zip(titles, sentences):
            full_text = " ".join(sent_list)

            if title in unique_docs:
                duplicate_count += 1
                continue

            unique_docs[title] = full_text

    documents = [{"title": k, "text": v} for k, v in unique_docs.items()]

    logger.success(
        f"Unique documents extracted: {len(documents):,}  "
        f"(duplicates skipped: {duplicate_count:,})"
    )
    return documents


# ════════════════════════════════════════════════════════════════════════════
# 1.3  Chunking
# ════════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    chunk_size: int = 150,
    overlap: int = 0,
    min_chunk_words: int = 50,
) -> List[str]:
    """
    Split `text` into word-level chunks of `chunk_size` words,
    with optional word-level `overlap` between consecutive chunks.

    Args:
        text           : raw document text
        chunk_size     : target words per chunk (100–300 recommended)
        overlap        : sliding-window overlap in words (0 = no overlap)
        min_chunk_words: discard chunks shorter than this (avoids tiny tail chunks)

    Returns:
        List of chunk strings
    """
    words  = text.split()
    chunks = []
    step   = max(chunk_size - overlap, 1)  # stride

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.split()) >= min_chunk_words:
            chunks.append(chunk)

    return chunks


def build_chunked_corpus(
    documents: List[Dict[str, str]],
    chunk_size: int = 150,
    overlap: int = 0,
    min_chunk_words: int = 50,
) -> List[Dict]:
    """
    Apply chunking to every document and return a flat list of chunk dicts.

    Each chunk dict:
        {
            "chunk_id": "<title>_<idx>",
            "text"    : "<chunk text>",
            "source"  : "<wikipedia title>"
        }
    """
    logger.info(
        f"Chunking {len(documents):,} documents "
        f"(chunk_size={chunk_size}, overlap={overlap}) …"
    )

    chunked_docs: List[Dict] = []

    for doc in tqdm(documents, desc="Chunking", unit="doc"):
        chunks = chunk_text(
            doc["text"],
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_words=min_chunk_words,
        )

        for idx, chunk in enumerate(chunks):
            chunked_docs.append({
                "chunk_id": f"{doc['title']}_{idx}",
                "text"    : chunk,
                "source"  : doc["title"],
            })

    logger.success(f"Total chunks produced: {len(chunked_docs):,}")
    return chunked_docs


# ════════════════════════════════════════════════════════════════════════════
# 1.4  Text Preprocessing
# ════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Light-touch normalisation:
        • lowercase
        • collapse extra whitespace / newlines
        • strip leading/trailing spaces

    ⚠️  Deliberately avoids removing stop-words or punctuation
        so as not to destroy semantic meaning for the LLM.
    """
    text = text.lower()
    text = re.sub(r"\n+",  " ", text)   # newlines → space
    text = re.sub(r"\s+",  " ", text)   # multiple spaces → single space
    return text.strip()


def preprocess_corpus(chunked_docs: List[Dict]) -> List[Dict]:
    """Apply clean_text() to every chunk's text in-place."""
    logger.info("Applying text preprocessing …")

    for doc in tqdm(chunked_docs, desc="Cleaning", unit="chunk"):
        doc["text"] = clean_text(doc["text"])

    logger.success("Preprocessing complete.")
    return chunked_docs


# ════════════════════════════════════════════════════════════════════════════
# 1.5  Keyword Extraction (Journal-Level Bonus)
# ════════════════════════════════════════════════════════════════════════════

def add_keywords(
    chunked_docs: List[Dict],
    top_n: int = 5,
    batch_size: int = 512,
) -> List[Dict]:
    """
    Extract top-N keywords per chunk using KeyBERT.
    Adds fields:
        • "keywords" : List[str]
        • "length"   : int  (word count)

    Falls back gracefully if keybert is not installed.
    """
    try:
        from keybert import KeyBERT
    except ImportError:
        logger.warning(
            "keybert not installed — skipping keyword extraction. "
            "Install with: pip install keybert"
        )
        for doc in chunked_docs:
            doc["keywords"] = []
            doc["length"]   = len(doc["text"].split())
        return chunked_docs

    logger.info(f"Extracting keywords (top_n={top_n}) with KeyBERT …")
    kw_model = KeyBERT()

    texts = [d["text"] for d in chunked_docs]

    # Process in batches to avoid OOM
    all_keywords: List[List[str]] = []
    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="KeyBERT batches",
        unit="batch",
    ):
        batch  = texts[i : i + batch_size]
        result = kw_model.extract_keywords(
            batch,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
        )
        # result: List[List[Tuple[keyword, score]]]
        for kw_scores in result:
            all_keywords.append([kw for kw, _ in kw_scores])

    for doc, kws in zip(chunked_docs, all_keywords):
        doc["keywords"] = kws
        doc["length"]   = len(doc["text"].split())

    logger.success("Keyword extraction complete.")
    return chunked_docs


# ════════════════════════════════════════════════════════════════════════════
# 1.6  Save Corpus
# ════════════════════════════════════════════════════════════════════════════

def save_corpus(
    chunked_docs: List[Dict],
    output_path: str = "data/corpus.json",
    stats_path:  str = "data/corpus_stats.json",
) -> None:
    """
    Persist the final corpus and a summary stats file.

    Args:
        chunked_docs : list of processed chunk dicts
        output_path  : path for corpus.json
        stats_path   : path for corpus_stats.json (human-readable summary)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Save full corpus ─────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)

    logger.success(f"Corpus saved → {output_path}  ({len(chunked_docs):,} chunks)")

    # ── Compute and save stats ───────────────────────────────────────────
    lengths      = [len(d["text"].split()) for d in chunked_docs]
    unique_srcs  = len({d["source"] for d in chunked_docs})
    total_words  = sum(lengths)
    avg_len      = total_words / len(lengths) if lengths else 0

    stats = {
        "total_chunks"       : len(chunked_docs),
        "unique_source_docs" : unique_srcs,
        "total_words"        : total_words,
        "avg_chunk_words"    : round(avg_len, 2),
        "min_chunk_words"    : min(lengths) if lengths else 0,
        "max_chunk_words"    : max(lengths) if lengths else 0,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Corpus stats  → {stats_path}")
    logger.info(
        f"  chunks={stats['total_chunks']:,}  "
        f"unique_docs={stats['unique_source_docs']:,}  "
        f"avg_words={stats['avg_chunk_words']}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Main entry-point
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  STEP 1 — Corpus Preparation  (Confidence-Aware Adaptive RAG)")
    logger.info("=" * 70)

    # ── Load config ──────────────────────────────────────────────────────
    cfg        = load_config("configs/config.yaml")
    ds_cfg     = cfg.get("dataset", {})
    corpus_cfg = get_corpus_config(cfg)

    max_train = ds_cfg.get("max_train_samples", None)
    max_val   = ds_cfg.get("max_val_samples", 1000)

    chunk_size      = corpus_cfg.get("chunk_size",       150)
    overlap         = corpus_cfg.get("overlap",          0)
    min_chunk_words = corpus_cfg.get("min_chunk_words",  50)
    output_path     = corpus_cfg.get("output_path",      "data/corpus.json")
    stats_path      = corpus_cfg.get("stats_path",       "data/corpus_stats.json")
    add_kw          = corpus_cfg.get("add_keywords",     True)
    keyword_top_n   = corpus_cfg.get("keyword_top_n",    5)

    # ── Step 1.1: Load dataset ───────────────────────────────────────────
    train_data, val_data = load_hotpotqa(
        max_train=max_train,
        max_val=max_val,
    )

    # ── Step 1.2: Extract unique documents ───────────────────────────────
    documents = extract_unique_documents(train_data)

    # ── Step 1.3: Chunk documents ────────────────────────────────────────
    chunked_docs = build_chunked_corpus(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_words=min_chunk_words,
    )

    # ── Step 1.4: Preprocess text ────────────────────────────────────────
    chunked_docs = preprocess_corpus(chunked_docs)

    # ── Step 1.5: Keyword extraction (journal-level bonus) ───────────────
    if add_kw:
        chunked_docs = add_keywords(chunked_docs, top_n=keyword_top_n)
    else:
        # Still add length metadata even without keywords
        for doc in chunked_docs:
            doc["keywords"] = []
            doc["length"]   = len(doc["text"].split())

    # ── Step 1.6: Save corpus ─────────────────────────────────────────────
    save_corpus(chunked_docs, output_path=output_path, stats_path=stats_path)

    # ── Quick sanity check: print a sample chunk ─────────────────────────
    logger.info("\n── Sample chunk (index 0) ──────────────────────────────────")
    sample = chunked_docs[0]
    for key, val in sample.items():
        if key == "text":
            logger.info(f"  {key:10}: {str(val)[:120]} …")
        else:
            logger.info(f"  {key:10}: {val}")

    logger.info("\n✅  STEP 1 COMPLETE — corpus.json is ready for STEP 2 (Indexing)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
