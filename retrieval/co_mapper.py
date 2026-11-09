"""Course Outcome (CO/LO) semantic mapping using SentenceTransformer embeddings.

Optimized with:
- Model singleton with thread-safe loading
- Batched embedding computation
- Vectorized cosine similarity (numpy matrix multiplication)
- Disk (.cache/ingestion/) and in-memory chunk embedding & CO score persistence
"""
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Optional

import numpy as np

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    MAX_CHUNK_WORDS,
    MIN_CHUNK_WORDS,
    OVERLAP_WORDS,
)

CACHE_DIR = Path(".cache/ingestion")
_model = None
_model_lock = threading.Lock()
_MEM_CHUNK_CACHE: dict[str, np.ndarray] = {}


def _m():
    """Lazy-load the SentenceTransformer embedding model once per process."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _model


def _get_cache_prefix(book_id: str, file_hash: Optional[str] = None) -> str:
    model_slug = EMBEDDING_MODEL_NAME.replace("/", "_")
    cfg_slug = f"w{MAX_CHUNK_WORDS}_o{OVERLAP_WORDS}_m{MIN_CHUNK_WORDS}"
    hash_part = f"_{file_hash}" if file_hash else ""
    return f"{book_id}{hash_part}_pf_v3_{model_slug}_{cfg_slug}"


def get_chunk_embeddings(chunks: list[dict[str, Any]], model=None) -> np.ndarray:
    """Batch compute or retrieve cached embeddings for chunks.

    Returns:
        np.ndarray of shape (len(chunks), embedding_dim) with L2 normalized rows.
    """
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)

    # 1. Try disk cache if chunks belong to a single book with cached embeddings
    book_ids = {c.get("book_id") for c in chunks if c.get("book_id")}
    if len(book_ids) == 1:
        bid = list(book_ids)[0]
        # Check if any chunk has file_hash
        f_hash = next((c.get("file_hash") for c in chunks if c.get("file_hash")), None)
        disk_cache_file = CACHE_DIR / f"{_get_cache_prefix(bid, f_hash)}.npy"
        if disk_cache_file.exists():
            try:
                emb_matrix = np.load(disk_cache_file)
                if len(emb_matrix) == len(chunks):
                    for c, row in zip(chunks, emb_matrix):
                        c["embedding"] = row
                    return emb_matrix
            except Exception:
                pass

    # 2. Check in-memory / per-chunk embedding cache
    missing_indices = []
    missing_texts = []
    embeddings_list: list[Optional[np.ndarray]] = [None] * len(chunks)

    for i, c in enumerate(chunks):
        if "embedding" in c and isinstance(c["embedding"], np.ndarray):
            embeddings_list[i] = c["embedding"]
            continue

        text = c.get("text", "")
        text_key = hashlib.sha256(f"{EMBEDDING_MODEL_NAME}::{text}".encode("utf-8")).hexdigest()
        if text_key in _MEM_CHUNK_CACHE:
            emb = _MEM_CHUNK_CACHE[text_key]
            c["embedding"] = emb
            embeddings_list[i] = emb
        else:
            missing_indices.append(i)
            missing_texts.append(text)

    # 3. Batch encode missing texts
    if missing_texts:
        m = model or _m()
        encoded = m.encode(
            missing_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if isinstance(encoded, list):
            encoded = np.array(encoded, dtype=np.float32)

        for idx, text, emb in zip(missing_indices, missing_texts, encoded):
            text_key = hashlib.sha256(f"{EMBEDDING_MODEL_NAME}::{text}".encode("utf-8")).hexdigest()
            _MEM_CHUNK_CACHE[text_key] = emb
            chunks[idx]["embedding"] = emb
            embeddings_list[idx] = emb

    res_matrix = np.array(embeddings_list, dtype=np.float32)

    # 4. Save to disk cache if book-level
    if len(book_ids) == 1:
        bid = list(book_ids)[0]
        f_hash = next((c.get("file_hash") for c in chunks if c.get("file_hash")), None)
        disk_cache_file = CACHE_DIR / f"{_get_cache_prefix(bid, f_hash)}.npy"
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(disk_cache_file, res_matrix)
        except Exception:
            pass

    return res_matrix


def score_chunks_against_outcomes(chunks: list[dict[str, Any]], outcomes: dict[str, str]):
    """Embed outcome descriptions and chunk texts to compute cosine similarities.

    Uses batch encoding, vectorized numpy matrix multiplication, and disk caching.
    Sets chunk["co_scores"] = {code: cosine_similarity} for every chunk and outcome.

    Args:
        chunks: list of chunk dicts (must contain 'text').
        outcomes: mapping of outcome code to description (e.g. {'CO1': '...', 'CO2': '...'}).

    Returns:
        The modified list of chunks with 'co_scores' populated.
    """
    if not chunks:
        return chunks
    if not outcomes:
        for c in chunks:
            c["co_scores"] = {}
        return chunks

    codes = list(outcomes.keys())
    outcomes_hash = hashlib.sha256(json.dumps(outcomes, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    # Check disk cache for precomputed co_scores
    book_ids = {c.get("book_id") for c in chunks if c.get("book_id")}
    if len(book_ids) == 1:
        bid = list(book_ids)[0]
        f_hash = next((c.get("file_hash") for c in chunks if c.get("file_hash")), None)
        co_cache_file = CACHE_DIR / f"{_get_cache_prefix(bid, f_hash)}_co_{outcomes_hash}.json"
        if co_cache_file.exists():
            try:
                with open(co_cache_file, "r", encoding="utf-8") as f:
                    cached_scores = json.load(f)
                if len(cached_scores) == len(chunks):
                    for c, scores in zip(chunks, cached_scores):
                        c["co_scores"] = scores
                    return chunks
            except Exception:
                pass

    # Batch embed chunks and outcomes
    model = _m()
    o_vecs = model.encode(
        list(outcomes.values()),
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if isinstance(o_vecs, list):
        o_vecs = np.array(o_vecs, dtype=np.float32)

    c_vecs = get_chunk_embeddings(chunks, model=model)

    # Vectorized similarity matrix computation (N_chunks x N_outcomes)
    sims = c_vecs @ o_vecs.T

    all_scores = []
    for c, row in zip(chunks, sims):
        scores = {code: float(s) for code, s in zip(codes, row)}
        c["co_scores"] = scores
        all_scores.append(scores)

    # Save to disk cache if book-level
    if len(book_ids) == 1:
        bid = list(book_ids)[0]
        f_hash = next((c.get("file_hash") for c in chunks if c.get("file_hash")), None)
        co_cache_file = CACHE_DIR / f"{_get_cache_prefix(bid, f_hash)}_co_{outcomes_hash}.json"
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(co_cache_file, "w", encoding="utf-8") as f:
                json.dump(all_scores, f)
        except Exception:
            pass

    return chunks
