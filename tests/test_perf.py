"""Performance and speed regression guards."""
import os
import random
import numpy as np
import pytest

from blueprint import build_blueprint
from generation.mock_client import MockClient
from generation.generator import generate_paper_by_slots
from retrieval.co_mapper import get_chunk_embeddings, score_chunks_against_outcomes, _m

SAMPLE_PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples", "sample_book.pdf")


def test_embedding_cache_hit_avoids_reencoding(monkeypatch):
    """Verify that a second call with unchanged chunks hits cache and does not call encode."""
    chunks = [
        {"chunk_id": "c1", "text": "Docker architecture consists of client, daemon, and container runtime components.", "book_id": "test_bk"},
        {"chunk_id": "c2", "text": "Kubernetes pod orchestration handles deployment and scaling of containers.", "book_id": "test_bk"},
    ]
    # First call encodes and caches
    embs_1 = get_chunk_embeddings(chunks)
    assert embs_1.shape == (2, 384)

    # Monkeypatch encode to fail if called
    model = _m()
    called = []
    orig_encode = model.encode

    def fail_encode(*args, **kwargs):
        called.append(True)
        raise RuntimeError("encode() was called despite chunks being cached!")

    monkeypatch.setattr(model, "encode", fail_encode)
    try:
        embs_2 = get_chunk_embeddings(chunks)
        assert len(called) == 0, "Embedding encode was invoked when cache should have hit."
        assert np.allclose(embs_1, embs_2, atol=1e-5)
    finally:
        monkeypatch.setattr(model, "encode", orig_encode)


def test_co_scores_matrix_multiplication_accuracy():
    """Verify outcome scoring produces expected accurate cosine similarities."""
    chunks = [
        {"chunk_id": "c1", "text": "Containerization provides lightweight isolation for microservices.", "book_id": "test_co_bk"},
        {"chunk_id": "c2", "text": "Database indexing optimizes SQL query execution plans.", "book_id": "test_co_bk"},
    ]
    outcomes = {
        "CO1": "Understand containerization and microservices isolation.",
        "CO2": "Analyze relational database query execution and indexing.",
    }
    scored = score_chunks_against_outcomes(chunks, outcomes)
    assert len(scored) == 2
    # c1 should match CO1 higher than CO2
    assert scored[0]["co_scores"]["CO1"] > scored[0]["co_scores"]["CO2"]
    # c2 should match CO2 higher than CO1
    assert scored[1]["co_scores"]["CO2"] > scored[1]["co_scores"]["CO1"]


def test_concurrent_generation_determinism_with_same_seed():
    """Verify deterministic question ordering across different worker runs with the same seed."""
    chunks = [
        {"chunk_id": "c1", "text": "Docker images are built from layered Dockerfiles with build caching.", "word_count": 80, "book_id": "b1", "page_start": 1, "page_end": 1, "chapter": "1", "display_label": "DevOps Handbook", "co_scores": {"CO1": 0.85}},
        {"chunk_id": "c2", "text": "Virtual networks connect containers through software bridge interfaces.", "word_count": 80, "book_id": "b1", "page_start": 2, "page_end": 2, "chapter": "2", "display_label": "DevOps Handbook", "co_scores": {"CO1": 0.80}},
    ]
    slots = [
        {"id": "1", "question_no": 1, "unit": 1, "level": "L2", "marks": 5, "co": "CO1", "knowledge_dimension": "Conceptual"},
        {"id": "2", "question_no": 2, "unit": 1, "level": "L2", "marks": 5, "co": "CO1", "knowledge_dimension": "Conceptual"},
    ]

    client = MockClient()
    res1 = generate_paper_by_slots(chunks=chunks, slots=slots, client=client, seed=42, workers=1)
    res2 = generate_paper_by_slots(chunks=chunks, slots=slots, client=client, seed=42, workers=2)

    q_texts1 = [q["question"] for q in res1["questions"]]
    q_texts2 = [q["question"] for q in res2["questions"]]
    assert q_texts1 == q_texts2
