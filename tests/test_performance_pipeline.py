"""Tests for performance optimizations in paper generation pipeline.

Verifies:
1. Output equivalence: mock client + fixed seed produces identical paper structure and output.
2. Multi-book disk cache skips embedding recomputation.
3. Ollama keep-alive and options remain constant across calls.
4. Truncated outputs are rejected.
"""
import pytest
import numpy as np
from blueprint import build_blueprint
from generation.mock_client import MockClient
from generation.generator import (
    generate_full_paper,
    generate_paper_by_slots,
    is_truncated_text,
)
from generation.ollama_client import OllamaClient
from paper_preset import make_preset
from retrieval.co_mapper import get_chunk_embeddings, score_chunks_against_outcomes
from tests.test_slots import _make_dummy_chunks


def test_output_equivalence_large_paper():
    """Verify mock client with fixed seed produces identical paper structure and content."""
    chunks = _make_dummy_chunks()
    preset = make_preset(
        "Choice50",
        units=3,
        marks_per_unit=15,
        options=2,
        sub_marks={1: [5, 10], 2: [5, 10], 3: [5, 10]},
        has_choice=True,
    )
    outcomes = {
        "CO1": "Understand containerization fundamentals and architecture",
        "CO2": "Design and manage container storage, networking, and security",
        "CO3": "Deploy and orchestrate containerized applications",
        "CO4": "Implement DevOps automation and CI/CD pipelines",
    }
    bp = build_blueprint(preset=preset, outcomes=list(outcomes.keys()), seed=42)

    client1 = MockClient()
    client2 = MockClient()

    res1 = generate_full_paper(chunks=chunks, blueprint=bp, outcomes=outcomes, unit_sources=None, client=client1, seed=42, workers=1)
    res2 = generate_full_paper(chunks=chunks, blueprint=bp, outcomes=outcomes, unit_sources=None, client=client2, seed=42, workers=3)

    qs1 = res1["questions"]
    qs2 = res2["questions"]

    assert len(qs1) == len(qs2) == len(bp["slots"])
    for q1, q2 in zip(qs1, qs2):
        assert q1["id"] == q2["id"]
        assert q1["unit"] == q2["unit"]
        assert q1["marks"] == q2["marks"]
        assert q1["level"] == q2["level"]
        assert q1["knowledge_dimension"] == q2["knowledge_dimension"]
        assert q1["question"] == q2["question"]
        assert q1["answer"] == q2["answer"]


def test_multi_book_disk_cache_skips_embedding(tmp_path, monkeypatch):
    """Verify that chunks across multiple books are disk-cached per book and hit on repeat runs."""
    import retrieval.co_mapper as cm
    monkeypatch.setattr(cm, "CACHE_DIR", tmp_path)

    # Chunks from two different books
    chunks = [
        {"chunk_id": "c1", "book_id": "b1", "file_hash": "hash1", "text": "Kubernetes pods encapsulate containers."},
        {"chunk_id": "c2", "book_id": "b1", "file_hash": "hash1", "text": "Services route traffic across pods."},
        {"chunk_id": "c3", "book_id": "b2", "file_hash": "hash2", "text": "Docker images define layered filesystems."},
        {"chunk_id": "c4", "book_id": "b2", "file_hash": "hash2", "text": "Containers isolate process execution."},
    ]

    outcomes = {
        "CO1": "Understand container orchestration and virtualization.",
        "CO2": "Manage container images and runtime environments.",
    }

    # First run: computes and saves cache files
    scored1 = score_chunks_against_outcomes(chunks, outcomes)
    assert all("co_scores" in c and len(c["co_scores"]) == 2 for c in scored1)

    # Verify disk files created for both books
    files = list(tmp_path.glob("*.json")) + list(tmp_path.glob("*.npy"))
    assert any("b1" in f.name for f in files)
    assert any("b2" in f.name for f in files)

    # Second run: mock embedder so that if encode is called, it raises an error
    class FailIfCalledModel:
        def encode(self, *args, **kwargs):
            raise AssertionError("encode should not be called on cache hit!")

    # Clear memory embeddings
    chunks_clean = [
        {"chunk_id": "c1", "book_id": "b1", "file_hash": "hash1", "text": "Kubernetes pods encapsulate containers."},
        {"chunk_id": "c2", "book_id": "b1", "file_hash": "hash1", "text": "Services route traffic across pods."},
        {"chunk_id": "c3", "book_id": "b2", "file_hash": "hash2", "text": "Docker images define layered filesystems."},
        {"chunk_id": "c4", "book_id": "b2", "file_hash": "hash2", "text": "Containers isolate process execution."},
    ]
    monkeypatch.setattr(cm, "_m", lambda: FailIfCalledModel())

    scored2 = score_chunks_against_outcomes(chunks_clean, outcomes)
    assert all("co_scores" in c and len(c["co_scores"]) == 2 for c in scored2)
    # Scores must match
    for c1, c2 in zip(scored1, scored2):
        assert c1["co_scores"] == c2["co_scores"]


def test_ollama_client_constant_options_and_metadata():
    """Verify OllamaClient maintains constant options, keep-alive, and logs metadata."""
    client = OllamaClient()
    assert client.keep_alive == "30m"

    # Verify payload construction
    recorded_payloads = []

    class DummySession:
        def post(self, url, json=None, timeout=None):
            recorded_payloads.append(json)
            class MockResponse:
                status_code = 200
                def raise_for_status(self): pass
                def json(self):
                    return {
                        "response": "Explain Docker architecture.",
                        "done_reason": "stop",
                        "total_duration": 1000000000,
                        "load_duration": 2000000,
                        "prompt_eval_count": 25,
                        "prompt_eval_duration": 300000000,
                        "eval_count": 50,
                        "eval_duration": 700000000,
                    }
            return MockResponse()

    client.session = DummySession()

    _ = client.generate("Test prompt 1", num_predict=120)
    _ = client.generate("Test prompt 2", num_predict=180)

    assert len(recorded_payloads) == 2
    for p in recorded_payloads:
        assert p["keep_alive"] == "30m"
        assert p["options"]["num_ctx"] == 2048
        assert p["options"]["temperature"] == 0.4

    # Verify metadata tracking
    assert len(client.call_history) == 2
    assert client.last_metadata["load_duration"] == 2000000
    assert client.last_metadata["eval_count"] == 50
    assert client.last_metadata["tokens_per_second"] > 0


def test_truncated_outputs_rejected():
    """Verify that truncated question and answer outputs are detected and rejected."""
    # Stems ending mid-sentence
    assert is_truncated_text("Explain how container namespaces provide", is_question=True)
    assert not is_truncated_text("Explain how container namespaces provide process isolation?", is_question=True)

    # Answers ending mid-sentence
    assert is_truncated_text("Docker images are built using layers that", is_question=False)
    assert not is_truncated_text("Docker images are built using layers that share base filesystems.", is_question=False)
