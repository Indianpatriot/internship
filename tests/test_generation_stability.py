"""Tests for generation pipeline stability, worker concurrency clamps, timeouts, and failure diagnostics."""
import pytest
from unittest.mock import patch, MagicMock

import config
from contracts import Slot
from generation.mock_client import MockClient
from generation.ollama_client import OllamaClient
from generation.generator import generate_paper_by_slots, resolve_worker_count


def _sample_chunks():
    return [
        {
            "id": "c1",
            "chunk_id": "c1",
            "text": "Docker engine uses a client-server architecture with a persistent daemon process dockerd. The daemon manages container objects such as images, containers, networks, and storage volumes. Clients communicate with dockerd using a REST API over UNIX sockets or a network interface.",
            "chapter": "Chapter 1",
            "page_start": 1,
            "page_end": 2,
            "book_id": "docker_guide",
            "co_scores": {"CO1": 0.85, "CO2": 0.30},
        }
    ]


def _sample_slots():
    return [
        Slot(
            id="1(a)",
            question_no=1,
            sub_index="a",
            unit=1,
            co="CO1",
            level="L2",
            marks=5,
            knowledge_dimension="Conceptual",
        ),
        Slot(
            id="1(b)",
            question_no=1,
            sub_index="b",
            unit=1,
            co="CO1",
            level="L3",
            marks=5,
            knowledge_dimension="Procedural",
        ),
    ]


def test_ollama_worker_clamp_forces_single_worker(monkeypatch):
    """Verify that Ollama client is clamped to workers=1 even if caller requests multiple workers."""
    monkeypatch.setattr(config, "ALLOW_PARALLEL_OLLAMA", False)
    ollama_client = OllamaClient(model="llama3.1:8b")

    # Direct clamp resolution check
    assert resolve_worker_count(ollama_client, requested_workers=4, slot_count=10) == 1
    assert resolve_worker_count(ollama_client, requested_workers=None, slot_count=10) == 1


def test_ollama_worker_clamp_override_with_flag(monkeypatch):
    """Verify that setting ALLOW_PARALLEL_OLLAMA allows requested worker count for Ollama."""
    monkeypatch.setattr(config, "ALLOW_PARALLEL_OLLAMA", True)
    ollama_client = OllamaClient(model="llama3.1:8b")

    assert resolve_worker_count(ollama_client, requested_workers=3, slot_count=10) == 3
    assert resolve_worker_count(ollama_client, requested_workers=None, slot_count=10) == min(2, config.GENERATION_WORKERS)


def test_mock_client_preserves_parallel_workers(monkeypatch):
    """Verify that MockClient retains parallel concurrency when multiple workers are requested."""
    monkeypatch.setattr(config, "ALLOW_PARALLEL_OLLAMA", False)
    mock_client = MockClient()

    assert resolve_worker_count(mock_client, requested_workers=4, slot_count=10) == 4
    assert resolve_worker_count(mock_client, requested_workers=None, slot_count=10) == min(config.GENERATION_WORKERS, 10)


def test_generate_paper_by_slots_runs_fast_with_mock():
    """Verify generate_paper_by_slots completes with MockClient without hanging."""
    mock_client = MockClient()
    chunks = _sample_chunks()
    slots = _sample_slots()

    res = generate_paper_by_slots(
        chunks=chunks,
        slots=slots,
        client=mock_client,
        workers=2,
    )
    assert len(res["questions"]) == 2
    assert all(q.get("valid") for q in res["questions"])

import json
from unittest.mock import MagicMock
from generation.ollama_client import OllamaTimeoutError


def test_ollama_client_streaming_accumulates_and_computes_tok_s():
    """Verify OllamaClient streams chunks and accurately calculates tokens/sec from metadata."""
    client = OllamaClient(model="llama3.1:8b")
    
    stream_lines = [
        json.dumps({"response": "Docker ", "done": False}),
        json.dumps({"response": "architecture.", "done": False}),
        json.dumps({
            "response": "",
            "done": True,
            "done_reason": "stop",
            "eval_count": 20,
            "eval_duration": 4_000_000_000,  # 4 seconds -> 5.0 tok/s
        }),
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.iter_lines.return_value = stream_lines

    with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
        output = client.generate("Explain Docker architecture", num_predict=50)
        
        assert output == "Docker architecture."
        assert mock_post.call_count == 1
        _, kwargs = mock_post.call_args
        assert kwargs.get("stream") is True
        assert kwargs.get("json", {}).get("stream") is True
        
        # Verify tokens_per_second calculation
        assert client.last_metadata["tokens_per_second"] == 5.0
        assert client.measured_tok_s == 5.0
        assert mock_resp.close.call_count >= 1


def test_ollama_client_inactivity_timeout_raises_timeout_error():
    """Verify that if tokens stop arriving within inactivity timeout, OllamaTimeoutError is raised."""
    client = OllamaClient(model="llama3.1:8b", retries=1)
    client.inactivity_timeout = 0.05  # 50ms for test speed

    def slow_generator():
        yield json.dumps({"response": "Initial token", "done": False})
        import time
        time.sleep(0.1)  # Exceeds inactivity timeout of 0.05s
        yield json.dumps({"response": "Late token", "done": True})

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.iter_lines.return_value = slow_generator()

    with patch.object(client.session, "post", return_value=mock_resp):
        with pytest.raises(OllamaTimeoutError):
            client.generate("Test prompt")
        assert mock_resp.close.call_count >= 1



def test_failing_slot_never_exceeds_call_budget(monkeypatch):
    """Verify that a failing slot strictly respects MAX_SLOT_CALL_BUDGET and halts retries."""
    budget = 4
    monkeypatch.setattr(config, "MAX_SLOT_CALL_BUDGET", budget)

    call_count = 0

    class FailingMockClient:
        def __init__(self):
            self.model = "mock"

        def generate(self, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return garbage that fails question validation
            return "This is not a valid question ending with a question mark"

    failing_client = FailingMockClient()
    chunks = _sample_chunks()
    slots = [
        Slot(
            id="1",
            question_no=1,
            unit=1,
            co="CO1",
            level="L2",
            marks=5,
            knowledge_dimension="Conceptual",
        )
    ]

    results = generate_paper_by_slots(
        chunks=chunks,
        slots=slots,
        client=failing_client,
        workers=1,
        keep_placeholders=True,
    )

    # Must produce a placeholder
    assert len(results["questions"]) == 1
    placeholder = results["questions"][0]
    assert placeholder.get("valid") is False
    assert placeholder.get("needs_review") is True
    # Assert call count strictly bounded by budget
    assert call_count <= budget
    assert placeholder.get("calls_used", 0) <= budget



def test_failure_injection_reason_codes():
    """Verify that simulated client failures correctly produce expected ReasonCodes."""
    from generation.diagnostics import ReasonCode

    chunks = _sample_chunks()
    slot = [
        Slot(
            id="1",
            question_no=1,
            unit=1,
            co="CO1",
            level="L2",
            marks=5,
            knowledge_dimension="Conceptual",
        )
    ]

    # 1. Timeout injection
    class TimeoutClient:
        model = "mock"
        def generate(self, *args, **kwargs):
            raise OllamaTimeoutError("Stream timed out after 60s")

    res_to = generate_paper_by_slots(chunks=chunks, slots=slot, client=TimeoutClient(), workers=1, keep_placeholders=True)
    assert len(res_to["questions"]) == 1
    assert res_to["questions"][0]["reason_code"] == ReasonCode.OLLAMA_TIMEOUT.value

    # 2. Empty response injection
    class EmptyClient:
        model = "mock"
        def generate(self, *args, **kwargs):
            return ""

    res_empty = generate_paper_by_slots(chunks=chunks, slots=slot, client=EmptyClient(), workers=1, keep_placeholders=True)
    assert len(res_empty["questions"]) == 1
    assert res_empty["questions"][0]["reason_code"] == ReasonCode.EMPTY_RESPONSE.value

    # 3. Bloom verb fail injection
    class BadVerbClient:
        model = "mock"
        def generate(self, *args, **kwargs):
            # Question with verb not matching L2
            return "Evaluate the performance and security risks of containerization?"

    res_verb = generate_paper_by_slots(chunks=chunks, slots=slot, client=BadVerbClient(), workers=1, keep_placeholders=True)
    assert len(res_verb["questions"]) == 1
    assert res_verb["questions"][0]["reason_code"] == ReasonCode.BLOOM_VERB_FAIL.value



def test_combined_answer_and_scheme_call_reduction():
    """Verify single-call generation of model answer and itemized marking scheme."""
    from generation.generator import generate_answer_and_scheme

    mock_client = MockClient()
    chunk = _sample_chunks()[0]

    ans_rec, scheme_rec = generate_answer_and_scheme(
        question="Explain the role of dockerd in Docker architecture?",
        chunk=chunk,
        marks=5,
        client=mock_client,
        level="L2",
    )

    assert ans_rec["valid"] is True
    assert len(ans_rec["answer"]) > 20
    assert scheme_rec is not None
    assert "criteria" in scheme_rec
    assert len(scheme_rec["criteria"]) >= 1
    # Verify sum of criteria equals marks exactly
    total_criteria_marks = sum(c["marks"] for c in scheme_rec["criteria"])
    assert total_criteria_marks == 5



def test_detailed_preflight_check_and_runtime_estimation():
    """Verify preflight check detects model residency, GPU offload %, RAM/swap, and warns if runtime > 90m."""
    client = OllamaClient(model="llama3.1:8b")
    client.measured_tok_s = 2.0  # Slow speed on partial GPU

    def mock_get(url, *args, **kwargs):
        mock_r = MagicMock()
        mock_r.ok = True
        mock_r.status_code = 200
        if url.endswith("/api/tags"):
            mock_r.json.return_value = {"models": [{"name": "llama3.1:8b:latest"}]}
        elif url.endswith("/api/ps"):
            # Partial offload: 16 of 32 layers ~ 50%
            mock_r.json.return_value = {
                "models": [{"name": "llama3.1:8b", "size": 4_000_000_000, "size_vram": 2_000_000_000}]
            }
        return mock_r

    def mock_post(url, *args, **kwargs):
        mock_r = MagicMock()
        mock_r.ok = True
        mock_r.status_code = 200
        mock_r.json.return_value = {"eval_count": 10, "eval_duration": 5_000_000_000}  # 2 tok/s
        return mock_r

    with patch.object(client.session, "get", side_effect=mock_get), \
         patch.object(client.session, "post", side_effect=mock_post):
        # 40 slots * 2 calls * 180 tokens / 2 tok/s = 7200s = 120 minutes (> 90m)
        status = client.detailed_preflight_check(slot_count=40)

        assert status["ok"] is True
        assert status["gpu_offload_pct"] == 50.0
        assert status["free_ram_mb"] is not None
        assert status["est_minutes"] >= 90.0
        assert status["suggest_small_model"] is True
        assert any("exceeds 90 minutes" in w for w in status["warnings"])
        assert any("only 50.0% offloaded to GPU" in w for w in status["warnings"])


def test_progress_tracking_records_elapsed_time_and_calls_used():
    """Verify on_progress callback receives records with elapsed_time and calls_used."""
    mock_client = MockClient()
    chunks = _sample_chunks()
    slots = _sample_slots()

    progress_records = []

    def on_prog(rec):
        progress_records.append(dict(rec))

    res = generate_paper_by_slots(
        chunks=chunks,
        slots=slots,
        client=mock_client,
        on_progress=on_prog,
        workers=1,
    )

    assert len(res["questions"]) == 2
    assert len(progress_records) == 2
    for r in progress_records:
        assert "elapsed_time" in r
        assert isinstance(r["elapsed_time"], (int, float))
        assert "calls_used" in r
        assert r["calls_used"] >= 1


def test_incremental_checkpoint_persists_to_disk_and_resumes(tmp_path):
    """Verify generation progress is saved to disk incrementally and resumed without re-running."""
    from generation.generator import (
        get_checkpoint_path,
        load_generation_checkpoint,
    )

    mock_client = MockClient()
    chunks = _sample_chunks()
    slots = _sample_slots()
    session_id = "test_resilience_session_1"
    cache_dir = str(tmp_path)

    # 1. Run generation with session_id
    res = generate_paper_by_slots(
        chunks=chunks,
        slots=slots,
        client=mock_client,
        session_id=session_id,
        cache_dir=cache_dir,
        workers=1,
    )
    assert len(res["questions"]) == 2

    # 2. Verify checkpoint exists on disk
    cp_path = get_checkpoint_path(session_id, cache_dir=cache_dir)
    assert cp_path.is_file()

    cp_data = load_generation_checkpoint(session_id, cache_dir=cache_dir)
    assert cp_data is not None
    assert cp_data["session_id"] == session_id
    assert len(cp_data["completed_slots"]) == 2
    assert 0 in cp_data["completed_slots"]
    assert 1 in cp_data["completed_slots"]
    assert cp_data["completed_slots"][0]["status"] == "ok"
    assert "elapsed_time" in cp_data["completed_slots"][0]

    # 3. Resume with a client that raises if invoked
    class ExplodingClient:
        def generate(self, *args, **kwargs):
            raise AssertionError("Client was called when it should have resumed from disk checkpoint!")

    resumed = generate_paper_by_slots(
        chunks=chunks,
        slots=slots,
        client=ExplodingClient(),
        session_id=session_id,
        cache_dir=cache_dir,
        force_regenerate=False,
        workers=1,
    )
    assert len(resumed["questions"]) == 2
    assert resumed["questions"][0]["question"] == res["questions"][0]["question"]
    assert resumed["questions"][1]["question"] == res["questions"][1]["question"]


def test_regenerate_missing_slots_updates_disk_checkpoint(tmp_path):
    """Verify regenerate_missing_slots resumes incomplete slots and updates disk checkpoint."""
    from generation.generator import (
        load_generation_checkpoint,
        regenerate_missing_slots,
    )

    mock_client = MockClient()
    chunks = _sample_chunks()
    slots = _sample_slots()
    session_id = "test_regen_checkpoint_session_2"
    cache_dir = str(tmp_path)

    # Create initial result with slot 0 valid and slot 1 incomplete placeholder
    initial_questions = [
        {
            "id": "1(a)",
            "question": "Explain the role of dockerd in managing container storage volumes?",
            "answer": "dockerd manages container storage volumes through storage driver plugins.",
            "marks": 5,
            "unit": 1,
            "co": "CO1",
            "level": "L2",
            "bloom": "L2",
            "bt_level": "L2",
            "valid": True,
            "needs_review": False,
            "calls_used": 1,
            "elapsed_time": 1.2,
        },
        {
            "id": "1(b)",
            "question": "[NEEDS REVIEW: criteria check failed]",
            "answer": "[NEEDS REVIEW: Select additional chapters]",
            "marks": 5,
            "unit": 1,
            "co": "CO1",
            "level": "L3",
            "bloom": "L3",
            "bt_level": "L3",
            "valid": False,
            "needs_review": True,
            "reason_code": "BUDGET_EXCEEDED",
            "calls_used": 6,
            "elapsed_time": 4.5,
        },
    ]

    ctx = {
        "scored_chunks": chunks,
        "slots": [s.to_dict() if hasattr(s, "to_dict") else dict(s) for s in slots],
        "blueprint": [s.to_dict() if hasattr(s, "to_dict") else dict(s) for s in slots],
        "client": mock_client,
        "seed": 42,
        "session_id": session_id,
        "cache_dir": cache_dir,
    }

    regen_result = regenerate_missing_slots(
        ctx=ctx,
        result={"questions": initial_questions, "unfilled": [], "rejected": []},
        session_id=session_id,
        cache_dir=cache_dir,
    )

    # Slot 0 should remain untouched
    assert regen_result["questions"][0]["question"] == initial_questions[0]["question"]
    # Slot 1 should now be generated and valid
    assert regen_result["questions"][1]["valid"] is True
    assert not str(regen_result["questions"][1]["question"]).startswith("[NEEDS REVIEW")

    # Checkpoint on disk should reflect both slots as completed ok
    cp = load_generation_checkpoint(session_id, cache_dir=cache_dir)
    assert cp is not None
    assert cp["completed_slots"][0]["status"] == "ok"
    assert cp["completed_slots"][1]["status"] == "ok"

