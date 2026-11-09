"""Tests for performance optimizations: parallel execution, caching, and fault tolerance."""
import random
from blueprint import build_blueprint
from generation.cache import clear_generation_cache
from generation.generator import generate_paper_by_slots
from generation.mock_client import MockClient
from tests.test_slots import _make_dummy_chunks


class CountingMockClient(MockClient):
    """Mock client that counts generate invocations."""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def generate(self, prompt, temperature=None, num_predict=120, format=None, **kwargs):
        self.call_count += 1
        return super().generate(prompt, temperature, num_predict, format=format, **kwargs)


class FailingMockClient(MockClient):
    """Mock client that throws an exception for a specific question level."""

    def __init__(self, fail_level="L5"):
        super().__init__()
        self.fail_level = fail_level

    def generate(self, prompt, temperature=None, num_predict=120, format=None, **kwargs):
        if f"Bloom's {self.fail_level}" in prompt:
            raise RuntimeError(f"Simulated LLM crash for {self.fail_level}")
        return super().generate(prompt, temperature, num_predict, format=format, **kwargs)


def test_output_equivalence_with_fixed_seed():
    """Confirm paper generated with mock client and fixed seed maintains identical slots, marks, and distribution."""
    chunks = _make_dummy_chunks()
    slots = [
        {"unit": 1, "id": "1(a)", "co": "CO1", "level": "L1", "marks": 5, "knowledge_dimension": "Factual"},
        {"unit": 1, "id": "1(b)", "co": "CO2", "level": "L2", "marks": 5, "knowledge_dimension": "Conceptual"},
        {"unit": 2, "id": "2(a)", "co": "CO1", "level": "L3", "marks": 7, "knowledge_dimension": "Procedural"},
        {"unit": 2, "id": "2(b)", "co": "CO2", "level": "L4", "marks": 3, "knowledge_dimension": "Conceptual"},
    ]

    client = MockClient()
    # Run with workers=1 (sequential)
    res_seq = generate_paper_by_slots(chunks, slots, client, seed=42, workers=1, force_regenerate=True)
    # Run with workers=3 (parallel)
    res_par = generate_paper_by_slots(chunks, slots, client, seed=42, workers=3, force_regenerate=True)

    qs_seq = res_seq["questions"]
    qs_par = res_par["questions"]

    assert len(qs_seq) == len(qs_par) == len(slots)
    for q_s, q_p, slot in zip(qs_seq, qs_par, slots):
        assert q_s["id"] == q_p["id"] == slot["id"]
        assert q_s["co"] == q_p["co"] == slot["co"]
        assert q_s["level"] == q_p["level"] == slot["level"]
        assert q_s["marks"] == q_p["marks"] == slot["marks"]
        assert q_s["knowledge_dimension"] == q_p["knowledge_dimension"] == slot["knowledge_dimension"]
        assert q_s["question"] == q_p["question"]


def test_parallel_generation_preserving_slot_order():
    """Assert parallel generation preserves the exact original slot sequence and numbering."""
    chunks = _make_dummy_chunks()
    slots = [
        {"unit": 1, "id": f"Slot_{i:02d}", "co": "CO1" if i % 2 == 0 else "CO2", "level": "L2", "marks": 5}
        for i in range(1, 9)
    ]
    client = MockClient()
    res = generate_paper_by_slots(chunks, slots, client, seed=123, workers=4, force_regenerate=True)

    qs = res["questions"]
    assert len(qs) == 8
    returned_ids = [q["id"] for q in qs]
    expected_ids = [s["id"] for s in slots]
    assert returned_ids == expected_ids


def test_cache_hit_skipping_llm_call():
    """Assert cache hit reuses stored response and skips calling the LLM."""
    clear_generation_cache()
    chunks = _make_dummy_chunks()
    slots = [
        {"unit": 1, "id": "Q1", "co": "CO1", "level": "L2", "marks": 5},
    ]

    counting_client = CountingMockClient()

    # 1. First run: cold cache -> calls LLM
    res1 = generate_paper_by_slots(chunks, slots, counting_client, seed=10, workers=1, force_regenerate=False)
    assert len(res1["questions"]) == 1
    calls_after_run1 = counting_client.call_count
    assert calls_after_run1 > 0

    # 2. Second run: warm cache -> skips LLM
    res2 = generate_paper_by_slots(chunks, slots, counting_client, seed=10, workers=1, force_regenerate=False)
    assert len(res2["questions"]) == 1
    calls_after_run2 = counting_client.call_count
    # Call count does NOT increase because responses came from cache!
    assert calls_after_run2 == calls_after_run1

    # 3. Third run with force_regenerate=True -> bypasses cache and calls LLM
    res3 = generate_paper_by_slots(chunks, slots, counting_client, seed=10, workers=1, force_regenerate=True)
    assert len(res3["questions"]) == 1
    assert counting_client.call_count > calls_after_run2


def test_one_failed_slot_does_not_crash_batch():
    """Assert a crashed or failing slot is safely placed into unfilled without killing other slots."""
    chunks = _make_dummy_chunks()
    slots = [
        {"unit": 1, "id": "Q1", "co": "CO1", "level": "L2", "marks": 5},
        {"unit": 1, "id": "Q2_Fail", "co": "CO1", "level": "L5", "marks": 5},  # Will fail/crash in FailingMockClient
        {"unit": 2, "id": "Q3", "co": "CO2", "level": "L2", "marks": 5},
    ]

    failing_client = FailingMockClient(fail_level="L5")
    res = generate_paper_by_slots(chunks, slots, failing_client, seed=99, workers=3, force_regenerate=True)

    questions = res["questions"]
    unfilled = res["unfilled"]

    assert len(questions) == 2
    assert [q["id"] for q in questions] == ["Q1", "Q3"]
    assert len(unfilled) == 1
    assert unfilled[0]["id"] == "Q2_Fail"

