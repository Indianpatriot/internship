"""Tests for regenerate_missing_slots and regeneration context reliability."""
import pytest
from generation.generator import regenerate_missing_slots, verify_paper_completeness
from generation.mock_client import MockClient


def _sample_chunks():
    """Build mock textbook chunks for testing."""
    return [
        {
            "chunk_id": "c1",
            "book_id": "docker_practice",
            "chapter": "Chapter 2: Docker Fundamentals",
            "page_start": 25,
            "page_end": 26,
            "text": (
                "Docker containers run isolated processes utilizing Linux cgroups and namespaces. "
                "The Docker daemon builds container images using layered Dockerfile instructions. "
                "Container images encapsulate runtime dependencies, system binaries, and configuration files. "
                "Port mapping binds container internal ports to host interfaces allowing external network access."
            ),
            "word_count": 50,
            "co_scores": {"CO1": 0.95, "CO2": 0.20},
        },
        {
            "chunk_id": "c2",
            "book_id": "docker_practice",
            "chapter": "Chapter 3: Docker Networking",
            "page_start": 45,
            "page_end": 46,
            "text": (
                "Bridge networks enable communication between containers on the same Docker host daemon. "
                "Overlay networks facilitate multi-host communication across distributed swarm nodes. "
                "Host networking removes network isolation by attaching container processes directly to the host stack."
            ),
            "word_count": 40,
            "co_scores": {"CO1": 0.30, "CO2": 0.92},
        },
        {
            "chunk_id": "c3",
            "book_id": "k8s_action",
            "chapter": "Chapter 4: Kubernetes Pods and ReplicaSets",
            "page_start": 80,
            "page_end": 82,
            "text": (
                "Kubernetes Pods represent the basic execution unit containing one or more co-located containers. "
                "ReplicaSets maintain a specified number of identical running Pod replicas using declarative selectors. "
                "Deployments manage declarative updates for Pods and ReplicaSets through rollouts and revisions."
            ),
            "word_count": 45,
            "co_scores": {"CO1": 0.15, "CO2": 0.95},
        },
    ]


def test_regenerating_fills_only_missing_slots_and_leaves_accepted_untouched():
    """Requirement: Regenerating with 2 missing slots fills only those 2 and leaves accepted questions untouched."""
    chunks = _sample_chunks()
    slots = [
        {"id": "1(a)", "question_no": 1, "sub_index": 1, "marks": 5, "co": "CO1", "level": "L2", "unit": 1},
        {"id": "1(b)", "question_no": 1, "sub_index": 2, "marks": 5, "co": "CO1", "level": "L2", "unit": 1},
        {"id": "2(a)", "question_no": 2, "sub_index": 1, "marks": 5, "co": "CO2", "level": "L2", "unit": 1},
        {"id": "2(b)", "question_no": 2, "sub_index": 2, "marks": 5, "co": "CO2", "level": "L2", "unit": 1},
    ]

    orig_q0_text = "Explain the fundamental architecture and process isolation of Docker containers?"
    orig_q0_ans = "Docker containers isolate processes using Linux cgroups and namespaces."
    orig_q2_text = "Describe how bridge networks enable communication between containers on the host?"
    orig_q2_ans = "Bridge networks provide a virtual software switch linking local container interfaces."

    accepted_q0 = {
        "id": "1(a)",
        "question_no": 1,
        "sub_index": 1,
        "marks": 5,
        "co": "CO1",
        "level": "L2",
        "unit": 1,
        "question": orig_q0_text,
        "answer": orig_q0_ans,
        "valid": True,
        "chunk_id": "c1",
        "book_id": "docker_practice",
        "page": 25,
    }
    placeholder_q1 = {
        "id": "1(b)",
        "question_no": 1,
        "sub_index": 2,
        "marks": 5,
        "co": "CO1",
        "level": "L2",
        "unit": 1,
        "question": "[NEEDS REVIEW: Generation retries exhausted]",
        "answer": "[NEEDS REVIEW: Model answer generation failed]",
        "valid": False,
        "needs_review": True,
        "review_flags": ["generation_failed: retries exhausted"],
    }
    accepted_q2 = {
        "id": "2(a)",
        "question_no": 2,
        "sub_index": 1,
        "marks": 5,
        "co": "CO2",
        "level": "L2",
        "unit": 1,
        "question": orig_q2_text,
        "answer": orig_q2_ans,
        "valid": True,
        "chunk_id": "c2",
        "book_id": "docker_practice",
        "page": 45,
    }
    placeholder_q3 = {
        "id": "2(b)",
        "question_no": 2,
        "sub_index": 2,
        "marks": 5,
        "co": "CO2",
        "level": "L2",
        "unit": 1,
        "question": "[NEEDS REVIEW: Generation retries exhausted]",
        "answer": "[NEEDS REVIEW: Model answer generation failed]",
        "valid": False,
        "needs_review": True,
        "review_flags": ["generation_failed: retries exhausted"],
    }

    ctx = {
        "scored_chunks": chunks,
        "slots": slots,
        "blueprint": slots,
        "client": MockClient(),
        "seed": 42,
    }
    result = {
        "questions": [accepted_q0, placeholder_q1, accepted_q2, placeholder_q3],
        "unfilled": [slots[1], slots[3]],
        "rejected": [],
    }

    regen_res = regenerate_missing_slots(ctx, result, seed_bump=1)
    res_qs = regen_res["questions"]

    assert len(res_qs) == 4
    # Accepted questions MUST remain completely untouched
    assert res_qs[0]["question"] == orig_q0_text
    assert res_qs[0]["answer"] == orig_q0_ans
    assert res_qs[2]["question"] == orig_q2_text
    assert res_qs[2]["answer"] == orig_q2_ans

    # Incomplete slots must now be validly filled
    assert res_qs[1]["valid"] is True
    assert not str(res_qs[1]["question"]).startswith("[NEEDS REVIEW")
    assert len(res_qs[1]["question"]) > 10
    assert len(res_qs[1]["answer"]) > 10

    assert res_qs[3]["valid"] is True
    assert not str(res_qs[3]["question"]).startswith("[NEEDS REVIEW")
    assert len(res_qs[3]["question"]) > 10
    assert len(res_qs[3]["answer"]) > 10

    # Unfilled list must now be empty
    assert len(regen_res["unfilled"]) == 0


def test_slot_order_numbering_and_marks_preserved_after_merge():
    """Requirement: Slot order, numbering (e.g. 5(a), 5(b), 5(c)), and marks match blueprint after merge."""
    chunks = _sample_chunks()
    slots = [
        {"id": "5(a)", "question_no": 5, "sub_index": 1, "marks": 5, "co": "CO1", "level": "L2", "unit": 3},
        {"id": "5(b)", "question_no": 5, "sub_index": 2, "marks": 5, "co": "CO1", "level": "L2", "unit": 3},
        {"id": "5(c)", "question_no": 5, "sub_index": 3, "marks": 10, "co": "CO2", "level": "L3", "unit": 3},
    ]

    q_5a = {
        "id": "5(a)",
        "question_no": 5,
        "sub_index": 1,
        "marks": 5,
        "co": "CO1",
        "level": "L2",
        "unit": 3,
        "question": "Explain Docker container runtime isolation.",
        "answer": "Process isolation via namespaces and cgroups.",
        "valid": True,
        "chunk_id": "c1",
    }
    q_5b_placeholder = {
        "id": "5(b)",
        "question_no": 5,
        "sub_index": 2,
        "marks": 5,
        "co": "CO1",
        "level": "L2",
        "unit": 3,
        "question": "[NEEDS REVIEW: Generation retries exhausted]",
        "answer": "[NEEDS REVIEW]",
        "valid": False,
        "needs_review": True,
    }
    q_5c = {
        "id": "5(c)",
        "question_no": 5,
        "sub_index": 3,
        "marks": 10,
        "co": "CO2",
        "level": "L3",
        "unit": 3,
        "question": "Apply Kubernetes ReplicaSets to maintain Pod availability.",
        "answer": "ReplicaSets use label selectors to manage desired pod counts.",
        "valid": True,
        "chunk_id": "c3",
    }

    ctx = {
        "scored_chunks": chunks,
        "slots": slots,
        "blueprint": slots,
        "client": MockClient(),
        "seed": 42,
    }
    result = {
        "questions": [q_5a, q_5b_placeholder, q_5c],
        "unfilled": [slots[1]],
        "rejected": [],
    }

    regen_res = regenerate_missing_slots(ctx, result)
    res_qs = regen_res["questions"]

    # Slot IDs must preserve exact original order and labels
    assert [q["id"] for q in res_qs] == ["5(a)", "5(b)", "5(c)"]
    assert [q["marks"] for q in res_qs] == [5, 5, 10]
    assert [q["question_no"] for q in res_qs] == [5, 5, 5]
    assert [q["sub_index"] for q in res_qs] == [1, 2, 3]
    assert [q["unit"] for q in res_qs] == [3, 3, 3]
    assert res_qs[1]["valid"] is True
    assert res_qs[1]["id"] == "5(b)"
    assert res_qs[1]["marks"] == 5


def test_no_duplicates_introduced_between_regenerated_and_existing():
    """Requirement: No duplicates are introduced between regenerated and existing questions."""
    chunks = _sample_chunks()
    slots = [
        {"id": "1(a)", "question_no": 1, "sub_index": 1, "marks": 5, "co": "CO1", "level": "L2", "unit": 1},
        {"id": "1(b)", "question_no": 1, "sub_index": 2, "marks": 5, "co": "CO1", "level": "L2", "unit": 1},
    ]

    existing_question = "Explain the fundamental architecture and process isolation of Docker containers?"
    q0 = {
        "id": "1(a)",
        "question_no": 1,
        "sub_index": 1,
        "marks": 5,
        "co": "CO1",
        "level": "L2",
        "unit": 1,
        "question": existing_question,
        "answer": "Docker isolates processes via namespaces.",
        "valid": True,
        "chunk_id": "c1",
    }
    q1_placeholder = {
        "id": "1(b)",
        "question_no": 1,
        "sub_index": 2,
        "marks": 5,
        "co": "CO1",
        "level": "L2",
        "unit": 1,
        "question": "[NEEDS REVIEW: duplicate question]",
        "answer": "[NEEDS REVIEW]",
        "valid": False,
        "needs_review": True,
    }

    ctx = {
        "scored_chunks": chunks,
        "slots": slots,
        "blueprint": slots,
        "client": MockClient(),
        "seed": 42,
    }
    result = {
        "questions": [q0, q1_placeholder],
        "unfilled": [slots[1]],
        "rejected": [],
    }

    regen_res = regenerate_missing_slots(ctx, result)
    res_qs = regen_res["questions"]

    # Regenerated question must not be identical to existing question
    assert res_qs[1]["question"].strip().lower() != existing_question.strip().lower()


def test_unfillable_slot_stays_unfilled_with_specific_reason():
    """Requirement: An unfillable slot (e.g. no chunks for that CO) stays unfilled with specific reason and hint."""
    chunks = _sample_chunks()
    unfillable_slot = {
        "id": "3(a)",
        "question_no": 3,
        "sub_index": 1,
        "marks": 10,
        "co": "CO99",  # CO99 does NOT exist in any chunks
        "level": "L4",
        "unit": 2,
    }
    slots = [unfillable_slot]

    placeholder_q = {
        "id": "3(a)",
        "question_no": 3,
        "sub_index": 1,
        "marks": 10,
        "co": "CO99",
        "level": "L4",
        "unit": 2,
        "question": "[NEEDS REVIEW: no chunks]",
        "answer": "[NEEDS REVIEW]",
        "valid": False,
        "needs_review": True,
    }

    ctx = {
        "scored_chunks": chunks,
        "slots": slots,
        "blueprint": slots,
        "client": MockClient(),
        "seed": 42,
    }
    result = {
        "questions": [placeholder_q],
        "unfilled": [unfillable_slot],
        "rejected": [],
    }

    regen_res = regenerate_missing_slots(ctx, result)

    # Must not crash!
    assert len(regen_res["unfilled"]) == 1
    unfilled_record = regen_res["unfilled"][0]
    # Check that a specific reason and hint are attached
    assert "reason" in unfilled_record
    assert "hint" in unfilled_record
    reason_str = unfilled_record["reason"].lower()
    assert "co99" in reason_str or "no content" in reason_str or "selected chapters" in reason_str or "slot constraints" in reason_str
    assert len(unfilled_record["hint"]) > 5

    # Check question placeholder in paper
    q_res = regen_res["questions"][0]
    assert q_res["valid"] is False
    assert str(q_res["question"]).startswith("[NEEDS REVIEW")


def test_empty_or_missing_context_raises_clean_value_error():
    """Requirement: Calling regenerate_missing_slots with empty or missing context raises ValueError with actionable message."""
    dummy_result = {"questions": [], "unfilled": [], "rejected": []}
    dummy_ctx = {"scored_chunks": [], "slots": [], "client": MockClient()}

    # 1. Missing context (None)
    with pytest.raises(ValueError) as exc_info:
        regenerate_missing_slots(None, dummy_result)
    assert "Please click Generate first" in str(exc_info.value)

    # 2. Empty context dict ({})
    with pytest.raises(ValueError) as exc_info:
        regenerate_missing_slots({}, dummy_result)
    assert "Please click Generate first" in str(exc_info.value)

    # 3. Missing result (None and no result in ctx)
    with pytest.raises(ValueError) as exc_info:
        regenerate_missing_slots(dummy_ctx, None)
    assert "Please click Generate first" in str(exc_info.value)

    # 4. Empty result dict ({})
    with pytest.raises(ValueError) as exc_info:
        regenerate_missing_slots(dummy_ctx, {})
    assert "Please click Generate first" in str(exc_info.value)


def test_all_slots_already_complete_returns_early():
    """When all slots are already complete, return immediately with clear message."""
    chunks = _sample_chunks()
    slots = [
        {"id": "1(a)", "question_no": 1, "sub_index": 1, "marks": 5, "co": "CO1", "level": "L2", "unit": 1},
    ]
    accepted_q0 = {
        "id": "1(a)",
        "question_no": 1,
        "sub_index": 1,
        "marks": 5,
        "co": "CO1",
        "level": "L2",
        "unit": 1,
        "question": "Explain Docker architecture?",
        "answer": "Process isolation via namespaces.",
        "valid": True,
        "chunk_id": "c1",
    }
    ctx = {
        "scored_chunks": chunks,
        "slots": slots,
        "blueprint": slots,
        "client": MockClient(),
        "seed": 42,
    }
    result = {
        "questions": [accepted_q0],
        "unfilled": [],
        "rejected": [],
    }

    regen_res = regenerate_missing_slots(ctx, result)
    assert "complete" in regen_res.get("message", "").lower()
    assert regen_res["questions"][0]["question"] == accepted_q0["question"]

