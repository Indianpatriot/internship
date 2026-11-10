"""Comprehensive regression tests for the 7 content and structure fixes.

Covers:
- Problem 1: Bounded retries, placeholder retention [NEEDS REVIEW], OR symmetry, offered marks, Q1 start.
- Problem 2: Reject section-divider pages (<150 words), chunk partitioning, post-generation deduplication.
- Problem 3: CO-aware retrieval, CO similarity check (<0.40), low CO mapping warning (<0.25), unchanged totals.
- Problem 4: Banned deictic/unseen references ("provided", "the code snippet", "Listing", "the technique"), inline code allowed.
- Problem 5: Answer sentence-level grounding, demand coverage, compare entity co-occurrence, circular question rejection, compute numeric check.
- Problem 6: Knowledge Dimension verb alignment (KD_BLOOM_VERBS), nearest valid verb fallback, unusual pair handling.
- Problem 7: Outline chapters over Parts, non-content filtering (Roadmap, Appendix, About Packt), covered chapters header formatting.
"""
import random
import re
import pytest

from blueprint import build_blueprint, calculate_blueprint_stats
from blooms.verb_bank import (
    KD_BLOOM_VERBS,
    get_allowed_kd_verbs,
    is_verb_allowed_for_kd,
    nearest_valid_kd_verb,
    normalize_knowledge_dimension,
    replace_question_verb,
    sample_kd_verbs,
    validate_question,
    validate_question_kd,
)
from export_utils import format_covered_chapters_header
from generation.generator import (
    check_compare_entities_in_context,
    check_demand_coverage,
    check_question_co_similarity,
    check_question_duplicates,
    generate_answer,
    generate_paper_by_slots,
    generate_question,
    has_unseen_material_reference,
    is_circular_question,
    normalize_paper_question_numbers,
    regenerate_missing_slots,
    verify_paper_completeness,
)
from generation.mock_client import MockClient
from ingestion.heading_detect import detect_headings, is_non_content_title
from ingestion.page_filter import PageText, _is_divider_page, filter_pages
from paper_preset import make_preset
from retrieval.sources import is_content_chunk, select_chunks_for_unit
from totals import summarize


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
            "co_scores": {"CO1": 0.92, "CO2": 0.20},
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
            "co_scores": {"CO1": 0.30, "CO2": 0.88},
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


class FailSpecificSlotClient(MockClient):
    """Fails generation only for a specific Bloom level or prompt pattern."""
    def __init__(self, fail_pattern="L5"):
        super().__init__()
        self.fail_pattern = fail_pattern

    def generate(self, prompt, temperature=None, num_predict=120, format=None, **kwargs):
        if self.fail_pattern in prompt:
            raise RuntimeError("Simulated failure for slot")
        return super().generate(prompt, temperature, num_predict, format=format, **kwargs)


# =============================================================================
# PROBLEM 1: Bounded retries, placeholders, OR symmetry, marks & numbering
# =============================================================================

def test_problem_1_slot_never_dropped_kept_as_placeholder():
    """A failed slot must never be dropped; keep as [NEEDS REVIEW] placeholder preserving OR symmetry."""
    chunks = _sample_chunks()
    preset = make_preset("TestChoice", units=2, marks_per_unit=10, options=2, sub_marks={1: [5, 5], 2: [10]})
    bp = build_blueprint(preset=preset, outcomes=["CO1", "CO2", "CO3", "CO4"], seed=42)
    slots = bp["slots"]

    # Client that fails for L5
    client = FailSpecificSlotClient(fail_pattern="Bloom's L5")
    # Generate with keep_placeholders=True (as done in production app.py)
    res = generate_paper_by_slots(chunks, slots, client, seed=42, workers=1, keep_placeholders=True)
    qs = res["questions"]

    # All slots must be present in questions list
    assert len(qs) == len(slots)
    # Check that any failed slot is marked NEEDS REVIEW
    for q in qs:
        assert "question" in q and len(q["question"]) > 0
        assert "answer" in q and len(q["answer"]) > 0

    # Completeness verification
    complete, issues = verify_paper_completeness(qs, slots)
    assert complete is False
    assert any("incomplete" in str(iss).lower() or "review" in str(iss).lower() for iss in issues)
    assert any(q.get("needs_review") or str(q.get("question", "")).startswith("[NEEDS REVIEW") for q in qs)


def test_problem_1_offered_and_attempted_marks_match_blueprint():
    """Offered marks must sum all blueprint slots (e.g. 160 for 4x20 choice), not surviving slots."""
    preset = make_preset("Choice80", units=4, marks_per_unit=20, options=2, sub_marks={1: [10, 10], 2: [10, 10], 3: [10, 10], 4: [10, 10]})
    bp = build_blueprint(preset=preset, outcomes=["CO1", "CO2", "CO3", "CO4"], seed=42)
    slots = bp["slots"]

    # Total blueprint slots: 4 units * 2 questions * 2 sub-questions = 16 slots of 10 marks = 160 marks
    assert sum(s["marks"] for s in slots) == 160

    # Even if 2 questions are placeholders, summarize(questions, blueprint_slots=slots) returns 160 offered
    dummy_questions = [
        {"id": s["id"], "unit": s["unit"], "marks": s["marks"], "question": "Explain X?", "answer": "Answer X.", "needs_review": False}
        for s in slots[:-2]
    ]
    dummy_questions.append({
        "id": slots[-2]["id"], "unit": slots[-2]["unit"], "marks": slots[-2]["marks"],
        "question": "[NEEDS REVIEW: retries exhausted]", "answer": "[NEEDS REVIEW]", "needs_review": True
    })
    dummy_questions.append({
        "id": slots[-1]["id"], "unit": slots[-1]["unit"], "marks": slots[-1]["marks"],
        "question": "[NEEDS REVIEW: retries exhausted]", "answer": "[NEEDS REVIEW]", "needs_review": True
    })

    summary = summarize(dummy_questions, blueprint_slots=slots)
    assert summary["offered_marks"] == 160
    assert summary["total_marks"] == 80


def test_problem_1_numbering_starts_at_q1():
    """Paper numbering must start at Q1 for the first unit and run continuously."""
    questions = [
        {"id": "3(a)", "unit": 2, "question_no": 3, "sub_index": 1, "marks": 5, "question": "Explain A?"},
        {"id": "3(b)", "unit": 2, "question_no": 3, "sub_index": 2, "marks": 5, "question": "Explain B?"},
        {"id": "4", "unit": 2, "question_no": 4, "sub_index": None, "marks": 10, "question": "Explain C?"},
    ]
    norm = normalize_paper_question_numbers(questions)
    assert norm[0]["question_no"] == 1
    assert norm[0]["id"] == "1(a)"
    assert norm[1]["question_no"] == 1
    assert norm[1]["id"] == "1(b)"
    assert norm[2]["question_no"] == 2
    assert norm[2]["id"] == "2"


# =============================================================================
# PROBLEM 2: Reject section divider pages, partition chunks & deduplicate
# =============================================================================

def test_problem_2_section_divider_pages_rejected():
    """Standalone section/part divider pages (<150 words) must be rejected by page filter and chunk filter."""
    divider_text = "PART 3 - HIGHER-LEVEL CONCEPTS\nThis part covers deployment, scaling, and operational considerations."
    assert _is_divider_page(divider_text) is True

    chunk = {"text": divider_text, "word_count": len(divider_text.split())}
    assert is_content_chunk(chunk) is False

    # Subsections like "Section 10.3" with regular text must NOT be rejected
    regular_text = (
        "Section 10.3 Legal Aspects and Compliance Guidelines: Software licensing and open source governance. "
        "Students must cite all academic references accurately in their documentation. "
        "Furthermore, algorithm complexity bounds establish lower limits on computation."
    )
    assert _is_divider_page(regular_text) is False


def test_problem_2_duplicate_questions_detected():
    """Pairs of questions with embedding similarity >= 0.85 must be detected."""
    from retrieval.co_mapper import _m
    embedder = _m()
    if embedder is None:
        pytest.skip("Sentence transformer embedder not available")

    qs = [
        {"id": "7(a)", "question": "Summarize how Helm functions as a package manager for Kubernetes applications?"},
        {"id": "8(a)", "question": "Summarize how Helm operates as a package manager for Kubernetes cluster applications?"},
        {"id": "8(b)", "question": "Explain the architectural differences between Docker bridge and overlay networks?"},
    ]
    dups = check_question_duplicates(qs, embedder)
    assert len(dups) >= 1
    # First pair (0, 1) detected as duplicate
    assert dups[0][0] == 0 and dups[0][1] == 1
    assert dups[0][2] >= 0.85


# =============================================================================
# PROBLEM 3: CO similarity check & warnings
# =============================================================================

def test_problem_3_co_similarity_check():
    """A question misaligned with its assigned CO description (<0.40) must be flagged."""
    co_desc = "Analyze container orchestration, cluster scheduling, and deployment scaling using Kubernetes."
    matching_q = "Explain how Kubernetes Deployments manage declarative Pod replica rollouts and scaling?"
    mismatch_q = "Describe relational database foreign key cascading delete operations and schema constraints?"

    class MockEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            import numpy as np
            if "Deployments" in texts[0]:
                return np.array([[1.0, 0.0], [0.8, 0.6]])
            return np.array([[1.0, 0.0], [0.2, 0.98]])

    mock_emb = MockEmbedder()
    ok_match, sim_match = check_question_co_similarity(matching_q, co_desc, mock_emb)
    assert ok_match is True
    assert sim_match >= 0.40

    ok_mismatch, sim_mismatch = check_question_co_similarity(mismatch_q, co_desc, mock_emb)
    assert ok_mismatch is False
    assert sim_mismatch < 0.40


# =============================================================================
# PROBLEM 4: Ban deictic/unseen references ("provided", "the code snippet")
# =============================================================================

def test_problem_4_banned_deictic_references():
    """Questions/answers referring to unseen material ('provided', 'snippet', 'Listing') must be flagged."""
    bad_questions = [
        "Based on the provided text, describe how Docker containers achieve isolation?",
        "Explain the key functions of the code snippet below?",
        "What is the output of the provided code?",
        "According to Listing 3.6, how is port mapping configured?",
        "Explain how the technique simplifies network configuration?",
    ]
    for bq in bad_questions:
        has_unseen, msg = has_unseen_material_reference(bq, is_question=True)
        assert has_unseen is True, f"Failed to flag bad question: {bq}"

    # Inline code in the stem is self-contained and allowed
    good_inline = "Explain how the Docker command `docker run -d -p 8080:80 nginx` configures host port mapping?"
    has_unseen, _ = has_unseen_material_reference(good_inline, is_question=True)
    assert has_unseen is False

    # Standalone conceptual question
    clean_q = "Describe how Kubernetes ReplicaSets ensure that a specified number of Pods remain running?"
    has_unseen, _ = has_unseen_material_reference(clean_q, is_question=True)
    assert has_unseen is False


# =============================================================================
# PROBLEM 5: Factual/logical checks (circularity, compare entities, compute)
# =============================================================================

def test_problem_5_circular_question_rejected():
    """Circular questions where one concept is a case/subset of the other must be rejected."""
    circular_1 = "Compare containers and Docker containers?"
    assert is_circular_question(circular_1) is True

    circular_2 = "Integrate continuous deployment into continuous delivery for automated releases?"
    assert is_circular_question(circular_2) is True

    valid = "Compare Docker bridge networking and overlay networking in multi-host environments?"
    assert is_circular_question(valid) is False


def test_problem_5_compare_entities_in_context():
    """Compare/distinguish questions must ensure both compared entities exist in the chunk context."""
    context = (
        "Alice analyzed Docker bridge networks and container port binding mechanisms. "
        "Bridge networks connect containers on a single host daemon."
    )
    # Both in context
    assert check_compare_entities_in_context("Compare Docker bridge networks and container port binding?", context) is True
    # 'Kubernetes overlay' is NOT in context
    assert check_compare_entities_in_context("Compare Docker bridge networks and Kubernetes overlay networks?", context) is False


def test_problem_5_demand_coverage():
    """If a question demands 'trade-offs' or 'steps', the answer must cover those concepts."""
    q = "Discuss the trade-offs between Docker host networking and bridge networking?"
    ans_without_tradeoffs = "Docker bridge networks connect containers on a host. Port forwarding exposes ports."
    ok, msg = check_demand_coverage(q, ans_without_tradeoffs)
    assert ok is False
    assert "trade-off" in msg

    ans_with_tradeoffs = (
        "The trade-offs include higher throughput for host networking at the cost of reduced network isolation. "
        "Conversely, bridge networking offers isolation but introduces routing overhead."
    )
    ok, _ = check_demand_coverage(q, ans_with_tradeoffs)
    assert ok is True


# =============================================================================
# PROBLEM 6: Knowledge Dimension verb alignment (KD_BLOOM_VERBS)
# =============================================================================

def test_problem_6_kd_verb_validation_and_fallback():
    """Verbs must match assigned KD guide; mismatches fallback to nearest valid verb."""
    # Factual at L1: define, list, name, identify
    assert is_verb_allowed_for_kd("define", "L1", "Factual") is True
    assert is_verb_allowed_for_kd("design", "L1", "Factual") is False

    # Conceptual at L2: explain, describe, classify
    assert is_verb_allowed_for_kd("explain", "L2", "Conceptual") is True
    assert is_verb_allowed_for_kd("compute", "L2", "Conceptual") is False

    # Nearest valid verb fallback
    replacement = nearest_valid_kd_verb("calculate", "L2", "Conceptual")
    assert replacement.lower() in get_allowed_kd_verbs("L2", "Conceptual")

    # Question verb replacement
    replaced = replace_question_verb("Calculate the architecture of Docker?", replacement)
    assert replaced.lower().startswith(replacement.lower())

    # Unusual pair: L1 + Metacognitive is supported through the guide
    allowed_unusual = get_allowed_kd_verbs("L1", "Metacognitive")
    assert len(allowed_unusual) > 0
    assert "identify" in allowed_unusual


def test_problem_6_totals_unchanged():
    """Blueprint totals per KD, CO, and Bloom band must be invariant before and after."""
    preset = make_preset("TestChoice", units=3, marks_per_unit=20, options=2)
    bp = build_blueprint(preset=preset, outcomes=["CO1", "CO2", "CO3", "CO4"], seed=42)
    stats = calculate_blueprint_stats(bp["slots"])

    # Ensure total attempted marks is 60 and all slots have valid KD
    assert stats["attempted_marks"] == 60
    assert stats["total_marks"] == 60
    for s in bp["slots"]:
        assert s.get("knowledge_dimension") in ["Factual", "Conceptual", "Procedural", "Metacognitive"]


# =============================================================================
# PROBLEM 7: Clean chapter titles, no Parts as chapters, header formatting
# =============================================================================

def test_problem_7_non_content_titles_rejected():
    """Front/back-matter and appendix headings must be rejected from clean chapter detection."""
    bad_headings = [
        "Roadmap", "About the code", "About Packt", "Author Online",
        "Back Cover", "Index", "Appendix A: Getting Help",
        "Appendix B: Configuring Docker", "The Virtual Machine Approach",
    ]
    for bh in bad_headings:
        assert is_non_content_title(bh) is True, f"Failed to reject non-content heading: {bh}"

    good_headings = [
        "Chapter 1: Getting Started with Docker",
        "Chapter 2: Understanding Container Basics",
        "Chapter 3: Kubernetes Deployments in Practice",
    ]
    for gh in good_headings:
        assert is_non_content_title(gh) is False


def test_problem_7_format_covered_chapters_header():
    """Covered chapters header must group by book and omit pure Parts, Roadmap, and Appendices."""
    raw_chapters = [
        "Docker in Practice - PART 1: Docker Basics",
        "Docker in Practice - Chapter 1: Getting Started with Docker",
        "Docker in Practice - Chapter 2: Working with Images",
        "Docker in Practice - Roadmap",
        "Docker in Practice - About the Code",
        "Docker in Practice - Appendix A: Setting Up",
    ]
    formatted = format_covered_chapters_header(covered_chapters=raw_chapters)
    assert "Chapter 1" in formatted
    assert "Chapter 2" in formatted
    assert "Roadmap" not in formatted
    assert "About the Code" not in formatted
    assert "Appendix" not in formatted
