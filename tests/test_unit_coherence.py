"""Tests for PART G: Unit Coherence, OR Chapter/Section Partitioning, and Coverage Balance."""
import pytest
from generation.generator import check_chapter_coverage_balance, generate_paper_by_slots
from generation.mock_client import MockClient
from retrieval.sources import select_chunks_for_unit


def _create_coherent_unit_chunks():
    chunks = [
        # Unit 1 - Chapter 1
        {
            "chunk_id": "u1_ch1_c1",
            "book_id": "book1",
            "chapter": "Chapter 1",
            "section": "Section 1.1",
            "text": "Docker container lifecycle execution environments and daemon architecture overview. " * 6,
            "page_start": 10,
            "word_count": 60,
            "co_scores": {"CO1": 0.95},
        },
        # Unit 1 - Chapter 2
        {
            "chunk_id": "u1_ch2_c1",
            "book_id": "book1",
            "chapter": "Chapter 2",
            "section": "Section 2.1",
            "text": "Docker storage drivers and volume management primitives with persistent layers. " * 6,
            "page_start": 25,
            "word_count": 60,
            "co_scores": {"CO1": 0.90},
        },
        # Unit 2 - Chapter 3
        {
            "chunk_id": "u2_ch3_c1",
            "book_id": "book2",
            "chapter": "Chapter 3",
            "section": "Section 3.1",
            "text": "Kubernetes pod networking and overlay bridge interface routing models. " * 6,
            "page_start": 40,
            "word_count": 60,
            "co_scores": {"CO2": 0.95},
        },
    ]
    return chunks


def test_strict_unit_sources_enforcement():
    """Verify that select_chunks_for_unit strictly enforces mapped chapters and returns empty if no match, without silent fallback."""
    chunks = _create_coherent_unit_chunks()
    unit_sources = {
        1: {"book1": ["Chapter 1"]},
        2: {"book2": ["Chapter 99"]},  # Non-existent chapter in book2
    }

    # Unit 1 gets only Chapter 1
    u1_chunks = select_chunks_for_unit(chunks, unit_sources, unit=1)
    assert len(u1_chunks) == 1
    assert u1_chunks[0]["chapter"] == "Chapter 1"

    # Unit 2 mapped to non-existent chapter strictly gets 0 chunks (does NOT leak book1 chunks)
    u2_chunks = select_chunks_for_unit(chunks, unit_sources, unit=2)
    assert len(u2_chunks) == 0


def test_or_branches_partition_across_chapters():
    """Verify OR branches (Option A vs Option B) partition across distinct chapters in the unit."""
    chunks = _create_coherent_unit_chunks()
    unit_sources = {
        1: {"book1": ["Chapter 1", "Chapter 2"]},
    }
    slots = [
        {"unit": 1, "question_no": 1, "sub_index": 1, "level": "L2", "co": "CO1", "marks": 10, "alternative_group": "A", "kd": "Conceptual"},
        {"unit": 1, "question_no": 2, "sub_index": 1, "level": "L2", "co": "CO1", "marks": 10, "alternative_group": "B", "kd": "Conceptual"},
    ]
    res = generate_paper_by_slots(chunks, slots, client=MockClient(), unit_sources=unit_sources, seed=42)
    assert len(res["questions"]) == 2

    q_a = next(q for q in res["questions"] if q["question_no"] == 1)
    q_b = next(q for q in res["questions"] if q["question_no"] == 2)

    assert q_a["chapter"] != q_b["chapter"]
    assert {q_a["chapter"], q_b["chapter"]} == {"Chapter 1", "Chapter 2"}


def test_or_branches_partition_across_sections():
    """When a unit has a single chapter with multiple sections, OR branches partition across sections."""
    chunks = [
        {
            "chunk_id": "sec1",
            "book_id": "book1",
            "chapter": "Chapter 1",
            "section": "Section 1.1",
            "text": "Docker storage drivers and overlayfs mechanics explained. " * 6,
            "page_start": 10,
            "word_count": 60,
            "co_scores": {"CO1": 0.9},
        },
        {
            "chunk_id": "sec2",
            "book_id": "book1",
            "chapter": "Chapter 1",
            "section": "Section 1.2",
            "text": "Docker volume mount points and bind mount security considerations. " * 6,
            "page_start": 20,
            "word_count": 60,
            "co_scores": {"CO1": 0.9},
        },
    ]
    unit_sources = {
        1: {"book1": ["Chapter 1"]},
    }
    slots = [
        {"unit": 1, "question_no": 1, "sub_index": 1, "level": "L2", "co": "CO1", "marks": 10, "alternative_group": "A", "kd": "Conceptual"},
        {"unit": 1, "question_no": 2, "sub_index": 1, "level": "L2", "co": "CO1", "marks": 10, "alternative_group": "B", "kd": "Conceptual"},
    ]
    res = generate_paper_by_slots(chunks, slots, client=MockClient(), unit_sources=unit_sources, seed=42)
    assert len(res["questions"]) == 2

    q_a = next(q for q in res["questions"] if q["question_no"] == 1)
    q_b = next(q for q in res["questions"] if q["question_no"] == 2)

    assert q_a["page"] != q_b["page"]


def test_chapter_coverage_balance_check():
    """Verify check_chapter_coverage_balance emits warnings when mapped chapters have 0 questions."""
    unit_sources = {
        1: {"book1": ["Chapter 1", "Chapter 2"]},
    }
    questions = [
        {"unit": 1, "chapter": "Chapter 1", "id": "1(a)"},
        {"unit": 1, "chapter": "Chapter 1", "id": "1(b)"},
    ]
    warnings, coverage = check_chapter_coverage_balance(questions, unit_sources)
    assert len(warnings) == 1
    assert "Chapter 2" in warnings[0]
    assert coverage[1]["Chapter 1"] == 2
    assert coverage[1]["Chapter 2"] == 0

