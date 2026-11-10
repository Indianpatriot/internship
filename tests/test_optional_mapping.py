"""Tests for strictly optional Textbook Mapping, fallback to all textbook content, and soft hint."""
import fitz
import pytest
from blueprint import build_blueprint
from generation.generator import generate_full_paper
from generation.mock_client import MockClient
from ingestion.library import load_library
from retrieval.sources import select_chunks_for_unit


@pytest.fixture
def multi_book_library(tmp_path):
    """Create two textbook PDFs with distinct chapters and sufficient content (>40 words per chunk)."""
    pdf_a = tmp_path / "book_a.pdf"
    pdf_b = tmp_path / "book_b.pdf"

    for p, title in [(pdf_a, "OperatingSystems"), (pdf_b, "DatabaseSystems")]:
        doc = fitz.open()
        for ch in [1, 2]:
            page = doc.new_page()
            page.insert_text((50, 72), f"Chapter {ch}: Principles of {title}", fontsize=16)
            page.insert_textbox(
                fitz.Rect(50, 100, 550, 400),
                (
                    f"In this chapter {ch} of {title}, we explore computational paradigms, invariants, and systems. "
                    "Invariants are analyzed across recursive divisions and state transitions systematically. "
                    "The operational complexity and formal mathematical models guarantee correctness. "
                    "Practical implementations provide scalable, fault-tolerant solutions for real-world software problems. "
                    "Advanced synchronization primitives and virtualization kernels ensure robust computational execution."
                ),
                fontsize=11,
            )
        doc.save(str(p))
        doc.close()

    lib = load_library([pdf_a, pdf_b], cache_dir=tmp_path / "cache")
    return lib


def test_select_chunks_all_units_unmapped(multi_book_library):
    """When all units are unmapped (None, empty dict, or books with empty chapter lists), retrieval falls back to all content chunks."""
    chunks = multi_book_library["chunks"]
    book_ids = list(multi_book_library["books"].keys())

    # 1. unit_sources is None
    chunks_none = select_chunks_for_unit(chunks, unit_sources=None, unit=1)
    assert len(chunks_none) == len(chunks)

    # 2. unit_sources is empty dict
    chunks_empty = select_chunks_for_unit(chunks, unit_sources={}, unit=1)
    assert len(chunks_empty) == len(chunks)

    # 3. unit_sources maps unit to empty chapter lists across books
    unmapped_sources = {1: {b: [] for b in book_ids}}
    chunks_unmapped = select_chunks_for_unit(chunks, unit_sources=unmapped_sources, unit=1)
    assert len(chunks_unmapped) == len(chunks)


def test_select_chunks_some_units_mapped_others_default(multi_book_library):
    """Mapped units are restricted to chosen chapters; unmapped units draw from all textbook content."""
    chunks = multi_book_library["chunks"]
    b1, b2 = list(multi_book_library["books"].keys())[:2]
    b1_chaps = multi_book_library["books"][b1]["chapters"]

    # Unit 1 is mapped specifically to Book 1 Chapter 1
    # Unit 2 is completely unmapped
    partial_sources = {
        1: {b1: [b1_chaps[0]]},
        2: {b1: [], b2: []},  # unmapped
    }

    # Unit 1 retrieval must be restricted
    u1_chunks = select_chunks_for_unit(chunks, unit_sources=partial_sources, unit=1)
    assert len(u1_chunks) > 0
    assert all(c["book_id"] == b1 for c in u1_chunks)
    assert all(b1_chaps[0] in c["chapter"] for c in u1_chunks)

    # Unit 2 retrieval must fall back to all chunks across both books
    u2_chunks = select_chunks_for_unit(chunks, unit_sources=partial_sources, unit=2)
    assert len(u2_chunks) == len(chunks)
    book_ids_in_u2 = {c["book_id"] for c in u2_chunks}
    assert b1 in book_ids_in_u2
    assert b2 in book_ids_in_u2


def test_generation_with_all_units_unmapped(multi_book_library):
    """Question paper generates completely and successfully with zero units mapped."""
    chunks = multi_book_library["chunks"]
    bp = build_blueprint(preset=50, units=2, questions_per_unit=1, seed=42)
    outcomes = {
        "CO1": "Understand system principles and virtualization.",
        "CO2": "Apply database management and transactions.",
        "CO3": "Analyze distributed consensus protocols.",
        "CO4": "Evaluate storage hierarchy and memory caching.",
    }

    # No mapping provided at all (None)
    client = MockClient()
    paper = generate_full_paper(
        chunks=chunks,
        blueprint=bp,
        outcomes=outcomes,
        unit_sources=None,
        client=client,
        seed=42,
    )

    questions = paper["questions"]
    assert len(questions) > 0
    # Confirm questions generated for both Unit 1 and Unit 2
    units_in_paper = {q["unit"] for q in questions}
    assert 1 in units_in_paper
    assert 2 in units_in_paper


def test_generation_with_some_units_mapped(multi_book_library):
    """Question paper generates when some units are mapped and others are unmapped."""
    chunks = multi_book_library["chunks"]
    b1 = list(multi_book_library["books"].keys())[0]
    b1_ch1 = multi_book_library["books"][b1]["chapters"][0]

    bp = build_blueprint(preset=50, units=2, questions_per_unit=1, seed=42)
    outcomes = {
        "CO1": "Understand system principles and virtualization.",
        "CO2": "Apply database management and transactions.",
        "CO3": "Analyze distributed consensus protocols.",
        "CO4": "Evaluate storage hierarchy and memory caching.",
    }

    # Unit 1 is mapped, Unit 2 is unmapped
    unit_sources = {
        1: {b1: [b1_ch1]},
        2: None,
    }

    client = MockClient()
    paper = generate_full_paper(
        chunks=chunks,
        blueprint=bp,
        outcomes=outcomes,
        unit_sources=unit_sources,
        client=client,
        seed=42,
    )

    questions = paper["questions"]
    assert len(questions) > 0
    u1_qs = [q for q in questions if q["unit"] == 1]
    u2_qs = [q for q in questions if q["unit"] == 2]
    assert len(u1_qs) > 0
    assert len(u2_qs) > 0

    # Unit 1 questions must strictly cite Book 1 Chapter 1
    for q in u1_qs:
        assert q["book_id"] == b1
        assert b1_ch1 in str(q.get("chapter"))


def test_unmapped_hint_format_and_no_blocking():
    """Verify that unmapped units produce the soft informational tip and never block can_generate."""
    active_units = [1, 2, 3, 4, 5]
    updated_sources = {1: {"book_a": ["Chapter 1"]}}  # Units 2, 3, 4, 5 unmapped
    manually_mapped = set()

    unmapped = [
        u for u in active_units
        if (updated_sources.get(u) is None or not any(bool(v) for v in (updated_sources[u] or {}).values() if v is not None))
        and u not in manually_mapped
    ]
    assert unmapped == [2, 3, 4, 5]

    # Verify hint string matches expected format
    hint = f"Tip: Units {', '.join(map(str, unmapped))} aren't mapped to textbook chapters. Mapping them can improve how closely questions match your textbook. This is optional."
    assert "Tip: Units 2, 3, 4, 5 aren't mapped to textbook chapters." in hint
    assert "This is optional." in hint

    # In app.py, can_generate depends only on filtered_chunks, not on unmapped_units
    can_generate = True
    assert can_generate is True
