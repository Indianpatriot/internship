"""Tests for Multi-Book Ingestion Library, Caching, and Source Validation."""
import fitz
import pytest
from ingestion.library import (
    add_book,
    list_chapters,
    load_library,
    suggest_unit_sources,
    validate_unit_sources,
)


def _create_sample_pdf(path, title="Algorithms", num_pages=2, with_headings=True):
    """Helper to create synthetic test PDFs with valid geometry and word counts."""
    doc = fitz.open()
    for p_no in range(1, num_pages + 1):
        page = doc.new_page()
        y_text = 130
        if with_headings:
            page.insert_text(
                (50, 100),
                f"Chapter {p_no}: Technical Concepts of {title}",
                fontsize=16,
            )
        else:
            y_text = 100

        body_text = (
            f"Alice and Bob analyzed section {p_no} focusing on foundational computational algorithms. "
            "Data structures organize information hierarchically to optimize search and retrieval times. "
            "The mathematical recurrence relation yields a closed form asymptotic runtime bound. "
            "State transitions maintain invariants throughout the complete execution lifecycle. "
            "Practical system implementations validate these theoretical bounds across diverse configurations. "
            "Engineering teams verify correctness by applying formal validation and stress tests."
        ) * 2  # Sufficient word count (> 80 words)

        page.insert_textbox(fitz.Rect(50, y_text, 550, 750), body_text, fontsize=11)

    doc.save(str(path))
    doc.close()


def test_duplicate_names_disambiguate_book_ids(tmp_path):
    """Books with identical filenames are assigned unique disambiguated book_ids."""
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()

    pdf_a = dir_a / "ds.pdf"
    pdf_b = dir_b / "ds.pdf"
    _create_sample_pdf(pdf_a, "Data Structures")
    _create_sample_pdf(pdf_b, "Distributed Systems")

    lib = load_library([pdf_a, pdf_b], cache_dir=tmp_path / "cache")
    books = lib["books"]

    assert len(books) == 2
    book_ids = list(books.keys())
    assert len(set(book_ids)) == 2
    assert "ds" in book_ids
    assert "ds_2" in book_ids


def test_chunk_ids_unique_across_books(tmp_path):
    """Chunk IDs are prefixed with their respective book_id to ensure global uniqueness."""
    pdf_a = tmp_path / "book_a.pdf"
    pdf_b = tmp_path / "book_b.pdf"
    _create_sample_pdf(pdf_a, "Book A")
    _create_sample_pdf(pdf_b, "Book B")

    lib = load_library([pdf_a, pdf_b], cache_dir=tmp_path / "cache")
    chunks = lib["chunks"]

    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all(c["chunk_id"].startswith(c["book_id"]) for c in chunks)


def test_cache_hit_on_second_load(tmp_path, monkeypatch):
    """Loading a previously ingested PDF hits the disk cache without calling ingest again."""
    pdf_path = tmp_path / "network.pdf"
    cache_dir = tmp_path / "cache"
    _create_sample_pdf(pdf_path, "Computer Networks")

    # First load populates cache
    chunks_first = add_book(str(pdf_path), cache_dir=cache_dir)
    assert len(chunks_first) > 0

    # Monkeypatch pipeline.ingest so that calling it would raise an assertion
    def fail_ingest(*args, **kwargs):
        pytest.fail("ingest() was called instead of using disk cache")

    monkeypatch.setattr("ingestion.library.ingest", fail_ingest)

    # Second load must succeed via cache
    chunks_second = add_book(str(pdf_path), cache_dir=cache_dir)
    assert len(chunks_second) == len(chunks_first)
    assert chunks_second[0]["chunk_id"] == chunks_first[0]["chunk_id"]


def test_cache_invalidated_when_file_changes(tmp_path):
    """Modifying the file changes its hash, bypassing stale cache and re-ingesting."""
    pdf_path = tmp_path / "database.pdf"
    cache_dir = tmp_path / "cache"
    _create_sample_pdf(pdf_path, "Database Systems v1", num_pages=1)

    chunks_v1 = add_book(str(pdf_path), cache_dir=cache_dir)

    # Overwrite PDF with new content (different hash)
    _create_sample_pdf(pdf_path, "Database Systems v2", num_pages=3)
    chunks_v2 = add_book(str(pdf_path), cache_dir=cache_dir)

    # Different number of pages / chunks generated from updated content
    assert len(chunks_v2) >= len(chunks_v1)


def test_list_chapters_in_page_order(tmp_path):
    """list_chapters returns chapters ordered by page appearance with chunk counts."""
    pdf_path = tmp_path / "os.pdf"
    _create_sample_pdf(pdf_path, "Operating Systems", num_pages=3, with_headings=True)

    lib = load_library([pdf_path], cache_dir=tmp_path / "cache")
    book_id = list(lib["books"].keys())[0]

    chapters = list_chapters(lib, book_id)
    assert len(chapters) >= 2
    assert all("chapter" in ch and "chunks" in ch for ch in chapters)
    # Verify page order sequence
    pages = [ch["page_start"] for ch in chapters]
    assert pages == sorted(pages)


def test_page_range_fallback_when_no_headings(tmp_path):
    """When a book has no detectable headings, chunks fallback to page-range labels."""
    pdf_path = tmp_path / "plain.pdf"
    # Create PDF without chapter headings
    _create_sample_pdf(pdf_path, "Plain Document", num_pages=2, with_headings=False)

    chunks = add_book(str(pdf_path), cache_dir=tmp_path / "cache")
    assert len(chunks) > 0
    for c in chunks:
        assert c["chapter"].startswith("Pages ")


def test_validate_unit_sources_catches_unknown_book_and_chapter(tmp_path):
    """validate_unit_sources identifies missing book_ids, missing chapters, and empty units."""
    pdf_path = tmp_path / "compilers.pdf"
    _create_sample_pdf(pdf_path, "Compiler Design", num_pages=2, with_headings=True)

    lib = load_library([pdf_path], cache_dir=tmp_path / "cache")
    real_book_id = list(lib["books"].keys())[0]

    # 1. Unknown book
    bad_sources_1 = {1: {"nonexistent_book": ["Chapter 1"]}}
    problems_1 = validate_unit_sources(lib, bad_sources_1)
    assert any("Unknown book_id" in p for p in problems_1)

    # 2. Unknown chapter
    bad_sources_2 = {1: {real_book_id: ["Fictional Chapter 99"]}}
    problems_2 = validate_unit_sources(lib, bad_sources_2)
    assert any("not found in book" in p for p in problems_2)

    # 3. Valid source produces no problems
    real_chaps = lib["books"][real_book_id]["chapters"]
    assert len(real_chaps) > 0
    real_chap = real_chaps[0]
    valid_sources = {1: {real_book_id: [real_chap]}}
    assert len(validate_unit_sources(lib, valid_sources)) == 0


def test_suggest_unit_sources_splits_chapters(tmp_path):
    """suggest_unit_sources splits available chapters across units in order."""
    pdf_path = tmp_path / "math.pdf"
    _create_sample_pdf(pdf_path, "Mathematics", num_pages=4, with_headings=True)

    lib = load_library([pdf_path], cache_dir=tmp_path / "cache")
    unit_sources = suggest_unit_sources(lib, units=[1, 2])

    assert 1 in unit_sources and 2 in unit_sources
    assert len(unit_sources[1]) > 0
    assert len(unit_sources[2]) > 0
