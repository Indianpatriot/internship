"""Regression tests for the 4 identified issues:
1. Problem 1: Non-content front/back matter (preface, TOC, index) dropped from chunks & chapters.
2. Problem 2: Multiple textbook upload, unique IDs, chapter disambiguation, and unit source routing.
3. Problem 3: 100 Marks (Choice) blueprint structure, 26 slots, 200 offered / 100 attempted marks, OR symmetry, KD distribution.
4. Problem 4: Progress bar callbacks execute strictly on the main/caller thread without thread-context freezes.
"""
import threading
import fitz
import pytest

from blueprint import build_blueprint, calculate_blueprint_stats, validate_weightage_totals
from chapter_filter import filter_chunks, get_chapter_metadata
from generation.generator import generate_paper_by_slots
from generation.mock_client import MockClient
from ingestion.heading_detect import is_non_content_title
from ingestion.library import add_book, load_library, validate_unit_sources
from ingestion.pipeline import ingest
from retrieval.sources import select_chunks_for_unit
from scheme import parse_scheme_text


def _create_synthetic_book_with_matter(path):
    """Create a PDF with front-matter (preface, toc), real chapters, and back-matter (index)."""
    doc = fitz.open()

    # Page 1: Copyright / ISBN front matter
    p1 = doc.new_page()
    p1.insert_text((50, 100), "Copyright © 2026 Academic Press. All rights reserved. ISBN 978-0-123456-78-9.", fontsize=11)

    # Page 2: Preface
    p2 = doc.new_page()
    p2.insert_text((50, 80), "Preface", fontsize=18)
    p2.insert_textbox(
        fitz.Rect(50, 110, 550, 400),
        "This textbook introduces students to principles of computer science. "
        "We thank our colleagues and students for their valuable feedback and support. "
        "The authors hope this comprehensive guide assists educators in structuring modern curricula.",
        fontsize=11,
    )

    # Page 3: Table of Contents
    p3 = doc.new_page()
    p3.insert_text((50, 80), "Table of Contents", fontsize=18)
    p3.insert_textbox(
        fitz.Rect(50, 110, 550, 400),
        "Chapter 1: Foundational Algorithms ......................... 4\n"
        "Chapter 2: Advanced Data Structures ....................... 5\n"
        "Index .................................................... 6",
        fontsize=11,
    )

    # Page 4: Chapter 1
    p4 = doc.new_page()
    p4.insert_text((50, 80), "Chapter 1: Foundational Algorithms", fontsize=18)
    p4.insert_textbox(
        fitz.Rect(50, 110, 550, 600),
        (
            "Divide and conquer algorithms partition large problems into smaller independent subproblems. "
            "Merge sort divides arrays recursively into single element subarrays before merging them in sorted order. "
            "Binary search repeatedly bifurcates sorted search spaces to locate keys in logarithmic asymptotic time. "
            "Recurrence relations model recursive runtime overhead rigorously using the Master Theorem framework."
        ) * 2,
        fontsize=11,
    )

    # Page 5: Chapter 2
    p5 = doc.new_page()
    p5.insert_text((50, 80), "Chapter 2: Advanced Data Structures", fontsize=18)
    p5.insert_textbox(
        fitz.Rect(50, 110, 550, 600),
        (
            "Balanced binary search trees maintain logarithmic depth invariants to optimize search and update speeds. "
            "Red-black trees enforce coloring constraints to restrict path length ratios across all leaf nodes. "
            "Hash tables employ collision resolution policies such as linear probing and separate chaining. "
            "Amortized analysis guarantees that dynamic array resizing maintains constant expected operation costs."
        ) * 2,
        fontsize=11,
    )

    # Page 6: Index
    p6 = doc.new_page()
    p6.insert_text((50, 80), "Index", fontsize=18)
    p6.insert_textbox(
        fitz.Rect(50, 110, 550, 400),
        "Algorithms, 4\nBinary search, 4\nHash tables, 5\nMerge sort, 4\nRed-black trees, 5\n",
        fontsize=11,
    )

    doc.save(str(path))
    doc.close()


def test_problem1_front_matter_dropped_completely(tmp_path):
    """Problem 1: Front and back matter pages must be dropped; no non-content chunks or chapters."""
    pdf_path = tmp_path / "textbook_with_matter.pdf"
    _create_synthetic_book_with_matter(pdf_path)

    chunks = ingest(str(pdf_path))
    assert len(chunks) > 0

    # No chunk should originate from non-content pages (1, 2, 3, or 6)
    for c in chunks:
        p_start = c.get("page_start", 0)
        p_end = c.get("page_end", p_start)
        assert p_start not in {1, 2, 3, 6}, f"Chunk {c['chunk_id']} from front/back page {p_start}"
        assert p_end not in {1, 2, 3, 6}, f"Chunk {c['chunk_id']} ends on front/back page {p_end}"

        chap = c.get("chapter", "")
        assert not is_non_content_title(chap), f"Chunk assigned non-content chapter '{chap}'"

    # Chapter metadata should only contain Chapter 1 and Chapter 2
    meta = get_chapter_metadata(chunks)
    detected_chaps = [m["chapter"] for m in meta]
    assert all("Chapter" in ch for ch in detected_chaps)
    assert not any(is_non_content_title(ch) for ch in detected_chaps)
    assert not any("Preface" in ch or "Contents" in ch or "Index" in ch for ch in detected_chaps)


def test_problem2_multiple_textbook_upload(tmp_path):
    """Problem 2: Multiple textbooks ingested with unique IDs, disambiguated chapters, and source routing."""
    pdf_a = tmp_path / "operating_systems.pdf"
    pdf_b = tmp_path / "database_systems.pdf"

    # Create Book A
    doc_a = fitz.open()
    p_a = doc_a.new_page()
    p_a.insert_text((50, 80), "Chapter 1: Process Scheduling", fontsize=18)
    p_a.insert_textbox(
        fitz.Rect(50, 110, 550, 600),
        (
            "Operating systems schedule processes using round-robin and multi-level feedback queues. "
            "Context switches preserve processor registers and virtual memory translation tables. "
            "Thread synchronization utilizes semaphores, mutex locks, and condition variables."
        ) * 3,
        fontsize=11,
    )
    doc_a.save(str(pdf_a))
    doc_a.close()

    # Create Book B
    doc_b = fitz.open()
    p_b = doc_b.new_page()
    p_b.insert_text((50, 80), "Chapter 1: Relational Modeling", fontsize=18)
    p_b.insert_textbox(
        fitz.Rect(50, 110, 550, 600),
        (
            "Relational database management systems organize data into structured normalized tables. "
            "Transactions preserve atomicity, consistency, isolation, and durability guarantees. "
            "Two-phase locking protocols guarantee serializable schedules during concurrent queries."
        ) * 3,
        fontsize=11,
    )
    doc_b.save(str(pdf_b))
    doc_b.close()

    # Load both books into library
    lib = load_library([pdf_a, pdf_b], cache_dir=tmp_path / "cache")
    assert len(lib["books"]) == 2
    all_chunks = lib["chunks"]

    # Verify unique chunk IDs and textbook/book_id metadata
    chunk_ids = [c["chunk_id"] for c in all_chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    for c in all_chunks:
        assert c.get("textbook") in ["Operating Systems", "Database Systems"]
        assert c.get("book_id") in ["operating-systems", "database-systems"]

    # Verify get_chapter_metadata disambiguates chapters across books
    meta = get_chapter_metadata(all_chunks)
    chap_names = [m["chapter"] for m in meta]
    assert any("Operating Systems - Chapter 1" in ch for ch in chap_names)
    assert any("Database Systems - Chapter 1" in ch for ch in chap_names)

    # Verify filter_chunks matches disambiguated chapter correctly
    os_filtered = filter_chunks(all_chunks, selected_chapters=[chap_names[0]])
    assert all(c["book_id"] == meta[0]["book_id"] for c in os_filtered)

    # Verify select_chunks_for_unit routes chunks by book_id and chapter
    unit_sources = {
        1: {"operating-systems": ["Chapter 1: Process Scheduling"]},
        2: {"database-systems": ["Chapter 1: Relational Modeling"]},
    }
    problems = validate_unit_sources(lib, unit_sources)
    assert len(problems) == 0

    u1_chunks = select_chunks_for_unit(all_chunks, unit_sources, unit=1)
    assert all(c["book_id"] == "operating-systems" for c in u1_chunks)

    u2_chunks = select_chunks_for_unit(all_chunks, unit_sources, unit=2)
    assert all(c["book_id"] == "database-systems" for c in u2_chunks)


def test_problem3_100_mark_choice_preset_slots():
    """Problem 3: 100-mark choice preset with sub-marks generates exactly 26 slots, 200 offered / 100 attempted marks."""
    sub_marks = {
        1: [5, 5, 10],
        2: [7, 7, 6],
        3: [10, 10],
        4: [7, 7, 6],
        5: [12, 8],
    }
    outcomes = ["CO1", "CO2", "CO3", "CO4", "CO5"]
    bp = build_blueprint(
        preset=100,
        sub_question_marks=sub_marks,
        outcomes=outcomes,
        seed=42,
    )
    slots = bp["slots"]

    # 1. Exactly 26 slots
    assert len(slots) == 26

    # 2. All 5 units present
    units_in_slots = sorted(list({s["unit"] for s in slots}))
    assert units_in_slots == [1, 2, 3, 4, 5]

    # 3. Offered and attempted marks
    stats = calculate_blueprint_stats(slots)
    assert stats["offered_marks"] == 200
    assert stats["attempted_marks"] == 100
    assert stats["is_choice"] is True

    # 4. Each unit offers 40 marks (20 per option) and attempts 20 marks
    for u in [1, 2, 3, 4, 5]:
        assert stats["unit_marks"][u] == 40
        assert stats["unit_attempted_marks"][u] == 20

    # 5. Question numbering runs 1(a)..10(b)
    slot_ids = [s["id"] for s in slots]
    assert slot_ids[0] == "1(a)"
    assert slot_ids[-1] == "10(b)"

    # 6. Mirror symmetry: Option questions mirror the primary question marks and BT levels
    by_q = {}
    for s in slots:
        by_q.setdefault(s["question_no"], []).append(s)

    for primary_q, choice_q in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]:
        p_slots = by_q[primary_q]
        c_slots = by_q[choice_q]
        assert len(p_slots) == len(c_slots)
        assert [s["marks"] for s in p_slots] == [s["marks"] for s in c_slots]
        assert [s["level"] for s in p_slots] == [s["level"] for s in c_slots]
        assert [s["co"] for s in p_slots] == [s["co"] for s in c_slots]

    # 7. Cognitive & Knowledge Dimension coverage
    assert len(stats["bt_counts"]) >= 3
    assert all(kd in stats["kd_counts"] for kd in ["Factual", "Conceptual", "Procedural", "Metacognitive"])


def test_problem4_progress_callback_main_thread():
    """Problem 4: Progress callback executes strictly on caller thread, exactly once per slot."""
    slots = [
        {"id": "1", "unit": 1, "level": "L1", "co": "CO1", "marks": 5, "knowledge_dimension": "Factual"},
        {"id": "2", "unit": 1, "level": "L2", "co": "CO1", "marks": 5, "knowledge_dimension": "Conceptual"},
        {"id": "3", "unit": 2, "level": "L3", "co": "CO2", "marks": 10, "knowledge_dimension": "Procedural"},
    ]
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "book_id": "book1",
            "chapter": "Chapter 1",
            "text": f"Algorithmic computational principles Section {i} verify complexity bounds rigorously.",
            "word_count": 45,
            "co_scores": {"CO1": 0.9, "CO2": 0.9},
            "page_start": i * 10,
        }
        for i in range(1, 4)
    ]

    calling_thread_id = threading.get_ident()
    callback_threads = []
    completed_slots = []

    def on_progress(rec):
        callback_threads.append(threading.get_ident())
        completed_slots.append(rec.get("id") or rec.get("slot_id"))

    client = MockClient()
    res = generate_paper_by_slots(
        chunks=chunks,
        slots=slots,
        client=client,
        seed=42,
        on_progress=on_progress,
        workers=3,
        force_regenerate=True,
    )

    # Must invoke on_progress once per slot
    assert len(callback_threads) == len(slots)
    # Every invocation must happen on the caller/main thread
    assert all(t_id == calling_thread_id for t_id in callback_threads)
    assert len(res["questions"]) == len(slots)


def test_change1_pasted_scheme_exact_outcomes():
    """Change 1: Pasted scheme preserves Course Outcomes EXACTLY as pasted."""
    scheme_text = """
    Course Outcomes:
    CO1: Understand core principles & foundations (e.g. O(n log n), memory hierarchy).
    CO2 - Analyze multi-threaded synchronization primitives; mutexes/semaphores!
    CO3. Design distributed microservices [High-Availability & Fault-Tolerant].
    CO4: Formulate real-time task scheduling policies (EDF vs RMS).
    CO5 - Evaluate cryptographic hash algorithms: SHA-256 / AES-GCM @ scale.

    Unit 1: Foundations (8 Hours)
    Topics in algorithmic analysis.
    """
    parsed = parse_scheme_text(scheme_text)
    outcomes = parsed["outcomes"]
    assert len(outcomes) == 5
    assert outcomes["CO1"] == "Understand core principles & foundations (e.g. O(n log n), memory hierarchy)."
    assert outcomes["CO2"] == "Analyze multi-threaded synchronization primitives; mutexes/semaphores!"
    assert outcomes["CO3"] == "Design distributed microservices [High-Availability & Fault-Tolerant]."
    assert outcomes["CO4"] == "Formulate real-time task scheduling policies (EDF vs RMS)."
    assert outcomes["CO5"] == "Evaluate cryptographic hash algorithms: SHA-256 / AES-GCM @ scale."

    # Test numbered list format under Course Outcomes
    scheme_numbered = """
    Course Outcomes:
    1. Analyze dynamic data structures (trees, graphs, heaps).
    2. Evaluate worst-case runtime bounds: Big-O, Theta, Omega.
    3. Implement greedy & divide-and-conquer paradigms.
    4. Synthesize randomized approximation algorithms.
    """
    parsed_num = parse_scheme_text(scheme_numbered)
    out_num = parsed_num["outcomes"]
    assert len(out_num) == 4
    assert out_num["CO1"] == "Analyze dynamic data structures (trees, graphs, heaps)."
    assert out_num["CO2"] == "Evaluate worst-case runtime bounds: Big-O, Theta, Omega."
    assert out_num["CO3"] == "Implement greedy & divide-and-conquer paradigms."
    assert out_num["CO4"] == "Synthesize randomized approximation algorithms."

    # Test that fewer than 4 or more than 6 COs are NOT dropped or overwritten by defaults
    scheme_fewer = """
    Course Outcomes:
    CO1: First custom outcome.
    CO2: Second custom outcome.
    CO3: Third custom outcome.
    """
    parsed_fewer = parse_scheme_text(scheme_fewer)
    assert len(parsed_fewer["outcomes"]) == 3
    assert parsed_fewer["outcomes"]["CO1"] == "First custom outcome."
    assert parsed_fewer["outcomes"]["CO3"] == "Third custom outcome."


def test_change2_weightage_validation_on_apply():
    """Change 2: Validate existing CO and BT band totals equal exactly 100% on Apply, allow +/-0.01 tolerance."""
    existing_cos = ["CO1", "CO2", "CO3", "CO4"]

    # 1. Valid totals = 100% pass
    co_valid = {"CO1": 25.0, "CO2": 25.0, "CO3": 25.0, "CO4": 25.0}
    bt_valid = {"L1-L2": 40.0, "L3-L4": 40.0, "L5-L6": 20.0}
    ok, errors = validate_weightage_totals(co_valid, bt_valid, existing_cos)
    assert ok is True
    assert len(errors) == 0

    # 2. CO total != 100% (85%) fails with explanatory message
    co_85 = {"CO1": 20.0, "CO2": 25.0, "CO3": 20.0, "CO4": 20.0}
    ok, errors = validate_weightage_totals(co_85, bt_valid, existing_cos)
    assert ok is False
    assert any("CO weightages total 85%. They must total exactly 100% (-15%). Adjust and apply again." in e for e in errors)

    # 3. Bloom's total != 100% (110%) fails with explanatory message
    bt_110 = {"L1-L2": 50.0, "L3-L4": 40.0, "L5-L6": 20.0}
    ok, errors = validate_weightage_totals(co_valid, bt_110, existing_cos)
    assert ok is False
    assert any("Bloom's weightages total 110%. They must total exactly 100% (+10%). Adjust and apply again." in e for e in errors)

    # 4. Only existing COs count (deleted CO5 ignored)
    co_with_stale = {"CO1": 25.0, "CO2": 25.0, "CO3": 25.0, "CO4": 25.0, "CO5": 50.0}
    ok, errors = validate_weightage_totals(co_with_stale, bt_valid, existing_cos)
    assert ok is True
    assert len(errors) == 0

    # 5. Float tolerance (+/- 0.01)
    co_float_ok = {"CO1": 25.0, "CO2": 25.0, "CO3": 25.0, "CO4": 24.999}
    ok, errors = validate_weightage_totals(co_float_ok, bt_valid, existing_cos, tolerance=0.01)
    assert ok is True

    # 99% fails
    co_99 = {"CO1": 25.0, "CO2": 25.0, "CO3": 25.0, "CO4": 24.0}
    ok, errors = validate_weightage_totals(co_99, bt_valid, existing_cos, tolerance=0.01)
    assert ok is False

    # 101% fails
    co_101 = {"CO1": 25.0, "CO2": 26.0, "CO3": 25.0, "CO4": 25.0}
    ok, errors = validate_weightage_totals(co_101, bt_valid, existing_cos, tolerance=0.01)
    assert ok is False


def test_change3_textbook_names_upload_order_no_hex(tmp_path):
    """Change 3: Number uploaded textbooks in upload order ('1 <book>', '2 <book>'), without hex hashes."""
    import hashlib

    # Create two synthetic textbooks
    def _make_book(path, title, chapter_name):
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((50, 50), chapter_name, fontsize=16)
        body_text = (
            f"Content for {title} discussing computational theories and architectures in depth. "
            "Data structures organize information hierarchically to optimize search and retrieval times. "
            "The mathematical recurrence relation yields a closed form asymptotic runtime bound. "
            "State transitions maintain invariants throughout the complete execution lifecycle. "
            "Engineering teams verify correctness by applying formal validation and stress tests."
        ) * 2
        p.insert_textbox(
            fitz.Rect(50, 80, 550, 400),
            body_text,
            fontsize=11,
        )
        doc.save(str(path))
        doc.close()

    raw_hash_a = hashlib.sha256(b"Book ABC Data").hexdigest()[:12]
    raw_hash_b = hashlib.sha256(b"Book DEF Data").hexdigest()[:12]

    up_a = tmp_path / f"{raw_hash_a}_ABC.pdf"
    up_b = tmp_path / f"{raw_hash_b}_DEF.pdf"
    _make_book(up_a, "ABC", "Chapter 1: Foundational ABC")
    _make_book(up_b, "DEF", "Chapter 1: Advanced DEF")

    cache_dir = tmp_path / "cache"

    # Load both books in upload order ABC then DEF
    lib = load_library([up_a, up_b], cache_dir=cache_dir)
    books = lib["books"]
    assert len(books) == 2

    # Verify display labels in upload order: "1 ABC" and "2 DEF"
    labels = [binfo.get("display_label") or binfo.get("title") for binfo in books.values()]
    assert labels == ["1 ABC", "2 DEF"]

    # Verify NO SHA-256 hex hash appears in any user-facing label, title, or chunk textbook
    for bid, binfo in books.items():
        assert raw_hash_a not in binfo["title"]
        assert raw_hash_b not in binfo["title"]
        assert raw_hash_a not in binfo["display_label"]
        assert raw_hash_b not in binfo["display_label"]

    for c in lib["chunks"]:
        assert raw_hash_a not in c["book_title"]
        assert raw_hash_b not in c["book_title"]
        assert raw_hash_a not in c["textbook"]
        assert raw_hash_b not in c["textbook"]

    # Verify chapter metadata has no hex hash and has clear book labels
    meta = get_chapter_metadata(lib["chunks"])
    for m in meta:
        assert raw_hash_a not in m["chapter"]
        assert raw_hash_b not in m["chapter"]
        assert raw_hash_a not in m["display_label"]
        assert raw_hash_b not in m["display_label"]

    # Verify mapping works
    book_ids = list(books.keys())
    unit_sources = {
        1: {book_ids[0]: [books[book_ids[0]]["chapters"][0]]},
        2: {book_ids[1]: [books[book_ids[1]]["chapters"][0]]},
    }
    problems = validate_unit_sources(lib, unit_sources)
    assert len(problems) == 0

    # Removing "ABC" leaves only "DEF", renumbering remaining contiguous in order ("1 DEF")
    lib_after_removal = load_library([up_b], cache_dir=cache_dir)
    assert len(lib_after_removal["books"]) == 1
    rem_binfo = list(lib_after_removal["books"].values())[0]
    rem_label = rem_binfo.get("display_label") or rem_binfo.get("title")
    assert rem_label == "1 DEF"


# -----------------------------------------------------------------------------
# Regression Tests for 8 Quality Issues (Problems 1 - 8)
# -----------------------------------------------------------------------------


def test_quality_problem1_front_matter_exclusion():
    """Problem 1: Non-content sections (about this book, cover, etc.) must be detected and rejected."""
    non_content_samples = [
        "about this book",
        "about the cover illustration",
        "about the author",
        "about the authors",
        "about the technical editor",
        "welcome",
        "preface",
        "foreword",
        "acknowledgements",
        "dedication",
        "table of contents",
        "brief contents",
        "contents",
        "to get the most out of this book",
        "conventions used",
        "get in touch",
        "reviews",
        "other books you may enjoy",
        "assessments",
        "answers",
        "index",
        "bibliography",
        "references",
        "glossary",
    ]
    for sample in non_content_samples:
        assert is_non_content_title(sample) is True, f"Failed to detect non-content title: {sample}"

    # Verify select_chunks_for_unit drops chunks with non-content chapter or section
    chunks = [
        {"chunk_id": "c1", "chapter": "About this book", "section": "Overview", "text": "Audience info.", "book_id": "b1"},
        {"chunk_id": "c2", "chapter": "Chapter 1: Containers", "section": "About the author", "text": "Author bio.", "book_id": "b1"},
        {"chunk_id": "c3", "chapter": "Chapter 1: Containers", "section": "1.1 Architecture", "text": "Real content here.", "book_id": "b1"},
    ]
    filtered = select_chunks_for_unit(chunks, unit_sources=None, unit=None)
    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == "c3"


def test_quality_problem2_source_labels_and_chapter_dedup():
    """Problem 2: Discard TOC entries, strip bullets, unify Chapter N: format, and prevent book-as-chapter."""
    from ingestion.heading_detect import normalize_chapter_title

    assert normalize_chapter_title("• Chapter 1: Docker In Practice") == "Chapter 1: Docker In Practice"
    assert normalize_chapter_title("•   Chapter 2, Kubernetes Architecture") == "Chapter 2: Kubernetes Architecture"
    assert normalize_chapter_title("Chapter 3: Pods and Services .............. 45") == "Chapter 3: Pods and Services"
    assert normalize_chapter_title("Chapter 4: Deployments      120") == "Chapter 4: Deployments"

    # Verify formatting helper format_source_label or export cleaner
    from export_utils import clean_book_name
    assert clean_book_name("6a1831f8f825_Docker_In_Practice") == "Docker In Practice"
    assert clean_book_name("• 6a1831f8f825: 1. Docker In Practice") == "1. Docker In Practice"


def test_quality_problem3_hex_prefix_cleanup():
    """Problem 3: Pass inputs with hex prefixes to export pipeline and verify no hex leak in outputs."""
    from export_utils import clean_book_name, create_docx_exam_paper, create_pdf_exam_paper

    raw_hex = "6a1831f8f825"
    hex_course = f"{raw_hex}_Docker_In_Practice"
    clean_course = clean_book_name(hex_course)
    assert raw_hex not in clean_course
    assert clean_course == "Docker In Practice"

    questions = [
        {
            "id": "1(a)",
            "question": "Explain container isolation mechanisms?",
            "marks": 10,
            "level": "L2",
            "co": "CO1",
            "unit": 1,
            "chapter": f"{raw_hex}: Chapter 1 Docker In Practice",
            "book_title": f"{raw_hex}_Docker_In_Practice",
            "page_start": 25,
            "answer": "Containers leverage Linux cgroups and namespaces.",
        }
    ]

    docx_bytes = create_docx_exam_paper(
        questions=questions,
        course_name=hex_course,
        covered_chapters=[f"{raw_hex}: Chapter 1 Docker In Practice"],
    )
    assert raw_hex.encode("utf-8") not in docx_bytes

    pdf_bytes = create_pdf_exam_paper(
        questions=questions,
        course_name=hex_course,
        covered_chapters=[f"{raw_hex}: Chapter 1 Docker In Practice"],
    )
    assert raw_hex.encode("utf-8") not in pdf_bytes


def test_quality_problem4_truncation_detection_and_retry():
    """Problem 4: Detect cut-off questions/answers (missing terminal punctuation, dangling words) and retry."""
    from generation.generator import generate_question, is_truncated_text

    # Dangling endings
    assert is_truncated_text("Compare and contrast pods, use cases, and impl", is_question=True) is True
    assert is_truncated_text("Ensuring high availab", is_question=False) is True
    assert is_truncated_text("Scalable applications that cannot be scaled ho", is_question=False) is True
    assert is_truncated_text("Explain the deployment architecture", is_question=False) is True  # no punctuation
    assert is_truncated_text("Explain the deployment architecture.", is_question=False) is False
    assert is_truncated_text("How do pods scale in Kubernetes?", is_question=True) is False

    # Retry when mock client returns truncated then complete
    class FlakyMockClient:
        model = "mock"
        attempts = 0

        def generate(self, prompt, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                return "Explain the container lifecycle and impl"
            return "Explain the container lifecycle and operational mechanisms in modern clusters?"

    flaky_client = FlakyMockClient()
    chunk = {
        "text": "Containers manage process lifecycles and operational mechanisms in modern clusters.",
        "chunk_id": "c1",
        "word_count": 25,
    }
    rec = generate_question(chunk, "L2", flaky_client, max_retries=2, force_regenerate=True)
    assert rec["valid"] is True
    assert rec["attempts"] >= 2
    assert rec["question"].endswith("?")


def test_quality_problem5_banned_answer_phrases_and_or_separator():
    """Problem 5: Banned apology phrases in answers rejected; OR separators converted to Alternative approach."""
    from generation.generator import clean_answer_text, has_banned_answer_phrases

    banned = [
        "The text does not explicitly mention standard Kubernetes commands.",
        "Based on general knowledge, pods can be scaled with kubectl.",
        "The provided text cannot be answered with precision.",
        "The passage doesn't mention replication controllers.",
    ]
    for b in banned:
        assert has_banned_answer_phrases(b) is True

    clean_ans = "Kubernetes provides kubectl scale deployment to adjust replica counts."
    assert has_banned_answer_phrases(clean_ans) is False

    # Test clean_answer_text replaces OR with Alternative approach and leaves no '― OR ―'
    raw_with_or = "Approach 1: HPA scales on CPU metrics.\n\n― OR ―\n\nApproach 2: HPA scales on custom memory metrics."
    full, alts = clean_answer_text(raw_with_or)
    assert "― OR ―" not in full
    assert "— OR —" not in full
    assert len(alts) == 2
    assert "Alternative approach:" in full


def test_quality_problem6_textbook_independence():
    """Problem 6: Stems and answers referencing physical textbook/author/chapters are rejected."""
    from generation.generator import has_textbook_reference

    bad_stems = [
        "Name the primary target audience for this book?",
        "Mention the main aim of Chapter 17?",
        "According to the author in Section 3.2, explain pods?",
        "As described in the text, discuss deployment strategies?",
        "Look at Figure 4.2 and describe the architecture?",
    ]
    for stem in bad_stems:
        assert has_textbook_reference(stem) is True, f"Failed to reject textbook-dependent stem: {stem}"

    good_stem = "Describe the deployment strategies available in Kubernetes?"
    assert has_textbook_reference(good_stem) is False


def test_quality_problem7_marks_blooms_kd_alignment():
    """Problem 7: High marks (>= 10) never get L1; low marks (<= 3) never get L5/L6; no Compute on non-numbers."""
    # Build 100-mark paper blueprint
    bp = build_blueprint(preset=100, sub_question_counts=2, seed=42)
    slots = bp["slots"]
    for s in slots:
        m = s["marks"]
        lvl = s["level"]
        if m >= 10:
            assert lvl != "L1", f"Slot {s['id']} has marks {m} >= 10 but was allocated L1 (Remember)"
        if m <= 3:
            assert lvl not in ("L5", "L6"), f"Slot {s['id']} has marks {m} <= 3 but was allocated {lvl}"

    # Verb sanity: context without numbers cannot sample or keep compute/calculate
    from generation.generator import generate_question
    non_numeric_chunk = {
        "text": "Docker images encapsulate runtime dependencies, system binaries, and configuration files.",
        "chunk_id": "c_cli",
        "word_count": 20,
    }
    client = MockClient()
    rec = generate_question(non_numeric_chunk, "L3", client, marks=5, force_regenerate=True)
    assert rec["valid"] is True
    assert rec["verb"].lower() not in ("compute", "calculate")


def test_quality_problem8_paper_header_values():
    """Problem 8: Choice paper header shows attempted marks (100) and unit count, not offered marks (200) or leaf count (26)."""
    from export_utils import _get_paper_header_values

    # Simulate 100-mark choice paper with 5 units x 2 alternative questions (200 offered marks)
    questions = []
    for u in range(1, 6):
        for q_no in [u * 2 - 1, u * 2]:
            questions.append({
                "id": f"{q_no}(a)",
                "unit": u,
                "question_no": q_no,
                "marks": 10,
                "alternative_group": f"unit_{u}_choice",
                "question": f"Question {q_no} part a",
            })
            questions.append({
                "id": f"{q_no}(b)",
                "unit": u,
                "question_no": q_no,
                "marks": 10,
                "alternative_group": f"unit_{u}_choice",
                "question": f"Question {q_no} part b",
            })

    # Total offered is 5 units x 2 choice questions x 20 marks = 200 marks
    assert sum(q["marks"] for q in questions) == 200
    assert len(questions) == 20

    attempted_marks, q_count_text = _get_paper_header_values(questions)
    assert attempted_marks == 100, f"Expected 100 attempted marks, got {attempted_marks}"
    assert "Total Units: 5 (Choice in each)" in q_count_text
    assert "20" not in q_count_text
    assert "26" not in q_count_text


def test_clean_book_name_not_shadowed_in_app():
    """Verify that clean_book_name imported in app.py is never assigned as a string variable."""
    import ast
    with open("app.py", "r") as f:
        tree = ast.parse(f.read(), filename="app.py")

    assigned_targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_targets.add(target.id)

    assert "clean_book_name" not in assigned_targets, (
        "clean_book_name imported function is shadowed by a variable assignment in app.py"
    )


