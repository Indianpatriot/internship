"""Tests for Syllabus / Scheme Parsing and Semantic Topic Mapping (Task 2)."""
from scheme import parse_scheme_text


def _sample_full_scheme_text() -> str:
    return """
Course Title: Advanced Data Systems and Algorithms
Course Code: CS501

Course Outcomes:
CO1: Understand asymptotic complexity and recurrence analysis methodologies.
CO2: Apply balanced search trees and graph algorithms to optimization tasks.
CO3: Analyze relational database schema normalization and ACID transactional guarantees.
CO4: Design distributed storage systems with replication and consensus protocols.

Unit 1: Divide and Conquer Algorithms (8 Hours)
Recurrence relations, master theorem, merge sort, quicksort, median finding, order statistics.

Unit II: Advanced Data Structures [10 Hours]
AVL trees, Red-black trees, B-trees, Disjoint-set data structures, Binomial heaps, Fibonacci heaps.

Module 3: Graph Algorithms (8 hrs)
Breadth-first search, depth-first search, Dijkstra's shortest path, Bellman-Ford algorithm, Minimum spanning trees.

Unit IV: Relational Database Principles (8 Hours)
Relational algebra, SQL query optimization, 1NF to BCNF normalization, Multi-valued dependencies.

Module 5: Transaction Management and Recovery (6 Hours)
ACID properties, two-phase locking, serializability, write-ahead logging, deadlock detection mechanisms.

Text Books / References:
1. Cormen, Leiserson, Rivest, Stein, "Introduction to Algorithms", Fourth Edition, MIT Press.
2. Silberschatz, Korth, Sudarshan, "Database System Concepts", McGraw-Hill.
"""


def test_full_scheme_parsed_successfully():
    """Verify that 5 units with mixed Roman/Arabic numerals, topics, hours, COs, and textbooks are parsed."""
    text = _sample_full_scheme_text()
    res = parse_scheme_text(text)

    # 1. Units
    units = res["units"]
    assert len(units) == 5
    unit_nums = [u["unit"] for u in units]
    assert unit_nums == [1, 2, 3, 4, 5]

    # Hours
    assert units[0]["hours"] == 8
    assert units[1]["hours"] == 10
    assert units[4]["hours"] == 6

    # Topics
    assert len(units[0]["topics"]) >= 4
    assert any("merge sort" in t.lower() for t in units[0]["topics"])
    assert any("avl trees" in t.lower() for t in units[1]["topics"])

    # 2. Outcomes
    outcomes = res["outcomes"]
    assert len(outcomes) == 4
    assert "CO1" in outcomes and "CO4" in outcomes
    assert "asymptotic complexity" in outcomes["CO1"].lower()

    # 3. Textbooks
    textbooks = res["textbooks"]
    assert len(textbooks) >= 2
    assert any("Algorithms" in tb for tb in textbooks)

    # 4. Confidence and Warnings
    assert res["confidence"] >= 0.8
    assert len(res["warnings"]) == 0


def test_garbage_text_returns_empty_with_warnings():
    """Unstructured garbage text returns empty data structures, low confidence, and warnings without raising."""
    garbage = "The weather was sunny in London yesterday. Alice and Bob went for a walk in the park."
    res = parse_scheme_text(garbage)

    assert res["units"] == []
    assert res["outcomes"] == {}
    assert res["confidence"] == 0.0
    assert len(res["warnings"]) > 0
    assert any("No units" in w for w in res["warnings"])
    assert any("No course outcomes" in w for w in res["warnings"])


def test_confidence_higher_for_full_scheme_than_partial():
    """Confidence score accurately reflects the completeness of detected scheme sections."""
    full_text = _sample_full_scheme_text()
    full_res = parse_scheme_text(full_text)

    partial_text = """
    Unit 1: Introduction to Programming
    Variables, loops, conditionals, functions.
    """
    partial_res = parse_scheme_text(partial_text)

    assert full_res["confidence"] > partial_res["confidence"]
    assert partial_res["confidence"] < 0.6
    assert any("No course outcomes" in w for w in partial_res["warnings"])


# ---------------------------------------------------------------------------
# Step 2.2 / 2.3: prefill + manual fallback
# ---------------------------------------------------------------------------
import fitz  # noqa: E402

from scheme import (  # noqa: E402
    manual_scheme_template,
    map_topics_to_chapters,
    merge_manual_entry,
    parse_scheme,
    prefill_unit_sources,
)


def _fake_library():
    body = lambda words: " ".join(words * 12)  # noqa: E731
    chunks = [
        {"book_id": "algo", "chapter": "Sorting and Recurrences", "page_start": 1, "page_end": 2,
         "text": body(["merge", "sort", "recurrence", "master", "theorem", "quicksort"])},
        {"book_id": "algo", "chapter": "Graph Search", "page_start": 3, "page_end": 4,
         "text": body(["breadth", "first", "search", "dijkstra", "shortest", "path", "graph"])},
        {"book_id": "db", "chapter": "Transactions", "page_start": 1, "page_end": 2,
         "text": body(["acid", "locking", "serializability", "logging", "transaction"])},
    ]
    books = {
        "algo": {"title": "Introduction To Algorithms", "chapters": ["Sorting and Recurrences", "Graph Search"],
                 "n_chunks": 2, "page_range": (1, 4), "path": "algo.pdf"},
        "db": {"title": "Database System Concepts", "chapters": ["Transactions"],
               "n_chunks": 1, "page_range": (1, 2), "path": "db.pdf"},
    }
    return {"books": books, "chunks": chunks}


_SMALL_SCHEME = """
Course Outcomes:
CO1: Understand recurrence analysis.
CO2: Apply graph algorithms.
CO3: Analyze transaction schedules.

Unit 1: Divide and Conquer (8 Hours)
Merge sort, quicksort, recurrence relations, master theorem.

Unit 2: Graph Algorithms (8 Hours)
Breadth first search, Dijkstra shortest path, graph traversal.

Unit 3: Transactions (6 Hours)
ACID properties, locking, serializability, logging.

Text Books:
1. Cormen et al., Introduction To Algorithms, MIT Press.
"""


def test_prefill_maps_units_to_best_chapters_across_books():
    lib = _fake_library()
    res = parse_scheme_text(_SMALL_SCHEME, library=lib)
    src = res["unit_sources"]
    assert src[1] == {"algo": ["Sorting and Recurrences"]}
    assert src[2] == {"algo": ["Graph Search"]}
    assert src[3] == {"db": ["Transactions"]}
    assert res["needs_manual"] is False


def test_map_topics_returns_sorted_scores():
    lib = _fake_library()
    res = parse_scheme_text(_SMALL_SCHEME)
    m = map_topics_to_chapters(res["units"], lib)
    for matches in m.values():
        scores = [x["score"] for x in matches]
        assert scores == sorted(scores, reverse=True)


def test_prefill_falls_back_to_page_order_when_nothing_matches():
    lib = _fake_library()
    scheme = {"units": [{"unit": 1, "title": "Zzz", "topics": ["qqq"], "hours": None}],
              "unit_sources": {1: None}}
    out = prefill_unit_sources(scheme, lib, min_score=10.0)
    assert out[1]  # non-empty suggestion from suggest_unit_sources


def test_garbage_text_requests_manual_entry():
    res = parse_scheme_text("The weather was sunny in London yesterday.")
    assert res["needs_manual"] is True


def test_parse_scheme_pdf_roundtrip_and_unreadable_fallback(tmp_path):
    pdf = tmp_path / "scheme.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 550, 800), _SMALL_SCHEME, fontsize=10)
    doc.save(str(pdf))
    doc.close()
    res = parse_scheme(str(pdf), library=_fake_library())
    assert [u["unit"] for u in res["units"]] == [1, 2, 3]
    assert set(res["outcomes"]) == {"CO1", "CO2", "CO3"}
    assert res["needs_manual"] is False

    # missing file -> manual fallback, no exception
    bad = parse_scheme(str(tmp_path / "missing.pdf"))
    assert bad["needs_manual"] is True and bad["units"] == []
    assert any("Could not read" in w for w in bad["warnings"])

    # blank (scanned-like) PDF -> manual fallback
    blank = tmp_path / "blank.pdf"
    d = fitz.open()
    d.new_page()
    d.save(str(blank))
    d.close()
    res_blank = parse_scheme(str(blank))
    assert res_blank["needs_manual"] is True


def test_manual_entry_overrides_failed_parse():
    failed = parse_scheme_text("nothing useful here")
    tpl = manual_scheme_template(2, 2)
    assert list(tpl["outcomes"]) == ["CO1", "CO2"]
    merged = merge_manual_entry(
        failed,
        units=[{"unit": 1, "title": "A", "topics": ["x"], "hours": 4}],
        outcomes={"CO1": "Understand A", "CO2": "  "},
        unit_sources={1: {"algo": None}},
    )
    assert merged["outcomes"] == {"CO1": "Understand A"}
    assert merged["needs_manual"] is False and merged["confirmed"] is True
