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
