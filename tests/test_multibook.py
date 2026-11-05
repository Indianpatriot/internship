"""Tests for Multi-Book Retrieval and Assembly (Task 3)."""
from blueprint import build_blueprint
from generation.generator import generate_full_paper, generate_paper_by_slots
from generation.mock_client import MockClient
from retrieval.sources import select_chunks_for_unit


def _create_multi_book_chunks():
    chunks = []
    # Book A
    chunks.append({
        "chunk_id": "book_a_c1",
        "book_id": "book_a",
        "chapter": "Chapter 1",
        "text": (
            "Alice formulated divide and conquer algorithms including merge sort and binary search. "
            "Recursive recurrence relations determine asymptotic runtime bounds accurately. "
            "Master theorem provides closed-form solutions for divide and conquer algorithms. "
            "Optimal worst-case complexity remains bounded by O(n log n) operations."
        ),
        "word_count": 45,
        "co_scores": {"CO1": 0.9, "CO2": 0.2},
        "page_start": 10,
    })
    chunks.append({
        "chunk_id": "book_a_c2",
        "book_id": "book_a",
        "chapter": "Chapter 2",
        "text": (
            "Alice implemented dynamic programming solutions for knapsack and sequence alignment. "
            "Optimal substructure and overlapping subproblems characterize dynamic programming. "
            "Memoization caches intermediate results to avoid redundant recursive calculations. "
            "Table bottom-up tabulation ensures polynomial runtime guarantees for problems."
        ),
        "word_count": 45,
        "co_scores": {"CO1": 0.85, "CO2": 0.3},
        "page_start": 25,
    })
    chunks.append({
        "chunk_id": "book_a_c3",
        "book_id": "book_a",
        "chapter": "Chapter 3",
        "text": (
            "Alice studied greedy heuristics for minimum spanning trees using Prim and Kruskal. "
            "Disjoint-set data structures track connected components with path compression. "
            "Locally optimal choices guarantee global optimality for matroid problem classes. "
            "Edge weights are sorted iteratively to construct minimal acyclic subgraphs."
        ),
        "word_count": 45,
        "co_scores": {"CO1": 0.8, "CO2": 0.25},
        "page_start": 40,
    })

    # Book B
    chunks.append({
        "chunk_id": "book_b_c1",
        "book_id": "book_b",
        "chapter": "Chapter 1",
        "text": (
            "Bob designed relational database architectures with transactional atomicity and durability. "
            "Relational algebra operations select, project, and join relations mathematically. "
            "Entity relationship diagrams map business requirements into normalized schemas. "
            "Foreign key constraints enforce referential integrity across tabular relations."
        ),
        "word_count": 45,
        "co_scores": {"CO1": 0.2, "CO2": 0.95},
        "page_start": 5,
    })
    chunks.append({
        "chunk_id": "book_b_c2",
        "book_id": "book_b",
        "chapter": "Chapter 2",
        "text": (
            "Bob explored concurrency control mechanisms using two-phase locking protocols. "
            "Deadlock detection algorithms construct wait-for graphs to break cyclic dependencies. "
            "Serializability ensures concurrent transaction execution equivalent to serial orders. "
            "Write-ahead logging guarantees durability and atomicity during system crashes."
        ),
        "word_count": 45,
        "co_scores": {"CO1": 0.25, "CO2": 0.9},
        "page_start": 30,
    })
    return chunks


def test_select_chunks_for_unit():
    """Verify select_chunks_for_unit filters by book_id and chapter, with fallback for missing unit."""
    chunks = _create_multi_book_chunks()
    unit_sources = {
        1: {"book_a": ["Chapter 1", "Chapter 2"]},
        2: {"book_b": None},  # All chapters in book_b
    }

    # Unit 1 -> Book A, Chapters 1 and 2 only
    u1_chunks = select_chunks_for_unit(chunks, unit_sources, unit=1)
    assert len(u1_chunks) == 2
    assert all(c["book_id"] == "book_a" for c in u1_chunks)
    assert {c["chapter"] for c in u1_chunks} == {"Chapter 1", "Chapter 2"}

    # Unit 2 -> Book B, all chapters
    u2_chunks = select_chunks_for_unit(chunks, unit_sources, unit=2)
    assert len(u2_chunks) == 2
    assert all(c["book_id"] == "book_b" for c in u2_chunks)

    # Unit 3 missing from unit_sources -> returns all chunks
    u3_chunks = select_chunks_for_unit(chunks, unit_sources, unit=3)
    assert len(u3_chunks) == len(chunks)


def test_every_question_respects_unit_sources():
    """Verify that questions generated for each unit strictly respect the unit's mapped textbooks and chapters."""
    chunks = _create_multi_book_chunks()
    unit_sources = {
        1: {"book_a": ["Chapter 1"]},
        2: {"book_b": ["Chapter 2"]},
    }
    slots = [
        {"unit": 1, "question_no": 1, "sub_index": 1, "level": "L1", "co": "CO1", "marks": 10},
        {"unit": 2, "question_no": 2, "sub_index": 1, "level": "L2", "co": "CO2", "marks": 10},
    ]

    client = MockClient()
    res = generate_paper_by_slots(chunks, slots, client=client, unit_sources=unit_sources, seed=42)

    assert len(res["questions"]) == 2
    q1 = next(q for q in res["questions"] if q["unit"] == 1)
    q2 = next(q for q in res["questions"] if q["unit"] == 2)

    assert q1["book_id"] == "book_a"
    assert q1["chapter"] == "Chapter 1"

    assert q2["book_id"] == "book_b"
    assert q2["chapter"] == "Chapter 2"


def test_unit_missing_from_sources_uses_all_chunks():
    """When a unit is omitted from unit_sources, all available chunks are eligible."""
    chunks = _create_multi_book_chunks()
    unit_sources = {
        1: {"book_a": ["Chapter 1"]},
    }

    # select_chunks_for_unit for unit 3 (omitted) returns all chunks
    selected = select_chunks_for_unit(chunks, unit_sources, unit=3)
    assert len(selected) == len(chunks)
    assert {c["book_id"] for c in selected} == {"book_a", "book_b"}

    # In paper generation, a slot with missing unit is successfully filled from all available chunks
    slots = [
        {"unit": 3, "question_no": 1, "sub_index": 1, "level": "L1", "co": "CO1", "marks": 10},
    ]
    client = MockClient()
    res = generate_paper_by_slots(chunks, slots, client=client, unit_sources=unit_sources, seed=42)
    assert len(res["questions"]) == 1
    assert len(res["unfilled"]) == 0
    assert res["questions"][0]["book_id"] in ["book_a", "book_b"]


def test_end_to_end_100_marks_two_books():
    """End-to-end integration: 100-mark preset, 2 books, MockClient -> 10 questions, all valid with answers, exactly 20 marks each, unfilled empty."""
    chunks = []
    for u in range(1, 6):
        book_id = "book_a" if u <= 3 else "book_b"
        for ch in [1, 2]:
            chunks.append({
                "chunk_id": f"{book_id}_u{u}_c{ch}",
                "book_id": book_id,
                "chapter": f"Chapter {ch}",
                "text": (
                    f"Unit {u} section {ch} describes algorithmic concepts and data system designs in detail. "
                    f"Students learn foundational principles, operational mechanisms, and architectural trade-offs. "
                    f"Methodologies provide step-by-step guidance for solving complex engineering challenges. "
                    f"Practical validation verifies correctness and runtime complexity bounds systematically."
                ),
                "word_count": 48,
                "co_scores": {f"CO{u}": 0.95, "CO1": 0.5},
                "page_start": u * 20 + ch,
            })

    unit_sources = {
        1: {"book_a": ["Chapter 1", "Chapter 2"]},
        2: {"book_a": ["Chapter 1", "Chapter 2"]},
        3: {"book_a": ["Chapter 1", "Chapter 2"]},
        4: {"book_b": ["Chapter 1", "Chapter 2"]},
        5: {"book_b": ["Chapter 1", "Chapter 2"]},
    }

    outcomes = {
        f"CO{i}": f"Outcome {i} mastery of algorithmic concepts and engineering methods"
        for i in range(1, 6)
    }

    # 100-mark preset with 1 sub-question per question (10 questions x 20 marks = 200 offered, 100 attempted)
    bp = build_blueprint(preset=100, sub_question_counts=1, seed=42)
    client = MockClient()

    res = generate_full_paper(
        chunks=chunks,
        blueprint=bp,
        outcomes=outcomes,
        unit_sources=unit_sources,
        client=client,
        seed=42,
    )

    questions = res["questions"]
    assert len(questions) == 10
    assert len(res["unfilled"]) == 0

    for q in questions:
        assert q["valid"] is True
        assert q["marks"] == 20
        assert isinstance(q["answer"], str) and len(q["answer"].strip()) > 0
        assert len(q["answer_alternatives"]) >= 1
        assert q["book_id"] in ["book_a", "book_b"]
        assert q["chapter"] in ["Chapter 1", "Chapter 2"]
        assert q["page"] is not None

    by_q = {}
    for q in questions:
        by_q.setdefault(q["question_no"], 0)
        by_q[q["question_no"]] += q["marks"]

    assert len(by_q) == 10
    for q_num, total_m in by_q.items():
        assert total_m == 20
