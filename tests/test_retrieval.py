import pytest
from retrieval.co_mapper import score_chunks_against_outcomes


def test_co_mapper_scores_expected_chunk_highest():
    """Assert the expected chunk scores highest for each outcome using generic dummy data."""
    pytest.importorskip(
        "sentence_transformers",
        reason="sentence_transformers requires Python 3.12 venv (Python 3.15 has no torch wheels)",
    )
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Alice explained binary search trees and recursive tree traversal techniques such as inorder traversal.",
        },
        {
            "chunk_id": "c2",
            "text": "Bob discussed relational database schema design, normalization forms, and executing complex SQL queries.",
        },
        {
            "chunk_id": "c3",
            "text": "John demonstrated digital logic circuits, NAND gates, boolean simplification, and Karnaugh maps.",
        },
    ]

    outcomes = {
        "CO1": "Understand binary search trees, traversal algorithms, and tree data structures.",
        "CO2": "Design relational databases, write SQL queries, and perform normalization.",
    }

    scored = score_chunks_against_outcomes(chunks, outcomes)

    # Each chunk should have co_scores dictionary containing both CO1 and CO2
    for c in scored:
        assert "co_scores" in c
        assert "CO1" in c["co_scores"]
        assert "CO2" in c["co_scores"]
        assert isinstance(c["co_scores"]["CO1"], float)
        assert isinstance(c["co_scores"]["CO2"], float)

    # Chunk 1 (trees) should score highest for CO1
    best_co1_chunk = max(scored, key=lambda c: c["co_scores"]["CO1"])
    assert best_co1_chunk["chunk_id"] == "c1"
    assert scored[0]["co_scores"]["CO1"] > scored[1]["co_scores"]["CO1"]
    assert scored[0]["co_scores"]["CO1"] > scored[2]["co_scores"]["CO1"]

    # Chunk 2 (databases) should score highest for CO2
    best_co2_chunk = max(scored, key=lambda c: c["co_scores"]["CO2"])
    assert best_co2_chunk["chunk_id"] == "c2"
    assert scored[1]["co_scores"]["CO2"] > scored[0]["co_scores"]["CO2"]
    assert scored[1]["co_scores"]["CO2"] > scored[2]["co_scores"]["CO2"]


def test_co_mapper_edge_cases():
    """Test empty chunks and empty outcomes handling."""
    assert score_chunks_against_outcomes([], {"CO1": "Something"}) == []

    chunks = [{"text": "Sample text by Alice"}]
    res = score_chunks_against_outcomes(chunks, {})
    assert res[0]["co_scores"] == {}
