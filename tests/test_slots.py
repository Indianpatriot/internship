"""Tests for slot-based question paper generation."""
from config import MIN_CONTEXT_WORDS
from generation.generator import generate_paper_by_slots
from generation.mock_client import MockClient


class SelectiveMockClient(MockClient):
    """Mock client that fails on an impossible level or prompt to test unfilled slots."""

    def __init__(self, fail_on_level="L5"):
        self.fail_on_level = fail_on_level

    def generate(self, prompt, temperature=None, num_predict=120):
        if f"Bloom's {self.fail_on_level}" in prompt:
            # Output that fails validation (no valid Bloom's verb, doesn't end with ?)
            return "This response contains no valid verb and has no question mark"
        return super().generate(prompt, temperature, num_predict)


def _make_dummy_chunks():
    """Create sample chunks with sufficient length and dummy names."""
    alice_text = (
        "Alice explored binary search trees, balanced height AVL trees, and red-black trees. "
        "The recursive algorithms traverse left and right subtrees systematically. "
        "Each tree node stores a key, left child reference, right child reference, and height factor. "
        "Tree traversal includes inorder, preorder, and postorder traversal methods."
    )
    bob_text = (
        "Bob designed relational database schemas with primary keys, foreign keys, and indexes. "
        "SQL queries join tables using inner, left, and outer join operations efficiently. "
        "Database normalization rules from 1NF to BCNF eliminate anomalies and redundancy. "
        "Transactions maintain ACID properties across distributed database systems."
    )
    john_text = (
        "John analyzed computational complexity, big O notation, and asymptotic bounds. "
        "Divide and conquer algorithms like merge sort achieve optimal O(n log n) running time. "
        "Dynamic programming and greedy heuristics solve optimization problems across graph topologies. "
        "P versus NP problems define fundamental boundaries of efficient algorithmic computation."
    )

    chunks = [
        {
            "chunk_id": "chunk_alice_001",
            "text": alice_text,
            "word_count": len(alice_text.split()),
            "co_scores": {"CO1": 0.92, "CO2": 0.15},
        },
        {
            "chunk_id": "chunk_bob_002",
            "text": bob_text,
            "word_count": len(bob_text.split()),
            "co_scores": {"CO1": 0.18, "CO2": 0.95},
        },
        {
            "chunk_id": "chunk_john_003",
            "text": john_text,
            "word_count": len(john_text.split()),
            "co_scores": {"CO1": 0.45, "CO2": 0.35},
        },
    ]
    for c in chunks:
        assert c["word_count"] >= MIN_CONTEXT_WORDS
    return chunks


def test_generate_paper_by_slots_success():
    """Assert each returned question matches slot co, level, marks, and ids are unique."""
    client = MockClient()
    chunks = _make_dummy_chunks()

    slots = [
        {"co": "CO1", "level": "L1", "marks": 5},
        {"co": "CO2", "level": "L2", "marks": 10},
        {"co": "CO1", "level": "L3", "marks": 7},
    ]

    result = generate_paper_by_slots(chunks, slots, client, seed=42)

    questions = result["questions"]
    assert len(questions) == len(slots)
    assert len(result["unfilled"]) == 0

    # Assert matching properties
    for q, slot in zip(questions, slots):
        assert q["co"] == slot["co"]
        assert q["level"] == slot["level"]
        assert q["marks"] == slot["marks"]
        assert q["valid"] is True
        assert q["question"].endswith("?")

    # Assert unique IDs
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    assert ids == ["Q001", "Q002", "Q003"]


def test_generate_paper_by_slots_unfilled_slot():
    """Assert an impossible slot (no valid question generated) lands in 'unfilled'."""
    client = SelectiveMockClient(fail_on_level="L5")
    chunks = _make_dummy_chunks()

    slots = [
        {"co": "CO1", "level": "L1", "marks": 5},
        {"co": "CO2", "level": "L5", "marks": 15},  # Impossible: client fails validation
    ]

    result = generate_paper_by_slots(chunks, slots, client, seed=42)

    # CO1/L1 succeeds
    assert len(result["questions"]) == 1
    assert result["questions"][0]["co"] == "CO1"
    assert result["questions"][0]["level"] == "L1"

    # CO2/L5 is impossible and lands in unfilled
    assert len(result["unfilled"]) == 1
    assert result["unfilled"][0]["co"] == "CO2"
    assert result["unfilled"][0]["level"] == "L5"
    assert result["unfilled"][0]["marks"] == 15
    assert len(result["rejected"]) > 0


def test_generate_paper_by_slots_no_usable_chunks():
    """Assert slot lands in unfilled when no chunks meet MIN_CONTEXT_WORDS."""
    client = MockClient()
    short_chunks = [
        {
            "chunk_id": "short_001",
            "text": "Short chunk with Alice.",
            "word_count": 4,
            "co_scores": {"CO1": 0.9},
        }
    ]
    slots = [{"co": "CO1", "level": "L1", "marks": 5}]
    result = generate_paper_by_slots(short_chunks, slots, client, seed=42)
    assert len(result["questions"]) == 0
    assert len(result["unfilled"]) == 1
    assert result["unfilled"][0]["co"] == "CO1"
