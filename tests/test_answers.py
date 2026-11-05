"""Tests for Model Answer Generation and Grounding (Task 2)."""
from generation.generator import generate_answer, generate_paper_by_slots
from generation.mock_client import MockClient


class UnrelatedAnswerClient:
    """Client returning answers completely unrelated to the academic context."""

    def is_up(self):
        return True

    def generate(self, prompt, temperature=None, num_predict=120):
        return "Pizza is topped with tomato sauce and mozzarella cheese baked in a wood-fired oven."


def _sample_chunk():
    return {
        "chunk_id": "c_alice_01",
        "book_id": "algorithms_vol1",
        "chapter": "Chapter 3",
        "text": (
            "Alice analyzed balanced binary search trees and self-balancing AVL trees. "
            "Tree nodes maintain balance factors through single and double rotations. "
            "Inorder tree traversal produces items in strictly ascending sorted order. "
            "Logarithmic search complexity O(log n) guarantees fast lookup performance."
        ),
        "word_count": 45,
        "co_scores": {"CO1": 0.95},
    }


def test_generated_questions_have_non_empty_answers():
    """Verify that every generated question has a non-empty answer and alternatives list."""
    client = MockClient()
    chunk = _sample_chunk()
    slots = [{"co": "CO1", "level": "L2", "marks": 5, "unit": 1, "question_no": 1, "sub_index": 1}]

    res = generate_paper_by_slots([chunk], slots, client=client, seed=42)
    assert len(res["questions"]) == 1
    q = res["questions"][0]

    assert "answer" in q and len(q["answer"].strip()) > 0
    assert "answer_alternatives" in q and len(q["answer_alternatives"]) >= 1
    assert q["answer_alternatives"][0] in q["answer"]


def test_or_splitting_produces_alternatives():
    """Ensure that multi-answer responses split on 'OR' produce separate alternatives."""
    client = MockClient()
    chunk = _sample_chunk()
    question = "Compare binary search trees and balanced AVL tree rotations?"

    ans_rec = generate_answer(question, chunk, marks=5, client=client)
    assert ans_rec["valid"] is True
    assert "OR" in ans_rec["answer"]
    assert len(ans_rec["answer_alternatives"]) >= 2
    for alt in ans_rec["answer_alternatives"]:
        assert len(alt.strip()) > 0
        assert "OR" not in alt.strip().splitlines()


def test_answer_length_grows_with_marks():
    """Verify model answer length scales proportionally with question marks."""
    client = MockClient()
    chunk = _sample_chunk()
    q = "Explain the fundamental operations of binary search trees?"

    ans_2 = generate_answer(q, chunk, marks=2, client=client)
    ans_5 = generate_answer(q, chunk, marks=5, client=client)
    ans_10 = generate_answer(q, chunk, marks=10, client=client)

    words_2 = len(ans_2["answer"].split())
    words_5 = len(ans_5["answer"].split())
    words_10 = len(ans_10["answer"].split())

    assert words_2 < words_5 < words_10


def test_grounded_check_rejects_unrelated_answer():
    """When similarity checking is enabled, unrelated answers must be rejected."""
    client = UnrelatedAnswerClient()
    chunk = _sample_chunk()
    q = "Explain AVL tree node rotation mechanisms?"

    # check_similarity=True enforces cosine similarity evaluation
    ans_rec = generate_answer(q, chunk, marks=5, client=client, check_similarity=True)
    assert ans_rec["valid"] is False
    assert "similarity" in ans_rec["reason"]
