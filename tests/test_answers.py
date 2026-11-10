"""Tests for Model Answer Generation and Grounding (Task 2)."""
from generation.generator import (
    check_or_twin_answer_depth_parity,
    extract_scheme_points_and_marks,
    generate_answer,
    generate_paper_by_slots,
    validate_evaluation_scheme,
)
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


def test_scheme_validation_point_sum_and_single_sentence():
    """PART D: Verify marking scheme validators enforce sum(points) == marks and reject single sentence answers."""
    # 1. Single sentence rejected for 5 marks
    ok, msg, flags = validate_evaluation_scheme("Docker containers share the host Linux kernel for isolation.", marks=5)
    assert ok is False
    assert "single-sentence" in msg
    assert "RUBRIC" in flags

    # 2. Sum mismatch rejected
    bad_split = (
        "Point 1 (2M): Containers share the host OS kernel.\n"
        "Point 2 (2M): Namespaces isolate process execution.\n"
        "Point 3 (2M): Cgroups constrain memory usage."
    )
    ok, msg, flags = validate_evaluation_scheme(bad_split, marks=5)
    assert ok is False
    assert "sum 6 != expected 5M" in msg
    assert "RUBRIC" in flags

    # 3. Exact sum accepted
    good_split = (
        "Point 1 (2M): Containers share the host OS kernel.\n"
        "Point 2 (2M): Namespaces isolate process execution.\n"
        "Point 3 (1M): Cgroups constrain memory usage."
    )
    ok, msg, flags = validate_evaluation_scheme(good_split, marks=5)
    assert ok is True
    assert msg == "ok"


def test_scheme_validation_minimum_points_rule():
    """PART D: Marks >= 5 requires at least ceil(marks / 2) points."""
    # 5 marks requires ceil(5/2) = 3 points. 2 points fails.
    two_points = (
        "Point 1 (3M): Containers share the host OS kernel.\n"
        "Point 2 (2M): Namespaces isolate process execution."
    )
    ok, msg, flags = validate_evaluation_scheme(two_points, marks=5)
    assert ok is False
    assert "below minimum 3" in msg

    # 10 marks requires ceil(10/2) = 5 points
    four_points_10m = (
        "Point 1 (3M): First point.\n"
        "Point 2 (3M): Second point.\n"
        "Point 3 (2M): Third point.\n"
        "Point 4 (2M): Fourth point."
    )
    ok, msg, flags = validate_evaluation_scheme(four_points_10m, marks=10, level="L2")
    assert ok is False
    assert "below minimum 5" in msg


def test_scheme_validation_l5_l6_rubrics():
    """PART D: L5 and L6 answers must adhere to structured evaluation or creation rubrics."""
    # L5 without criteria/verdict/justification/tradeoffs fails
    plain_l5 = (
        "Point 1 (2M): Option A performs fast.\n"
        "Point 2 (2M): Option B consumes more memory.\n"
        "Point 3 (1M): Both options are valid."
    )
    ok, msg, flags = validate_evaluation_scheme(plain_l5, marks=5, level="L5")
    assert ok is False
    assert "rubric" in msg.lower()
    assert "RUBRIC" in flags

    # L5 with structured rubric passes
    structured_l5 = (
        "Criteria (2M): Scalability, fault-tolerance, and latency benchmarks.\n"
        "Verdict (1M): Kubernetes stateful sets are superior for database persistence.\n"
        "Justification (1M): Automated failover and persistent volume claims ensure data durability.\n"
        "Trade-offs (1M): Increased control-plane complexity compared to standalone pods."
    )
    ok, msg, _ = validate_evaluation_scheme(structured_l5, marks=5, level="L5")
    assert ok is True

    # L6 with structured creation rubric passes
    structured_l6 = (
        "Architecture / Plan Steps (2M): Define modular microservices with ingress controllers.\n"
        "Rationale (2M): Decouples deployment velocity and isolates resource contention.\n"
        "Trade-offs (1M): Requires distributed tracing to monitor inter-service latency."
    )
    ok, msg, _ = validate_evaluation_scheme(structured_l6, marks=5, level="L6")
    assert ok is True


def test_mock_client_outputs_itemized_points_and_exact_sum():
    """PART D: MockClient answers are itemized with mark splits exactly summing to marks."""
    client = MockClient()
    chunk = _sample_chunk()

    # 5-mark answer
    ans_5 = generate_answer("Explain AVL tree rotations?", chunk, marks=5, client=client, level="L2")
    assert ans_5["valid"] is True
    pts_5, marks_5 = extract_scheme_points_and_marks(ans_5["answer"])
    assert len(pts_5) >= 3  # ceil(5/2) = 3
    assert sum(marks_5) == 5

    # 10-mark L5 answer
    ans_10_l5 = generate_answer("Evaluate AVL trees versus Red-Black trees?", chunk, marks=10, client=client, level="L5")
    assert ans_10_l5["valid"] is True
    pts_10, marks_10 = extract_scheme_points_and_marks(ans_10_l5["answer"])
    assert sum(marks_10) == 10
    assert any("criteria" in p.lower() for p in pts_10)
    assert any("verdict" in p.lower() for p in pts_10)

    # 10-mark L6 answer
    ans_10_l6 = generate_answer("Design an auto-balancing indexing tree structure?", chunk, marks=10, client=client, level="L6")
    assert ans_10_l6["valid"] is True
    pts_l6, marks_l6 = extract_scheme_points_and_marks(ans_10_l6["answer"])
    assert sum(marks_l6) == 10
    assert any("plan" in p.lower() or "architecture" in p.lower() for p in pts_l6)


def test_or_twin_answer_depth_parity():
    """PART D: Disparity in answer depth between OR twins triggers RUBRIC review flag."""
    questions = [
        {
            "id": "1(a)",
            "unit": 1,
            "alternative_group": "U1_A",
            "marks": 10,
            "answer": "Point 1 (2M): Short answer text only here.",
        },
        {
            "id": "2(a)",
            "unit": 1,
            "alternative_group": "U1_B",
            "marks": 10,
            "answer": (
                "Point 1 (2M): Extremely comprehensive detailed analysis of algorithms with extensive proof.\n"
                "Point 2 (2M): Thorough discussion of time complexity and execution bounds.\n"
                "Point 3 (2M): Space overhead exploration in hardware registers.\n"
                "Point 4 (2M): Comparison benchmarks across diverse data distribution datasets.\n"
                "Point 5 (2M): Concrete real-world production engineering deployment recommendations."
            ),
        },
    ]
    disparities = check_or_twin_answer_depth_parity(questions)
    assert len(disparities) == 1
    assert "RUBRIC: answer depth disparity" in disparities[0][2]

