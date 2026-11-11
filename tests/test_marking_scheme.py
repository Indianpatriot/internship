"""Tests for Examiner Marking Scheme breakdown, contracts, generation, and export."""
import io
import docx
import fitz
import pytest

import config
from contracts import MarkingScheme, SchemeCriterion, validate_marking_scheme
from generation.mock_client import MockClient
from generation.scheme_generator import (
    build_fallback_scheme,
    extract_json_object,
    generate_scheme,
    repair_scheme_marks,
)
from generation.generator import generate_paper_by_slots
from blueprint import build_blueprint
from export_utils import create_docx_exam_paper, create_pdf_exam_paper


def _sample_chunk():
    return {
        "chunk_id": "c_alice_01",
        "book_id": "docker_practice",
        "chapter": "Chapter 2: Docker Architecture",
        "text": (
            "Alice designed the container execution runtime and daemon architecture. "
            "The Docker daemon listens for Docker API requests and manages Docker objects such as images, "
            "containers, networks, and storage volumes. The client and daemon communicate using a REST API "
            "over UNIX sockets or a network interface. Registries store container images securely."
        ),
        "word_count": 55,
        "co_scores": {"CO1": 0.95},
    }


class BrokenJsonClient:
    """Mock client returning malformed or broken JSON text."""

    def is_up(self):
        return True

    def generate(self, prompt, **kwargs):
        return '{"total_marks": 10, "criteria": [{"no": 1, "criterion": "Unterminated string'


class WrongSumClient:
    """Mock client returning valid JSON but with widely inaccurate mark sum."""

    def is_up(self):
        return True

    def generate(self, prompt, **kwargs):
        return """
        {
          "total_marks": 10,
          "criteria": [
            {
              "no": 1,
              "criterion": "Huge criterion",
              "expected": "Unexpected allocation",
              "marks": 99,
              "kind": "points",
              "reason": "Invalid allocation sum",
              "partial_credit": null,
              "alternatives": []
            }
          ],
          "notes": "Testing wrong sum"
        }
        """


class EmptyClient:
    """Mock client returning completely empty text."""

    def is_up(self):
        return True

    def generate(self, prompt, **kwargs):
        return ""


def test_validate_marking_scheme():
    """Verify validate_marking_scheme rules: exact sum, off-by-one, reasons, negative/half marks."""
    # 1. Exact sum passes
    valid_scheme: MarkingScheme = {
        "total_marks": 10,
        "criteria": [
            {
                "no": 1,
                "criterion": "Definition of container daemon",
                "expected": "Identifies core daemon role and socket communication.",
                "marks": 4,
                "kind": "definition",
                "reason": "L2 recall of core architecture; 4 marks.",
                "partial_credit": "2 marks for partial definition",
                "alternatives": [],
            },
            {
                "no": 2,
                "criterion": "Components and workflow",
                "expected": "Explains client, daemon, and registry interactions.",
                "marks": 6,
                "kind": "explanation",
                "reason": "Demonstrates comprehensive component comprehension; 6 marks.",
                "partial_credit": "3 marks for partial interaction flow",
                "alternatives": [],
            },
        ],
        "notes": "Accept equivalent architectural terminology.",
    }
    ok, problems = validate_marking_scheme(valid_scheme, 10)
    assert ok is True, f"Expected valid scheme to pass, got: {problems}"

    # 2. Off-by-one sum fails
    bad_sum_scheme = dict(valid_scheme)
    bad_sum_scheme["total_marks"] = 10
    bad_sum_scheme["criteria"] = [
        dict(valid_scheme["criteria"][0], marks=4),
        dict(valid_scheme["criteria"][1], marks=5),  # Sum = 9 != 10
    ]
    ok, problems = validate_marking_scheme(bad_sum_scheme, 10)
    assert ok is False
    assert any("Criteria marks sum to 9.0" in p for p in problems)

    # Off-by-one total_marks vs expected_total
    ok, problems = validate_marking_scheme(valid_scheme, 11)
    assert ok is False
    assert any("does not match expected total" in p for p in problems)

    # 3. Criteria with missing or empty reason fail
    missing_reason_scheme = dict(valid_scheme)
    missing_reason_scheme["criteria"] = [
        dict(valid_scheme["criteria"][0], reason=""),
        dict(valid_scheme["criteria"][1]),
    ]
    ok, problems = validate_marking_scheme(missing_reason_scheme, 10)
    assert ok is False
    assert any("missing or empty 'reason'" in p for p in problems)

    # Missing expected text fails
    missing_expected_scheme = dict(valid_scheme)
    missing_expected_scheme["criteria"] = [
        dict(valid_scheme["criteria"][0], expected="   "),
        dict(valid_scheme["criteria"][1]),
    ]
    ok, problems = validate_marking_scheme(missing_expected_scheme, 10)
    assert ok is False
    assert any("missing or empty 'expected'" in p for p in problems)

    # 4. Negative or zero marks fail
    zero_mark_scheme = dict(valid_scheme)
    zero_mark_scheme["criteria"] = [
        dict(valid_scheme["criteria"][0], marks=0),
        dict(valid_scheme["criteria"][1], marks=10),
    ]
    ok, problems = validate_marking_scheme(zero_mark_scheme, 10)
    assert ok is False
    assert any("marks must be greater than zero" in p for p in problems)

    neg_mark_scheme = dict(valid_scheme)
    neg_mark_scheme["criteria"] = [
        dict(valid_scheme["criteria"][0], marks=-2),
        dict(valid_scheme["criteria"][1], marks=12),
    ]
    ok, problems = validate_marking_scheme(neg_mark_scheme, 10)
    assert ok is False
    assert any("marks must be greater than zero" in p for p in problems)

    # 5. Half marks rejected when ALLOW_HALF_MARKS is False
    config.ALLOW_HALF_MARKS = False
    half_mark_scheme: MarkingScheme = {
        "total_marks": 5,
        "criteria": [
            {
                "no": 1,
                "criterion": "Definition",
                "expected": "Core concept",
                "marks": 2.5,
                "kind": "definition",
                "reason": "Recall of concept",
                "partial_credit": None,
                "alternatives": [],
            },
            {
                "no": 2,
                "criterion": "Explanation",
                "expected": "Mechanism",
                "marks": 2.5,
                "kind": "explanation",
                "reason": "Comprehension of mechanism",
                "partial_credit": None,
                "alternatives": [],
            },
        ],
    }
    ok, problems = validate_marking_scheme(half_mark_scheme, 5)
    assert ok is False
    assert any("fractional marks" in p for p in problems)

    # 6. Half marks allowed when ALLOW_HALF_MARKS is True
    config.ALLOW_HALF_MARKS = True
    try:
        ok, problems = validate_marking_scheme(half_mark_scheme, 5)
        assert ok is True, f"Expected half marks to pass when ALLOW_HALF_MARKS is True, got: {problems}"
    finally:
        config.ALLOW_HALF_MARKS = False


def test_generate_scheme_mock_client_marks_and_levels():
    """Verify generate_scheme with MockClient for marks 5, 10, 20 and levels L1-L5."""
    client = MockClient()
    chunk = _sample_chunk()

    for marks in [5, 10, 20]:
        for level in ["L1", "L2", "L3", "L4", "L5"]:
            q_rec = {
                "question": f"Explain the core runtime architecture of container daemon for {marks} marks?",
                "level": level,
                "marks": marks,
                "answer": (
                    f"Point 1 ({marks // 2}M): The container daemon manages images, containers, and networks.\n"
                    f"Point 2 ({marks - (marks // 2)}M): The client communicates over REST API socket."
                ),
            }

            scheme = generate_scheme(q_rec, chunk, client=client)
            assert scheme is not None
            assert scheme["total_marks"] == marks

            ok, problems = validate_marking_scheme(scheme, marks)
            assert ok is True, f"Failed for level {level}, marks {marks}: {problems}"

            c_sum = sum(c["marks"] for c in scheme["criteria"])
            assert c_sum == marks

            for c in scheme["criteria"]:
                assert len(c["reason"].strip()) > 0
                assert len(c["expected"].strip()) > 0
                assert c["marks"] > 0


def test_diagram_criterion_inclusion_rule():
    """Verify question containing 'draw' or 'diagram' gets a diagram criterion; others do not."""
    client = MockClient()
    chunk = _sample_chunk()

    # Question with 'draw' and 'diagram'
    q_draw = {
        "question": "Draw and explain the architecture diagram of Docker client, daemon, and registry?",
        "level": "L2",
        "marks": 10,
        "answer": "Architecture diagram consists of client, daemon, and registry communicating via REST API.",
    }
    scheme_draw = generate_scheme(q_draw, chunk, client=client)
    has_diag = any(c.get("kind") == "diagram" for c in scheme_draw["criteria"])
    assert has_diag is True, "Expected diagram criterion for question asking to draw architecture diagram"

    # Question without drawing / diagram
    q_no_draw = {
        "question": "Describe the operational responsibilities of the Docker container daemon?",
        "level": "L2",
        "marks": 10,
        "answer": "The Docker container daemon manages containers, storage volumes, and virtual networks.",
    }
    scheme_no_draw = generate_scheme(q_no_draw, chunk, client=client)
    has_no_diag = any(c.get("kind") == "diagram" for c in scheme_no_draw["criteria"])
    assert has_no_diag is False, "Expected NO diagram criterion for question without draw/diagram request"


def test_malformed_model_output_fallback():
    """Verify malformed model output (broken JSON, wrong sum, empty) falls back to valid rule-based scheme."""
    chunk = _sample_chunk()
    q_rec = {
        "question": "Explain the interaction between Docker client and daemon?",
        "level": "L2",
        "marks": 10,
        "answer": (
            "Alice analyzed the container daemon architecture. "
            "The client communicates over REST API sockets. "
            "Images are pulled securely from the registry."
        ),
    }

    # 1. Broken JSON
    scheme_broken = generate_scheme(q_rec, chunk, client=BrokenJsonClient())
    assert scheme_broken is not None
    assert scheme_broken.get("is_fallback") is True
    ok, problems = validate_marking_scheme(scheme_broken, 10)
    assert ok is True, f"Fallback for broken JSON failed validation: {problems}"
    assert sum(c["marks"] for c in scheme_broken["criteria"]) == 10

    # 2. Inaccurate mark sum beyond repair
    scheme_wrong_sum = generate_scheme(q_rec, chunk, client=WrongSumClient())
    assert scheme_wrong_sum is not None
    assert scheme_wrong_sum.get("is_fallback") is True
    ok, problems = validate_marking_scheme(scheme_wrong_sum, 10)
    assert ok is True, f"Fallback for wrong sum failed validation: {problems}"
    assert sum(c["marks"] for c in scheme_wrong_sum["criteria"]) == 10

    # 3. Empty client output
    scheme_empty = generate_scheme(q_rec, chunk, client=EmptyClient())
    assert scheme_empty is not None
    assert scheme_empty.get("is_fallback") is True
    ok, problems = validate_marking_scheme(scheme_empty, 10)
    assert ok is True, f"Fallback for empty output failed validation: {problems}"
    assert len(scheme_empty["criteria"]) >= 2


def test_sub_questions_schemes_summing_to_each_marks():
    """Verify sub-questions (e.g. 20-mark split 10+10 or 7+7+6) get schemes summing to each sub-question's marks."""
    client = MockClient()
    chunk = _sample_chunk()

    # Case A: 20-mark question split 10 + 10
    slots_10_10 = [
        {"id": "1(a)", "unit": 1, "question_no": 1, "sub_index": "a", "marks": 10, "level": "L2", "co": "CO1"},
        {"id": "1(b)", "unit": 1, "question_no": 1, "sub_index": "b", "marks": 10, "level": "L3", "co": "CO1"},
    ]
    res_a = generate_paper_by_slots([chunk], slots_10_10, client=client, seed=42)
    assert len(res_a["questions"]) == 2
    for q in res_a["questions"]:
        assert q["marks"] == 10
        scheme = q.get("marking_scheme")
        assert scheme is not None
        assert scheme["total_marks"] == 10
        ok, problems = validate_marking_scheme(scheme, 10)
        assert ok is True, f"Sub-question {q['id']} scheme failed validation: {problems}"
        assert sum(c["marks"] for c in scheme["criteria"]) == 10

    # Case B: 20-mark question split 7 + 7 + 6
    slots_7_7_6 = [
        {"id": "2(a)", "unit": 1, "question_no": 2, "sub_index": "a", "marks": 7, "level": "L1", "co": "CO1"},
        {"id": "2(b)", "unit": 1, "question_no": 2, "sub_index": "b", "marks": 7, "level": "L2", "co": "CO1"},
        {"id": "2(c)", "unit": 1, "question_no": 2, "sub_index": "c", "marks": 6, "level": "L4", "co": "CO1"},
    ]
    res_b = generate_paper_by_slots([chunk], slots_7_7_6, client=client, seed=101)
    assert len(res_b["questions"]) == 3
    for q in res_b["questions"]:
        expected_m = q["marks"]
        scheme = q.get("marking_scheme")
        assert scheme is not None
        assert scheme["total_marks"] == expected_m
        ok, problems = validate_marking_scheme(scheme, expected_m)
        assert ok is True, f"Sub-question {q['id']} scheme failed validation: {problems}"
        assert sum(c["marks"] for c in scheme["criteria"]) == expected_m


def test_end_to_end_100_mark_preset_with_schemes():
    """Verify 100-mark preset with mock client yields questions, every one with a valid marking_scheme."""
    client = MockClient()
    chunk = _sample_chunk()

    # 100-mark preset: 5 units x 2 choice questions = 10 questions of 20 marks each
    blueprint = build_blueprint(preset=100, sub_question_counts=1, seed=42)
    slots = blueprint["slots"]
    assert len(slots) == 10

    # Ensure chunks have CO scores for all Course Outcomes (CO1..CO5)
    all_cos = {"CO1": 0.95, "CO2": 0.95, "CO3": 0.95, "CO4": 0.95, "CO5": 0.95}
    chunks = [dict(chunk, chunk_id=f"chunk_{i}", co_scores=all_cos) for i in range(20)]
    res = generate_paper_by_slots(chunks, slots, client=client, seed=42)

    assert len(res["questions"]) == 10
    for idx, q in enumerate(res["questions"], 1):
        assert q.get("valid") is True
        scheme = q.get("marking_scheme")
        assert scheme is not None, f"Question {q.get('id')} missing marking_scheme"
        assert scheme["total_marks"] == q["marks"]

        ok, problems = validate_marking_scheme(scheme, q["marks"])
        assert ok is True, f"Question {q.get('id')} failed scheme validation: {problems}"

        crit_sum = sum(c["marks"] for c in scheme["criteria"])
        assert crit_sum == q["marks"]


def test_export_smoke_test_docx_and_pdf_with_schemes():
    """Verify DOCX and PDF with schemes generate without error and contain criterion text and total."""
    client = MockClient()
    chunk = _sample_chunk()

    q_rec = {
        "id": "1",
        "unit": 1,
        "question_no": 1,
        "question": "Explain Docker architecture and daemon role?",
        "marks": 10,
        "level": "L2",
        "co": "CO1",
        "chapter": "Chapter 2: Docker Architecture",
        "page_start": 42,
        "display_label": "Docker in Practice",
        "answer": "The Docker daemon executes client API commands and controls container lifecycle.",
        "marking_scheme": {
            "total_marks": 10,
            "criteria": [
                {
                    "no": 1,
                    "criterion": "Definition of container daemon",
                    "expected": "Daemon listens for Docker API requests and manages objects.",
                    "marks": 4,
                    "kind": "definition",
                    "reason": "Direct recall of architecture required for L2.",
                    "partial_credit": "2 marks for partial definition",
                    "alternatives": ["Accept alternate socket explanation"],
                },
                {
                    "no": 2,
                    "criterion": "Component interaction workflow",
                    "expected": "Client interacts with daemon over REST API socket.",
                    "marks": 6,
                    "kind": "explanation",
                    "reason": "Comprehension of runtime execution flow.",
                    "partial_credit": "3 marks for partial description",
                    "alternatives": [],
                },
            ],
            "notes": "Evaluate technical terminology accurately.",
            "is_fallback": False,
        },
    }

    # 1. Test DOCX generation
    docx_bytes = create_docx_exam_paper(
        questions=[q_rec],
        course_name="Cloud Engineering",
        subject_code="CS-501",
        include_answers=True,
        include_marking_scheme=True,
        include_reasons=True,
    )
    assert docx_bytes is not None
    assert len(docx_bytes) > 1000

    doc = docx.Document(io.BytesIO(docx_bytes))
    docx_text = "\n".join([p.text for p in doc.paragraphs])
    for table in doc.tables:
        for row in table.rows:
            docx_text += "\n" + " | ".join(cell.text for cell in row.cells)

    assert "Marking Scheme & Evaluation Breakdown" in docx_text
    assert "Definition of container daemon" in docx_text
    assert "What Examiner Looks For" in docx_text
    assert "Reason" in docx_text
    assert "Total Marks" in docx_text

    # 2. Test PDF generation
    pdf_bytes = create_pdf_exam_paper(
        questions=[q_rec],
        course_name="Cloud Engineering",
        subject_code="CS-501",
        include_answers=True,
        include_marking_scheme=True,
        include_reasons=True,
    )
    assert pdf_bytes is not None
    assert len(pdf_bytes) > 1000

    doc_pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    pdf_text = "\n".join([page.get_text() for page in doc_pdf])
    norm_pdf_text = " ".join(pdf_text.split())

    assert "Marking Scheme & Evaluation Breakdown" in norm_pdf_text
    assert "Definition of container daemon" in norm_pdf_text
    assert "Total Marks" in norm_pdf_text
