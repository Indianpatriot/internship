"""Tests for DOCX and PDF question paper export."""
import io
import docx
import fitz
from export_utils import create_docx_exam_paper, create_pdf_exam_paper
from blueprint import get_50_mark_blueprint, build_blueprint, calculate_blueprint_stats
from totals import summarize


def test_create_docx_exam_paper():
    questions = [
        {"id": "1", "part": "Part A", "question": "Define machine learning algorithms?", "level": "L1", "co": "CO1", "marks": 2},
        {"id": "2", "part": "Part A", "question": "Explain gradient descent optimization?", "level": "L2", "co": "CO2", "marks": 2},
        {"id": "6(a)", "part": "Part B", "question": "Demonstrate the backpropagation algorithm with an example?", "level": "L3", "co": "CO1", "marks": 5},
        {"id": "6(b)", "part": "Part B", "question": "Analyze overfitting mitigation techniques?", "level": "L4", "co": "CO2", "marks": 5},
    ]
    outcomes = {"CO1": "Understand ML concepts", "CO2": "Analyze optimization"}
    docx_bytes = create_docx_exam_paper(questions, course_name="Machine Learning", outcomes=outcomes)
    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 1000
    # Check standard docx zip signature (PK..)
    assert docx_bytes.startswith(b"PK\x03\x04")


def test_create_pdf_exam_paper():
    questions = [
        {"id": "1", "part": "Part A", "question": "Define machine learning algorithms?", "level": "L1", "co": "CO1", "marks": 2},
        {"id": "2", "part": "Part A", "question": "Explain gradient descent optimization?", "level": "L2", "co": "CO2", "marks": 2},
        {"id": "6(a)", "part": "Part B", "question": "Demonstrate the backpropagation algorithm with an example?", "level": "L3", "co": "CO1", "marks": 5},
        {"id": "6(b)", "part": "Part B", "question": "Analyze overfitting mitigation techniques?", "level": "L4", "co": "CO2", "marks": 5},
    ]
    outcomes = {"CO1": "Understand ML concepts", "CO2": "Analyze optimization"}
    pdf_bytes = create_pdf_exam_paper(questions, course_name="Machine Learning", outcomes=outcomes)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    # Check standard PDF header (%PDF-)
    assert pdf_bytes.startswith(b"%PDF-")


def test_50_mark_blueprint_structure():
    slots = get_50_mark_blueprint(["CO1", "CO2", "CO3"])
    stats = calculate_blueprint_stats(slots)
    assert stats["total_marks"] == 50
    assert stats["is_standard_50"] is True
    assert stats["part_marks"]["Part A"] == 10
    assert stats["part_marks"]["Part B"] == 40
    assert len(slots) == 13  # 5 in Part A + 8 in Part B (4 questions x 2 subparts)


def test_export_answers_and_or_format():
    """Substep 2.1: Verify model answers in DOCX and PDF, including OR choice and answer alternatives."""
    questions = [
        {
            "id": "1(a)",
            "unit": 1,
            "question_no": 1,
            "sub_index": 1,
            "alternative_group": "U1",
            "question": "Explain Docker containerization architecture in detail?",
            "marks": 10,
            "level": "L2",
            "knowledge_dimension": "Conceptual",
            "co": "CO1",
            "answer": "Containers provide OS-level virtualization sharing the host kernel.\nOR\nDocker encapsulates microservices inside isolated runtime namespaces.",
        },
        {
            "id": "2(a)",
            "unit": 1,
            "question_no": 2,
            "sub_index": 1,
            "alternative_group": "U1",
            "question": "Contrast virtual machines with operating system containers?",
            "marks": 10,
            "level": "L4",
            "knowledge_dimension": "Conceptual",
            "co": "CO1",
            "answer": "Virtual machines emulate entire hardware systems with hypervisors.",
        },
    ]
    outcomes = {"CO1": "Understand virtualization and containers"}

    # 1. DOCX Export
    docx_bytes = create_docx_exam_paper(
        questions,
        course_name="Cloud Systems",
        outcomes=outcomes,
        include_answers=True,
        include_totals=False,
    )
    doc = docx.Document(io.BytesIO(docx_bytes))
    full_docx_text = "\n".join(
        [p.text for p in doc.paragraphs]
        + [c.text for t in doc.tables for row in t.rows for c in row.cells]
    )

    assert "SCHEME OF EVALUATION & MODEL ANSWERS" in full_docx_text
    assert "Containers provide OS-level virtualization" in full_docx_text
    assert "Docker encapsulates microservices" in full_docx_text
    assert "— OR —" in full_docx_text

    # 2. PDF Export
    pdf_bytes = create_pdf_exam_paper(
        questions,
        course_name="Cloud Systems",
        outcomes=outcomes,
        include_answers=True,
        include_totals=False,
    )
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pdf_text = "\n".join([page.get_text() for page in pdf_doc])

    assert "SCHEME OF EVALUATION & MODEL ANSWERS" in pdf_text
    assert "Containers provide OS-level virtualization" in pdf_text
    assert "Docker encapsulates microservices" in pdf_text
    assert "— OR —" in pdf_text


def test_export_cumulative_totals():
    """Substep 2.2: Add optional totals table at the end (enabled vs disabled)."""
    questions = [
        {"id": "1", "part": "Part A", "question": "Define machine learning?", "level": "L1", "co": "CO1", "marks": 2},
        {"id": "2", "part": "Part A", "question": "Explain gradient descent?", "level": "L2", "co": "CO2", "marks": 2},
        {"id": "3", "part": "Part B", "question": "Implement backpropagation?", "level": "L3", "co": "CO1", "marks": 5},
    ]
    outcomes = {"CO1": "Foundations", "CO2": "Optimization"}

    # Test when include_totals=True
    d_with = create_docx_exam_paper(questions, outcomes=outcomes, include_totals=True)
    doc_with = docx.Document(io.BytesIO(d_with))
    text_with = "\n".join(
        [p.text for p in doc_with.paragraphs]
        + [c.text for t in doc_with.tables for row in t.rows for c in row.cells]
    )
    assert "CUMULATIVE" in text_with
    assert "Course Outcome" in text_with
    assert "Bloom's Taxonomy" in text_with

    p_with = create_pdf_exam_paper(questions, outcomes=outcomes, include_totals=True)
    pdf_doc_with = fitz.open(stream=p_with, filetype="pdf")
    p_text_with = "\n".join([page.get_text() for page in pdf_doc_with])
    assert "CUMULATIVE" in p_text_with
    assert "Course Outcomes (CO) Distribution" in p_text_with
    assert "Bloom's Taxonomy (BT) Cognitive Distribution" in p_text_with

    # Test when include_totals=False
    d_without = create_docx_exam_paper(questions, outcomes=outcomes, include_totals=False)
    doc_without = docx.Document(io.BytesIO(d_without))
    text_without = "\n".join([p.text for p in doc_without.paragraphs])
    assert "CUMULATIVE" not in text_without

    p_without = create_pdf_exam_paper(questions, outcomes=outcomes, include_totals=False)
    pdf_doc_without = fitz.open(stream=p_without, filetype="pdf")
    p_text_without = "\n".join([page.get_text() for page in pdf_doc_without])
    assert "CUMULATIVE" not in p_text_without


def test_export_100_mark_preset_choice_paper():
    """Verify 100-mark paper export with 5 units, internal choice questions, and totals."""
    bp = build_blueprint(preset=100, sub_question_counts=2, seed=42)
    slots = bp["slots"]
    assert len(slots) == 20  # 5 units x 2 questions x 2 subparts
    questions = []
    for s in slots:
        q_copy = dict(s)
        q_copy["question"] = f"Explain concepts relating to unit {s['unit']} outcome {s['co']}?"
        q_copy["answer"] = f"Standard model solution for unit {s['unit']} problem."
        questions.append(q_copy)

    summary = summarize(questions, attempted_only=True)
    assert summary["total_marks"] == 100
    assert summary["offered_marks"] == 200

    docx_bytes = create_docx_exam_paper(
        questions,
        course_name="Distributed Computing",
        subject_code="CS-501",
        include_answers=True,
        include_totals=True,
        totals_summary=summary,
    )
    assert len(docx_bytes) > 20000
    assert docx_bytes.startswith(b"PK\x03\x04")

    pdf_bytes = create_pdf_exam_paper(
        questions,
        course_name="Distributed Computing",
        subject_code="CS-501",
        include_answers=True,
        include_totals=True,
        totals_summary=summary,
    )
    assert len(pdf_bytes) > 5000
    assert pdf_bytes.startswith(b"%PDF-")
