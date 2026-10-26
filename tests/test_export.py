"""Tests for DOCX and PDF question paper export."""
from export_utils import create_docx_exam_paper, create_pdf_exam_paper
from blueprint import get_50_mark_blueprint, calculate_blueprint_stats


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
