"""Tests for DOCX and PDF question paper export."""
from export_utils import create_docx_exam_paper, create_pdf_exam_paper


def test_create_docx_exam_paper():
    questions = [
        {"id": "Q001", "question": "Define machine learning algorithms?", "level": "L1", "co": "CO1", "marks": 5},
        {"id": "Q002", "question": "Explain how gradient descent optimizes parameters?", "level": "L2", "co": "CO2", "marks": 10},
    ]
    outcomes = {"CO1": "Understand ML concepts", "CO2": "Analyze optimization"}
    docx_bytes = create_docx_exam_paper(questions, course_name="Machine Learning", outcomes=outcomes)
    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 1000
    # Check standard docx zip signature (PK..)
    assert docx_bytes.startswith(b"PK\x03\x04")


def test_create_pdf_exam_paper():
    questions = [
        {"id": "Q001", "question": "Define machine learning algorithms?", "level": "L1", "co": "CO1", "marks": 5},
        {"id": "Q002", "question": "Explain how gradient descent optimizes parameters?", "level": "L2", "co": "CO2", "marks": 10},
    ]
    outcomes = {"CO1": "Understand ML concepts", "CO2": "Analyze optimization"}
    pdf_bytes = create_pdf_exam_paper(questions, course_name="Machine Learning", outcomes=outcomes)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    # Check standard PDF header (%PDF-)
    assert pdf_bytes.startswith(b"%PDF-")
