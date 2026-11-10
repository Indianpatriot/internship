"""Tests for DOCX and PDF question paper export."""
import io
import docx
import fitz
from export_utils import (
    create_docx_exam_paper,
    create_pdf_exam_paper,
    create_student_docx_paper,
    create_student_pdf_paper,
    create_faculty_docx_paper,
    create_faculty_pdf_paper,
)
from blueprint import get_50_mark_blueprint, build_blueprint, calculate_blueprint_stats
from totals import summarize


def test_create_docx_exam_paper():
    questions = [
        {"id": "1", "unit": 1, "question": "Define machine learning algorithms?", "level": "L1", "co": "CO1", "marks": 2},
        {"id": "2", "unit": 1, "question": "Explain gradient descent optimization?", "level": "L2", "co": "CO2", "marks": 2},
        {"id": "6(a)", "unit": 2, "question": "Demonstrate the backpropagation algorithm with an example?", "level": "L3", "co": "CO1", "marks": 5},
        {"id": "6(b)", "unit": 2, "question": "Analyze overfitting mitigation techniques?", "level": "L4", "co": "CO2", "marks": 5},
    ]
    outcomes = {"CO1": "Understand ML concepts", "CO2": "Analyze optimization"}
    docx_bytes = create_docx_exam_paper(questions, course_name="Machine Learning", outcomes=outcomes)
    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 1000
    # Check standard docx zip signature (PK..)
    assert docx_bytes.startswith(b"PK\x03\x04")


def test_create_pdf_exam_paper():
    questions = [
        {"id": "1", "unit": 1, "question": "Define machine learning algorithms?", "level": "L1", "co": "CO1", "marks": 2},
        {"id": "2", "unit": 1, "question": "Explain gradient descent optimization?", "level": "L2", "co": "CO2", "marks": 2},
        {"id": "6(a)", "unit": 2, "question": "Demonstrate the backpropagation algorithm with an example?", "level": "L3", "co": "CO1", "marks": 5},
        {"id": "6(b)", "unit": 2, "question": "Analyze overfitting mitigation techniques?", "level": "L4", "co": "CO2", "marks": 5},
    ]
    outcomes = {"CO1": "Understand ML concepts", "CO2": "Analyze optimization"}
    pdf_bytes = create_pdf_exam_paper(questions, course_name="Machine Learning", outcomes=outcomes)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    # Check standard PDF header (%PDF-)
    assert pdf_bytes.startswith(b"%PDF-")


def test_50_mark_blueprint_structure():
    slots = get_50_mark_blueprint(["CO1", "CO2", "CO3", "CO4"])
    stats = calculate_blueprint_stats(slots)
    assert stats["total_marks"] == 50
    assert stats["is_standard_50"] is True
    assert stats["unit_marks"][1] == 10
    assert stats["unit_marks"][2] == 10
    assert stats["unit_marks"][5] == 10
    assert len(slots) == 10


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
        {"id": "1", "unit": 1, "question": "Define machine learning?", "level": "L1", "co": "CO1", "marks": 2},
        {"id": "2", "unit": 1, "question": "Explain gradient descent?", "level": "L2", "co": "CO2", "marks": 2},
        {"id": "3", "unit": 2, "question": "Implement backpropagation?", "level": "L3", "co": "CO1", "marks": 5},
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


def test_no_section_or_part_in_100_mark_export():
    """Verify 100-mark generated paper contains UNIT headers and NO Part/Section text."""
    from paper_preset import make_preset

    preset = make_preset(
        "100 marks",
        units=5,
        marks_per_unit=20,
        options=2,
        sub_marks={1: [10, 10], 2: [10, 10], 3: [10, 10], 4: [10, 10], 5: [10, 10]},
    )
    bp = build_blueprint(preset=preset, seed=42)
    slots = bp["slots"]
    questions = []
    for s in slots:
        q_copy = dict(s)
        q_copy["question"] = f"Explain the fundamental mechanism for unit {s['unit']}?"
        questions.append(q_copy)

    docx_bytes = create_docx_exam_paper(questions, course_name="Computer Science")
    doc = docx.Document(io.BytesIO(docx_bytes))
    docx_text = "\n".join(
        [p.text for p in doc.paragraphs]
        + [c.text for t in doc.tables for row in t.rows for c in row.cells]
    )

    # Must NOT contain Section A/B or Part A/B
    assert "part a" not in docx_text.lower()
    assert "part b" not in docx_text.lower()
    assert "section a" not in docx_text.lower()
    assert "section b" not in docx_text.lower()
    assert "UNIT 1" in docx_text
    assert "UNIT 2" in docx_text
    assert "UNIT 5" in docx_text
    assert "— OR —" in docx_text

    pdf_bytes = create_pdf_exam_paper(questions, course_name="Computer Science")
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pdf_text = "\n".join([page.get_text() for page in pdf_doc])

    assert "part a" not in pdf_text.lower()
    assert "part b" not in pdf_text.lower()
    assert "section a" not in pdf_text.lower()
    assert "section b" not in pdf_text.lower()
    assert "UNIT 1" in pdf_text
    assert "UNIT 2" in pdf_text
    assert "UNIT 5" in pdf_text
    assert "— OR —" in pdf_text


def test_export_pdf_preserves_code_bracketed_tokens_and_newlines():
    """PART B Regression: Verify ReportLab PDF preserves <container-id> and code block newlines."""
    dockerfile_code = "FROM alpine:3.18\nRUN apk add --no-cache curl\nCMD [\"curl\", \"http://localhost\"]"
    questions = [
        {
            "id": "1(a)",
            "unit": 1,
            "question_no": 1,
            "sub_index": 1,
            "alternative_group": "U1",
            "question": "Run `docker inspect <container-id>` with `<placeholder-token>` to extract container configuration?",
            "marks": 5,
            "level": "L3",
            "knowledge_dimension": "Procedural",
            "co": "CO1",
            "answer": (
                "Point 1 (2M): Execute `docker inspect <container-id>` to inspect low-level container metadata.\n"
                "Point 2 (3M): Here is the corresponding Dockerfile specification:\n"
                f"```dockerfile\n{dockerfile_code}\n```"
            ),
        }
    ]

    # 1. PDF Export verification with PyMuPDF (fitz)
    pdf_bytes = create_pdf_exam_paper(
        questions,
        course_name="Container Technologies",
        include_answers=True,
    )
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pdf_text = "\n".join([page.get_text() for page in pdf_doc])

    # Assert <container-id> and <placeholder-token> survived intact without XML parsing error or loss
    assert "<container-id>" in pdf_text
    assert "<placeholder-token>" in pdf_text

    # Assert Dockerfile newlines survived intact
    assert "FROM alpine:3.18" in pdf_text
    assert "RUN apk add --no-cache curl" in pdf_text
    assert "CMD [\"curl\", \"http://localhost\"]" in pdf_text
    # Verify newlines between lines in the code block
    assert "FROM alpine:3.18\nRUN apk add" in pdf_text or "RUN apk add --no-cache curl\nCMD" in pdf_text

    # 2. DOCX Export verification
    docx_bytes = create_docx_exam_paper(
        questions,
        course_name="Container Technologies",
        include_answers=True,
    )
    doc = docx.Document(io.BytesIO(docx_bytes))
    docx_text = "\n".join(
        [p.text for p in doc.paragraphs]
        + [c.text for t in doc.tables for row in t.rows for c in row.cells]
    )
    assert "<container-id>" in docx_text
    assert "<placeholder-token>" in docx_text
    assert "FROM alpine:3.18\nRUN apk add" in docx_text or ("FROM alpine:3.18" in docx_text and "RUN apk add" in docx_text)


def test_student_and_faculty_paper_profiles():
    """PART C: Student copy must omit sources, chapters, and answers, while Faculty copy includes them with QA flags and KD totals."""
    questions = [
        {
            "id": "1(a)",
            "unit": 1,
            "question_no": 1,
            "sub_index": 1,
            "question": "Demonstrate Kubernetes service discovery mechanisms?",
            "marks": 10,
            "level": "L3",
            "knowledge_dimension": "Procedural",
            "co": "CO2",
            "chapter": "Chapter 4: Services and Networking",
            "page_start": 123,
            "display_label": "Kubernetes in Action",
            "answer": "Kubernetes provisions cluster-internal DNS and environment variables.",
            "review_flags": ["DUPLICATE", "CO_MISMATCH"],
        }
    ]
    outcomes = {"CO2": "Configure networking and services"}
    summary = summarize(questions, attempted_only=True)

    # 1. Student PDF
    s_pdf = create_student_pdf_paper(
        questions,
        course_name="1_Docker_in_Practice_&_2_Kubernetes_in_Action_Exam_Paper",
        subject_code="CS-601",
        outcomes=outcomes,
        totals_summary=summary,
        covered_chapters=["Chapter 4: Services and Networking"],
    )
    s_doc = fitz.open(stream=s_pdf, filetype="pdf")
    s_text = "\n".join([page.get_text() for page in s_doc])

    assert "USN:" in s_text
    assert "SEMESTER END EXAMINATION (SEE)" in s_text
    assert "Docker in Practice & Kubernetes in Action" in s_text
    assert "1_Docker_in_Practice" not in s_text
    # Student paper must NOT contain internal details
    assert "Chapter 4: Services and Networking" not in s_text
    assert "p.123" not in s_text
    assert "SCHEME OF EVALUATION" not in s_text
    assert "Kubernetes provisions cluster-internal DNS" not in s_text
    assert "CUMULATIVE" not in s_text
    assert "QA Flags" not in s_text

    # 2. Student DOCX
    s_docx = create_student_docx_paper(
        questions,
        course_name="1_Docker_in_Practice_&_2_Kubernetes_in_Action_Exam_Paper",
        subject_code="CS-601",
        outcomes=outcomes,
        totals_summary=summary,
        covered_chapters=["Chapter 4: Services and Networking"],
    )
    doc_s = docx.Document(io.BytesIO(s_docx))
    s_docx_text = "\n".join(
        [p.text for p in doc_s.paragraphs]
        + [c.text for t in doc_s.tables for row in t.rows for c in row.cells]
    )
    assert "USN:" in s_docx_text
    assert "Docker in Practice & Kubernetes in Action" in s_docx_text
    assert "Chapter 4: Services and Networking" not in s_docx_text
    assert "p.123" not in s_docx_text
    assert "SCHEME OF EVALUATION" not in s_docx_text
    assert "Kubernetes provisions cluster-internal DNS" not in s_docx_text
    assert "QA Flags" not in s_docx_text

    # 3. Faculty PDF
    f_pdf = create_faculty_pdf_paper(
        questions,
        course_name="1_Docker_in_Practice_&_2_Kubernetes_in_Action_Exam_Paper",
        subject_code="CS-601",
        outcomes=outcomes,
        totals_summary=summary,
        covered_chapters=["Chapter 4: Services and Networking"],
    )
    f_doc = fitz.open(stream=f_pdf, filetype="pdf")
    f_text = "\n".join([page.get_text() for page in f_doc])

    assert "Syllabus / Covered Chapters" in f_text
    assert "Source" in f_text
    assert "QA Flags" in f_text
    assert "DUPLICATE" in f_text
    assert "CO_MISMATCH" in f_text
    assert "SCHEME OF EVALUATION & MODEL ANSWERS" in f_text
    assert "Kubernetes provisions cluster-internal DNS" in f_text
    assert "Knowledge Dimension (KD) Distribution" in f_text

    # 4. Faculty DOCX
    f_docx = create_faculty_docx_paper(
        questions,
        course_name="1_Docker_in_Practice_&_2_Kubernetes_in_Action_Exam_Paper",
        subject_code="CS-601",
        outcomes=outcomes,
        totals_summary=summary,
        covered_chapters=["Chapter 4: Services and Networking"],
    )
    doc_f = docx.Document(io.BytesIO(f_docx))
    f_docx_text = "\n".join(
        [p.text for p in doc_f.paragraphs]
        + [c.text for t in doc_f.tables for row in t.rows for c in row.cells]
    )
    assert "Syllabus / Covered Chapters" in f_docx_text
    assert "QA Flags" in f_docx_text
    assert "DUPLICATE" in f_docx_text
    assert "CO_MISMATCH" in f_docx_text
    assert "SCHEME OF EVALUATION & MODEL ANSWERS" in f_docx_text
    assert "Kubernetes provisions cluster-internal DNS" in f_docx_text
    assert "Knowledge Dimension (KD) Distribution" in f_docx_text



