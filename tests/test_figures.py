"""Tests for textbook figure extraction, retrieval, and marking scheme integration."""
import os
import fitz  # PyMuPDF
import numpy as np
import pytest

from ingestion.figures import extract_figures, link_figures_to_chunks
from retrieval.figure_search import (
    attach_figures_to_scheme,
    find_figures_for_question,
    question_needs_diagram,
)
from export_utils import create_docx_exam_paper, create_pdf_exam_paper
from retrieval.co_mapper import _m


def _create_test_pdf_with_figures(pdf_path: str) -> None:
    """Generate a realistic textbook PDF containing:

    - Page 1: Raster image with caption "Figure 1.1 Docker architecture" and text
    - Page 2: Vector drawing (boxes and arrows) with caption "Figure 2.3 Dockerfile layers"
    - Page 3: Text-only page without figures
    - Repeated small header logo on all pages (should be filtered out)
    """
    doc = fitz.open()

    # Small 30x30 logo pixmap
    logo_pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 30, 30), 0)
    logo_pix.clear_with(0x80)

    # Large 200x120 raster image pixmap
    raster_pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 120), 0)
    raster_pix.clear_with(0x30)

    # PAGE 1: Embedded raster image + Caption + Surrounding text
    p1 = doc.new_page(width=595, height=842)
    p1.insert_image(fitz.Rect(50, 20, 80, 50), pixmap=logo_pix)  # Header logo
    p1.insert_text((50, 100), "Chapter 1: Container Architecture\nDocker containerization architecture decouples applications from infrastructure.", fontsize=11)
    p1.insert_image(fitz.Rect(100, 160, 360, 320), pixmap=raster_pix)
    p1.insert_text((100, 340), "Figure 1.1 Docker architecture", fontsize=10)
    p1.insert_text((50, 380), "The client communicates with the Docker daemon via REST API over UNIX sockets or network interfaces.", fontsize=10)

    # PAGE 2: Vector diagram (drawing operations) + Caption + Labels
    p2 = doc.new_page(width=595, height=842)
    p2.insert_image(fitz.Rect(50, 20, 80, 50), pixmap=logo_pix)  # Header logo
    p2.insert_text((50, 100), "Chapter 2: Docker Images and Layering\nImages are composed of read-only layers generated from instructions.", fontsize=11)

    # Draw boxes & arrows representing layers
    shape = p2.new_shape()
    shape.draw_rect(fitz.Rect(120, 160, 380, 200))
    shape.draw_rect(fitz.Rect(120, 220, 380, 260))
    shape.draw_rect(fitz.Rect(120, 280, 380, 320))
    shape.draw_line(fitz.Point(250, 200), fitz.Point(250, 220))
    shape.draw_line(fitz.Point(250, 260), fitz.Point(250, 280))
    shape.finish(fill=(0.9, 0.9, 0.95), color=(0.1, 0.2, 0.5), width=1.5)
    shape.commit()

    p2.insert_text((150, 185), "Container Layer (Read-Write)", fontsize=9)
    p2.insert_text((150, 245), "Application Dependencies (Read-Only)", fontsize=9)
    p2.insert_text((150, 305), "Base Operating System (Read-Only)", fontsize=9)
    p2.insert_text((120, 345), "Figure 2.3 Dockerfile layers", fontsize=10)

    # PAGE 3: Text-only page
    p3 = doc.new_page(width=595, height=842)
    p3.insert_image(fitz.Rect(50, 20, 80, 50), pixmap=logo_pix)  # Header logo
    p3.insert_text((50, 120), "Chapter 3: Production Deployment\nContinuous integration pipelines automate the testing and packaging of microservices without manual deployment intervention.", fontsize=11)

    doc.save(pdf_path)
    doc.close()


def test_extract_figures_filters_junk_and_parses_captions(tmp_path):
    pdf_path = str(tmp_path / "docker_guide.pdf")
    _create_test_pdf_with_figures(pdf_path)

    figures = extract_figures(pdf_path, book_id="docker_guide")

    # Assert raster and vector figures were extracted, and page 3 has none
    assert len(figures) == 2, f"Expected exactly 2 figures, got {len(figures)}"

    kinds = {f["kind"] for f in figures}
    assert "raster" in kinds
    assert "vector" in kinds

    # Verify captions and labels parsed
    captions = [f["caption"] for f in figures]
    assert any("Figure 1.1" in c for c in captions)
    assert any("Figure 2.3" in c for c in captions)

    # Verify PNG files exist on disk and open
    for fig in figures:
        img_p = fig["image_path"]
        assert os.path.exists(img_p)
        assert os.path.getsize(img_p) > 500
        pix = fitz.Pixmap(img_p)
        assert pix.width > 0 and pix.height > 0
        pix = None


def test_extract_figures_caching(tmp_path):
    pdf_path = str(tmp_path / "docker_guide.pdf")
    _create_test_pdf_with_figures(pdf_path)

    # First extraction
    figs_1 = extract_figures(pdf_path, book_id="docker_guide")
    mtime_1 = os.path.getmtime(figs_1[0]["image_path"])

    # Second extraction on identical file should hit cache without overwriting
    figs_2 = extract_figures(pdf_path, book_id="docker_guide")
    mtime_2 = os.path.getmtime(figs_2[0]["image_path"])

    assert len(figs_1) == len(figs_2)
    assert mtime_1 == mtime_2


def test_link_figures_to_chunks(tmp_path):
    pdf_path = str(tmp_path / "docker_guide.pdf")
    _create_test_pdf_with_figures(pdf_path)
    figures = extract_figures(pdf_path, book_id="docker_guide")

    chunks = [
        {"chunk_id": "c1", "page_start": 1, "page_end": 1, "chapter": "Chapter 1", "section": "1.1"},
        {"chunk_id": "c2", "page_start": 2, "page_end": 2, "chapter": "Chapter 2", "section": "2.1"},
    ]
    link_figures_to_chunks(figures, chunks)

    assert "figure_ids" in chunks[0]
    assert len(chunks[0]["figure_ids"]) == 1
    assert figures[0]["chapter"] == "Chapter 1"


def test_find_figures_for_diagram_question(tmp_path):
    pdf_path = str(tmp_path / "docker_guide.pdf")
    _create_test_pdf_with_figures(pdf_path)
    figures = extract_figures(pdf_path, book_id="docker_guide")
    embedder = _m()

    # 1. Diagram question picks relevant figure
    q_diag = {
        "question": "Explain the architecture of Docker with a diagram?",
        "book_id": "docker_guide",
        "page": 1,
        "chapter": "Chapter 1",
    }
    chunk = {"book_id": "docker_guide", "page_start": 1, "page_end": 1, "chapter": "Chapter 1", "book_title": "Docker in Action"}
    scheme = {
        "criteria": [
            {"criterion": "Architecture Diagram", "expected": "Docker client, daemon, containers", "kind": "diagram", "marks": 3}
        ]
    }
    matched = find_figures_for_question(q_diag, chunk, scheme, figures, embedder)
    assert len(matched) >= 1
    assert "Figure 1.1" in matched[0]["caption"] or "Docker architecture" in matched[0]["caption"]

    # 2. Non-diagram question returns empty list
    q_nodraw = {
        "question": "List the benefits of continuous integration in software engineering?",
        "book_id": "docker_guide",
        "page": 3,
    }
    assert find_figures_for_question(q_nodraw, chunk, None, figures, embedder) == []

    # 3. Question on an unrelated topic below threshold returns empty list
    q_unrelated = {
        "question": "Illustrate photosynthesis light-dependent reactions with a chloroplast diagram?",
        "book_id": "docker_guide",
        "page": 1,
    }
    unrelated_scheme = {
        "criteria": [
            {"criterion": "Chloroplast diagram", "expected": "Thylakoid membrane and stroma", "kind": "diagram", "marks": 3}
        ]
    }
    assert find_figures_for_question(q_unrelated, chunk, unrelated_scheme, figures, embedder, min_score=0.60) == []


def test_scheme_figure_attachment_and_no_figure_available():
    scheme = {
        "total_marks": 5,
        "criteria": [
            {"no": 1, "criterion": "Concept definition", "expected": "Virtualization definition", "marks": 2, "kind": "definition"},
            {"no": 2, "criterion": "Architecture diagram", "expected": "Daemon and client components", "marks": 3, "kind": "diagram"},
        ],
    }
    mock_figure = {
        "figure_id": "fig_1",
        "image_path": "fake/path/p1_f1.png",
        "caption": "Figure 1.1 Architecture",
        "book_id": "b1",
        "book_title": "Textbook",
        "page": 1,
    }
    attach_figures_to_scheme(scheme, [mock_figure])
    assert "figures" in scheme["criteria"][1]
    assert len(scheme["criteria"][1]["figures"]) == 1


def test_export_with_and_without_figures_smoke(tmp_path):
    pdf_path = str(tmp_path / "docker_guide.pdf")
    _create_test_pdf_with_figures(pdf_path)
    figures = extract_figures(pdf_path, book_id="docker_guide")

    sample_img_path = figures[0]["image_path"]

    questions = [
        {
            "id": "1",
            "question": "Explain Docker architecture with a diagram?",
            "marks": 5,
            "level": "L2",
            "co": "CO1",
            "answer": "Docker uses a client-server architecture.",
            "marking_scheme": {
                "total_marks": 5,
                "criteria": [
                    {
                        "no": 1,
                        "criterion": "Architecture Diagram",
                        "expected": "Daemon, client, container runtime",
                        "marks": 5,
                        "kind": "diagram",
                        "figures": [
                            {
                                "figure_id": "fig_1",
                                "image_path": sample_img_path,
                                "caption": "Figure 1.1 Docker architecture",
                                "book_id": "b1",
                                "book_title": "Docker in Action",
                                "page": 1,
                            }
                        ],
                    }
                ],
            },
        }
    ]

    # Export with figures
    docx_bytes = create_docx_exam_paper(questions, include_figures=True)
    pdf_bytes = create_pdf_exam_paper(questions, include_figures=True)
    assert len(docx_bytes) > 1000
    assert len(pdf_bytes) > 1000

    # Export without figures
    docx_no_fig = create_docx_exam_paper(questions, include_figures=False)
    pdf_no_fig = create_pdf_exam_paper(questions, include_figures=False)
    assert len(docx_no_fig) > 1000
    assert len(pdf_no_fig) > 1000
