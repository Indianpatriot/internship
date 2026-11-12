"""Benchmark pipeline stages on data/samples/sample_book.pdf."""
import os
import sys
from perf import timed, get_timings, format_timings_table

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.pipeline import ingest
from retrieval.co_mapper import score_chunks_against_outcomes, _m
from blueprint import build_blueprint
from generation.mock_client import MockClient
from generation.generator import generate_paper_by_slots
from export_utils import create_docx_exam_paper, create_pdf_exam_paper

SAMPLE_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples", "sample_book.pdf")


def run_benchmark():
    if not os.path.exists(SAMPLE_PDF):
        print(f"Sample PDF not found at {SAMPLE_PDF}")
        return

    print("=== Running Baseline Performance Benchmark ===")
    
    # 1. Ingestion
    with timed("PDF text extraction & chunking"):
        chunks = ingest(SAMPLE_PDF)
    print(f"Ingested {len(chunks)} chunks.")

    # 2. Embedding & Outcomes scoring
    outcomes = {
        "CO1": "Understand containerization principles and operating system virtualization.",
        "CO2": "Analyze container image layering and build reproducible environments.",
        "CO3": "Evaluate container networking models and service discovery.",
    }
    with timed("Chunk embedding & outcome scoring"):
        scored_chunks = score_chunks_against_outcomes(chunks, outcomes)

    # 3. Blueprint generation
    with timed("Blueprint allocation"):
        bp = build_blueprint(preset=50, sub_question_counts=2, seed=42)
        slots = bp["slots"]
    print(f"Generated {len(slots)} slots.")

    # 4. Question & answer & scheme generation (with MockClient for reproducible baseline)
    client = MockClient()
    with timed("Generation (questions, answers, schemes)"):
        res = generate_paper_by_slots(
            chunks=scored_chunks,
            slots=slots,
            client=client,
            seed=42,
        )
    questions = res["questions"]
    print(f"Generated {len(questions)} questions.")

    # 5. Export
    with timed("DOCX export"):
        docx_bytes = create_docx_exam_paper(questions, outcomes=outcomes)
    with timed("PDF export"):
        pdf_bytes = create_pdf_exam_paper(questions, outcomes=outcomes)
    print(f"DOCX size: {len(docx_bytes)} bytes, PDF size: {len(pdf_bytes)} bytes.")

    print("\n=== Baseline Timings Table ===")
    table = format_timings_table()
    total_time = sum(get_timings().values())
    print(f"{'Stage':<45} | {'Duration (s)':<12} | {'Share (%)':<10}")
    print("-" * 73)
    for row in table:
        print(f"{row['Stage']:<45} | {row['Duration (s)']:<12.3f} | {row['Share (%)']:<10.1f}")
    print("-" * 73)
    print(f"{'Total':<45} | {total_time:<12.3f} | 100.0%\n")


if __name__ == "__main__":
    run_benchmark()
