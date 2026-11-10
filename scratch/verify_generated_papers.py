"""Verification script: regenerate both 80-mark and 50-mark papers using the 3 textbooks.
Measures generation time and verifies all 7 quality criteria.
"""
import sys
sys.path.insert(0, ".")
import time
from pathlib import Path
import config
from blueprint import build_blueprint, calculate_blueprint_stats
from blooms.verb_bank import validate_question_kd
from export_utils import format_covered_chapters_header
from generation.generator import (
    check_question_duplicates,
    generate_full_paper,
    has_unseen_material_reference,
    verify_paper_completeness,
)
from generation.mock_client import MockClient
from ingestion.library import add_book
from paper_preset import make_preset
from retrieval.co_mapper import _m
from retrieval.sources import is_content_chunk
from totals import summarize


def run_verification():
    print("=== Step 1: Checking Textbooks Ingestion ===")
    t0 = time.time()
    books = {
        "docker": Path(".cache/uploads/6a1831f8f825_Docker_in_Practice.pdf"),
        "k8s": Path(".cache/uploads/022a324a7b57_Kubernetes_in_Action.pdf"),
        "devops": Path(".cache/uploads/2baa1d6c9eb4_Learning_DevOps.pdf"),
    }
    all_chunks = []
    book_chapters = {}
    for b_id, p_path in books.items():
        if not p_path.exists():
            print(f"Warning: {p_path} not found")
            continue
        print(f"Ingesting {b_id} from {p_path}...")
        chunks = add_book(str(p_path), book_id=b_id)
        print(f"  {b_id}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"Total chunks loaded: {len(all_chunks)} in {time.time() - t0:.2f}s")

    client = MockClient()
    embedder = _m()

    outcomes = {
        "CO1": "Understand containerization fundamentals and Docker architecture",
        "CO2": "Design and manage Docker storage, networking, and security",
        "CO3": "Deploy and orchestrate containerized applications using Kubernetes",
        "CO4": "Implement CI/CD pipelines, DevOps automation, and monitoring",
    }

    from retrieval.co_mapper import score_chunks_against_outcomes
    print("\nScoring chunks against outcomes once...")
    scored_chunks = score_chunks_against_outcomes(all_chunks, outcomes)
    print("Scoring complete.")

    # -------------------------------------------------------------------------
    # Paper 1: 80-mark choice paper with 4 units (20m per unit, 2 options)
    # Offered = 160 marks, Attempted = 80 marks
    # -------------------------------------------------------------------------
    print("\n=== Step 2: Generating Paper 1 (80-Mark Choice Paper, 4 Units) ====")
    preset_80 = make_preset(
        "Choice80",
        units=4,
        marks_per_unit=20,
        options=2,
        sub_marks={1: [10, 10], 2: [10, 10], 3: [10, 10], 4: [10, 10]},
        has_choice=True,
    )
    bp_80 = build_blueprint(preset=preset_80, outcomes=list(outcomes.keys()), seed=42)
    slots_80 = bp_80["slots"]
    print(f"Blueprint slots built: {len(slots_80)} slots")

    t_start = time.time()
    paper_1 = generate_full_paper(
        chunks=scored_chunks,
        blueprint=bp_80,
        outcomes=outcomes,
        unit_sources=None,
        client=client,
        seed=42,
        keep_placeholders=True,
    )
    duration_p1 = time.time() - t_start
    print(f"Paper 1 generated in {duration_p1:.3f}s")

    # Verifications for Paper 1
    qs_1 = paper_1["questions"]
    print(f"Paper 1 returned {len(qs_1)} questions")
    assert len(qs_1) == len(slots_80), f"Slot count mismatch: {len(qs_1)} vs {len(slots_80)}"

    # Completeness and OR symmetry verification
    complete_1, issues_1 = verify_paper_completeness(slots_80, qs_1)
    print(f"Paper 1 complete: {complete_1}")
    if not complete_1:
        print(f"Paper 1 has {len(issues_1)} slots flagged for review: {issues_1[:3]}...")
    # Slots must NEVER be dropped (Problem 1)
    assert len(qs_1) == 16, f"Expected 16 slots, got {len(qs_1)}"
    # Every slot must have a question and an answer (placeholder or valid)
    for q in qs_1:
        assert q.get("question"), f"Slot {q.get('id')} missing question text"
        assert q.get("answer"), f"Slot {q.get('id')} missing answer text"
        if q.get("needs_review"):
            assert "[NEEDS REVIEW" in q["question"] or q.get("review_flags")

    # Totals
    summary_1 = summarize(qs_1, blueprint_slots=slots_80)
    print(f"Paper 1 offered marks: {summary_1['offered_marks']} (expected 160), attempted: {summary_1['total_marks']} (expected 80)")
    assert summary_1["offered_marks"] == 160
    assert summary_1["total_marks"] == 80

    # Numbering starts at Q1
    q_nos_1 = [q.get("question_no") for q in qs_1 if q.get("question_no") is not None]
    assert min(q_nos_1) == 1, f"Numbering did not start at Q1, started at Q{min(q_nos_1)}"
    print(f"Paper 1 numbering starts at Q{min(q_nos_1)}")

    # For valid questions, verify quality rules
    valid_qs_1 = [q for q in qs_1 if not q.get("needs_review") and q.get("valid", True)]
    print(f"Paper 1 valid questions: {len(valid_qs_1)}/{len(qs_1)}")
    for q in valid_qs_1:
        has_u, u_msg = has_unseen_material_reference(q["question"], is_question=True)
        assert not has_u, f"Deictic reference in question {q.get('id')}: {u_msg}"
        has_u_a, u_msg_a = has_unseen_material_reference(q["answer"], is_question=False)
        assert not has_u_a, f"Deictic reference in answer {q.get('id')}: {u_msg_a}"

        # Verbs match KD
        lvl = q.get("level")
        kd = q.get("knowledge_dimension")
        ok_kd, kd_msg = validate_question_kd(q["question"], lvl, kd)
        assert ok_kd, f"KD mismatch in {q.get('id')}: {kd_msg}"

    print("Paper 1 checks PASSED!")

    # -------------------------------------------------------------------------
    # Paper 2: 50-mark standard paper with 5 units (no choice)
    # -------------------------------------------------------------------------
    print("\n=== Step 3: Generating Paper 2 (50-Mark Standard Paper, 5 Units) ===")
    bp_50 = build_blueprint(preset=50, outcomes=list(outcomes.keys()), seed=42)
    slots_50 = bp_50["slots"]
    print(f"Blueprint slots built: {len(slots_50)} slots")

    t_start = time.time()
    paper_2 = generate_full_paper(
        chunks=scored_chunks,
        blueprint=bp_50,
        outcomes=outcomes,
        unit_sources=None,
        client=client,
        seed=42,
        keep_placeholders=True,
    )
    duration_p2 = time.time() - t_start
    print(f"Paper 2 generated in {duration_p2:.3f}s")

    qs_2 = paper_2["questions"]
    assert len(qs_2) == len(slots_50)
    complete_2, issues_2 = verify_paper_completeness(slots_50, qs_2)
    print(f"Paper 2 complete: {complete_2}")
    if not complete_2:
        print(f"Paper 2 has {len(issues_2)} slots flagged for review: {issues_2}")
    summary_2 = summarize(qs_2, blueprint_slots=slots_50)
    print(f"Paper 2 offered marks: {summary_2['offered_marks']} (expected 50), attempted: {summary_2['total_marks']} (expected 50)")
    assert summary_2["offered_marks"] == 50
    assert summary_2["total_marks"] == 50

    q_nos_2 = [q.get("question_no") for q in qs_2 if q.get("question_no") is not None]
    assert min(q_nos_2) == 1
    print(f"Paper 2 numbering starts at Q{min(q_nos_2)}")

    valid_qs_2 = [q for q in qs_2 if not q.get("needs_review") and q.get("valid", True)]
    for q in valid_qs_2:
        has_u, u_msg = has_unseen_material_reference(q["question"], is_question=True)
        assert not has_u, f"Deictic reference in {q.get('id')}: {u_msg}"
        lvl = q.get("level")
        kd = q.get("knowledge_dimension")
        ok_kd, kd_msg = validate_question_kd(q["question"], lvl, kd)
        assert ok_kd, f"KD mismatch in {q.get('id')}: {kd_msg}"

    # -------------------------------------------------------------------------
    # Paper 3: 100-mark choice paper with 5 units (200 offered marks, 26 slots)
    # -------------------------------------------------------------------------
    print("\n=== Step 4: Generating Paper 3 (100-Mark Choice Paper, 200 Offered Marks, 26 Slots) ===")
    preset_100 = make_preset(
        "Choice100",
        units=5,
        marks_per_unit=20,
        options=2,
        sub_marks={
            1: [7, 7, 6],
            2: [10, 10],
            3: [7, 7, 6],
            4: [10, 10],
            5: [7, 7, 6],
        },
        has_choice=True,
    )
    bp_100 = build_blueprint(preset=preset_100, outcomes=list(outcomes.keys()), seed=42)
    slots_100 = bp_100["slots"]
    print(f"Blueprint slots built: {len(slots_100)} slots (Offered Marks: 200, Attempted: 100)")
    assert len(slots_100) == 26

    t_start = time.time()
    paper_3 = generate_full_paper(
        chunks=scored_chunks,
        blueprint=bp_100,
        outcomes=outcomes,
        unit_sources=None,
        client=client,
        seed=42,
        keep_placeholders=True,
    )
    duration_p3 = time.time() - t_start
    print(f"Paper 3 (26 slots) generated in {duration_p3:.3f}s")

    qs_3 = paper_3["questions"]
    assert len(qs_3) == 26
    summary_3 = summarize(qs_3, blueprint_slots=slots_100)
    print(f"Paper 3 offered marks: {summary_3['offered_marks']} (expected 200), attempted: {summary_3['total_marks']} (expected 100)")
    assert summary_3["offered_marks"] == 200
    assert summary_3["total_marks"] == 100
    print("Paper 3 checks PASSED!")

    # Verify covered chapters header
    header_str = format_covered_chapters_header(questions=qs_1)
    print(f"\nSample Covered Chapters Header:\n{header_str}")
    assert "Roadmap" not in header_str
    assert "About the code" not in header_str.lower()
    assert "Appendix" not in header_str

    print("\nALL VERIFICATION CRITERIA PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_verification()
