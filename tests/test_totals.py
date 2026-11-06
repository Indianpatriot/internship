"""Tests for Cumulative Totals, Summary Table Rows, and Integration Pipeline (Task 3)."""
import fitz
from blueprint import build_blueprint
from generation.generator import generate_full_paper
from generation.mock_client import MockClient
from ingestion.library import load_library, suggest_unit_sources
from scheme import parse_scheme_text
from totals import summarize, totals_table_rows


def test_hand_computed_summary_for_small_paper():
    """Verify hand-computed marks, percentages, question counts, and band aggregations."""
    questions = [
        {"id": "1", "question_no": 1, "marks": 5, "co": "CO1", "level": "L1", "alternative_group": None},
        {"id": "2", "question_no": 2, "marks": 5, "co": "CO1", "level": "L2", "alternative_group": None},
        {"id": "3", "question_no": 3, "marks": 10, "co": "CO2", "level": "L4", "alternative_group": None},
    ]
    # Total marks = 20
    # CO1 = 10 marks (50.0%), CO2 = 10 marks (50.0%)
    # L1-L2 = 10 marks (50.0%), L3-L4 = 10 marks (50.0%), L5-L6 = 0 marks (0.0%)

    targets = {
        "bt": {"L1-L2": 0.40, "L3-L4": 0.40, "L5-L6": 0.20},
        "co": {"CO1": 0.50, "CO2": 0.50},
    }

    res = summarize(questions, targets=targets, attempted_only=True)

    assert res["total_marks"] == 20
    assert res["offered_marks"] == 20

    # by_co checks
    assert res["by_co"]["CO1"]["marks"] == 10
    assert res["by_co"]["CO1"]["pct"] == 50.0
    assert res["by_co"]["CO1"]["questions"] == 2
    assert res["by_co"]["CO2"]["marks"] == 10
    assert res["by_co"]["CO2"]["pct"] == 50.0

    # by_bt checks
    assert res["by_bt"]["L1"]["marks"] == 5
    assert res["by_bt"]["L2"]["marks"] == 5
    assert res["by_bt"]["L4"]["marks"] == 10
    assert res["by_bt"]["L5"]["marks"] == 0

    # by_bt_band checks
    assert res["by_bt_band"]["L1-L2"]["marks"] == 10
    assert res["by_bt_band"]["L1-L2"]["pct"] == 50.0
    assert res["by_bt_band"]["L3-L4"]["marks"] == 10
    assert res["by_bt_band"]["L3-L4"]["pct"] == 50.0
    assert res["by_bt_band"]["L5-L6"]["marks"] == 0
    assert res["by_bt_band"]["L5-L6"]["pct"] == 0.0

    # targets_diff: achieved - target in percentage points
    # L1-L2: 50.0% - 40.0% = +10.0
    assert res["targets_diff"]["bt_band"]["L1-L2"] == 10.0
    # L5-L6: 0.0% - 20.0% = -20.0
    assert res["targets_diff"]["bt_band"]["L5-L6"] == -20.0
    # CO1: 50.0% - 50.0% = 0.0
    assert res["targets_diff"]["co"]["CO1"] == 0.0


def test_100_mark_alternative_mirroring_counts_one_per_group():
    """In 100-mark papers with alternative groups, attempted_only counts one question per group."""
    # 5 units x 2 questions = 10 questions of 20 marks each (offered = 200, attempted = 100)
    questions = []
    for u in range(1, 6):
        # Primary question
        questions.append({
            "id": f"{2*u - 1}",
            "question_no": 2 * u - 1,
            "unit": u,
            "marks": 20,
            "co": f"CO{u}",
            "level": "L2",
            "alternative_group": f"U{u}",
        })
        # Alternative choice question
        questions.append({
            "id": f"{2*u}",
            "question_no": 2 * u,
            "unit": u,
            "marks": 20,
            "co": f"CO{u}",
            "level": "L2",
            "alternative_group": f"U{u}",
        })

    # attempted_only = True -> counts 5 questions = 100 marks
    res_attempted = summarize(questions, attempted_only=True)
    assert res_attempted["total_marks"] == 100
    assert res_attempted["offered_marks"] == 200

    # attempted_only = False -> counts all 10 questions = 200 marks
    res_all = summarize(questions, attempted_only=False)
    assert res_all["total_marks"] == 200
    assert res_all["offered_marks"] == 200


def test_empty_paper_does_not_crash():
    """Summarizing an empty list of questions safely returns zero totals without division-by-zero."""
    res = summarize([])
    assert res["total_marks"] == 0
    assert res["offered_marks"] == 0
    assert all(b["pct"] == 0.0 for b in res["by_bt"].values())
    assert all(b["pct"] == 0.0 for b in res["by_bt_band"].values())


def test_totals_table_rows_shape():
    """totals_table_rows produces structured list-of-lists for CO and BT tables with totals."""
    questions = [
        {"id": "1", "question_no": 1, "marks": 10, "co": "CO1", "level": "L1", "alternative_group": None},
        {"id": "2", "question_no": 2, "marks": 10, "co": "CO2", "level": "L3", "alternative_group": None},
    ]
    summary = summarize(questions)
    tables = totals_table_rows(summary)

    assert "co_table" in tables and "bt_table" in tables

    co_table = tables["co_table"]
    # Header + 2 CO rows + Total row
    assert len(co_table) == 4
    assert co_table[0] == ["Course Outcome", "Marks", "Percentage", "Questions Count"]
    assert co_table[-1][0] == "Total"
    assert co_table[-1][1] == 20

    bt_table = tables["bt_table"]
    assert len(bt_table) >= 10
    assert bt_table[0] == ["Bloom's Taxonomy Band / Level", "Marks", "Percentage", "Questions Count"]
    assert bt_table[-1][0] == "Total"
    assert bt_table[-1][1] == 20


def test_end_to_end_library_scheme_generation_totals(tmp_path):
    """Integration test: library of 2 books + scheme -> suggest sources -> generate_full_paper -> summarize."""
    # 1. Create two fake textbook PDFs with sufficient content
    pdf_a = tmp_path / "book_a.pdf"
    pdf_b = tmp_path / "book_b.pdf"

    for p, title in [(pdf_a, "Algorithms"), (pdf_b, "Databases")]:
        doc = fitz.open()
        for ch in [1, 2, 3]:
            page = doc.new_page()
            page.insert_text((50, 72), f"Chapter {ch}: Fundamentals of {title}", fontsize=16)
            page.insert_text(
                (50, 120),
                (
                    f"In this chapter {ch} of {title}, students explore algorithmic paradigms, structures, and systems. "
                    "Invariants are analyzed across recursive divisions and state transitions systematically. "
                    "The operational complexity and formal mathematical models guarantee correctness. "
                    "Practical implementations provide scalable, fault-tolerant solutions for computational problems."
                ),
                fontsize=11,
            )
        doc.save(str(p))
        doc.close()

    lib = load_library([pdf_a, pdf_b], cache_dir=tmp_path / "cache")
    assert len(lib["books"]) == 2

    # 2. Parse scheme text
    scheme_text = """
    Course: Computer Science Engineering
    Course Outcomes:
    CO1: Understand algorithms and complexities.
    CO2: Apply database normalization and transactions.
    
    Unit 1: Algorithmic Foundations (8 Hours)
    Divide and conquer, recurrence relations, complexity.
    
    Unit 2: Database Systems (8 Hours)
    Relational algebra, SQL, normalization, transactions.
    """
    scheme = parse_scheme_text(scheme_text, library=lib)
    assert len(scheme["units"]) == 2

    # 3. Suggest unit sources
    suggested_sources = suggest_unit_sources(lib, units=[1, 2])
    assert 1 in suggested_sources and 2 in suggested_sources

    # 4. Build blueprint for 50 marks across 2 units
    bp = build_blueprint(preset=50, units=2, questions_per_unit=1, seed=42)

    # 5. Generate full paper using Nitin's generator
    client = MockClient()
    paper = generate_full_paper(
        chunks=lib["chunks"],
        blueprint=bp,
        outcomes=scheme["outcomes"],
        unit_sources=suggested_sources,
        client=client,
        seed=42,
    )

    questions = paper["questions"]
    assert len(questions) > 0

    # 6. Summarize results
    summary = summarize(questions, attempted_only=True)
    expected_sum = sum(q["marks"] for q in questions)
    assert summary["total_marks"] == expected_sum
    assert summary["offered_marks"] == expected_sum


def test_marks_per_co_and_level_with_choice_and_targets_diff():
    qs = [
        {"id": "1", "question_no": 1, "marks": 10, "co": "CO1", "level": "L2", "alternative_group": "U1"},
        {"id": "2", "question_no": 2, "marks": 10, "co": "CO2", "level": "L5", "alternative_group": "U1"},
        {"id": "3", "question_no": 3, "marks": 20, "co": "CO2", "level": "L3", "alternative_group": None},
    ]
    s = summarize(qs, targets={"bt": {"L1-L2": 0.2, "L3-L4": 0.6, "L5-L6": 0.2}, "co": {"CO1": 0.5, "CO2": 0.5}})
    assert (s["total_marks"], s["offered_marks"]) == (30, 40)  # one of the pair counted
    assert s["by_co"]["CO1"]["marks"] == 10 and s["by_co"]["CO2"]["marks"] == 20
    assert s["by_bt"]["L2"]["marks"] == 10 and s["by_bt"]["L3"]["marks"] == 20
    assert s["by_bt_band"]["L1-L2"]["pct"] == 33.33
    assert s["targets_diff"]["co"] == {"CO1": -16.67, "CO2": 16.67}
    assert s["targets_diff"]["bt_band"]["L3-L4"] == 6.67
    import json
    json.dumps(s)  # plain data only: usable by the screen and the export
    s_all = summarize(qs, attempted_only=False)
    assert s_all["total_marks"] == 40
