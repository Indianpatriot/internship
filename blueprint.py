"""Blueprint module for defining exam structures (e.g., 50-mark standard paper with subparts).

Implements Step 1 of Paper Assembly:
- Part A: Short answer (e.g. 5 questions x 2 marks = 10 marks)
- Part B: Long/Descriptive questions with a/b subparts (e.g. 4 questions x 10 marks = 40 marks)
- Dynamic Course Outcome (CO) mapping
- Bloom's Taxonomy (BT) distribution and spread calculation
"""
from typing import Any


def get_50_mark_blueprint(cos: list[str] = None) -> list[dict[str, Any]]:
    """Return standard 50-mark exam paper blueprint.

    Structure:
    - Part A: 5 questions x 2 marks = 10 marks (L1, L2 foundation)
    - Part B: 4 main questions with (a) and (b) subparts = 40 marks (L2-L5 higher order)
    Total = 50 marks.
    """
    if not cos:
        cos = ["CO1", "CO2"]

    num_cos = len(cos)

    def co_at(index: int) -> str:
        return cos[index % num_cos]

    slots = [
        # --- PART A: Short Answer (10 Marks) ---
        {
            "part": "Part A",
            "id": "1",
            "co": co_at(0),
            "level": "L1",
            "marks": 2,
            "desc": "Short Answer (Foundation)",
        },
        {
            "part": "Part A",
            "id": "2",
            "co": co_at(1),
            "level": "L1",
            "marks": 2,
            "desc": "Short Answer (Definition / Recall)",
        },
        {
            "part": "Part A",
            "id": "3",
            "co": co_at(2 if num_cos > 2 else 0),
            "level": "L2",
            "marks": 2,
            "desc": "Short Answer (Explanation / Principle)",
        },
        {
            "part": "Part A",
            "id": "4",
            "co": co_at(3 if num_cos > 3 else 1),
            "level": "L2",
            "marks": 2,
            "desc": "Short Answer (Concept Distinction)",
        },
        {
            "part": "Part A",
            "id": "5",
            "co": co_at(4 if num_cos > 4 else 0),
            "level": "L2",
            "marks": 2,
            "desc": "Short Answer (Application Summary)",
        },
        # --- PART B: Long Answer with subparts (a) & (b) (40 Marks) ---
        # Question 6
        {
            "part": "Part B",
            "id": "6(a)",
            "co": co_at(0),
            "level": "L2",
            "marks": 5,
            "desc": "Descriptive / Analytical",
        },
        {
            "part": "Part B",
            "id": "6(b)",
            "co": co_at(1),
            "level": "L3",
            "marks": 5,
            "desc": "Problem Solving / Application",
        },
        # Question 7
        {
            "part": "Part B",
            "id": "7(a)",
            "co": co_at(2 if num_cos > 2 else 0),
            "level": "L3",
            "marks": 5,
            "desc": "Application & Demonstration",
        },
        {
            "part": "Part B",
            "id": "7(b)",
            "co": co_at(3 if num_cos > 3 else 1),
            "level": "L4",
            "marks": 5,
            "desc": "Analysis & Comparison",
        },
        # Question 8
        {
            "part": "Part B",
            "id": "8(a)",
            "co": co_at(0),
            "level": "L3",
            "marks": 5,
            "desc": "Design / Implementation",
        },
        {
            "part": "Part B",
            "id": "8(b)",
            "co": co_at(1),
            "level": "L4",
            "marks": 5,
            "desc": "Critical Analysis",
        },
        # Question 9
        {
            "part": "Part B",
            "id": "9(a)",
            "co": co_at(2 if num_cos > 2 else 0),
            "level": "L4",
            "marks": 5,
            "desc": "Case Analysis / Evaluation",
        },
        {
            "part": "Part B",
            "id": "9(b)",
            "co": co_at(3 if num_cos > 3 else 1),
            "level": "L5",
            "marks": 5,
            "desc": "Synthesis & Evaluation",
        },
    ]
    return slots


def get_25_mark_blueprint(cos: list[str] = None) -> list[dict[str, Any]]:
    """Return 25-mark internal assessment / quiz blueprint."""
    if not cos:
        cos = ["CO1", "CO2"]

    num_cos = len(cos)

    def co_at(index: int) -> str:
        return cos[index % num_cos]

    return [
        {"part": "Part A", "id": "1", "co": co_at(0), "level": "L1", "marks": 2},
        {"part": "Part A", "id": "2", "co": co_at(1), "level": "L2", "marks": 2},
        {"part": "Part A", "id": "3", "co": co_at(0), "level": "L2", "marks": 2},
        {"part": "Part B", "id": "4(a)", "co": co_at(0), "level": "L3", "marks": 5},
        {"part": "Part B", "id": "4(b)", "co": co_at(1), "level": "L3", "marks": 5},
        {"part": "Part B", "id": "5(a)", "co": co_at(0), "level": "L4", "marks": 5},
        {"part": "Part B", "id": "5(b)", "co": co_at(1), "level": "L4", "marks": 4},
    ]


def calculate_blueprint_stats(slots: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze blueprint slots for total marks, part distribution, BT spread, and CO coverage."""
    total_marks = sum(int(s.get("marks", 0)) for s in slots)
    part_marks: dict[str, int] = {}
    bt_marks: dict[str, int] = {}
    bt_counts: dict[str, int] = {}
    co_marks: dict[str, int] = {}
    co_counts: dict[str, int] = {}

    for s in slots:
        m = int(s.get("marks", 0))
        part = s.get("part", "General")
        part_marks[part] = part_marks.get(part, 0) + m

        lvl = s.get("level", "L1").upper()
        bt_marks[lvl] = bt_marks.get(lvl, 0) + m
        bt_counts[lvl] = bt_counts.get(lvl, 0) + 1

        co = s.get("co", "CO1")
        co_marks[co] = co_marks.get(co, 0) + m
        co_counts[co] = co_counts.get(co, 0) + 1

    bt_percent = {
        lvl: round((marks / total_marks * 100), 1) if total_marks > 0 else 0.0
        for lvl, marks in bt_marks.items()
    }

    return {
        "total_marks": total_marks,
        "total_questions": len(slots),
        "part_marks": part_marks,
        "bt_marks": bt_marks,
        "bt_counts": bt_counts,
        "bt_percent": bt_percent,
        "co_marks": co_marks,
        "co_counts": co_counts,
        "is_standard_50": total_marks == 50,
    }

