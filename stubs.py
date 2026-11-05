"""Stub implementations for teammate modules.

OWNERSHIP RULES:
- parse_scheme: OWNER: Alice
- summarize: OWNER: Bob
"""
from typing import Any, Optional


def parse_scheme(pdf_path: str) -> dict[str, Any]:
    """OWNER: Alice
    STUB - replace

    Parse syllabus / scheme PDF to extract units, course outcomes, and textbook mappings.
    """
    return {
        "units": [1, 2, 3, 4, 5],
        "outcomes": {
            "CO1": "Understand foundational principles and definitions",
            "CO2": "Apply algorithms and methods to solve core problems",
            "CO3": "Analyze computational complexity and structural designs",
            "CO4": "Evaluate architectural trade-offs and performance criteria",
            "CO5": "Design and synthesize robust software and engineering solutions",
        },
        "unit_sources": {
            1: {"book_a": ["Chapter 1", "Chapter 2"]},
            2: {"book_a": ["Chapter 3"], "book_b": ["Chapter 1"]},
            3: {"book_b": ["Chapter 2", "Chapter 3"]},
            4: {"book_b": ["Chapter 4", "Chapter 5"]},
            5: {"book_c": None},
        },
    }


def summarize(questions: list[dict[str, Any]], targets: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """OWNER: Bob
    STUB - replace

    Summarize question distribution across COs, BT levels, total marks, and target differences.
    """
    total_marks = sum(q.get("marks", 0) for q in questions)
    by_co: dict[str, int] = {}
    by_bt: dict[str, int] = {}
    for q in questions:
        co = q.get("co", "CO1")
        bt = q.get("level", "L1")
        m = q.get("marks", 0)
        by_co[co] = by_co.get(co, 0) + m
        by_bt[bt] = by_bt.get(bt, 0) + m

    targets_diff: dict[str, float] = {}
    if targets:
        for k, target_val in targets.items():
            targets_diff[k] = 0.0

    return {
        "by_co": by_co,
        "by_bt": by_bt,
        "total_marks": total_marks,
        "targets_diff": targets_diff,
    }
