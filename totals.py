"""Cumulative totals and question paper summary analytics.

Computes:
- Total attempted marks vs total offered marks
- Course Outcome (CO) distributions (marks, percentage, question counts)
- Bloom's Taxonomy cognitive level (L1-L6) and band (L1-L2, L3-L4, L5-L6) distributions
- Differences against target weight specifications (percentage points)
- Plain list-of-lists table representations for document export
"""
from typing import Any, Optional

from blooms.verb_bank import LEVELS
from contracts import Weights

BT_BANDS = ["L1-L2", "L3-L4", "L5-L6"]
BAND_MAP = {
    "L1": "L1-L2",
    "L2": "L1-L2",
    "L3": "L3-L4",
    "L4": "L3-L4",
    "L5": "L5-L6",
    "L6": "L5-L6",
}
LEVEL_NAMES = {
    "L1": "Remember",
    "L2": "Understand",
    "L3": "Apply",
    "L4": "Analyze",
    "L5": "Evaluate",
    "L6": "Create",
}


def _filter_attempted_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select attempted questions by picking one question per alternative choice group."""
    # Group records by alternative_group
    group_map: dict[str, list[dict[str, Any]]] = {}
    compulsory: list[dict[str, Any]] = []

    for q in questions:
        alt_grp = q.get("alternative_group")
        if alt_grp:
            group_map.setdefault(alt_grp, []).append(q)
        else:
            compulsory.append(q)

    attempted: list[dict[str, Any]] = list(compulsory)

    for alt_grp, group_questions in group_map.items():
        # Identify distinct questions within this alternative group
        # Look at question_no first
        q_nos = [q.get("question_no") for q in group_questions if q.get("question_no") is not None]
        if q_nos:
            chosen_q_no = min(q_nos)
            chosen_records = [q for q in group_questions if q.get("question_no") == chosen_q_no]
        else:
            # Fall back to question id prefix or the first record
            first_id = group_questions[0].get("id", "")
            base_id = first_id.split("(")[0] if first_id else ""
            if base_id:
                chosen_records = [q for q in group_questions if q.get("id", "").startswith(base_id)]
            else:
                chosen_records = [group_questions[0]]

        attempted.extend(chosen_records)

    return attempted


def summarize(
    questions: list[dict[str, Any]],
    targets: Optional[dict[str, Any]] = None,
    attempted_only: bool = True,
) -> dict[str, Any]:
    """Summarize question paper totals across COs, BT levels, and target differences.

    Args:
        questions: List of generated question dictionaries / QuestionRecords.
        targets: Target Weights dictionary (e.g. contracts.Weights).
        attempted_only: When True, counts only one question per alternative choice group.
            When False, counts all questions offered in the paper.

    Returns:
        {
            "by_co": {code: {"marks": int, "pct": float, "questions": int}},
            "by_bt": {"L1": ..., ..., "L6": ...},
            "by_bt_band": {"L1-L2": ..., "L3-L4": ..., "L5-L6": ...},
            "total_marks": int,
            "offered_marks": int,
            "targets_diff": {"bt_band": {...}, "co": {...}}
        }
    """
    offered_marks = sum(int(q.get("marks", 0)) for q in questions)

    if attempted_only:
        evaluated_questions = _filter_attempted_questions(questions)
    else:
        evaluated_questions = list(questions)

    total_marks = sum(int(q.get("marks", 0)) for q in evaluated_questions)

    # Initialize by_bt for all L1-L6
    by_bt: dict[str, dict[str, Any]] = {
        lvl: {"marks": 0, "pct": 0.0, "questions": 0} for lvl in LEVELS
    }

    # Initialize by_bt_band
    by_bt_band: dict[str, dict[str, Any]] = {
        band: {"marks": 0, "pct": 0.0, "questions": 0} for band in BT_BANDS
    }

    # Initialize by_co from questions and targets
    by_co: dict[str, dict[str, Any]] = {}
    if targets and "co" in targets and isinstance(targets["co"], dict):
        for co in targets["co"]:
            by_co[co] = {"marks": 0, "pct": 0.0, "questions": 0}

    # Aggregate counts and marks
    for q in evaluated_questions:
        m = int(q.get("marks", 0))
        co = q.get("co", "CO1")
        lvl = q.get("level", "L1").upper()

        # Update CO
        if co not in by_co:
            by_co[co] = {"marks": 0, "pct": 0.0, "questions": 0}
        by_co[co]["marks"] += m
        by_co[co]["questions"] += 1

        # Update BT level
        if lvl not in by_bt:
            by_bt[lvl] = {"marks": 0, "pct": 0.0, "questions": 0}
        by_bt[lvl]["marks"] += m
        by_bt[lvl]["questions"] += 1

        # Update BT band
        band = BAND_MAP.get(lvl, "L1-L2")
        by_bt_band[band]["marks"] += m
        by_bt_band[band]["questions"] += 1

    # Compute percentages (guard against division by zero)
    if total_marks > 0:
        for co, data in by_co.items():
            data["pct"] = round((data["marks"] / total_marks) * 100, 2)
        for lvl, data in by_bt.items():
            data["pct"] = round((data["marks"] / total_marks) * 100, 2)
        for band, data in by_bt_band.items():
            data["pct"] = round((data["marks"] / total_marks) * 100, 2)

    # Compute targets_diff if targets provided
    targets_diff: dict[str, dict[str, float]] = {"bt_band": {}, "co": {}}
    if targets:
        # Extract BT targets
        if "bt" in targets and isinstance(targets["bt"], dict):
            bt_target = targets["bt"]
        else:
            bt_target = targets

        for band in BT_BANDS:
            if band in bt_target:
                t_pct = bt_target[band] * 100
                achieved = by_bt_band[band]["pct"]
                targets_diff["bt_band"][band] = round(achieved - t_pct, 2)

        # Extract CO targets
        if isinstance(targets, dict) and "co" in targets and isinstance(targets["co"], dict):
            for co, frac in targets["co"].items():
                t_pct = frac * 100
                achieved = by_co.get(co, {}).get("pct", 0.0)
                targets_diff["co"][co] = round(achieved - t_pct, 2)

    return {
        "by_co": by_co,
        "by_bt": by_bt,
        "by_bt_band": by_bt_band,
        "total_marks": total_marks,
        "offered_marks": offered_marks,
        "targets_diff": targets_diff,
    }


def totals_table_rows(summary: dict[str, Any]) -> dict[str, list[list[Any]]]:
    """Convert summary analytics into plain list-of-lists tables for Word/PDF export.

    Returns:
        {
            "co_table": [[header...], [row1...], ..., [total_row...]],
            "bt_table": [[header...], [row1...], ..., [total_row...]]
        }
    """
    total_m = summary.get("total_marks", 0)

    # 1. Course Outcomes Table
    co_header = ["Course Outcome", "Marks", "Percentage", "Questions Count"]
    co_rows = []
    by_co = summary.get("by_co", {})
    sorted_cos = sorted(by_co.keys())
    for co in sorted_cos:
        info = by_co[co]
        co_rows.append([co, info["marks"], f"{info['pct']:.1f}%", info["questions"]])

    total_co_q = sum(info["questions"] for info in by_co.values())
    co_total_row = ["Total", total_m, "100.0%" if total_m > 0 else "0.0%", total_co_q]
    co_table = [co_header] + co_rows + [co_total_row]

    # 2. Bloom's Taxonomy Table
    bt_header = ["Bloom's Taxonomy Band / Level", "Marks", "Percentage", "Questions Count"]
    bt_rows = []
    by_bt = summary.get("by_bt", {})
    by_band = summary.get("by_bt_band", {})

    band_order = [
        ("L1-L2", ["L1", "L2"]),
        ("L3-L4", ["L3", "L4"]),
        ("L5-L6", ["L5", "L6"]),
    ]

    for band, sub_levels in band_order:
        b_info = by_band.get(band, {"marks": 0, "pct": 0.0, "questions": 0})
        bt_rows.append([band, b_info["marks"], f"{b_info['pct']:.1f}%", b_info["questions"]])
        for lvl in sub_levels:
            l_info = by_bt.get(lvl, {"marks": 0, "pct": 0.0, "questions": 0})
            name = LEVEL_NAMES.get(lvl, "")
            label = f"  {lvl} ({name})" if name else f"  {lvl}"
            bt_rows.append([label, l_info["marks"], f"{l_info['pct']:.1f}%", l_info["questions"]])

    total_bt_q = sum(b_info["questions"] for b_info in by_band.values())
    bt_total_row = ["Total", total_m, "100.0%" if total_m > 0 else "0.0%", total_bt_q]
    bt_table = [bt_header] + bt_rows + [bt_total_row]

    return {
        "co_table": co_table,
        "bt_table": bt_table,
    }
