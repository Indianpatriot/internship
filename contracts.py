"""Data contracts and schemas for question paper generation.

Defines the core data structures shared across ingestion, blueprinting,
retrieval, generation, and export modules.
"""
from typing import Any, Optional, TypedDict

DEFAULT_BT_WEIGHTS: dict[str, float] = {
    "L1-L2": 0.40,
    "L3-L4": 0.40,
    "L5-L6": 0.20,
}

TEACHER_BT_RANGES: dict[str, tuple[float, float]] = {
    "L1-L2": (0.30, 0.40),
    "L3-L4": (0.30, 0.40),
    "L5-L6": (0.10, 0.20),
}


class Slot(TypedDict, total=False):
    """Specification for a single question or sub-question slot in a blueprint."""
    unit: int
    question_no: int
    sub_index: int | str
    co: str
    level: str
    marks: int
    alternative_group: Optional[str]
    # Optional compatibility and export metadata
    id: str
    part: str
    knowledge_dimension: str
    kd: str


class QuestionRecord(TypedDict, total=False):
    """Generated question record with model answer and contextual provenance."""
    id: str
    unit: Optional[int]
    question_no: Optional[int]
    sub_index: Optional[int | str]
    question: str
    answer: str
    answer_alternatives: list[str]
    level: str
    co: str
    marks: int
    book_id: Optional[str]
    chapter: Optional[str | int]
    page: Optional[int]
    valid: bool
    reason: str
    attempts: int
    # Additional pipeline metadata
    verb: str
    chunk_id: Optional[str]
    section: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    knowledge_dimension: str
    kd: str
    part: str
    alternative_group: Optional[str]


class Weights(TypedDict, total=False):
    """Weight specification for Bloom's Taxonomy bands and Course Outcomes."""
    bt: dict[str, float]
    co: dict[str, float]


def validate_weights(weights: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate BT and CO weight fractions.

    Requires fractions to sum to 1.0 (within 1e-6 tolerance).
    Returns (ok, warnings) where warnings flags any BT band weight
    outside the teacher's allowed ranges (TEACHER_BT_RANGES).
    """
    warnings: list[str] = []
    ok = True

    if not weights:
        return False, ["Weights dictionary is empty."]

    # Support either {"bt": {...}, "co": {...}} or direct {"L1-L2": 0.40, ...}
    if "bt" in weights and isinstance(weights["bt"], dict):
        bt_map = weights["bt"]
        co_map = weights.get("co")
    else:
        bt_map = weights
        co_map = None

    # Validate BT sum
    bt_sum = sum(bt_map.values())
    if abs(bt_sum - 1.0) > 1e-6:
        ok = False
        warnings.append(f"BT weights sum to {bt_sum:.4f}, expected 1.0.")

    # Validate CO sum if provided
    if co_map and isinstance(co_map, dict):
        co_sum = sum(co_map.values())
        if abs(co_sum - 1.0) > 1e-6:
            ok = False
            warnings.append(f"CO weights sum to {co_sum:.4f}, expected 1.0.")

    # Check teacher allowed ranges (warn, do not invalidate ok if sum was 1.0)
    for band, (low, high) in TEACHER_BT_RANGES.items():
        if band in bt_map:
            val = bt_map[band]
            if val < low - 1e-6 or val > high + 1e-6:
                warnings.append(
                    f"BT weight for {band} is {val:.1%}, outside teacher allowed range [{low:.0%}, {high:.0%}]."
                )

    return ok, warnings
