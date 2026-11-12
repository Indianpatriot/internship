"""Data contracts and schemas for question paper generation.

Defines the core data structures shared across ingestion, blueprinting,
retrieval, generation, and export modules.
"""
from typing import Any, Optional, TypedDict, Union

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

VALID_CRITERION_KINDS: set[str] = {
    "definition",
    "explanation",
    "points",
    "diagram",
    "example",
    "steps",
    "code",
    "comparison",
    "analysis",
    "justification",
    "conclusion",
}


class SchemeCriterion(TypedDict, total=False):
    """A single examiner-checkable criterion within a marking scheme."""
    no: int
    criterion: str
    expected: str
    marks: Union[int, float]
    kind: str
    reason: str
    partial_credit: Optional[str]
    alternatives: Optional[list[str]]
    figures: Optional[list[dict[str, Any]]]


class MarkingScheme(TypedDict, total=False):
    """Examiner marking scheme breaking down expected criteria, marks, and reasons."""
    total_marks: Union[int, float]
    criteria: list[SchemeCriterion]
    notes: Optional[str]
    is_fallback: Optional[bool]
    reference_figures: Optional[list[dict[str, Any]]]


class Figure(TypedDict, total=False):
    """Extracted figure record from a textbook page."""
    figure_id: str
    book_id: str
    page: int
    bbox: tuple[float, float, float, float]
    image_path: str
    caption: str
    figure_label: str
    nearby_text: str
    chapter: Optional[str]
    section: Optional[str]
    kind: str  # "raster" | "vector"
    width: int
    height: int



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
    marking_scheme: Optional[MarkingScheme]


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


def validate_marking_scheme(
    scheme: dict[str, Any],
    expected_total: Union[int, float],
) -> tuple[bool, list[str]]:
    """Validate a MarkingScheme structure against the expected question marks.

    Rules:
    - criteria marks must sum EXACTLY to total_marks.
    - total_marks must equal expected_total.
    - Default granularity is whole marks; 0.5 marks are supported only when
      config.ALLOW_HALF_MARKS is True.
    - Every criterion must have a positive mark (> 0), a non-empty 'reason',
      and a non-empty 'expected' string.
    - Returns (ok, problems).
    """
    import config

    problems: list[str] = []
    if not isinstance(scheme, dict):
        return False, ["Marking scheme must be a dictionary."]

    total_marks = scheme.get("total_marks")
    if total_marks is None:
        problems.append("Marking scheme missing 'total_marks'.")
    else:
        try:
            t_val = float(total_marks)
            if abs(t_val - float(expected_total)) > 1e-6:
                problems.append(
                    f"Scheme total_marks ({total_marks}) does not match expected total ({expected_total})."
                )
        except (ValueError, TypeError):
            problems.append(f"Invalid total_marks value: {total_marks}")

    criteria = scheme.get("criteria")
    if not isinstance(criteria, list) or len(criteria) == 0:
        problems.append("Marking scheme must contain a non-empty 'criteria' list.")
        return False, problems

    criteria_sum = 0.0
    allow_half = getattr(config, "ALLOW_HALF_MARKS", False)

    for i, c in enumerate(criteria, 1):
        if not isinstance(c, dict):
            problems.append(f"Criterion #{i} is not a dictionary.")
            continue

        c_no = c.get("no", i)
        c_marks = c.get("marks")
        if c_marks is None:
            problems.append(f"Criterion #{c_no} missing 'marks'.")
        else:
            try:
                m_val = float(c_marks)
                if m_val <= 0:
                    problems.append(f"Criterion #{c_no} marks must be greater than zero, got {m_val}.")
                else:
                    criteria_sum += m_val
                    if not allow_half:
                        if not float(m_val).is_integer():
                            problems.append(
                                f"Criterion #{c_no} has fractional marks ({m_val}) but ALLOW_HALF_MARKS is False."
                            )
                    else:
                        # Allow only multiples of 0.5
                        if abs((m_val * 2) - round(m_val * 2)) > 1e-6:
                            problems.append(
                                f"Criterion #{c_no} marks ({m_val}) must be a multiple of 0.5."
                            )
            except (ValueError, TypeError):
                problems.append(f"Criterion #{c_no} has invalid marks: {c_marks}.")

        reason = c.get("reason")
        if not reason or not str(reason).strip():
            problems.append(f"Criterion #{c_no} missing or empty 'reason'.")

        expected = c.get("expected")
        if not expected or not str(expected).strip():
            problems.append(f"Criterion #{c_no} missing or empty 'expected'.")

        kind = c.get("kind")
        if kind and kind not in VALID_CRITERION_KINDS:
            problems.append(f"Criterion #{c_no} has invalid kind '{kind}'.")

    if total_marks is not None:
        try:
            if abs(criteria_sum - float(total_marks)) > 1e-6:
                problems.append(
                    f"Criteria marks sum to {criteria_sum}, expected {total_marks}."
                )
        except (ValueError, TypeError):
            pass

    return len(problems) == 0, problems

