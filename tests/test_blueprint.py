"""Tests for Blueprint Engine (Task 1)."""
import pytest
from blueprint import build_blueprint


def test_50_preset_structure():
    """Verify 50-mark preset has 5 questions: 5+5, 5+5, 5+5, 10, 10, totaling 50 marks."""
    bp = build_blueprint(preset=50)
    slots = bp["slots"]

    by_q = {}
    for s in slots:
        by_q.setdefault(s["question_no"], []).append(s["marks"])

    assert len(by_q) == 5
    assert by_q[1] == [5, 5]
    assert by_q[2] == [5, 5]
    assert by_q[3] == [5, 5]
    assert by_q[4] == [10]
    assert by_q[5] == [10]

    assert sum(s["marks"] for s in slots) == 50
    assert all(s["alternative_group"] is None for s in slots)


def test_100_preset_structure():
    """Verify 100-mark preset has 5 units x 2 questions = 10 questions, each exactly 20 marks."""
    bp = build_blueprint(preset=100)
    slots = bp["slots"]

    by_q = {}
    for s in slots:
        by_q.setdefault(s["question_no"], []).append(s)

    assert len(by_q) == 10
    for q_no, q_slots in by_q.items():
        assert sum(s["marks"] for s in q_slots) == 20

    units = {s["unit"] for s in slots}
    assert len(units) == 5
    assert bp["report"]["total_attempted_marks"] == 100
    assert bp["report"]["total_offered_marks"] == 200


def test_changing_sub_question_counts_keeps_20_marks():
    """Adjusting sub-question counts keeps every question at exactly 20 marks."""
    for count in [1, 2, 3, 4]:
        bp = build_blueprint(preset=100, sub_question_counts=count)
        by_q = {}
        for s in bp["slots"]:
            by_q.setdefault(s["question_no"], []).append(s["marks"])

        for q_no, marks_list in by_q.items():
            assert sum(marks_list) == 20
            assert len(marks_list) == count

    # 20 over 3 should give [7, 7, 6]
    bp3 = build_blueprint(preset=100, sub_question_counts=3)
    assert [s["marks"] for s in bp3["slots"] if s["question_no"] == 1] == [7, 7, 6]


def test_mirror_property_in_alternative_pairs():
    """In 100-mark preset, choice questions of each unit mirror each other in marks and BT levels."""
    bp = build_blueprint(preset=100, sub_question_counts=3, seed=42)
    slots = bp["slots"]

    for u in range(1, 6):
        q1_slots = [s for s in slots if s["unit"] == u and s["question_no"] == (2 * u - 1)]
        q2_slots = [s for s in slots if s["unit"] == u and s["question_no"] == (2 * u)]

        assert len(q1_slots) == len(q2_slots)
        assert [s["marks"] for s in q1_slots] == [s["marks"] for s in q2_slots]
        assert [s["level"] for s in q1_slots] == [s["level"] for s in q2_slots]
        assert all(s["alternative_group"] == f"U{u}" for s in q1_slots)
        assert all(s["alternative_group"] == f"U{u}" for s in q2_slots)


def test_achieved_percentages_within_5_points():
    """Achieved BT and CO percentages must fall within 5 percentage points of targets."""
    # Default weights
    bp_default = build_blueprint(preset=100, seed=42)
    rep_def = bp_default["report"]
    assert rep_def["max_deviation"] <= 5.0
    for band_info in rep_def["bt"].values():
        assert band_info["deviation"] <= 5.0

    # Custom weights
    custom_bt = {"L1-L2": 0.35, "L3-L4": 0.35, "L5-L6": 0.30}
    bp_custom = build_blueprint(preset=100, bt_weights=custom_bt, seed=42)
    rep_cust = bp_custom["report"]
    assert rep_cust["max_deviation"] <= 5.0


def test_invalid_weights_rejected():
    """Weights that do not sum to 1.0 must be rejected with ValueError."""
    with pytest.raises(ValueError, match="Invalid weights"):
        build_blueprint(preset=100, bt_weights={"L1-L2": 0.4, "L3-L4": 0.4, "L5-L6": 0.4})


def test_out_of_range_weights_warn_but_still_build():
    """Weights summing to 1.0 but outside teacher allowed ranges issue warnings but succeed."""
    # L1-L2 = 0.50 (outside teacher range 30-40%)
    out_of_range_bt = {"L1-L2": 0.50, "L3-L4": 0.30, "L5-L6": 0.20}
    bp = build_blueprint(preset=100, bt_weights=out_of_range_bt)
    assert len(bp["slots"]) == 20
    assert any("outside teacher allowed range" in w for w in bp["warnings"])


def test_determinism_with_seed():
    """Same seed yields identical blueprint slot sequences."""
    bp1 = build_blueprint(preset=100, seed=999)
    bp2 = build_blueprint(preset=100, seed=999)

    assert bp1["slots"] == bp2["slots"]
    assert bp1["report"] == bp2["report"]


def test_explicit_sub_question_marks_validation():
    """Explicit sub_question_marks must sum correctly or raise ValueError."""
    # Valid explicit marks summing to 20
    bp = build_blueprint(preset=100, sub_question_marks=[12, 8])
    for s in bp["slots"][:2]:
        assert s["marks"] in [12, 8]

    # Invalid explicit marks not summing to 20
    with pytest.raises(ValueError, match="Sub-question marks"):
        build_blueprint(preset=100, sub_question_marks=[10, 5])
