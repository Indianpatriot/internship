"""Tests for Blueprint Engine (Task 1)."""
import pytest
import config
from blueprint import build_blueprint, validate_outcomes



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


def test_stale_co_weights_ignored_with_warning():
    """4 outcomes (CO1-CO4) but co_weights containing a stale CO5 -> builds, CO5 ignored, warning returned."""
    outcomes = [f"CO{i}" for i in range(1, config.MIN_COS + 1)]
    co_weights = {f"CO{i}": 0.25 for i in range(1, config.MIN_COS + 1)}
    co_weights["CO5"] = 0.25

    bp = build_blueprint(preset=100, outcomes=outcomes, co_weights=co_weights)
    assert bp["slots"]
    assert "CO5" not in bp["report"]["co"]
    assert any("CO5" in w for w in bp["warnings"])
    assert sum(info["target_pct"] for info in bp["report"]["co"].values()) == pytest.approx(100.0, abs=1e-2)


def test_missing_co_weight_gets_equal_share():
    """5 outcomes but co_weights only for CO1-CO4 -> builds, CO5 gets a share, weights sum to 1.0."""
    outcomes = [f"CO{i}" for i in range(1, config.MIN_COS + 2)]  # 5 outcomes
    co_weights = {f"CO{i}": 0.25 for i in range(1, config.MIN_COS + 1)}  # CO1-CO4

    bp = build_blueprint(preset=100, outcomes=outcomes, co_weights=co_weights)
    assert bp["slots"]
    assert "CO5" in bp["report"]["co"]
    assert bp["report"]["co"]["CO5"]["target_pct"] > 0
    assert sum(info["target_pct"] for info in bp["report"]["co"].values()) == pytest.approx(100.0, abs=1e-2)


def test_outcome_count_upper_limit():
    """MAX_COS outcomes builds; MAX_COS + 1 outcomes raises ValueError mentioning the maximum."""
    valid_cos = [f"CO{i}" for i in range(1, config.MAX_COS + 1)]
    bp = build_blueprint(preset=100, outcomes=valid_cos)
    assert len(bp["slots"]) > 0

    too_many_cos = [f"CO{i}" for i in range(1, config.MAX_COS + 2)]
    with pytest.raises(ValueError, match=f"at most {config.MAX_COS} course outcomes"):
        build_blueprint(preset=100, outcomes=too_many_cos)


def test_outcome_count_lower_limit():
    """Fewer than MIN_COS outcomes raises ValueError mentioning minimum; zero outcomes raises same error."""
    too_few_cos = [f"CO{i}" for i in range(1, config.MIN_COS)]
    with pytest.raises(ValueError, match=f"at least {config.MIN_COS} course outcomes"):
        build_blueprint(preset=100, outcomes=too_few_cos)

    with pytest.raises(ValueError, match=f"at least {config.MIN_COS} course outcomes"):
        build_blueprint(preset=100, outcomes=[])


def test_unit_co_map_nonexistent_code_dropped_with_warning():
    """unit_co_map referencing a nonexistent code -> builds with a warning."""
    outcomes = [f"CO{i}" for i in range(1, config.MIN_COS + 1)]
    unit_co_map = {1: ["CO1", "NONEXISTENT_CO"], 2: ["CO2"], 3: ["CO3"], 4: ["CO4"], 5: ["CO1"]}
    bp = build_blueprint(preset=100, outcomes=outcomes, unit_co_map=unit_co_map)
    assert bp["slots"]
    assert any("NONEXISTENT_CO" in w for w in bp["warnings"])
    assert all(s["co"] != "NONEXISTENT_CO" for s in bp["slots"])


def test_rename_outcome_scenario():
    """Rename scenario: outcome CO3 renamed to LO3 -> no KeyError, weights sum to 1.0."""
    outcomes = ["CO1", "CO2", "LO3", "CO4"]
    stale_co_weights = {"CO1": 0.25, "CO2": 0.25, "CO3": 0.25, "CO4": 0.25}

    bp = build_blueprint(preset=100, outcomes=outcomes, co_weights=stale_co_weights)
    assert bp["slots"]
    assert "LO3" in bp["report"]["co"]
    assert "CO3" not in bp["report"]["co"]
    assert sum(info["target_pct"] for info in bp["report"]["co"].values()) == pytest.approx(100.0, abs=1e-2)


def test_duplicate_or_blank_codes_not_double_counted():
    """Duplicate or blank codes are not double-counted."""
    outcomes = ["CO1", "CO2", "CO3", "CO4", "CO4", "", "   "]
    bp = build_blueprint(preset=100, outcomes=outcomes)
    assert len(bp["report"]["co"]) == config.MIN_COS
    assert set(bp["report"]["co"].keys()) == {"CO1", "CO2", "CO3", "CO4"}

    # If unique non-empty count drops below MIN_COS, raises ValueError
    insufficient = ["CO1", "CO1", "CO2", "CO3", "", "  "]
    with pytest.raises(ValueError, match=f"at least {config.MIN_COS} course outcomes"):
        build_blueprint(preset=100, outcomes=insufficient)


def test_validate_outcomes_helper():
    """Test UI-level validate_outcomes helper: 3 -> invalid, 4 -> valid, 6 -> valid, 7 -> invalid."""
    ok_below, msg_below = validate_outcomes(config.MIN_COS - 1)
    assert not ok_below
    assert f"at least {config.MIN_COS}" in msg_below

    ok_min, msg_min = validate_outcomes(config.MIN_COS)
    assert ok_min
    assert msg_min == ""

    ok_max, msg_max = validate_outcomes(config.MAX_COS)
    assert ok_max
    assert msg_max == ""

    ok_above, msg_above = validate_outcomes(config.MAX_COS + 1)
    assert not ok_above
    assert f"at most {config.MAX_COS}" in msg_above

    # Also test list inputs
    assert not validate_outcomes([f"CO{i}" for i in range(1, config.MIN_COS)])[0]
    assert validate_outcomes([f"CO{i}" for i in range(1, config.MIN_COS + 1)])[0]
    assert validate_outcomes([f"CO{i}" for i in range(1, config.MAX_COS + 1)])[0]
    assert not validate_outcomes([f"CO{i}" for i in range(1, config.MAX_COS + 2)])[0]

