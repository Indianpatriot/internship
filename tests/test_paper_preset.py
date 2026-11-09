"""Tests for paper_preset module and multi-unit exam presets."""
import pytest
from paper_preset import QuestionPlan, UnitPlan, PaperPreset, make_preset, format_paper, PRESETS
from blueprint import build_blueprint


def test_question_plan_validation():
    # Valid with sub-marks
    qp = QuestionPlan(marks=20, sub_marks=[10, 10])
    qp.validate()

    # Valid with 3 subparts
    qp3 = QuestionPlan(marks=20, sub_marks=[8, 6, 6])
    qp3.validate()

    # Valid with no sub-parts (empty list)
    qp_empty = QuestionPlan(marks=20, sub_marks=[])
    qp_empty.validate()

    # Invalid marks sum
    qp_bad = QuestionPlan(marks=20, sub_marks=[10, 5])
    with pytest.raises(ValueError, match="Sub-question marks"):
        qp_bad.validate()


def test_unit_plan_validation():
    # Valid unit with 2 options of 20 marks each
    up = UnitPlan(
        unit=1,
        options=[QuestionPlan(20, [10, 10]), QuestionPlan(20, [10, 10])]
    )
    up.validate()

    # Needs at least 2 options
    with pytest.raises(ValueError, match="needs at least 2 options"):
        UnitPlan(unit=1, options=[QuestionPlan(20, [10, 10])]).validate()

    # Options must carry the same marks
    with pytest.raises(ValueError, match="all options must carry the same marks"):
        UnitPlan(
            unit=1,
            options=[QuestionPlan(20, [10, 10]), QuestionPlan(15, [10, 5])]
        ).validate()


def test_paper_preset_properties_and_make_preset():
    # Standard 100 marks preset
    p100 = PRESETS["100 marks"]
    assert p100.attempt_marks == 100
    assert p100.generated_marks == 200
    assert len(p100.units) == 5

    # Standard 50 marks preset
    p50 = PRESETS["50 marks"]
    assert p50.attempt_marks == 50
    assert p50.generated_marks == 100
    assert len(p50.units) == 5

    # Custom preset
    preset = make_preset("custom 100", units=5, marks_per_unit=20, sub_marks={2: [8, 6, 6], 4: [10, 10]})
    assert preset.attempt_marks == 100
    assert preset.generated_marks == 200

    u2 = preset.units[1]
    assert u2.options[0].sub_marks == [8, 6, 6]
    assert u2.options[1].sub_marks == [8, 6, 6]

    u4 = preset.units[3]
    assert u4.options[0].sub_marks == [10, 10]
    assert u4.options[1].sub_marks == [10, 10]

    # Unit 1 has no sub-questions ([] -> full 20m)
    u1 = preset.units[0]
    assert u1.options[0].sub_marks == []


def test_build_blueprint_with_paper_preset():
    # Build blueprint using custom PaperPreset
    preset = make_preset(
        "custom 100",
        units=5,
        marks_per_unit=20,
        sub_marks={
            1: [],            # No sub-questions: single 20m question
            2: [8, 6, 6],     # 3 sub-questions: 8, 6, 6
            3: [10, 10],      # 2 sub-questions: 10, 10
            4: [12, 8],       # 2 sub-questions: 12, 8
            5: [],            # No sub-questions: single 20m question
        }
    )
    bp = build_blueprint(preset=preset, seed=42)
    slots = bp["slots"]

    # In 5 units with 2 options each = 10 questions total
    by_q = {}
    for s in slots:
        by_q.setdefault(s["question_no"], []).append(s)

    assert len(by_q) == 10

    # Unit 1: Q1 and Q2 each have 1 slot of 20 marks
    assert [s["marks"] for s in by_q[1]] == [20]
    assert by_q[1][0]["id"] == "1"
    assert [s["marks"] for s in by_q[2]] == [20]
    assert by_q[2][0]["id"] == "2"

    # Unit 2: Q3 and Q4 each have 3 slots [8, 6, 6]
    assert [s["marks"] for s in by_q[3]] == [8, 6, 6]
    assert [s["id"] for s in by_q[3]] == ["3(a)", "3(b)", "3(c)"]
    assert [s["marks"] for s in by_q[4]] == [8, 6, 6]
    assert [s["id"] for s in by_q[4]] == ["4(a)", "4(b)", "4(c)"]

    # Unit 3: Q5 and Q6 each have 2 slots [10, 10]
    assert [s["marks"] for s in by_q[5]] == [10, 10]
    assert [s["id"] for s in by_q[5]] == ["5(a)", "5(b)"]
    assert [s["marks"] for s in by_q[6]] == [10, 10]
    assert [s["id"] for s in by_q[6]] == ["6(a)", "6(b)"]

    # Report marks
    assert bp["report"]["total_attempted_marks"] == 100
    assert bp["report"]["total_offered_marks"] == 200


def test_format_paper_layout():
    # Build questions layout test
    sample_questions = [
        # Unit 1 Q1 & Q2 (with subparts a, b)
        {"unit": 1, "question_no": 1, "sub_index": 1, "id": "1(a)", "question": "Explain memory hierarchy.", "marks": 10, "co": "CO1", "level": "L2"},
        {"unit": 1, "question_no": 1, "sub_index": 2, "id": "1(b)", "question": "Differentiate SRAM and DRAM.", "marks": 10, "co": "CO1", "level": "L3"},
        {"unit": 1, "question_no": 2, "sub_index": 1, "id": "2(a)", "question": "Describe cache mapping.", "marks": 10, "co": "CO2", "level": "L2"},
        {"unit": 1, "question_no": 2, "sub_index": 2, "id": "2(b)", "question": "Analyze miss rates.", "marks": 10, "co": "CO2", "level": "L4"},
        # Unit 2 Q3 & Q4 (no subparts for Q3, subparts for Q4)
        {"unit": 2, "question_no": 3, "sub_index": 1, "id": "3", "question": "Explain pipeline hazards in detail.", "marks": 20, "co": "CO2", "level": "L2"},
        {"unit": 2, "question_no": 4, "sub_index": 1, "id": "4(a)", "question": "Describe branch prediction.", "marks": 10, "co": "CO3", "level": "L3"},
        {"unit": 2, "question_no": 4, "sub_index": 2, "id": "4(b)", "question": "Evaluate forwarding techniques.", "marks": 10, "co": "CO3", "level": "L4"},
    ]

    text = format_paper(sample_questions)
    assert "UNIT 1" in text
    assert "Q1.  (a) Explain memory hierarchy.  [10 marks]  CO1  L2" in text
    assert "     (b) Differentiate SRAM and DRAM.  [10 marks]  CO1  L3" in text
    assert "OR" in text
    assert "Q2.  (a) Describe cache mapping.  [10 marks]  CO2  L2" in text
    assert "UNIT 2" in text
    assert "Q3.  Explain pipeline hazards in detail.  [20 marks]  CO2  L2" in text
    assert "Q4.  (a) Describe branch prediction.  [10 marks]  CO3  L3" in text

