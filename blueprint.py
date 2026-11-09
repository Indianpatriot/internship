"""Blueprint module for defining exam structures and building dynamic blueprints.

Implements:
- build_blueprint: Deficit-based greedy allocation for 50 and 100-mark presets (and custom presets)
- Presets: 50 marks (compulsory Q1-Q5) and 100 marks (5 units x 2 mirrored choice questions)
- Revised Bloom's Taxonomy (Cognitive L1-L6 & Knowledge Dimensions)
- Legacy helpers: get_50_mark_blueprint, get_25_mark_blueprint, calculate_blueprint_stats
"""
import random
from typing import Any, Optional

import config
from blooms.revised_taxonomy import (
    KNOWLEDGE_DIMENSIONS,
    compute_2d_taxonomy_matrix,
    normalize_knowledge_dimension,
)
from contracts import (
    DEFAULT_BT_WEIGHTS,
    TEACHER_BT_RANGES,
    Slot,
    Weights,
    validate_weights,
)


PRESETS: dict[Any, dict[str, Any]] = {
    50: {
        "total_marks": 50,
        "units": 5,
        "questions_per_unit": 1,
        "question_marks": 10,
        "sub_question_marks": {
            1: [5, 5],
            2: [5, 5],
            3: [5, 5],
            4: [10],
            5: [10],
        },
        "sub_question_counts": {
            1: 2,
            2: 2,
            3: 2,
            4: 1,
            5: 1,
        },
        "is_choice": False,
    },
    100: {
        "total_marks": 100,
        "units": 5,
        "questions_per_unit": 2,
        "question_marks": 20,
        "sub_question_counts": 2,
        "is_choice": True,
    },
}
# Also allow string keys
PRESETS["50"] = PRESETS[50]
PRESETS["100"] = PRESETS[100]

BT_BANDS = ["L1-L2", "L3-L4", "L5-L6"]
BAND_LEVELS = {
    "L1-L2": ["L1", "L2"],
    "L3-L4": ["L3", "L4"],
    "L5-L6": ["L5", "L6"],
}


def _split_marks(total_m: int, count: int) -> list[int]:
    """Split total marks into count sub-questions as evenly as possible.

    Remainder is spread across the first sub-questions (e.g. 20 over 3 -> 7, 7, 6).
    """
    if count <= 0:
        raise ValueError("Sub-question count must be at least 1")
    base = total_m // count
    rem = total_m % count
    return [base + 1 if i < rem else base for i in range(count)]


def validate_outcomes(outcomes: Any, require_desc: bool = False) -> tuple[bool, str]:
    """Validate course outcomes count against MIN_COS and MAX_COS project constraints.

    Args:
        outcomes: An int count, list/tuple/set of codes or dicts, or a dict.
        require_desc: If True, outcomes with empty descriptions do not count.

    Returns:
        (ok, message): (True, "") if MIN_COS <= count <= MAX_COS, else (False, error_message).
    """
    if isinstance(outcomes, int):
        n = outcomes
    elif isinstance(outcomes, dict):
        if require_desc:
            valid = [k for k, v in outcomes.items() if str(k).strip() and str(v).strip()]
        else:
            valid = [k for k in outcomes.keys() if str(k).strip()]
        n = len(set(valid))
    elif hasattr(outcomes, "__iter__") and not isinstance(outcomes, (str, bytes)):
        seen = set()
        for item in outcomes:
            if isinstance(item, dict):
                code = str(item.get("code", "")).strip()
                desc = str(item.get("desc", "")).strip()
                if code and (not require_desc or desc):
                    seen.add(code)
            else:
                code = str(item).strip()
                if code:
                    seen.add(code)
        n = len(seen)
    else:
        n = 0

    if n < config.MIN_COS:
        return False, f"Add at least {config.MIN_COS} course outcomes (you have {n})"
    if n > config.MAX_COS:
        return False, f"A paper can have at most {config.MAX_COS} course outcomes (you have {n})"
    return True, ""


def validate_weightage_totals(
    co_weights: dict[str, float],
    bt_weights: dict[str, float],
    existing_cos: list[str],
    tolerance: float = 0.01,
) -> tuple[bool, list[str]]:
    """Validate that existing CO weightages and BT band weightages both sum to exactly 100%.

    Args:
        co_weights: Mapping of CO code -> percentage (e.g. {"CO1": 25.0, ...}).
        bt_weights: Mapping of BT band -> percentage (e.g. {"L1-L2": 40.0, ...}).
        existing_cos: List of currently existing CO codes to include in sum.
        tolerance: Float tolerance for 100% check (default 0.01).

    Returns:
        (ok, errors): ok is True only if both totals equal 100 (within tolerance).
    """
    errors: list[str] = []

    # 1. Sum only existing COs
    co_sum = sum(float(co_weights.get(c, 0.0)) for c in existing_cos)
    if abs(co_sum - 100.0) > tolerance:
        diff = co_sum - 100.0
        sign = "+" if diff > 0 else ""
        errors.append(
            f"CO weightages total {co_sum:g}%. They must total exactly 100% ({sign}{diff:g}%). Adjust and apply again."
        )

    # 2. Sum BT bands
    bt_sum = sum(float(v) for v in bt_weights.values())
    if abs(bt_sum - 100.0) > tolerance:
        diff = bt_sum - 100.0
        sign = "+" if diff > 0 else ""
        errors.append(
            f"Bloom's weightages total {bt_sum:g}%. They must total exactly 100% ({sign}{diff:g}%). Adjust and apply again."
        )

    return len(errors) == 0, errors


def build_blueprint(
    preset: Optional[Any] = None,
    total_marks: Optional[int] = None,
    units: Optional[Any] = None,
    questions_per_unit: Optional[int] = None,
    sub_question_counts: Optional[Any] = None,
    sub_question_marks: Optional[Any] = None,
    bt_weights: Optional[dict[str, float]] = None,
    co_weights: Optional[dict[str, float]] = None,
    unit_co_map: Optional[dict[Any, list[str]]] = None,
    seed: Optional[int] = None,
    outcomes: Optional[Any] = None,
) -> dict[str, Any]:
    """Build a balanced question paper blueprint using deficit-based greedy allocation.

    Args:
        preset: 50, 100, "50", "100", or None.
        total_marks: Target attempted marks (e.g. 50 or 100).
        units: Number of units (int) or list of unit identifiers.
        questions_per_unit: Questions per unit (1 for compulsory, 2 for choice).
        sub_question_counts: Int, list, or dict specifying sub-questions per question.
        sub_question_marks: Explicit sub-question marks (list or dict).
        bt_weights: Fraction per BT band (L1-L2, L3-L4, L5-L6).
        co_weights: Fraction per Course Outcome.
        unit_co_map: Mapping of unit -> list of allowed CO codes.
        seed: Random seed for deterministic allocation.
        outcomes: Explicit list of CO codes or outcome dicts defining paper outcomes.

    Returns:
        dict with "slots", "report", and "warnings".
    """
    # 1. Determine preset and base parameters
    from paper_preset import PaperPreset

    is_paper_preset = isinstance(preset, PaperPreset)
    if is_paper_preset:
        preset.validate()
        num_units = len(preset.units)
        unit_list = [u.unit for u in preset.units]
        q_per_unit = max((len(u.options) for u in preset.units), default=2)
        is_choice = True
        expected_q_marks = preset.units[0].options[0].marks if preset.units and preset.units[0].options else 20
        attempted_marks_pool = preset.attempt_marks
        preset_config = {
            "total_marks": preset.attempt_marks,
            "units": num_units,
            "questions_per_unit": q_per_unit,
            "question_marks": expected_q_marks,
            "is_choice": True,
        }
    else:
        selected_preset_key = None
        if preset is not None:
            selected_preset_key = preset
        elif total_marks == 100:
            selected_preset_key = 100
        elif total_marks == 50:
            selected_preset_key = 50
        else:
            selected_preset_key = 50

        preset_config = PRESETS.get(selected_preset_key, PRESETS[50])

        num_units = units if isinstance(units, int) else (len(units) if isinstance(units, (list, tuple)) else preset_config["units"])
        unit_list = list(range(1, num_units + 1)) if not isinstance(units, (list, tuple)) else list(units)
        q_per_unit = questions_per_unit if questions_per_unit is not None else preset_config["questions_per_unit"]
        is_choice = preset_config.get("is_choice", q_per_unit > 1)

        expected_q_marks = preset_config["question_marks"]
        if total_marks is not None and not is_choice:
            expected_q_marks = total_marks // (num_units * q_per_unit)
        elif total_marks is not None and is_choice:
            expected_q_marks = total_marks // num_units

        # Attempted marks pool
        attempted_marks_pool = total_marks if total_marks is not None else (num_units * expected_q_marks)


    warnings: list[str] = []

    # 2. Setup COs and Weights
    # Derive all_cos from a SINGLE source of truth
    raw_codes = None
    if outcomes is not None:
        if isinstance(outcomes, dict):
            raw_codes = list(outcomes.keys())
        elif isinstance(outcomes, (list, tuple, set)):
            raw_codes = []
            for item in outcomes:
                if isinstance(item, dict) and "code" in item:
                    raw_codes.append(item["code"])
                else:
                    raw_codes.append(item)
        else:
            raw_codes = [outcomes]
    elif co_weights is not None:
        raw_codes = list(co_weights.keys())

    if raw_codes is not None:
        all_cos: list[str] = []
        seen_cos = set()
        for c in raw_codes:
            code = str(c).strip()
            if code and code not in seen_cos:
                seen_cos.add(code)
                all_cos.append(code)
    else:
        default_count = min(max(num_units, config.MIN_COS), config.MAX_COS)
        all_cos = [f"CO{u}" for u in range(1, default_count + 1)]

    # Validate outcome count
    n_cos = len(all_cos)
    if n_cos < config.MIN_COS:
        raise ValueError(f"Add at least {config.MIN_COS} course outcomes (you have {n_cos})")
    if n_cos > config.MAX_COS:
        raise ValueError(f"A paper can have at most {config.MAX_COS} course outcomes (you have {n_cos})")

    # Clean unit_co_map: if it references a code that does not exist, drop it with a warning
    cleaned_unit_co_map: Optional[dict[Any, list[str]]] = None
    if unit_co_map:
        cleaned_unit_co_map = {}
        for u_key, co_list in unit_co_map.items():
            valid_list = []
            for c in co_list:
                c_str = str(c).strip()
                if c_str in all_cos:
                    valid_list.append(c_str)
                else:
                    warnings.append(
                        f"Unit {u_key} references nonexistent course outcome '{c}', dropped."
                    )
            cleaned_unit_co_map[u_key] = valid_list

    # Build eff_co_weights over exactly all_cos
    if co_weights is not None:
        for k in co_weights.keys():
            k_str = str(k).strip()
            if k_str not in all_cos:
                warnings.append(
                    f"Ignored weight for outcome '{k}' in co_weights not present in paper outcomes."
                )

        present_cos = [c for c in all_cos if c in co_weights]
        missing_cos = [c for c in all_cos if c not in co_weights]

        raw_co_weights: dict[str, float] = {}
        for c in present_cos:
            raw_co_weights[c] = float(co_weights[c])

        if missing_cos:
            present_sum = sum(raw_co_weights.values())
            if 0 < present_sum < 1.0 - 1e-6:
                rem_weight = 1.0 - present_sum
                share = rem_weight / len(missing_cos)
            else:
                share = (present_sum / len(present_cos)) if present_cos and present_sum > 0 else (1.0 / len(all_cos))
            for c in missing_cos:
                raw_co_weights[c] = share

        tot_co_weight = sum(raw_co_weights.values())
        if tot_co_weight > 0:
            eff_co_weights = {c: raw_co_weights[c] / tot_co_weight for c in all_cos}
        else:
            equal_share = 1.0 / len(all_cos)
            eff_co_weights = {c: equal_share for c in all_cos}
    else:
        equal_share = 1.0 / len(all_cos)
        eff_co_weights = {c: equal_share for c in all_cos}

    eff_bt_weights = dict(bt_weights) if bt_weights is not None else dict(DEFAULT_BT_WEIGHTS)

    # Validate weights
    ok, weight_warnings = validate_weights({"bt": eff_bt_weights, "co": eff_co_weights})
    if not ok:
        raise ValueError(f"Invalid weights: {weight_warnings}")

    warnings.extend(weight_warnings)


    # 3. Determine structure and marks for every question
    # Questions are ordered by question_no = 1, 2, ...
    question_specs: list[dict[str, Any]] = []
    q_no = 1
    for u_idx, u_val in enumerate(unit_list):
        u_plan = preset.units[u_idx] if is_paper_preset and u_idx < len(preset.units) else None
        for q_in_u in range(q_per_unit):
            q_plan = u_plan.options[q_in_u] if u_plan and q_in_u < len(u_plan.options) else None

            if q_plan is not None:
                q_expected_marks = q_plan.marks
                if q_plan.sub_marks:
                    q_sub_marks = list(q_plan.sub_marks)
                else:
                    q_sub_marks = [q_expected_marks]
            else:
                q_expected_marks = expected_q_marks
                # Check explicit sub_question_marks
                q_sub_marks = None
                if sub_question_marks is not None:
                    if isinstance(sub_question_marks, list):
                        if sub_question_marks and isinstance(sub_question_marks[0], list):
                            # List of lists per question
                            if q_no - 1 < len(sub_question_marks):
                                val = sub_question_marks[q_no - 1]
                                q_sub_marks = list(val) if val else [q_expected_marks]
                        else:
                            # Single list applied to all questions
                            q_sub_marks = list(sub_question_marks) if sub_question_marks else [q_expected_marks]
                    elif isinstance(sub_question_marks, dict):
                        # If keys are unit numbers (e.g. 1..N for N units), map by unit
                        if u_val in sub_question_marks and (
                            set(sub_question_marks.keys()).issubset(set(unit_list))
                            or (q_per_unit > 1 and max((k for k in sub_question_marks if isinstance(k, int)), default=0) <= len(unit_list))
                        ):
                            val = sub_question_marks[u_val]
                            q_sub_marks = list(val) if val else [q_expected_marks]
                        elif q_no in sub_question_marks:
                            val = sub_question_marks[q_no]
                            q_sub_marks = list(val) if val else [q_expected_marks]
                        elif u_val in sub_question_marks:
                            val = sub_question_marks[u_val]
                            q_sub_marks = list(val) if val else [q_expected_marks]

                # If not explicitly given, check sub_question_counts or preset defaults
                if q_sub_marks is None:
                    sq_count = None
                    if sub_question_counts is not None:
                        if isinstance(sub_question_counts, int):
                            sq_count = sub_question_counts
                        elif isinstance(sub_question_counts, list) and q_no - 1 < len(sub_question_counts):
                            sq_count = sub_question_counts[q_no - 1]
                        elif isinstance(sub_question_counts, dict):
                            if u_val in sub_question_counts and (
                                set(sub_question_counts.keys()).issubset(set(unit_list))
                                or (q_per_unit > 1 and max((k for k in sub_question_counts if isinstance(k, int)), default=0) <= len(unit_list))
                            ):
                                sq_count = sub_question_counts[u_val]
                            elif q_no in sub_question_counts:
                                sq_count = sub_question_counts[q_no]
                            else:
                                sq_count = sub_question_counts.get(u_val)
                    elif "sub_question_marks" in preset_config and q_no in preset_config["sub_question_marks"]:
                        q_sub_marks = list(preset_config["sub_question_marks"][q_no])
                    elif "sub_question_counts" in preset_config:
                        cfg_sq = preset_config["sub_question_counts"]
                        if isinstance(cfg_sq, dict):
                            sq_count = cfg_sq.get(q_no, 1)
                        else:
                            sq_count = cfg_sq

                    if q_sub_marks is None:
                        count_to_use = sq_count if sq_count is not None else 1
                        if count_to_use <= 0:
                            q_sub_marks = [q_expected_marks]
                        else:
                            q_sub_marks = _split_marks(q_expected_marks, count_to_use)

            # Validate that sub-question marks sum to the question's expected marks
            if sum(q_sub_marks) != q_expected_marks:
                raise ValueError(
                    f"Sub-question marks {q_sub_marks} for question {q_no} sum to {sum(q_sub_marks)}, "
                    f"expected exactly {q_expected_marks} marks."
                )

            alt_grp = f"U{u_val}" if is_choice else None
            is_primary = (q_in_u == 0)

            question_specs.append({
                "unit": u_val,
                "unit_index": u_idx,
                "question_no": q_no,
                "question_in_unit": q_in_u,
                "is_primary": is_primary,
                "marks": q_expected_marks,
                "sub_marks": q_sub_marks,
                "alternative_group": alt_grp,
            })
            q_no += 1

    # 4. Greedy Deficit-Based Allocation for Primary (Attempted) Slots
    # Prepare attempted slots pool
    primary_slots_to_allocate: list[dict[str, Any]] = []
    for q_spec in question_specs:
        if q_spec["is_primary"]:
            for sub_i, m in enumerate(q_spec["sub_marks"], 1):
                primary_slots_to_allocate.append({
                    "unit": q_spec["unit"],
                    "question_no": q_spec["question_no"],
                    "sub_index": sub_i,
                    "marks": m,
                    "alternative_group": q_spec["alternative_group"],
                })

    # Sort largest marks first for allocation, with stable tie-breaking
    rng = random.Random(seed if seed is not None else 42)
    # Tag with original index for stable reproduction
    for idx, s in enumerate(primary_slots_to_allocate):
        s["_orig_idx"] = idx

    # Sort descending by marks, with deterministic secondary ordering
    sorted_slots = sorted(
        primary_slots_to_allocate,
        key=lambda s: (-s["marks"], s["_orig_idx"]),
    )

    # Targets
    target_bt = {b: eff_bt_weights[b] * attempted_marks_pool for b in BT_BANDS}
    target_co = {c: eff_co_weights[c] * attempted_marks_pool for c in all_cos}

    allocated_bt = {b: 0 for b in BT_BANDS}
    allocated_co = {c: 0 for c in all_cos}
    allocated_kd = {kd: 0 for kd in KNOWLEDGE_DIMENSIONS}

    band_level_alternation = {b: 0 for b in BT_BANDS}

    KD_BY_LEVEL = {
        "L1": ["Factual", "Conceptual"],
        "L2": ["Conceptual", "Factual"],
        "L3": ["Procedural", "Conceptual"],
        "L4": ["Procedural", "Conceptual"],
        "L5": ["Metacognitive", "Procedural"],
        "L6": ["Metacognitive", "Procedural"],
    }

    for s in sorted_slots:
        m = s["marks"]
        u = s["unit"]

        # Select BT Band with largest remaining deficit
        # Deficit = target - allocated
        best_band = max(
            BT_BANDS,
            key=lambda b: (target_bt[b] - allocated_bt[b], -allocated_bt[b]),
        )
        allocated_bt[best_band] += m
        lvl_idx = band_level_alternation[best_band]
        assigned_level = BAND_LEVELS[best_band][lvl_idx]
        band_level_alternation[best_band] = (lvl_idx + 1) % len(BAND_LEVELS[best_band])
        s["level"] = assigned_level
        s["_band"] = best_band

        # Select KD matching level
        candidates_kd = KD_BY_LEVEL.get(assigned_level, KNOWLEDGE_DIMENSIONS)
        assigned_kd = min(candidates_kd, key=lambda k: allocated_kd[k])
        allocated_kd[assigned_kd] += m
        s["knowledge_dimension"] = assigned_kd

        # Select CO
        if cleaned_unit_co_map and u in cleaned_unit_co_map and cleaned_unit_co_map[u]:
            candidate_cos = cleaned_unit_co_map[u]
        else:
            candidate_cos = all_cos

        best_co = max(
            candidate_cos,
            key=lambda c: (target_co[c] - allocated_co[c], -allocated_co[c]),
        )
        allocated_co[best_co] += m
        s["co"] = best_co

    # Put primary slots back into question_specs
    # Group allocated slots by question_no
    allocated_by_q: dict[int, list[dict[str, Any]]] = {}
    for s in sorted_slots:
        allocated_by_q.setdefault(s["question_no"], []).append(s)

    # Ensure sub-questions in each question maintain sub_index order
    for q_num, s_list in allocated_by_q.items():
        s_list.sort(key=lambda s: s["sub_index"])

    # 5. Build final slots list, applying Mirror Property for alternative questions
    final_slots: list[Slot] = []
    # Map from unit -> primary question's allocated sub-slots
    primary_unit_slots: dict[Any, list[dict[str, Any]]] = {}
    for q_spec in question_specs:
        if q_spec["is_primary"]:
            primary_unit_slots[q_spec["unit"]] = allocated_by_q.get(q_spec["question_no"], [])

    for q_spec in question_specs:
        q_num = q_spec["question_no"]
        u = q_spec["unit"]
        alt_grp = q_spec["alternative_group"]

        if q_spec["is_primary"]:
            cur_q_slots = allocated_by_q.get(q_num, [])
            for s in cur_q_slots:
                sub_i = s["sub_index"]
                total_subs = len(cur_q_slots)
                sub_label = chr(96 + sub_i) if total_subs > 1 else ""
                slot_id = f"{q_num}({sub_label})" if sub_label else str(q_num)
                slot_rec: Slot = {
                    "unit": u,
                    "question_no": q_num,
                    "sub_index": sub_i,
                    "co": s["co"],
                    "level": s["level"],
                    "marks": s["marks"],
                    "alternative_group": alt_grp,
                    "id": slot_id,
                    "knowledge_dimension": s.get("knowledge_dimension", "Conceptual"),
                    "kd": s.get("knowledge_dimension", "Conceptual"),
                }
                final_slots.append(slot_rec)
        else:
            # Choice question: MIRROR primary question of the same unit
            partner_slots = primary_unit_slots[u]
            # Ensure same sub-question marks and same BT level sequence
            for sub_i, p_slot in enumerate(partner_slots, 1):
                total_subs = len(partner_slots)
                sub_label = chr(96 + sub_i) if total_subs > 1 else ""
                slot_id = f"{q_num}({sub_label})" if sub_label else str(q_num)
                slot_rec: Slot = {
                    "unit": u,
                    "question_no": q_num,
                    "sub_index": sub_i,
                    "co": p_slot["co"],
                    "level": p_slot["level"],
                    "marks": p_slot["marks"],
                    "alternative_group": alt_grp,
                    "id": slot_id,
                    "knowledge_dimension": p_slot.get("knowledge_dimension", "Conceptual"),
                    "kd": p_slot.get("knowledge_dimension", "Conceptual"),
                }
                final_slots.append(slot_rec)

    # 6. Assert Invariant: Every question's sub-question marks must sum to question_marks
    total_offered_marks = sum(s["marks"] for s in final_slots)
    by_q: dict[int, list[Slot]] = {}
    for s in final_slots:
        by_q.setdefault(s["question_no"], []).append(s)

    expected_marks_by_q = {q["question_no"]: q["marks"] for q in question_specs}
    for q_num, q_slots in by_q.items():
        q_sum = sum(s["marks"] for s in q_slots)
        exp_m = expected_marks_by_q.get(q_num, expected_q_marks)
        assert q_sum == exp_m, (
            f"Question {q_num} marks invariant violated: sum {q_sum} != {exp_m}"
        )

    # 7. Compute Report on Attempted Marks
    bt_report: dict[str, dict[str, Any]] = {}
    bt_deviations: list[float] = []
    for b in BT_BANDS:
        achieved_m = allocated_bt[b]
        achieved_pct = round((achieved_m / attempted_marks_pool) * 100, 2)
        target_pct = round(eff_bt_weights[b] * 100, 2)
        dev = round(abs(achieved_pct - target_pct), 2)
        bt_deviations.append(dev)
        bt_report[b] = {
            "target_pct": target_pct,
            "achieved_pct": achieved_pct,
            "deviation": dev,
            "achieved_marks": achieved_m,
            "target_marks": round(target_bt[b], 2),
        }

    co_report: dict[str, dict[str, Any]] = {}
    co_deviations: list[float] = []
    for c in all_cos:
        achieved_m = allocated_co[c]
        achieved_pct = round((achieved_m / attempted_marks_pool) * 100, 2)
        target_pct = round(eff_co_weights[c] * 100, 2)
        dev = round(abs(achieved_pct - target_pct), 2)
        co_deviations.append(dev)
        co_report[c] = {
            "target_pct": target_pct,
            "achieved_pct": achieved_pct,
            "deviation": dev,
            "achieved_marks": achieved_m,
            "target_marks": round(target_co[c], 2),
        }

    max_deviation = max(bt_deviations + (co_deviations or [0.0]))
    if max_deviation > 5.0:
        warnings.append(
            f"Maximum deviation {max_deviation:.1f}% exceeds 5 percentage points threshold."
        )

    report = {
        "bt": bt_report,
        "co": co_report,
        "max_deviation": max_deviation,
        "total_attempted_marks": attempted_marks_pool,
        "total_offered_marks": total_offered_marks,
    }

    return {
        "slots": final_slots,
        "report": report,
        "warnings": warnings,
    }


def get_50_mark_blueprint(cos: list[str] = None) -> list[dict[str, Any]]:
    """Return standard 50-mark exam paper blueprint aligned with Revised Bloom's Taxonomy.

    Structure: 5 units x 10 marks = 50 marks across Factual, Conceptual, Procedural, Metacognitive dimensions.
    """
    if not cos:
        cos = [f"CO{i}" for i in range(1, config.MIN_COS + 1)]

    num_cos = len(cos)

    def co_at(index: int) -> str:
        return cos[index % num_cos]

    slots = [
        # Unit 1: 10 marks
        {
            "unit": 1,
            "question_no": 1,
            "sub_index": 1,
            "id": "1(a)",
            "co": co_at(0),
            "level": "L1",
            "knowledge_dimension": "Factual",
            "marks": 5,
            "desc": "Short Answer (Terminology / Factual Recall)",
        },
        {
            "unit": 1,
            "question_no": 1,
            "sub_index": 2,
            "id": "1(b)",
            "co": co_at(1),
            "level": "L2",
            "knowledge_dimension": "Conceptual",
            "marks": 5,
            "desc": "Explanation & Principles",
        },
        # Unit 2: 10 marks
        {
            "unit": 2,
            "question_no": 2,
            "sub_index": 1,
            "id": "2(a)",
            "co": co_at(1),
            "level": "L2",
            "knowledge_dimension": "Conceptual",
            "marks": 5,
            "desc": "Concept Modeling",
        },
        {
            "unit": 2,
            "question_no": 2,
            "sub_index": 2,
            "id": "2(b)",
            "co": co_at(2 if num_cos > 2 else 0),
            "level": "L3",
            "knowledge_dimension": "Procedural",
            "marks": 5,
            "desc": "Procedure Application & Problem Solving",
        },
        # Unit 3: 10 marks
        {
            "unit": 3,
            "question_no": 3,
            "sub_index": 1,
            "id": "3(a)",
            "co": co_at(2 if num_cos > 2 else 0),
            "level": "L3",
            "knowledge_dimension": "Procedural",
            "marks": 5,
            "desc": "Analytical Execution",
        },
        {
            "unit": 3,
            "question_no": 3,
            "sub_index": 2,
            "id": "3(b)",
            "co": co_at(3 if num_cos > 3 else 1),
            "level": "L4",
            "knowledge_dimension": "Conceptual",
            "marks": 5,
            "desc": "Architectural Analysis & Differentiation",
        },
        # Unit 4: 10 marks
        {
            "unit": 4,
            "question_no": 4,
            "sub_index": 1,
            "id": "4(a)",
            "co": co_at(0),
            "level": "L4",
            "knowledge_dimension": "Metacognitive",
            "marks": 5,
            "desc": "Trade-off Assessment",
        },
        {
            "unit": 4,
            "question_no": 4,
            "sub_index": 2,
            "id": "4(b)",
            "co": co_at(1),
            "level": "L5",
            "knowledge_dimension": "Procedural",
            "marks": 5,
            "desc": "Algorithm Efficiency Judgement",
        },
        # Unit 5: 10 marks
        {
            "unit": 5,
            "question_no": 5,
            "sub_index": 1,
            "id": "5(a)",
            "co": co_at(2 if num_cos > 2 else 0),
            "level": "L5",
            "knowledge_dimension": "Metacognitive",
            "marks": 5,
            "desc": "Methodological Reflection & Evaluation",
        },
        {
            "unit": 5,
            "question_no": 5,
            "sub_index": 2,
            "id": "5(b)",
            "co": co_at(3 if num_cos > 3 else 1),
            "level": "L6",
            "knowledge_dimension": "Procedural",
            "marks": 5,
            "desc": "System & Workflow Design (Creation)",
        },
    ]
    return slots


def get_25_mark_blueprint(cos: list[str] = None) -> list[dict[str, Any]]:
    """Return 25-mark internal assessment blueprint across units."""
    if not cos:
        cos = [f"CO{i}" for i in range(1, config.MIN_COS + 1)]

    num_cos = len(cos)

    def co_at(index: int) -> str:
        return cos[index % num_cos]

    return [
        {"unit": 1, "id": "1", "co": co_at(0), "level": "L1", "knowledge_dimension": "Factual", "marks": 5},
        {"unit": 1, "id": "2", "co": co_at(1), "level": "L2", "knowledge_dimension": "Conceptual", "marks": 5},
        {"unit": 2, "id": "3(a)", "co": co_at(2 if num_cos > 2 else 0), "level": "L3", "knowledge_dimension": "Procedural", "marks": 5},
        {"unit": 2, "id": "3(b)", "co": co_at(3 if num_cos > 3 else 1), "level": "L4", "knowledge_dimension": "Conceptual", "marks": 5},
        {"unit": 3, "id": "4", "co": co_at(0), "level": "L5", "knowledge_dimension": "Metacognitive", "marks": 5},
    ]


def calculate_blueprint_stats(slots: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze blueprint slots for total marks, unit distribution, BT spread, KD spread, and CO coverage."""
    from totals import _filter_attempted_questions

    is_choice = any(bool(s.get("alternative_group")) for s in slots)
    attempted_slots = _filter_attempted_questions(slots) if is_choice else slots

    offered_marks = sum(int(s.get("marks", 0)) for s in slots)
    attempted_marks = sum(int(s.get("marks", 0)) for s in attempted_slots)
    base_marks = attempted_marks if is_choice and attempted_marks > 0 else (offered_marks or 1)

    unit_marks: dict[int, int] = {}
    unit_attempted_marks: dict[int, int] = {}
    bt_marks: dict[str, int] = {}
    bt_counts: dict[str, int] = {}
    kd_marks: dict[str, int] = {kd: 0 for kd in KNOWLEDGE_DIMENSIONS}
    kd_counts: dict[str, int] = {kd: 0 for kd in KNOWLEDGE_DIMENSIONS}
    co_marks: dict[str, int] = {}
    co_counts: dict[str, int] = {}

    for s in slots:
        m = int(s.get("marks", 0))
        u = int(s.get("unit", 1))
        unit_marks[u] = unit_marks.get(u, 0) + m

    for s in attempted_slots:
        m = int(s.get("marks", 0))
        u = int(s.get("unit", 1))
        unit_attempted_marks[u] = unit_attempted_marks.get(u, 0) + m

        lvl = s.get("level", "L1").upper()
        bt_marks[lvl] = bt_marks.get(lvl, 0) + m
        bt_counts[lvl] = bt_counts.get(lvl, 0) + 1

        kd = normalize_knowledge_dimension(s.get("knowledge_dimension") or s.get("kd"))
        kd_marks[kd] = kd_marks.get(kd, 0) + m
        kd_counts[kd] = kd_counts.get(kd, 0) + 1

        co = s.get("co", "CO1")
        co_marks[co] = co_marks.get(co, 0) + m
        co_counts[co] = co_counts.get(co, 0) + 1

    bt_percent = {
        lvl: round((marks / base_marks * 100), 1)
        for lvl, marks in bt_marks.items()
    }

    kd_percent = {
        kd: round((marks / base_marks * 100), 1)
        for kd, marks in kd_marks.items()
    }

    matrix_2d = compute_2d_taxonomy_matrix(attempted_slots)

    return {
        "total_marks": attempted_marks if is_choice else offered_marks,
        "attempted_marks": attempted_marks,
        "offered_marks": offered_marks,
        "is_choice": is_choice,
        "total_questions": len(slots),
        "unit_marks": unit_marks,
        "unit_attempted_marks": unit_attempted_marks,
        "part_marks": {f"Unit {u}": m for u, m in unit_marks.items()},
        "bt_marks": bt_marks,
        "bt_counts": bt_counts,
        "bt_percent": bt_percent,
        "kd_marks": kd_marks,
        "kd_counts": kd_counts,
        "kd_percent": kd_percent,
        "matrix_2d": matrix_2d,
        "co_marks": co_marks,
        "co_counts": co_counts,
        "is_standard_50": (attempted_marks if is_choice else offered_marks) == 50,
    }
