"""Marking scheme generator module.

Generates examiner-checkable marking scheme criteria breakdowns with marks,
pedagogical reasons, and expected answers, backed by a deterministic rule-based
fallback mechanism.
"""
import json
import logging
import random
import re
from typing import Any, Optional, Union

from blooms.verb_bank import normalize_level
from contracts import MarkingScheme, SchemeCriterion, VALID_CRITERION_KINDS, validate_marking_scheme
from generation.prompts import build_scheme_prompt

logger = logging.getLogger(__name__)

DIAGRAM_KEYWORDS_RE = re.compile(
    r"\b(draw|diagram|flowchart|illustrate|schematic|sketch|block diagram|circuit diagram|architectural diagram)\b",
    re.I,
)


def extract_json_object(raw_text: str) -> Optional[dict[str, Any]]:
    """Robustly extract and parse a JSON dictionary from LLM output.

    Strips code fences, removes trailing commas, and extracts the first
    balanced curly brace object.
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()

    # 1. Strip markdown fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Find outermost balanced JSON object
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    depth = 0
    end_idx = -1
    in_string = False
    escape = False

    for i in range(start_idx, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

    if end_idx == -1:
        # Fallback: slice from first { to last }
        last_brace = text.rfind("}")
        if last_brace > start_idx:
            candidate = text[start_idx : last_brace + 1]
        else:
            return None
    else:
        candidate = text[start_idx:end_idx]

    # 3. Clean trailing commas before closing braces/brackets
    cleaned = re.sub(r",\s*([\}\]])", r"\1", candidate)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception as ex:
        logger.debug("Initial JSON parse failed: %s; attempting lenient regex repair", ex)

    # 4. Secondary repair: fix common formatting errors (unquoted keys or values)
    try:
        repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
        data = json.loads(repaired)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None


def repair_scheme_marks(
    scheme: dict[str, Any],
    expected_total: Union[int, float],
) -> bool:
    """Deterministically repair small discrepancy in criteria marks sum.

    Adjusts the largest non-diagram criterion so that criteria marks sum
    exactly to expected_total.
    """
    criteria = scheme.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        return False

    try:
        exp_m = float(expected_total)
    except (ValueError, TypeError):
        return False

    # Calculate current sum
    current_sum = 0.0
    for c in criteria:
        if isinstance(c, dict) and "marks" in c:
            try:
                current_sum += float(c["marks"])
            except (ValueError, TypeError):
                return False

    diff = exp_m - current_sum
    if abs(diff) < 1e-6:
        scheme["total_marks"] = int(exp_m) if exp_m.is_integer() else exp_m
        return True

    # Allow repair for discrepancies up to max(3, 40% of total)
    max_tolerated_diff = max(3.0, 0.4 * exp_m)
    if abs(diff) > max_tolerated_diff:
        return False

    # Prefer adjusting largest non-diagram criterion
    non_diagram_candidates = [
        c for c in criteria if isinstance(c, dict) and c.get("kind") != "diagram"
    ]
    candidates = non_diagram_candidates if non_diagram_candidates else criteria

    best_candidate = max(candidates, key=lambda c: float(c.get("marks", 0)))
    current_val = float(best_candidate.get("marks", 0))
    adjusted_val = current_val + diff

    if adjusted_val <= 0:
        return False

    final_m = int(adjusted_val) if adjusted_val.is_integer() else round(adjusted_val, 2)
    best_candidate["marks"] = final_m
    scheme["total_marks"] = int(exp_m) if exp_m.is_integer() else exp_m

    notes = scheme.get("notes") or ""
    adj_note = "Criteria marks adjusted to match question total."
    if adj_note not in notes:
        scheme["notes"] = f"{notes} {adj_note}".strip() if notes else adj_note

    ok, _ = validate_marking_scheme(scheme, expected_total)
    return ok


def build_fallback_scheme(
    question: str,
    level: str,
    marks: int,
    answer: str,
    chunk_text: str = "",
) -> MarkingScheme:
    """Build a deterministic, rule-based fallback marking scheme.

    Constructs criteria matching the question's Bloom's taxonomy level, marks,
    and diagram requirements, populated with content from the model answer.
    Guaranteed to sum exactly to marks and have non-empty reasons and expected text.
    """
    m = marks if marks is not None and marks > 0 else 5
    norm_level = normalize_level(level) if level else "L2"
    has_diagram = bool(DIAGRAM_KEYWORDS_RE.search(question or ""))

    # Extract clean substantive sentences from answer or chunk
    source_corpus = (answer or "") + " " + (chunk_text or "")
    raw_sentences = [
        s.strip()
        for s in re.split(r"(?:\n+|(?<=[.!?])\s+)", source_corpus)
        if s.strip()
    ]
    clean_sentences = []
    for s in raw_sentences:
        cleaned = re.sub(
            r"^(?:Point\s*\d+\s*(?:\([^)]*\))?:|Criteria\s*(?:\([^)]*\))?:|Verdict\s*(?:\([^)]*\))?:|"
            r"Justification\s*(?:\([^)]*\))?:|Trade-offs\s*(?:\([^)]*\))?:|Model Answer:|"
            r"Alternative approach\s*\d*:)\s*",
            "",
            s,
            flags=re.I,
        ).strip()
        if len(cleaned.split()) >= 3 and not re.match(r"^[—\-*•]", cleaned):
            clean_sentences.append(cleaned.rstrip(".,;:"))

    if not clean_sentences:
        clean_sentences = [
            f"Accurate and complete technical explanation of {question.strip().rstrip('?')}",
            "Identification of core system components and operational mechanisms",
            "Verification of operational steps, execution behavior, and expected results",
        ]

    # Decide diagram marks if question asks for drawing or diagram
    diagram_marks = 0
    if has_diagram and m >= 4:
        diagram_marks = max(2, min(int(round(0.25 * m)), 4))
    remaining_marks = m - diagram_marks

    criteria_specs: list[dict[str, Any]] = []

    if norm_level == "L1":
        # Recall / Remember
        if remaining_marks <= 2:
            criteria_specs.append({
                "criterion": "Definition and core terminology",
                "kind": "definition",
                "marks": remaining_marks,
                "reason": "Direct recall of definition and core terminology required for L1 recall.",
                "partial_credit": None,
            })
        else:
            def_m = 2 if remaining_marks >= 5 else 1
            pts_m = remaining_marks - def_m
            criteria_specs.append({
                "criterion": "Definition and terminology",
                "kind": "definition",
                "marks": def_m,
                "reason": "Foundational recall of definition; accurate terminology required.",
                "partial_credit": "1 mark for partial definition" if def_m > 1 else None,
            })
            criteria_specs.append({
                "criterion": "Enumerated key facts and properties",
                "kind": "points",
                "marks": pts_m,
                "reason": f"{pts_m} distinct checkable facts or properties, 1 mark each.",
                "partial_credit": "1 mark per distinct checkable fact correctly listed.",
            })

    elif norm_level == "L2":
        # Understand / Explain
        if remaining_marks <= 3:
            criteria_specs.append({
                "criterion": "Conceptual explanation",
                "kind": "explanation",
                "marks": remaining_marks,
                "reason": "Demonstrates clear conceptual understanding and system behavior.",
                "partial_credit": "1 mark for basic understanding without complete explanation.",
            })
        elif remaining_marks <= 5:
            criteria_specs.append({
                "criterion": "Concept definition and overview",
                "kind": "definition",
                "marks": 2,
                "reason": "L2 questions start with accurate conceptual definition.",
                "partial_credit": "1 mark for partial definition.",
            })
            criteria_specs.append({
                "criterion": "Mechanism and operational role",
                "kind": "explanation",
                "marks": remaining_marks - 2,
                "reason": "Technical explanation of system behavior and components.",
                "partial_credit": "1 mark per key operational aspect explained.",
            })
        else:
            def_m = 2
            ex_m = max(2, remaining_marks - 4)
            app_m = remaining_marks - def_m - ex_m
            criteria_specs.append({
                "criterion": "Definition and architectural overview",
                "kind": "definition",
                "marks": def_m,
                "reason": "Foundational conceptual recall and definition.",
                "partial_credit": "1 mark for partial definition.",
            })
            criteria_specs.append({
                "criterion": "Detailed explanation of components and mechanisms",
                "kind": "explanation",
                "marks": ex_m,
                "reason": "Comprehensive operational breakdown of internal mechanism.",
                "partial_credit": "1-2 marks for partial explanation.",
            })
            criteria_specs.append({
                "criterion": "Practical example or application context",
                "kind": "example",
                "marks": app_m,
                "reason": "Demonstrates practical comprehension in realistic system scenario.",
                "partial_credit": "1 mark for partial example.",
            })

    elif norm_level == "L3":
        # Apply / Solve
        if remaining_marks <= 3:
            criteria_specs.append({
                "criterion": "Application steps and implementation",
                "kind": "steps",
                "marks": remaining_marks,
                "reason": "Accurate application of procedure to solve the given problem.",
                "partial_credit": "1 mark per correct step.",
            })
        else:
            step_m = max(2, int(round(0.6 * remaining_marks)))
            res_m = remaining_marks - step_m
            criteria_specs.append({
                "criterion": "Procedural steps and commands",
                "kind": "steps",
                "marks": step_m,
                "reason": "Correct step-by-step technical methodology and execution commands.",
                "partial_credit": "1 mark per correct step or command.",
            })
            criteria_specs.append({
                "criterion": "Verification and expected result",
                "kind": "explanation",
                "marks": res_m,
                "reason": "Validation of applied result and operational verification.",
                "partial_credit": "1 mark for partial verification.",
            })

    elif norm_level == "L4":
        # Analyze / Compare
        p1 = max(1, remaining_marks // 2)
        p2 = remaining_marks - p1
        criteria_specs.append({
            "criterion": "Component and architectural analysis",
            "kind": "analysis",
            "marks": p1,
            "reason": "Analytical decomposition of distinct architectural elements.",
            "partial_credit": "1 mark per analyzed component.",
        })
        criteria_specs.append({
            "criterion": "Comparative distinction and interaction trade-offs",
            "kind": "comparison",
            "marks": p2,
            "reason": "Examination of relational trade-offs and structural differences.",
            "partial_credit": "1 mark per substantiated comparison point.",
        })

    else:
        # L5 (Evaluate) or L6 (Create)
        if remaining_marks <= 4:
            p1 = max(1, remaining_marks // 2)
            p2 = remaining_marks - p1
            criteria_specs.append({
                "criterion": "Evaluation criteria and benchmarks",
                "kind": "justification",
                "marks": p1,
                "reason": "Technical benchmarks and design standards grounded in context.",
                "partial_credit": "1 mark for basic criteria identification.",
            })
            criteria_specs.append({
                "criterion": "Trade-off analysis and final verdict",
                "kind": "conclusion",
                "marks": p2,
                "reason": "Evidence-based technical justification and final recommendation.",
                "partial_credit": "1 mark for partial justification.",
            })
        else:
            p1 = max(1, int(round(0.4 * remaining_marks)))
            p2 = max(1, int(round(0.4 * remaining_marks)))
            p3 = remaining_marks - p1 - p2
            if p3 <= 0:
                p1 = max(1, remaining_marks // 2)
                p2 = remaining_marks - p1
                criteria_specs.append({
                    "criterion": "Evaluation standards and technical criteria",
                    "kind": "justification",
                    "marks": p1,
                    "reason": "Technical benchmarks and evaluation criteria.",
                    "partial_credit": "1 mark per criterion.",
                })
                criteria_specs.append({
                    "criterion": "Technical justification and verdict",
                    "kind": "conclusion",
                    "marks": p2,
                    "reason": "Evidence-based justification and final conclusion.",
                    "partial_credit": "1 mark for partial verdict.",
                })
            else:
                criteria_specs.append({
                    "criterion": "Architectural / evaluation criteria",
                    "kind": "justification",
                    "marks": p1,
                    "reason": "Technical benchmarks and evaluation criteria.",
                    "partial_credit": "1 mark per criterion.",
                })
                criteria_specs.append({
                    "criterion": "System trade-offs and technical justification",
                    "kind": "analysis",
                    "marks": p2,
                    "reason": "Detailed trade-off analysis and system boundary constraints.",
                    "partial_credit": "1 mark per trade-off analyzed.",
                })
                criteria_specs.append({
                    "criterion": "Synthesized conclusion / design recommendation",
                    "kind": "conclusion",
                    "marks": p3,
                    "reason": "Definitive technical recommendation or synthesized conclusion.",
                    "partial_credit": "1 mark for partial conclusion.",
                })

    # Add diagram criterion if question requires drawing
    if diagram_marks > 0:
        struct_m = max(1, diagram_marks // 2)
        label_m = diagram_marks - struct_m
        criteria_specs.append({
            "criterion": "Diagram / Illustration and Labelling",
            "kind": "diagram",
            "marks": diagram_marks,
            "reason": (
                f"Diagram is required by the question and rewards correct structure/layout ({struct_m}M) "
                f"and labelling accuracy ({label_m}M)."
            ),
            "partial_credit": f"{struct_m} mark(s) for correct layout without complete labels.",
        })

    # Build final criteria with expected text grounded in answer sentences
    criteria: list[SchemeCriterion] = []
    sent_idx = 0
    for idx, spec in enumerate(criteria_specs, 1):
        if spec.get("kind") == "diagram":
            exp_text = (
                "Accurately drawn and labelled diagram illustrating the components, layout, "
                "and directional flow."
            )
        else:
            if clean_sentences:
                exp_text = clean_sentences[sent_idx % len(clean_sentences)]
                sent_idx += 1
            else:
                exp_text = f"Expected technical explanation for {spec['criterion']}."

        criterion_item: SchemeCriterion = {
            "no": idx,
            "criterion": spec["criterion"],
            "expected": exp_text,
            "marks": spec["marks"],
            "kind": spec["kind"],
            "reason": spec["reason"],
            "partial_credit": spec.get("partial_credit"),
            "alternatives": [],
        }
        criteria.append(criterion_item)

    fallback_scheme: MarkingScheme = {
        "total_marks": m,
        "criteria": criteria,
        "notes": "Auto-generated structure; review before use.",
        "is_fallback": True,
    }

    # Verify and guarantee exact total
    ok, _ = validate_marking_scheme(fallback_scheme, m)
    if not ok:
        repair_scheme_marks(fallback_scheme, m)

    return fallback_scheme


def generate_scheme(
    question_record: dict[str, Any],
    chunk: Optional[dict[str, Any]] = None,
    client: Any = None,
    rng: Optional[random.Random] = None,
    max_retries: int = 3,
    force_regenerate: bool = False,
) -> MarkingScheme:
    """Generate an itemized marking scheme for a QuestionRecord.

    Queries the client using build_scheme_prompt, parses and validates the JSON
    structure. If the output is malformed, invalid, or fails after retries,
    constructs a valid rule-based fallback scheme.
    """
    question = question_record.get("question", "")
    level = question_record.get("level") or question_record.get("bloom") or "L2"
    marks = question_record.get("marks", 5)
    answer = question_record.get("answer", "")
    chunk_text = (chunk or {}).get("text", "") if chunk else question_record.get("context_text", "")

    m = int(marks) if marks is not None and marks > 0 else 5

    # If client is unavailable or missing generate method, directly return fallback
    if client is None or not hasattr(client, "generate"):
        return build_fallback_scheme(question, level, m, answer, chunk_text)

    prompt = build_scheme_prompt(
        question=question,
        level=level,
        marks=m,
        answer=answer,
        context=chunk_text,
    )

    for attempt in range(max_retries):
        try:
            raw_response = client.generate(
                prompt,
                temperature=0.2,
                num_predict=450,
            )
            parsed = extract_json_object(raw_response)
            if parsed and isinstance(parsed, dict) and "criteria" in parsed:
                # First attempt direct validation
                ok, problems = validate_marking_scheme(parsed, m)
                if ok:
                    parsed["is_fallback"] = False
                    return parsed  # type: ignore

                # Attempt deterministic repair for small sum discrepancies
                if repair_scheme_marks(parsed, m):
                    parsed["is_fallback"] = False
                    return parsed  # type: ignore

                logger.debug(
                    "Scheme validation failed on attempt %d: %s",
                    attempt + 1,
                    "; ".join(problems),
                )
        except Exception as ex:
            logger.debug("Exception querying scheme client on attempt %d: %s", attempt + 1, ex)

    # All retries failed: return rule-based fallback
    logger.info("Using rule-based fallback marking scheme for question: %s", question[:60])
    return build_fallback_scheme(question, level, m, answer, chunk_text)
