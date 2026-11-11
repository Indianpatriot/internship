"""Offline stand-in for the LLM (tests / demo without Ollama)."""
import re
from typing import Optional
from generation.ollama_client import OllamaTimeoutError, OllamaConnectionError


def _extract_concept(ctx: str) -> str:
    """Extract a clean, substantive concept from a chunk, filtering out conversational preambles."""
    raw_sentences = [s.strip() for s in re.split(r"[\.\n]+", ctx) if s.strip()]
    candidates = []
    for s in raw_sentences:
        # Skip sentences that are purely preamble or visual references
        if re.match(
            r"^(look at|see figure|see table|in this (chapter|tutorial|section)|here is|the learners|this tutorial)",
            s,
            re.I,
        ) and len(s.split()) < 8:
            continue

        # Strip preamble phrases
        cleaned = re.sub(
            r"^(in this (?:chapter|section|book|volume)\s*(?:[0-9ivxlc]+)?(?:\s+of\s+[^,]+)?,\s*|"
            r"look at the following diagram\.\s*|look at\b.*?[\.:]\s*|"
            r"the learners of this tutorial are expected to know the basics of\s*|"
            r"this tutorial has introduced you to [^.]*\.\s*|"
            r"here is an example to test your current knowledge of [^:]*:\s*|"
            r"you may also use\s*|now,\s*you know that\s*|now,\s*|it shows\s*|it illustrates\s*)",
            "",
            s,
            flags=re.I,
        ).strip()

        # Remove leading conjunctions/articles if needed
        cleaned = re.sub(r"^(that|and|or|in addition,\s*|furthermore,\s*)\s*", "", cleaned, flags=re.I).strip()

        if len(cleaned.split()) >= 3:
            candidates.append(cleaned)

    if not candidates:
        candidates = raw_sentences or ["the core technical principles"]

    c = candidates[0].rstrip(".,:;!?")
    words = c.split()
    if len(words) > 12:
        c = " ".join(words[:12])

    c = re.sub(r"[\s,;:–—\-]+", " ", c).strip()
    c = re.sub(r"\b(in|of|to|for|on|at|with|by|from|and|or|that|as|the|a|an)\s*$", "", c, flags=re.I).strip()
    c = c.rstrip(".,:;!?")

    if len(c) > 1:
        c = c[0].lower() + c[1:]
    return c


def _format_question_with_verb(verb: str, concept: str) -> str:
    """Format a realistic, sensible exam question starting with the specified Bloom's verb."""
    v = verb.lower().strip()
    V = v.capitalize()

    if v == "define":
        return f"{V} the formal definition and technical scope of {concept}?"
    elif v == "state":
        return f"{V} the primary principle, rule, or theorem governing {concept}?"
    elif v == "recall":
        return f"{V} the foundational terms and memory models of {concept}?"
    elif v == "name":
        return f"{V} the core components and key attributes of {concept}?"
    elif v == "list":
        return f"{V} the main properties and prerequisites associated with {concept}?"
    elif v == "enumerate":
        return f"{V} the sequential phases and distinct elements of {concept}?"
    elif v == "identify":
        return f"{V} the critical factors and operational roles within {concept}?"
    elif v == "label":
        return f"{V} the functional parts and structural diagrams of {concept}?"
    elif v == "explain":
        return f"{V} how {concept} functions and behaves within the system?"
    elif v == "describe":
        return f"{V} the step-by-step process and detailed workflow of {concept}?"
    elif v == "discuss":
        return f"{V} the practical implications and trade-offs of using {concept}?"
    elif v == "clarify":
        return f"{V} the common misconceptions and distinct subtleties regarding {concept}?"
    elif v == "interpret":
        return f"{V} the meaning, trends, and significance revealed by {concept}?"
    elif v == "classify":
        return f"{V} the distinct taxonomy groups and types related to {concept}?"
    elif v == "categorize":
        return f"{V} the diverse approaches, tiers, and styles of {concept}?"
    elif v == "distinguish":
        return f"{V} the boundaries and contrasting properties separating {concept}?"
    elif v == "differentiate":
        return f"{V} between the specialized variants and regular forms of {concept}?"
    elif v in {"apply", "use"}:
        return f"{V} practical solutions and real-world techniques to solve {concept}?"
    elif v in {"demonstrate", "illustrate"}:
        return f"{V} an end-to-end working example showing {concept} in action?"
    elif v in {"implement", "execute"}:
        return f"{V} a complete algorithmic procedure realizing {concept}?"
    elif v in {"respond", "provide"}:
        return f"{V} appropriate design responses and valid strategies for {concept}?"
    elif v in {"solve", "calculate"}:
        return f"{V} quantitative equations and analytical problems involving {concept}?"
    elif v in {"compute", "determine"}:
        return f"{V} numerical bounds and computational metrics for {concept}?"
    elif v == "analyze":
        return f"{V} the architectural trade-offs and structural behavior of {concept}?"
    elif v == "deconstruct":
        return f"{V} the internal logic, module boundaries, and mechanisms of {concept}?"
    elif v == "examine":
        return f"{V} the underlying relationships and execution patterns inside {concept}?"
    elif v in {"investigate", "integrate"}:
        return f"{V} the composite behavior and integration points of {concept}?"
    elif v in {"compare", "contrast"}:
        return f"{V} the operational advantages, complexity, and differences in {concept}?"
    elif v == "evaluate":
        return f"{V} the overall performance, scalability, and efficiency of {concept}?"
    elif v == "assess":
        return f"{V} the feasibility, risk profile, and system impact of {concept}?"
    elif v == "justify":
        return f"{V} the engineering choices and theoretical rationale supporting {concept}?"
    elif v == "critique":
        return f"{V} the architectural flaws, edge cases, and limitations of {concept}?"
    elif v == "appraise":
        return f"{V} the industrial utility, reliability, and business value of {concept}?"
    elif v in {"check", "reflect", "judge"}:
        return f"{V} the correctness, quality benchmarks, and ethical facets of {concept}?"
    elif v in {"design", "propose"}:
        return f"{V} an innovative architecture and blueprint utilizing {concept}?"
    elif v in {"formulate", "develop"}:
        return f"{V} a rigorous methodology and structured model based on {concept}?"
    elif v in {"construct", "assemble"}:
        return f"{V} a robust modular pipeline integrating {concept} seamlessly?"
    elif v in {"generate", "create"}:
        return f"{V} novel test scenarios and synthetic artifacts reflecting {concept}?"
    elif v in {"predict", "infer", "forecast"}:
        return f"{V} the likely outcomes and behavioral effects stemming from {concept}?"
    elif v in {"relate", "connect", "associate"}:
        return f"{V} the interconnections, context, and dependencies linking {concept}?"
    elif v in {"show", "display", "present"}:
        return f"{V} clear evidence and practical manifestations exemplifying {concept}?"
    elif v in {"recommend", "advise", "suggest"}:
        return f"{V} prioritized strategies and best practices regarding {concept}?"
    elif v in {"summarize", "outline"}:
        return f"{V} a concise overview highlighting the essential tenets of {concept}?"
    else:
        return f"{V} the underlying mechanisms and significance of {concept}?"


class MockClient:
    model = "mock"

    def __init__(self, fail_mode: Optional[str] = None, fail_pattern: Optional[str] = None):
        self.fail_mode = fail_mode
        self.fail_pattern = fail_pattern
        self.call_history: list[dict] = []

    def is_up(self):
        return self.fail_mode != "connection"

    def generate(self, prompt, temperature=None, num_predict=120, format=None, **kwargs):
        # 1. Failure-injection modes
        if self.fail_pattern and self.fail_pattern in prompt:
            raise RuntimeError(f"Simulated failure for pattern '{self.fail_pattern}'")

        if self.fail_mode == "timeout":
            raise OllamaTimeoutError("Simulated Ollama request timeout")
        elif self.fail_mode == "connection":
            raise OllamaConnectionError("Simulated Ollama connection failure")
        elif self.fail_mode == "empty":
            return ""
        elif self.fail_mode == "bad_json":
            raise ValueError("JSON parse error: model returned invalid JSON")
        elif self.fail_mode == "ungrounded" and ("Model Answer:" in prompt or "authoritative model answer" in prompt):
            return (
                "Point 1 (3M): Quantum entanglement teleports subatomic particles through non-local wormholes. "
                "Point 2 (2M): Superconducting tachyons manipulate hyper-dimensional space-time manifolds."
            )
        elif self.fail_mode == "bloom_verb_fail" and not ("Model Answer:" in prompt or "authoritative model answer" in prompt):
            return "Predict the cosmological future evolution of quantum entanglement?"

        # Marking scheme prompt (strict JSON format)
        if "MARKING SCHEME in strict JSON format" in prompt or "Now generate the marking scheme JSON for:" in prompt:
            return self._generate_marking_scheme(prompt)

        # Unit-to-chapter mapping prompt
        if "map each unmapped syllabus unit" in prompt.lower() or "available textbook chapters" in prompt.lower():
            return self._generate_unit_mapping(prompt)

        # Answer generation prompt
        if "Model Answer:" in prompt or "authoritative model answer" in prompt:
            return self._generate_answer(prompt)

        # Question generation prompt
        verbs_match = re.search(r"verbs: ([^\n]+)", prompt)
        verbs = verbs_match.group(1).split(", ") if verbs_match else ["Explain"]
        ctx_match = re.search(r'"""\n(.*?)\n"""', prompt, re.S)
        ctx = ctx_match.group(1) if ctx_match else prompt

        concept = _extract_concept(ctx)
        selected_verb = verbs[0].strip()
        if "multi-part recall" in prompt.lower():
            return f"List and state the purpose of four core components and operational features of {concept}?"
        return _format_question_with_verb(selected_verb, concept)

    def _generate_answer(self, prompt: str) -> str:
        ctx_match = re.search(r'"""\n(.*?)\n"""', prompt, re.S)
        ctx = ctx_match.group(1).strip() if ctx_match else prompt

        q_match = re.search(r"Question:\s*\n(.*?)(?:\n\s*Model Answer:|$)", prompt, re.S)
        question = q_match.group(1).strip() if q_match else ""

        m_match = re.search(r"(\d+)-mark", prompt) or re.search(r"for questions of (\d+) marks", prompt, re.I)
        marks = int(m_match.group(1)) if m_match else 5

        lvl_match = re.search(r"Bloom's\s*(L[1-6])", prompt) or re.search(r"L([1-6])\s*\(", prompt) or re.search(r"\b(L[1-6])\b", prompt)
        level = lvl_match.group(1) if lvl_match else "L2"
        if not level.startswith("L"):
            level = f"L{level}"

        raw_sents = [s.strip() for s in re.split(r"[\.\n]+", ctx) if len(s.strip().split()) >= 3]
        cleaned_sents = []
        for s in raw_sents:
            cleaned = re.sub(
                r"\bin this (?:chapter|section|book|volume)\s*(?:[0-9ivxlc]+)?(?:\s+of\s+[^,]+)?,\s*|"
                r"look at the following diagram\.\s*|look at\b.*?[\.:]\s*",
                "",
                s,
                flags=re.I,
            ).strip()
            if len(cleaned.split()) >= 3:
                cleaned_sents.append(cleaned)
        if cleaned_sents:
            raw_sents = cleaned_sents
        elif not raw_sents:
            raw_sents = ["The technical system operates according to foundational computational principles."]

        def _make_structured_answer(sents: list[str], target_marks: int, offset: int = 0) -> str:
            # Term question under 4 marks
            if target_marks <= 2:
                s1 = sents[offset % len(sents)]
                s2 = sents[(offset + 1) % len(sents)]
                return f"Point 1 (1M): {s1}.\nPoint 2 (1M): {s2}."

            # Structured Rubric for L5 (Evaluate)
            if level == "L5":
                if target_marks <= 5:
                    splits = [("Criteria", 2), ("Verdict", 1), ("Justification", 1), ("Trade-offs", 1)]
                elif target_marks <= 8:
                    rem = target_marks - 6
                    splits = [("Criteria", 2), ("Verdict", 2), ("Justification", 2 + rem), ("Trade-offs", 2)]
                else:
                    base = target_marks // 5
                    rem = target_marks % 5
                    splits = [
                        ("Criteria", base + (1 if rem > 0 else 0)),
                        ("Verdict", base + (1 if rem > 1 else 0)),
                        ("Justification", base + (1 if rem > 2 else 0)),
                        ("System Trade-offs", base + (1 if rem > 3 else 0)),
                        ("Operational Limitations", base),
                    ]
                pts = [f"{name} ({p_pts}M): {sents[(offset + i) % len(sents)]}." for i, (name, p_pts) in enumerate(splits)]
                return "\n".join(pts)

            # Structured Rubric for L6 (Create/Design)
            if level == "L6":
                if target_marks <= 5:
                    splits = [("Architecture / Plan Steps", 2), ("Rationale", 2), ("Trade-offs", 1)]
                elif target_marks <= 8:
                    rem = target_marks - 6
                    splits = [("Architecture / Plan Steps", 2), ("Implementation Steps", 2), ("Rationale", 2), ("Trade-offs", rem)]
                else:
                    base = target_marks // 5
                    rem = target_marks % 5
                    splits = [
                        ("Architectural Blueprint", base + (1 if rem > 0 else 0)),
                        ("Step-by-Step Implementation Plan", base + (1 if rem > 1 else 0)),
                        ("Component Interactions", base + (1 if rem > 2 else 0)),
                        ("Technical Rationale", base + (1 if rem > 3 else 0)),
                        ("Design Trade-offs", base),
                    ]
                pts = [f"{name} ({p_pts}M): {sents[(offset + i) % len(sents)]}." for i, (name, p_pts) in enumerate(splits)]
                return "\n".join(pts)

            # Standard levels (L1 - L4)
            num_points = max(2, (target_marks + 1) // 2)
            base_pts = target_marks // num_points
            remainder = target_marks % num_points

            # Check demand keywords in question to satisfy check_demand_coverage
            q_low = question.lower()
            demand_prefix = ""
            if "trade-off" in q_low or "tradeoff" in q_low:
                demand_prefix = "Trade-offs and advantages indicate that "
            elif "difference" in q_low or "distinguish" in q_low:
                demand_prefix = "Key operational differences reveal that "
            elif "step" in q_low:
                demand_prefix = "Step-by-step procedure: "
            elif "justify" in q_low:
                demand_prefix = "Justification based on system behavior: "

            pts = []
            for i in range(num_points):
                p_m = base_pts + (1 if i < remainder else 0)
                sent = sents[(offset + i) % len(sents)]
                prefix = demand_prefix if (i == 1 and demand_prefix) else ""
                pts.append(f"Point {i + 1} ({p_m}M): {prefix}{sent}.")
            return "\n".join(pts)

        primary_ans = _make_structured_answer(raw_sents, marks, offset=0)

        q_lower = question.lower()
        has_or = any(w in q_lower for w in ["compare", "contrast", "differentiate", "distinguish", "alternative", " or "]) or "or" in q_lower.split()

        if has_or and len(raw_sents) >= 2:
            alt_ans = _make_structured_answer(raw_sents, marks, offset=1)
            return f"{primary_ans}\nOR\n{alt_ans}"

        return primary_ans

    def _generate_unit_mapping(self, prompt: str) -> str:
        """Mock structured JSON response for unit-to-chapter mapping."""
        import json

        ch_matches = re.findall(
            r'- ID: "(C\d+)" \| Book: "([^"]+)" \| Chapter: "([^"]+)"(?: \| Overview: (.*?))?$',
            prompt,
            re.MULTILINE,
        )
        u_matches = re.findall(
            r'- Unit (\d+): Title: "([^"]+)"(?: \| Topics: (.*?))?$',
            prompt,
            re.MULTILINE,
        )

        mappings = {}
        for u_num_str, u_title, u_topics in u_matches:
            u_text = f"{u_title} {u_topics}".lower()
            u_words = set(re.findall(r"[a-z0-9]{3,}", u_text)) - {
                "unit", "the", "and", "for", "with", "from", "general", "content"
            }

            best_cid = None
            best_score = 0
            for cid, book, chap, overview in ch_matches:
                ch_text = f"{book} {chap} {overview}".lower()
                ch_words = set(re.findall(r"[a-z0-9]{3,}", ch_text))
                overlap = len(u_words & ch_words)
                if overlap > best_score:
                    best_score = overlap
                    best_cid = cid

            if best_cid and best_score >= 1:
                mappings[u_num_str] = {
                    "chapter_ids": [best_cid],
                    "confidence": 0.92,
                }
            else:
                mappings[u_num_str] = {
                    "chapter_ids": [],
                    "confidence": 0.10,
                }

        return json.dumps(mappings, indent=2)

    def _generate_marking_scheme(self, prompt: str) -> str:
        """Mock valid JSON response for marking scheme generation."""
        import json

        q_match = re.search(r"Question:\s*(.*?)(?:\n\s*Bloom's Level:|\n\s*Total Marks:|$)", prompt, re.S)
        question = q_match.group(1).strip() if q_match else ""

        lvl_match = re.search(r"Bloom's Level:\s*(L[1-6])", prompt, re.I) or re.search(r"\b(L[1-6])\b", prompt)
        level = lvl_match.group(1).upper() if lvl_match else "L2"
        if not level.startswith("L"):
            level = f"L{level}"

        m_match = re.search(r"Total Marks:\s*(\d+)", prompt, re.I) or re.search(r"sum EXACTLY to (\d+)", prompt, re.I)
        marks = int(m_match.group(1)) if m_match else 5

        # Check diagram demand: include diagram criterion ONLY if question contains draw/diagram
        has_diagram = bool(
            re.search(r"\b(draw|diagram|flowchart|illustrate|schematic|sketch)\b", question, re.I)
        )

        diag_m = 0
        if has_diagram and marks >= 4:
            diag_m = max(2, min(int(round(0.25 * marks)), 4))
        rem_m = marks - diag_m

        criteria_specs: list[dict[str, Any]] = []

        if level == "L1":
            if rem_m <= 2:
                criteria_specs.append({
                    "criterion": "Definition and core terminology",
                    "kind": "definition",
                    "marks": rem_m,
                    "reason": "Direct recall of definition and core terminology required for L1.",
                    "expected": "Accurate formal definition and technical terms.",
                    "partial_credit": None,
                    "alternatives": ["Accept standard textbook definition or equivalent terminology."],
                })
            else:
                def_m = 2 if rem_m >= 5 else 1
                pts_m = rem_m - def_m
                criteria_specs.append({
                    "criterion": "Definition and terminology",
                    "kind": "definition",
                    "marks": def_m,
                    "reason": "Foundational recall of definition; accurate terminology required.",
                    "expected": "Formal definition of technical concept and primary scope.",
                    "partial_credit": "1 mark for partial definition.",
                    "alternatives": [],
                })
                criteria_specs.append({
                    "criterion": "Enumerated key facts and properties",
                    "kind": "points",
                    "marks": pts_m,
                    "reason": f"{pts_m} distinct checkable facts or properties, 1 mark each.",
                    "expected": f"List of {pts_m} specific technical properties or core components.",
                    "partial_credit": "1 mark per distinct checkable fact correctly listed.",
                    "alternatives": ["Accept valid alternative properties described in textbook."],
                })

        elif level == "L2":
            if rem_m <= 3:
                criteria_specs.append({
                    "criterion": "Conceptual explanation",
                    "kind": "explanation",
                    "marks": rem_m,
                    "reason": "Demonstrates clear conceptual understanding and system behavior.",
                    "expected": "Clear explanation of system principles and operational behavior.",
                    "partial_credit": "1 mark for basic understanding without complete explanation.",
                    "alternatives": ["Accept alternative operational phrasing or equivalent explanation."],
                })
            elif rem_m <= 5:
                criteria_specs.append({
                    "criterion": "Concept definition and overview",
                    "kind": "definition",
                    "marks": 2,
                    "reason": "L2 questions start with accurate conceptual definition.",
                    "expected": "Clear definition of core architecture and primary function.",
                    "partial_credit": "1 mark for partial definition.",
                    "alternatives": [],
                })
                criteria_specs.append({
                    "criterion": "Mechanism and operational role",
                    "kind": "explanation",
                    "marks": rem_m - 2,
                    "reason": "Technical explanation of system behavior and components.",
                    "expected": "Detailed explanation of component interactions and runtime behavior.",
                    "partial_credit": "1 mark per key operational aspect explained.",
                    "alternatives": ["Alternative functional description using system APIs."],
                })
            else:
                def_m = 2
                ex_m = max(2, rem_m - 4)
                app_m = rem_m - def_m - ex_m
                criteria_specs.append({
                    "criterion": "Definition and architectural overview",
                    "kind": "definition",
                    "marks": def_m,
                    "reason": "Foundational conceptual recall and definition.",
                    "expected": "Accurate formal definition of core architecture and purpose.",
                    "partial_credit": "1 mark for partial definition.",
                    "alternatives": [],
                })
                criteria_specs.append({
                    "criterion": "Detailed explanation of components and mechanisms",
                    "kind": "explanation",
                    "marks": ex_m,
                    "reason": "Comprehensive operational breakdown of internal mechanism.",
                    "expected": "Identifies key components and explains runtime execution flow.",
                    "partial_credit": "1-2 marks for partial explanation.",
                    "alternatives": ["Alternative component model with equivalent functional responsibilities."],
                })
                criteria_specs.append({
                    "criterion": "Practical example or application context",
                    "kind": "example",
                    "marks": app_m,
                    "reason": "Demonstrates practical comprehension in realistic system scenario.",
                    "expected": "Concrete practical example or realistic deployment scenario.",
                    "partial_credit": "1 mark for partial example.",
                    "alternatives": [],
                })

        elif level == "L3":
            if rem_m <= 3:
                criteria_specs.append({
                    "criterion": "Application procedure and execution",
                    "kind": "steps",
                    "marks": rem_m,
                    "reason": "Accurate application of procedure to solve the given problem.",
                    "expected": "Correct sequence of operational steps and technical commands.",
                    "partial_credit": "1 mark per correct step.",
                    "alternatives": ["Alternative valid execution commands."],
                })
            else:
                step_m = max(2, int(round(0.6 * rem_m)))
                res_m = rem_m - step_m
                criteria_specs.append({
                    "criterion": "Procedural steps and commands",
                    "kind": "steps",
                    "marks": step_m,
                    "reason": "Correct step-by-step technical methodology and execution commands.",
                    "expected": "Correct sequence of configuration steps and execution commands.",
                    "partial_credit": "1 mark per correct command or procedural step.",
                    "alternatives": ["Alternative command flags or script implementation."],
                })
                criteria_specs.append({
                    "criterion": "Verification and expected result",
                    "kind": "explanation",
                    "marks": res_m,
                    "reason": "Validation of applied result and operational verification.",
                    "expected": "Expected console output, verification check, or system state change.",
                    "partial_credit": "1 mark for partial verification.",
                    "alternatives": [],
                })

        elif level == "L4":
            p1 = max(1, rem_m // 2)
            p2 = rem_m - p1
            criteria_specs.append({
                "criterion": "Component and architectural analysis",
                "kind": "analysis",
                "marks": p1,
                "reason": "Analytical decomposition of distinct architectural elements.",
                "expected": "In-depth decomposition of component roles and subsystem boundaries.",
                "partial_credit": "1 mark per analyzed component.",
                "alternatives": [],
            })
            criteria_specs.append({
                "criterion": "Comparative distinction and interaction trade-offs",
                "kind": "comparison",
                "marks": p2,
                "reason": "Examination of relational trade-offs and structural differences.",
                "expected": "Contrasting trade-offs, performance implications, and limitations.",
                "partial_credit": "1 mark per substantiated comparison point.",
                "alternatives": ["Alternative comparative evaluation metrics."],
            })

        else:
            # L5 or L6
            if rem_m <= 4:
                p1 = max(1, rem_m // 2)
                p2 = rem_m - p1
                criteria_specs.append({
                    "criterion": "Evaluation criteria and benchmarks",
                    "kind": "justification",
                    "marks": p1,
                    "reason": "Technical benchmarks and design standards grounded in context.",
                    "expected": "Defines quantitative or qualitative technical benchmarks.",
                    "partial_credit": "1 mark for basic criteria identification.",
                    "alternatives": [],
                })
                criteria_specs.append({
                    "criterion": "Trade-off analysis and final verdict",
                    "kind": "conclusion",
                    "marks": p2,
                    "reason": "Evidence-based technical justification and final recommendation.",
                    "expected": "Definitive technical verdict supported by contextual evidence.",
                    "partial_credit": "1 mark for partial justification.",
                    "alternatives": ["Alternative justified recommendation based on different constraints."],
                })
            else:
                p1 = max(1, int(round(0.4 * rem_m)))
                p2 = max(1, int(round(0.4 * rem_m)))
                p3 = rem_m - p1 - p2
                if p3 <= 0:
                    p1 = max(1, rem_m // 2)
                    p2 = rem_m - p1
                    criteria_specs.append({
                        "criterion": "Evaluation standards and technical criteria",
                        "kind": "justification",
                        "marks": p1,
                        "reason": "Technical benchmarks and evaluation criteria.",
                        "expected": "Defines concrete evaluation metrics and design goals.",
                        "partial_credit": "1 mark per criterion.",
                        "alternatives": [],
                    })
                    criteria_specs.append({
                        "criterion": "Technical justification and verdict",
                        "kind": "conclusion",
                        "marks": p2,
                        "reason": "Evidence-based justification and final conclusion.",
                        "expected": "Synthesized evaluative verdict with supporting evidence.",
                        "partial_credit": "1 mark for partial verdict.",
                        "alternatives": ["Alternative justified verdict with trade-off analysis."],
                    })
                else:
                    criteria_specs.append({
                        "criterion": "Architectural / evaluation criteria",
                        "kind": "justification",
                        "marks": p1,
                        "reason": "Technical benchmarks and evaluation criteria.",
                        "expected": "Defines architectural standards and operational requirements.",
                        "partial_credit": "1 mark per criterion.",
                        "alternatives": [],
                    })
                    criteria_specs.append({
                        "criterion": "System trade-offs and technical justification",
                        "kind": "analysis",
                        "marks": p2,
                        "reason": "Detailed trade-off analysis and system boundary constraints.",
                        "expected": "In-depth trade-off analysis addressing latency, memory, or complexity.",
                        "partial_credit": "1 mark per trade-off analyzed.",
                        "alternatives": ["Alternative trade-off analysis focusing on scalability."],
                    })
                    criteria_specs.append({
                        "criterion": "Synthesized conclusion / design recommendation",
                        "kind": "conclusion",
                        "marks": p3,
                        "reason": "Definitive technical recommendation or synthesized conclusion.",
                        "expected": "Synthesized actionable design recommendation or verdict.",
                        "partial_credit": "1 mark for partial conclusion.",
                        "alternatives": [],
                    })

        # Append diagram criterion if question requires drawing
        if diag_m > 0:
            struct_m = max(1, diag_m // 2)
            label_m = diag_m - struct_m
            criteria_specs.append({
                "criterion": "Architecture diagram and labelling",
                "kind": "diagram",
                "marks": diag_m,
                "reason": (
                    f"Diagram is required by the question and rewards correct structure/layout ({struct_m}M) "
                    f"and labelling accuracy ({label_m}M)."
                ),
                "expected": "Diagram with labelled components and arrows showing the flow.",
                "partial_credit": f"{struct_m} mark for correct layout without complete labels.",
                "alternatives": [],
            })

        criteria_list = []
        for i, spec in enumerate(criteria_specs, 1):
            criteria_list.append({
                "no": i,
                "criterion": spec["criterion"],
                "expected": spec["expected"],
                "marks": spec["marks"],
                "kind": spec["kind"],
                "reason": spec["reason"],
                "partial_credit": spec.get("partial_credit"),
                "alternatives": spec.get("alternatives", []),
            })

        scheme_data = {
            "total_marks": marks,
            "criteria": criteria_list,
            "notes": "Examiner guidance: evaluate technical accuracy and key terminology.",
        }
        return json.dumps(scheme_data, indent=2)

