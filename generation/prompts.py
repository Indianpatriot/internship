"""Step 3b - prompt templates (one base template + per-level hints + Revised Bloom's knowledge dimension hints + answer generation)."""
from typing import Optional

from blooms.revised_taxonomy import normalize_knowledge_dimension
from blooms.verb_bank import LEVEL_NAMES, normalize_level

BASE = """You are an expert exam question setter.
Using ONLY the context below, write exactly ONE high-quality, clear, sensible {name}-level (Bloom's {level}) exam question.
Assigned Knowledge Dimension: {kd_name}.
The question MUST start with one of these verbs: {verbs}.
Rules:
- Use only technical facts present in the context. Do not use outside knowledge. Never accept or build upon false premises.
- The question must be completely SELF-CONTAINED and answerable by a student who does NOT have the textbook or notes in front of them.
- NEVER refer to "in the context of", "the context", "the provided context", "the passage", "the text", "the book", "the author", "the chapter", "introductory chapter", "earlier chapter", "in this chapter", "in this book", "Chapter", "Section", "Figure", "Table", "Listing", "the listing", "the example", "the technique", "this technique", "above", "below", or "provided".
- Explicitly name the technical subject, tool, or system. NEVER ask about "the tools", "the approach", "the technique", or "this method" without explicitly naming the technology or system.
- If the question involves code, a command, or a script, you MUST include the exact code or command inline in the question text. NEVER say "the code snippet", "the following code", "the describe method", or "the script" without writing it out inline.
- For comparison/contrast questions, compare concepts from the same technical category that both appear in the context, and never formulate circular questions (do not compare a thing with itself or its direct definition).
- For questions under 8 marks, do NOT ask "Which option would you choose and why?" or request choosing between alternatives. Limit demands to at most 2 items for 4-5 marks.
- Do NOT use "Calculate" or "Compute" unless the context contains concrete numbers, formulas, or quantitative metrics.{used_topics_rule}
- {hint}{kd_hint}{marks}
- Output only the question text ending with '?'. No numbering, no preamble, no answer.

Context:
\"\"\"
{context}
\"\"\"

Question:"""

LEVEL_HINTS = {
    "L1": "Test direct recall of a term, definition or fact stated in the context.",
    "L2": "Ask the student to explain or describe a concept in their own words.",
    "L3": "Require applying a concept or procedure from the context to a concrete situation.",
    "L4": "Require breaking the content into parts, or comparing and contrasting ideas from it.",
    "L5": "Require a justified judgement, recommendation, or critique grounded in the context.",
    "L6": "Ask the student to design or propose something new built on ideas from the context.",
}

KNOWLEDGE_HINTS = {
    "Factual": " Focus on core definitions, basic terms, symbols, and factual elements from the text.",
    "Conceptual": " Focus on system relationships, categories, principles, theories, and architectural structure.",
    "Procedural": " Focus on algorithms, techniques, methods of inquiry, and step-by-step procedures.",
    "Metacognitive": " Focus on strategic evaluation, justifying choices between alternatives, and assessing trade-offs.",
}

MARKS_HINTS = {
    2: " Scope the question appropriately for 2 marks (keep it short; answerable in 2-3 sentences).",
    3: " Scope the question appropriately for 3 marks (ask for 2-3 key points or a short clear description).",
    4: " Scope the question appropriately for 4 marks (ask for 2-3 specific features, steps, or aspects).",
    5: " Scope the question appropriately for 5 marks (ask for 3 specific key points, components, or steps; answerable in one well-developed paragraph).",
    6: " Scope the question appropriately for 6 marks (ask for 3-4 distinct points, mechanisms, or trade-offs).",
    8: " Scope the question appropriately for 8 marks (ask for 4 distinct points, components, or in-depth analytical aspects).",
    10: " Scope the question appropriately for 10 marks (ask for 5 distinct points, comprehensive architectural breakdown, or detailed end-to-end steps; detailed multi-point answer).",
}


def build_prompt(
    context: str,
    level: str,
    verbs: list[str],
    marks: Optional[int] = None,
    knowledge_dimension: Optional[str] = None,
    used_topics: Optional[list[str]] = None,
) -> str:
    level = normalize_level(level)
    kd_name = normalize_knowledge_dimension(knowledge_dimension) if knowledge_dimension else "Conceptual"
    kd_hint = KNOWLEDGE_HINTS.get(kd_name, "")
    used_topics_rule = ""
    if used_topics:
        used_topics_rule = f"\n- Avoid the following topics already tested in this exam unit: {', '.join(used_topics)}."

    lvl_hint = LEVEL_HINTS[level]
    if level == "L1" and marks and marks >= 8:
        lvl_hint = " Require multi-part recall (e.g. 'List and state the purpose of four ...' or 'Name and describe five core components of ...'), NOT a single-word answer."

    return BASE.format(
        name=LEVEL_NAMES[level],
        level=level,
        kd_name=kd_name,
        verbs=", ".join(verbs),
        hint=lvl_hint,
        kd_hint=kd_hint,
        marks=MARKS_HINTS.get(marks, ""),
        used_topics_rule=used_topics_rule,
        context=context.strip(),
    )


ANSWER_BASE = """You are an expert exam evaluator and question paper author.
Provide an authoritative model answer and itemized evaluation scheme for the exam question below using ONLY the provided context.

Rules:
- Answer ONLY from the context. Do not use outside knowledge or extrapolate beyond the text. State only verified facts from the context; a question's false premise must not be accepted.
- For term recall questions (e.g. 'Identify the term...', 'Name the term...'), provide ONLY the short precise term/name without repeating the question stem.
- For questions of {marks} marks, format the model answer as itemized key points with allocated marks summing up EXACTLY to {marks} marks:
  Point 1 (<mark>M): <technical explanation>
  Point 2 (<mark>M): <technical explanation>
{rubric_rule}- Minimum number of distinct points: {min_points} points. Never provide a single unbroken block or single sentence for questions of 5 or more marks.
{strict_rule}- Provide ONE clear, correct, authoritative model answer. Do not apologize, do not comment on what the context lacks (e.g. NEVER write "the text does not mention" or "not explicitly stated").
- The model answer must be completely self-contained. NEVER write phrases such as "according to the provided context", "in the provided example", "as shown in Listing", "as described in the passage", "introductory chapter", "earlier chapter", "in this chapter", "in this book", "the author", or local file paths.
- Address all parts of the question directly (trade-offs, advantages, consequences, steps, or justification).
- Target answer length: approximately {target_words} words.
- If multiple valid approaches exist, label them clearly with "Alternative approach:". NEVER use "OR" or "― OR ―" inside the answer text.
- Output only the model answer text. No conversational preamble, no greetings, no markdown fences.

Context:
\"\"\"
{context}
\"\"\"

Question:
{question}

Model Answer:"""


def build_answer_prompt(
    question: str,
    context: str,
    marks: Optional[int] = None,
    level: Optional[str] = None,
    strict_grounding: bool = False,
) -> str:
    """Build prompt for generating a grounded model answer."""
    m = marks if marks is not None and marks > 0 else 5
    target_words = max(30, min(500, m * 25))
    min_points = max(2, (m + 1) // 2) if m >= 5 else 1

    rubric_rule = ""
    norm_lvl = normalize_level(level) if level else None
    if norm_lvl == "L5":
        rubric_rule = (
            f"- For this L5 (Evaluate) question, structure the marking scheme as an evaluation rubric summing to {m}M:\n"
            "  1. Criteria (<mark>M): Technical benchmarks or evaluation standards from the context.\n"
            "  2. Verdict (<mark>M): Clear evaluative judgment or assessment.\n"
            "  3. Justification (<mark>M): Technical evidence from the context justifying the verdict.\n"
            "  4. Trade-offs / Limitations (<mark>M): System constraints, edge cases, or counterarguments.\n"
        )
    elif norm_lvl == "L6":
        rubric_rule = (
            f"- For this L6 (Create/Design) question, structure the marking scheme as a design rubric summing to {m}M:\n"
            "  1. Architecture / Plan Steps (<mark>M): Step-by-step technical plan or design specification.\n"
            "  2. Rationale (<mark>M): Technical reasons and justification for the proposed architecture.\n"
            "  3. Trade-offs / Limitations (<mark>M): Design trade-offs, edge cases, or boundary conditions.\n"
        )

    strict_rule = (
        "- STRICT GROUNDING: Answer strictly and only from the provided context. Do not invent facts, claims, or tools. "
        "Every statement must be directly verifiable in the context.\n"
        if strict_grounding
        else ""
    )
    return ANSWER_BASE.format(
        target_words=target_words,
        marks=m,
        min_points=min_points,
        rubric_rule=rubric_rule,
        strict_rule=strict_rule,
        context=context.strip(),
        question=question.strip(),
    )


SCHEME_BASE = """You are an expert exam evaluator creating a detailed, examiner-checkable MARKING SCHEME in strict JSON format.
Using the question, Bloom's Taxonomy cognitive level, marks, and model answer below, break down the evaluation into specific checkable criteria with allocated marks.

Allocation guidelines:
- Total criteria marks must sum EXACTLY to {marks}.
- More marks means more independent points, not longer ones. Prefer 1 mark per distinct checkable point; a typical criterion is 1-3 marks; avoid any single criterion above 40% of the total unless the question is a single procedure.
- If the question asks to draw, illustrate, represent, or provide a diagram/flowchart, include a diagram criterion (~20-30% of marks) split between correct structure/layout and labelling accuracy.
- Level-specific focus:
  * L1 (remember: list, record, define): definition/points mostly.
  * L2 (understand: explain, illustrate, articulate): definition + explanation + example/diagram.
  * L3 (apply: formulate, demonstrate, solve): steps/commands and result/verification.
  * L4 (analyze: infer, compare): analysis points + comparison/flow.
  * L5-L6 (evaluate, create): criteria/justification with example, conclusion.
- For command or code questions: allocate marks per correct step or command, plus output or verification.
- If the question allows alternative valid answers, list them in the criterion's "alternatives" list.
- Every criterion MUST have a non-empty "reason" explaining WHY it carries that many marks in terms of what the examiner is checking.

Allowed kinds for each criterion: "definition", "explanation", "points", "diagram", "example", "steps", "code", "comparison", "analysis", "justification", "conclusion".

Worked Example for a 10-mark L2 question:
```json
{{
  "total_marks": 10,
  "criteria": [
    {{
      "no": 1,
      "criterion": "Definition of core concept",
      "expected": "Clear definition of the core architecture and purpose.",
      "marks": 1,
      "kind": "definition",
      "reason": "L2 questions start with recall of the concept; one mark for a correct definition.",
      "partial_credit": null,
      "alternatives": []
    }},
    {{
      "no": 2,
      "criterion": "Component breakdown",
      "expected": "Identifies key components with their roles.",
      "marks": 5,
      "kind": "points",
      "reason": "Five components are expected, each independently verifiable, so 1 mark each.",
      "partial_credit": "1 mark per component identified",
      "alternatives": []
    }},
    {{
      "no": 3,
      "criterion": "Architecture diagram and labelling",
      "expected": "Diagram with labelled components and arrows showing the flow.",
      "marks": 2,
      "kind": "diagram",
      "reason": "Diagram is required by the question and rewards layout (1M) and labelling (1M).",
      "partial_credit": "1 mark for correct layout without complete labels",
      "alternatives": []
    }},
    {{
      "no": 4,
      "criterion": "Component interaction workflow",
      "expected": "Explanation of how components interact during execution.",
      "marks": 2,
      "kind": "explanation",
      "reason": "Shows understanding beyond listing.",
      "partial_credit": "1 mark for partial description",
      "alternatives": []
    }}
  ],
  "notes": "Accept equivalent technical terminology and valid component groupings."
}}
```

Now generate the marking scheme JSON for:
Question: {question}
Bloom's Level: {level}
Total Marks: {marks}
Model Answer:
\"\"\"
{answer}
\"\"\"{context_section}

Output ONLY valid JSON starting with '{{' and ending with '}}'. No preamble, no conversational text, no markdown fences."""


def build_scheme_prompt(
    question: str,
    level: str,
    marks: int,
    answer: str,
    context: Optional[str] = "",
) -> str:
    """Build prompt instructing Ollama llama3.1:8b to generate a strict JSON marking scheme."""
    norm_level = normalize_level(level) if level else "L2"
    m = marks if marks is not None and marks > 0 else 5
    ctx = (context or "").strip()
    context_section = f"\n\nContext:\n\"\"\"\n{ctx}\n\"\"\"" if ctx else ""

    return SCHEME_BASE.format(
        question=question.strip(),
        level=norm_level,
        marks=m,
        answer=(answer or "").strip(),
        context_section=context_section,
    )



COMBINED_ANSWER_SCHEME_BASE = """You are an expert academic evaluator. Given the question, context text, Bloom's Taxonomy cognitive level, and marks below, generate BOTH the grounded model answer AND an itemized marking scheme breakdown in strict JSON format.

Context:
\"\"\"
{context}
\"\"\"

Question:
{question}

Cognitive Level: {level}
Total Marks: {marks}

Instructions:
1. "answer": Provide a clear, comprehensive model answer strictly grounded in the context text above.
2. "criteria": Provide itemized checkable criteria for grading this answer.
   - Each criterion must have:
     * "no": integer sequence (1, 2, ...)
     * "criterion": concise title of expected point
     * "expected": what examiner checks for in the answer
     * "marks": integer or half-mark allocation (> 0)
     * "kind": one of "definition", "explanation", "points", "diagram", "example", "steps", "code", "comparison"
     * "reason": pedagogical reason why this criterion earns marks
   - CRITICAL REQUIREMENT: The sum of marks across all criteria MUST equal EXACTLY {marks}.
3. "notes": Any brief guidance for the examiner.

Respond ONLY with a valid JSON object starting with '{{' and ending with '}}':
{{
  "answer": "...",
  "answer_alternatives": [],
  "criteria": [
    {{
      "no": 1,
      "criterion": "...",
      "expected": "...",
      "marks": 2,
      "kind": "explanation",
      "reason": "..."
    }}
  ],
  "total_marks": {marks},
  "notes": "..."
}}
"""


def build_combined_answer_scheme_prompt(
    question: str,
    context: str,
    marks: Optional[int] = None,
    level: Optional[str] = None,
) -> str:
    """Build prompt for generating both model answer and marking scheme in a single LLM call."""
    m = marks if marks is not None and marks > 0 else 5
    lvl = normalize_level(level) if level else "L2"
    return COMBINED_ANSWER_SCHEME_BASE.format(
        context=context.strip(),
        question=question.strip(),
        marks=m,
        level=lvl,
    )
