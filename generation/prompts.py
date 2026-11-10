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
    2: " Keep it short; answerable in 2-3 sentences.",
    5: " It should be answerable in one well-developed paragraph.",
    10: " It should need a detailed, multi-point answer.",
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
Provide an authoritative model answer for the exam question below using ONLY the provided context.

Rules:
- Answer ONLY from the context. Do not use outside knowledge or extrapolate beyond the text. State only verified facts from the context; a question's false premise must not be accepted.
- For term recall questions (e.g. 'Identify the term...', 'Name the term...'), provide ONLY the short precise term/name (under 8 words) without repeating or echoing the question stem.
{strict_rule}- Provide ONE clear, correct, authoritative model answer. Do not apologize, do not comment on what the context lacks (e.g. NEVER write "the text does not mention" or "not explicitly stated"), and do not invent contradictory alternatives. Write the exact answer an examiner expects.
- The model answer must be completely self-contained. NEVER write phrases such as "according to the provided context", "in the provided example", "as shown in Listing", "as described in the passage", "introductory chapter", "earlier chapter", "in this chapter", "in this book", "the author", or local file paths (e.g. /home/..., C:\\Users\\...).
- Address all parts of the question directly (such as trade-offs, advantages, consequences, steps, or justification).
- Target answer length: approximately {target_words} words (appropriate for a {marks}-mark question).
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
    strict_grounding: bool = False,
) -> str:
    """Build prompt for generating a grounded model answer."""
    m = marks if marks is not None and marks > 0 else 5
    target_words = max(30, min(500, m * 25))
    strict_rule = (
        "- STRICT GROUNDING: Answer strictly and only from the provided context. Do not invent facts, claims, or tools. "
        "Every statement must be directly verifiable in the context.\n"
        if strict_grounding
        else ""
    )
    return ANSWER_BASE.format(
        target_words=target_words,
        marks=m,
        strict_rule=strict_rule,
        context=context.strip(),
        question=question.strip(),
    )
