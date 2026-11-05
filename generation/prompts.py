"""Step 3b - prompt templates (one base template + per-level hints + Revised Bloom's knowledge dimension hints + answer generation)."""
from typing import Optional

from blooms.revised_taxonomy import normalize_knowledge_dimension
from blooms.verb_bank import LEVEL_NAMES, normalize_level

BASE = """You are an expert exam question setter.
Using ONLY the context below, write exactly ONE high-quality, clear, sensible {name}-level (Bloom's {level}) exam question.
The question MUST start with one of these verbs: {verbs}.
Rules:
- Use only facts present in the context. Do not use outside knowledge.
- Do not mention "the context", "the passage", "the text", "the diagram", "the author", or "the tutorial" in the question.
- Focus purely on core technical concepts, definitions, algorithms, or mechanisms. The question must be realistic and complete.
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
    "L5": "Require a justified judgement or critique grounded in the context.",
    "L6": "Ask the student to design or propose something new built on ideas from the context.",
}

KNOWLEDGE_HINTS = {
    "Factual": " Focus on core definitions, basic terms, symbols, and factual elements from the text.",
    "Conceptual": " Focus on system relationships, categories, principles, theories, and architectural structure.",
    "Procedural": " Focus on algorithms, techniques, methods of inquiry, and step-by-step procedures.",
    "Metacognitive": " Focus on cognitive reflection, evaluating methodologies, and strategic thinking.",
}

MARKS_HINTS = {
    2: " Keep it short; answerable in 2-3 sentences.",
    5: " It should be answerable in one well-developed paragraph.",
    10: " It should need a detailed, multi-point answer.",
}


def build_prompt(context, level, verbs, marks=None, knowledge_dimension=None) -> str:
    level = normalize_level(level)
    kd_hint = ""
    if knowledge_dimension:
        kd = normalize_knowledge_dimension(knowledge_dimension)
        kd_hint = KNOWLEDGE_HINTS.get(kd, "")
    return BASE.format(
        name=LEVEL_NAMES[level],
        level=level,
        verbs=", ".join(verbs),
        hint=LEVEL_HINTS[level],
        kd_hint=kd_hint,
        marks=MARKS_HINTS.get(marks, ""),
        context=context.strip(),
    )


ANSWER_BASE = """You are an expert exam evaluator and question paper author.
Provide an authoritative model answer for the exam question below using ONLY the provided context.

Rules:
- Answer ONLY from the context. Do not use outside knowledge or extrapolate beyond the text.
- Give one primary, comprehensive model answer.
- Target answer length: approximately {target_words} words (appropriate for a {marks}-mark question).
- If the question genuinely admits more than one acceptable or alternative answer, provide each alternative after a separate line containing only "OR".
- Output only the model answer text (and "OR" lines between distinct valid alternatives). No conversational preamble, no greetings, no markdown fences.

Context:
\"\"\"
{context}
\"\"\"

Question:
{question}

Model Answer:"""


def build_answer_prompt(question: str, context: str, marks: Optional[int] = None) -> str:
    """Build prompt for generating a grounded model answer."""
    m = marks if marks is not None and marks > 0 else 5
    target_words = max(30, min(500, m * 25))
    return ANSWER_BASE.format(
        target_words=target_words,
        marks=m,
        context=context.strip(),
        question=question.strip(),
    )
