"""Offline stand-in for the LLM (tests / demo without Ollama)."""
import re


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
            r"^(look at the following diagram\.\s*|look at\b.*?[\.:]\s*|"
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

    if len(c) > 1:
        c = c[0].lower() + c[1:]
    return c


def _format_question_with_verb(verb: str, concept: str) -> str:
    """Format a realistic, sensible exam question starting with the specified Bloom's verb."""
    v = verb.lower().strip()
    V = v.capitalize()

    if v in {"define", "state", "recall", "name"}:
        return f"{V} the fundamental concept and properties of {concept}?"
    elif v in {"enumerate", "list", "identify", "label"}:
        return f"{V} the primary characteristics and elements of {concept}?"
    elif v in {"explain", "describe", "discuss", "clarify", "interpret"}:
        return f"{V} how {concept} operates within the system?"
    elif v in {"classify", "categorize", "distinguish", "differentiate"}:
        return f"{V} the various categories and approaches associated with {concept}?"
    elif v in {"apply", "demonstrate", "illustrate", "implement", "execute"}:
        return f"{V} the core principles of {concept} to solve practical problems?"
    elif v in {"solve", "calculate", "compute", "determine"}:
        return f"{V} analytical problems involving {concept} using standard methodologies?"
    elif v in {"analyze", "deconstruct", "examine", "investigate"}:
        return f"{V} the structural behavior, parameters, and trade-offs of {concept}?"
    elif v in {"compare", "contrast"}:
        return f"{V} the operational advantages and distinctions within {concept}?"
    elif v in {"evaluate", "assess", "justify", "critique", "appraise"}:
        return f"{V} the performance, efficiency, and practical limitations of {concept}?"
    elif v in {"design", "formulate", "construct", "propose", "develop"}:
        return f"{V} an effective framework utilizing {concept}?"
    else:
        return f"{V} the underlying mechanisms and significance of {concept}?"


class MockClient:
    model = "mock"

    def is_up(self):
        return True

    def generate(self, prompt, temperature=None, num_predict=120):
        verbs_match = re.search(r"verbs: ([^\n]+)", prompt)
        verbs = verbs_match.group(1).split(", ") if verbs_match else ["Explain"]
        ctx_match = re.search(r'"""\n(.*?)\n"""', prompt, re.S)
        ctx = ctx_match.group(1) if ctx_match else prompt

        concept = _extract_concept(ctx)
        selected_verb = verbs[0].strip()
        return _format_question_with_verb(selected_verb, concept)
