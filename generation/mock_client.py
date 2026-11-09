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
    else:
        return f"{V} the underlying mechanisms and significance of {concept}?"


class MockClient:
    model = "mock"

    def is_up(self):
        return True

    def generate(self, prompt, temperature=None, num_predict=120, format=None, **kwargs):
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
        return _format_question_with_verb(selected_verb, concept)

    def _generate_answer(self, prompt: str) -> str:
        ctx_match = re.search(r'"""\n(.*?)\n"""', prompt, re.S)
        ctx = ctx_match.group(1).strip() if ctx_match else prompt

        q_match = re.search(r"Question:\s*\n(.*?)(?:\n\s*Model Answer:|$)", prompt, re.S)
        question = q_match.group(1).strip() if q_match else ""

        m_match = re.search(r"(\d+)-mark", prompt)
        marks = int(m_match.group(1)) if m_match else 5

        raw_sents = [s.strip() for s in re.split(r"[\.\n]+", ctx) if len(s.strip().split()) >= 3]
        if not raw_sents:
            raw_sents = ["The technical system operates according to foundational computational principles."]

        # Target words scale strictly with marks
        target_words = max(20, min(400, marks * 22))
        words_accum: list[str] = []
        idx = 0
        while len(words_accum) < target_words:
            sent = raw_sents[idx % len(raw_sents)] + "."
            words_accum.extend(sent.split())
            idx += 1
            if idx > 30:
                break

        primary_ans = " ".join(words_accum)

        q_lower = question.lower()
        has_or = any(w in q_lower for w in ["compare", "contrast", "differentiate", "distinguish", "alternative", " or "]) or "or" in q_lower.split()

        if has_or and len(raw_sents) >= 2:
            alt_words_target = max(15, min(200, marks * 15))
            alt_accum: list[str] = []
            idx = 1
            while len(alt_accum) < alt_words_target:
                sent = raw_sents[idx % len(raw_sents)] + "."
                alt_accum.extend(sent.split())
                idx += 1
                if idx > 20:
                    break
            alt_ans = " ".join(alt_accum)
            return f"{primary_ans}\nOR\n{alt_ans}"

        return primary_ans
