"""Step 3c / Issue 4 - question and answer generation with validation, retry, de-duplication, and multi-book assembly.

Optimized with:
- Parallel slot generation via ThreadPoolExecutor with strict order preservation
- Fast LLM result caching (keyed by prompt, model, temperature, seed)
- Bounded token caps (num_predict) tailored to questions vs answers
- Targeted Bloom's verb retries
- Reused chunk embeddings for grounded answer cosine similarity
"""
import concurrent.futures
import random
import re
import threading
from typing import Any, Optional
import config

from blooms.revised_taxonomy import normalize_knowledge_dimension
from blooms.verb_bank import (
    first_word,
    get_allowed_kd_verbs,
    get_verbs,
    is_verb_allowed_for_kd,
    nearest_valid_kd_verb,
    normalize_level,
    replace_question_verb,
    sample_kd_verbs,
    sample_verbs,
    validate_question,
    validate_question_kd,
)
from config import (
    ANSWER_GROUNDING_SENTENCE_SIMILARITY,
    ANSWER_MAX_UNGROUNDED_SENTENCE_RATIO,
    ANSWER_MIN_SIMILARITY,
    CO_SIMILARITY_THRESHOLD,
    GENERATION_WORKERS,
    MAX_CONTEXT_WORDS,
    MIN_CONTEXT_WORDS,
    QUESTION_SIMILARITY_DUPLICATE_THRESHOLD,
)
from generation.cache import get_cached_response, store_cached_response
from generation.prompts import build_answer_prompt, build_prompt
from retrieval.sources import is_content_chunk, select_chunks_for_unit

DANGLING_ENDINGS = {
    "and", "the", "with", "impl", "ho", "availab", "by", "to", "or", "of", "in", "on",
    "at", "for", "from", "as", "is", "are", "a", "an", "that", "this", "which", "into",
    "onto", "be", "been", "being", "have", "has", "had", "do", "does", "did", "but",
    "so", "if", "because", "while", "where", "when", "how", "what", "who",
}

TERMINAL_PUNCTUATION = (".", "!", "?", '."', ".'", '!"', "!'", '?"', "?'", "```", ")", "]", "}")

TEXTBOOK_REF_PATTERNS = [
    r"\b(this|the)\s+(book|text|chapter|section|author|authors|passage|diagram above|diagram below|figure\s*\d|table\s*\d)\b",
    r"\bchapter\s+\d+\b",
    r"\bfigure\s*\d+(?:\.\d+)?\b",
    r"\btable\s*\d+(?:\.\d+)?\b",
    r"\btarget\s+audience\b",
    r"\baim\s+of\s+chapter\b",
    r"\bas\s+described\s+in\b",
    r"\baccording\s+to\s+(?:the\s+author|the\s+text|the\s+book|chapter)\b",
    r"\bcovered\s+in\s+chapter\b",
    r"\bmentioned\s+in\s+chapter\b",
    r"\blook\s+at\s+(?:the\s+)?(?:diagram|figure|table)\b",
]
TEXTBOOK_REF_RE = re.compile("|".join(TEXTBOOK_REF_PATTERNS), re.I)

# Problem 4: Banned deictic and relative reference patterns
UNSEEN_MATERIAL_PATTERNS = [
    r"\bprovided\b",
    r"\babove\b",
    r"\bbelow\b",
    r"\bthe\s+example\b",
    r"\bthe\s+technique\b",
    r"\bthis\s+technique\b",
    r"\bthe\s+listing\b",
    r"\bListing\s+\d+\b",
    r"\bthe\s+describe\s+method\b",
    r"\bthe\s+context\b",
    r"\bin\s+the\s+passage\b",
]
UNSEEN_MATERIAL_RE = re.compile("|".join(UNSEEN_MATERIAL_PATTERNS), re.I)

CODE_REFERENCE_PATTERNS = [
    r"\bthe\s+following\s+code\b",
    r"\bthe\s+given\s+code\b",
    r"\bthe\s+code\s+snippet\b",
    r"\bthe\s+script\b",
]
CODE_REFERENCE_RE = re.compile("|".join(CODE_REFERENCE_PATTERNS), re.I)

BANNED_ANSWER_PATTERNS = [
    r"\baccording\s+to\s+the\s+provided\s+context\b",
    r"\bin\s+the\s+provided\s+example\b",
    r"\bas\s+shown\s+in\s+Listing\b",
    r"\bthe\s+provided\s+text\b",
    r"\bthe\s+provided\s+context\b",
    r"\bthe\s+text\s+does\s+not\b",
    r"\bnot\s+explicitly\s+mentioned\b",
    r"\bbased\s+on\s+general\s+knowledge\b",
    r"\bcannot\s+be\s+answered\b",
    r"\bthe\s+passage\s+doesn'?t\b",
    r"\bnot\s+mentioned\s+in\s+the\b",
    r"\bdoes\s+not\s+provide\b",
    r"\bnot\s+specified\s+in\s+the\b",
    r"\blacks\s+information\b",
]
BANNED_ANSWER_RE = re.compile("|".join(BANNED_ANSWER_PATTERNS), re.I)


def has_unseen_material_reference(text: str, is_question: bool = True) -> tuple[bool, str]:
    """Check for dependencies on unseen material (Problem 4).

    Allows code/script references only if code is included inline in the question stem.
    """
    s = text or ""
    if is_question:
        # Check inline code
        has_inline_code = bool(re.search(r"`[^`]+`|```[\s\S]+?```", s))
        if CODE_REFERENCE_RE.search(s) and not has_inline_code:
            return True, "relative code reference without inline code in stem"
        if UNSEEN_MATERIAL_RE.search(s):
            return True, "unseen material reference"
    else:
        if BANNED_ANSWER_RE.search(s):
            return True, "banned phrase in model answer"
        if UNSEEN_MATERIAL_RE.search(s):
            return True, "unseen material reference in model answer"
    return False, ""


def is_circular_question(q: str) -> bool:
    """Problem 5.4: Reject stems where one concept is a case of the other or circular."""
    fw = first_word(q)
    if fw in ("integrate", "compare", "contrast", "differentiate", "distinguish"):
        m = re.search(
            r"\b(?:integrate|compare|contrast|differentiate|distinguish)\s+(?:the\s+role\s+of\s+|the\s+differences?\s+between\s+|between\s+)?(.+?)\s+(?:into|with(?:\s+that\s+of)?|and)\s+(.+?)(?:\?|,|\bin\b|\bterms\b|\busing\b|\bas\b|$)",
            q,
            re.I,
        )
        if m:
            left_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", m.group(1))]
            right_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", m.group(2))]
            stop = {"concept", "mechanism", "process", "approach", "system", "method", "using", "between", "role", "the", "for", "with", "into"}
            left = {w for w in left_words if w not in stop}
            right = {w for w in right_words if w not in stop}
            if left and right:
                if left == right or left.issubset(right) or right.issubset(left):
                    return True
                # Check for continuous deployment into continuous delivery
                if ("continuous" in left and "continuous" in right) and ("deployment" in left or "delivery" in left) and ("deployment" in right or "delivery" in right):
                    return True
    return False


def check_compare_entities_in_context(q: str, context: str) -> bool:
    """Problem 5.3: For compare/contrast/distinguish questions, both named entities must appear in chunk context."""
    fw = first_word(q)
    if fw in ("compare", "contrast", "differentiate", "distinguish"):
        m = re.search(
            r"\b(?:compare|contrast|differentiate|distinguish)\s+(?:the\s+role\s+of\s+|the\s+differences?\s+between\s+|between\s+)?(.+?)\s+(?:with(?:\s+that\s+of)?|and)\s+(.+?)(?:\?|,|\bin\b|\bterms\b|\busing\b|\bas\b|$)",
            q,
            re.I,
        )
        if m:
            e1 = m.group(1).strip().lower()
            e2 = m.group(2).strip().lower()
            stop = {"the", "a", "an", "role", "of", "process", "concept", "approach"}
            generic = {
                "network", "networks", "system", "systems", "component", "components",
                "method", "methods", "model", "models", "architecture", "architectures",
                "tool", "tools", "data", "technique", "techniques", "mechanism", "mechanisms",
                "pattern", "patterns", "design", "designs",
            }
            e1_words = [w for w in e1.split() if w not in stop]
            e2_words = [w for w in e2.split() if w not in stop]
            e1_substantive = [w for w in e1_words if w not in generic]
            e2_substantive = [w for w in e2_words if w not in generic]
            e1_check = e1_substantive if e1_substantive else e1_words
            e2_check = e2_substantive if e2_substantive else e2_words
            ctx_lower = context.lower()
            has_e1 = all(w in ctx_lower for w in e1_check) if e1_check else True
            has_e2 = all(w in ctx_lower for w in e2_check) if e2_check else True
            return has_e1 and has_e2
    return True


DEMAND_KEYWORDS = {
    "trade-offs": ["trade-off", "tradeoff", "advantage", "disadvantage", "pros", "cons", "cost", "benefit"],
    "trade-off": ["trade-off", "tradeoff", "advantage", "disadvantage", "pros", "cons", "cost", "benefit"],
    "justify": ["because", "reason", "due to", "justif", "why", "since", "therefore"],
    "differences": ["whereas", "while", "unlike", "differ", "instead of", "contrast", "distinction", "however"],
    "advantages": ["advantage", "benefit", "strength", "superior", "gain"],
    "steps": ["step", "first", "next", "then", "finally", "1.", "2.", "procedure", "stage"],
    "consequences": ["impact", "lead to", "result in", "consequence", "risk", "damage", "affect"],
    "solution": ["solution", "mitigat", "resolve", "prevent", "address", "remedy", "fix"],
}


def check_demand_coverage(question: str, answer: str) -> tuple[bool, Optional[str]]:
    """Problem 5.2: Check that demanded question parts (trade-offs, justify, steps, etc.) appear in the model answer."""
    q_lower = question.lower()
    a_lower = answer.lower()
    for demand, markers in DEMAND_KEYWORDS.items():
        if demand in q_lower:
            has_marker = any(m in a_lower for m in markers)
            if not has_marker:
                return False, f"missing expected '{demand}' content in answer"
    return True, None


def is_truncated_text(text: str, is_question: bool = False) -> bool:
    """Return True if text appears truncated (cut off mid-sentence or mid-word)."""
    s = (text or "").strip()
    if not s:
        return True
    if is_question:
        if not s.endswith(("?", '?"', "?'")):
            return True
        last_word = re.sub(r"[^\w]", "", s.split()[-1].lower())
        if last_word in DANGLING_ENDINGS:
            return True
        return False

    ends_punct = any(s.endswith(p) for p in TERMINAL_PUNCTUATION)
    if not ends_punct:
        return True
    words = s.split()
    if words:
        last_word = re.sub(r"[^\w]", "", words[-1].lower())
        if last_word in DANGLING_ENDINGS:
            return True
    return False


def truncate_to_last_complete_sentence(text: str) -> str:
    """Fallback: truncate to the last sentence ending with terminal punctuation."""
    s = (text or "").strip()
    matches = list(re.finditer(r"[\.!\?]+(?:\s+|$)", s))
    if matches:
        last_end = matches[-1].end()
        return s[:last_end].strip()
    return s


def has_textbook_reference(text: str) -> bool:
    """Check if text refers to physical textbook rather than concept."""
    return bool(TEXTBOOK_REF_RE.search(text or ""))


def has_banned_answer_phrases(text: str) -> bool:
    """Check if model answer contains meta commentary or apology phrases."""
    return bool(BANNED_ANSWER_RE.search(text or ""))


def clean_answer_text(text: str) -> tuple[str, list[str]]:
    """Clean model answer: replace '― OR ―' and 'OR' with 'Alternative approach:'.

    Ensures ONE authoritative model answer or distinct alternative approaches.
    Never leaves '― OR ―' inside answer text.
    """
    raw = (text or "").strip()
    if not raw:
        return "", []

    lines = raw.splitlines()
    cur_buf: list[str] = []
    alternatives: list[str] = []

    for line in lines:
        stripped = line.strip()
        if re.match(r"^(?:[—―\-]+\s*)?OR(?:\s*[—―\-]+)?[:\.]?$", stripped, re.I):
            alt_text = "\n".join(cur_buf).strip()
            if alt_text:
                alternatives.append(alt_text)
            cur_buf = []
        elif re.match(r"^Alternative approach(?:\s*\d+)?[:\.]?", stripped, re.I):
            alt_text = "\n".join(cur_buf).strip()
            if alt_text:
                alternatives.append(alt_text)
            content_after = re.sub(r"^Alternative approach(?:\s*\d+)?[:\.]?\s*", "", stripped, flags=re.I).strip()
            cur_buf = [content_after] if content_after else []
        else:
            cur_buf.append(line)

    if cur_buf:
        alt_text = "\n".join(cur_buf).strip()
        if alt_text:
            alternatives.append(alt_text)

    if not alternatives:
        alternatives = [raw]

    unique_alts: list[str] = []
    for a in alternatives:
        # Strip any internal '― OR ―'
        a_clean = re.sub(r"^[—―-]+\s*OR\s*[—―-]+$", "", a, flags=re.I | re.M).strip()
        a_clean = re.sub(r"[—―-]+\s*OR\s*[—―-]+", "Alternative approach:", a_clean, flags=re.I).strip()
        if a_clean and a_clean not in unique_alts:
            unique_alts.append(a_clean)

    if len(unique_alts) > 1:
        full = unique_alts[0] + "\n\n" + "\n\n".join(f"Alternative approach:\n{a}" for a in unique_alts[1:])
    else:
        full = unique_alts[0] if unique_alts else raw

    full = re.sub(r"[—―]+\s*OR\s*[—―]+", "Alternative approach:", full, flags=re.I)
    return full.strip(), unique_alts


def clean_output(raw: str) -> str:
    """Strip fences, preambles, numbering and quotes from LLM output."""
    text = re.sub(r"^```\w*|```$", "", (raw or "").strip(), flags=re.M).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    q = lines[0]
    for l in lines:
        if not re.match(r"^(here('s| is)|sure|okay|certainly)\b.*:$", l, re.I):
            q = l
            break
    q = re.sub(r"^(question|q)\s*\d*\s*[:.)-]\s*", "", q, flags=re.I)
    q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", q)
    return q.strip(" \"'“”*_`")


def _jaccard(a: str, b: str) -> float:
    x, y = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(x & y) / len(x | y) if x | y else 0.0


def is_duplicate(q, seen, threshold=0.8):
    return any(_jaccard(q, s) >= threshold for s in seen)


def generate_question(
    chunk,
    level,
    client,
    rng=None,
    max_retries=3,
    marks=None,
    k_verbs=4,
    knowledge_dimension=None,
    used_topics=None,
    force_regenerate=False,
):
    level = normalize_level(level)
    rng = rng or random.Random()
    kd = normalize_knowledge_dimension(knowledge_dimension) if knowledge_dimension else "Conceptual"
    context = " ".join(chunk["text"].split()[:MAX_CONTEXT_WORDS])
    has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?\b", context))
    q, reason, attempts = "", "no output", 0
    ok = False

    # Proportional token cap
    num_predict = getattr(config, "QUESTION_NUM_PREDICT", 120)
    client_model = getattr(client, "model", "llama3.1:8b")

    for attempts in range(1, max_retries + 1):
        # Sample verbs constrained by both Bloom's level and Knowledge Dimension (Problem 6)
        verbs = sample_kd_verbs(level, kd, k_verbs, rng)
        if not has_numbers:
            verbs = [v for v in verbs if v.lower() not in ("compute", "calculate")]
            if not verbs:
                all_allowed = get_allowed_kd_verbs(level, kd)
                verbs = [v for v in all_allowed if v.lower() not in ("compute", "calculate")]
                verbs = rng.sample(verbs, min(k_verbs, len(verbs))) if verbs else sample_verbs(level, k_verbs, rng)

        prompt = build_prompt(context, level, verbs, marks=marks, knowledge_dimension=kd, used_topics=used_topics)
        temp = min(0.3 + 0.2 * (attempts - 1), 0.9)

        cached = get_cached_response(prompt, client_model, temperature=temp, force_regenerate=force_regenerate)
        if cached:
            raw = cached
        else:
            raw = client.generate(prompt, temperature=temp, num_predict=num_predict)

        q = clean_output(raw)
        ok, reason = validate_question(q, level)

        # Check unseen material / relative references (Problem 4)
        if ok:
            has_unseen, unseen_msg = has_unseen_material_reference(q, is_question=True)
            if has_unseen:
                ok = False
                reason = unseen_msg

        # Check textbook references
        if ok and has_textbook_reference(q):
            ok = False
            reason = "question contains textbook references"

        # Check truncation
        if ok and is_truncated_text(q, is_question=True):
            ok = False
            reason = "question is truncated"

        # Check circularity (Problem 5.4)
        if ok and is_circular_question(q):
            ok = False
            reason = "circular question"

        # Check compare entities in context (Problem 5.3)
        if ok and not check_compare_entities_in_context(q, context):
            ok = False
            reason = "compared entities not found in context"

        # Check numeric context for compute/calculate (Problem 5.5)
        if ok and not has_numbers and first_word(q) in ("compute", "calculate"):
            old_v = first_word(q)
            new_v = nearest_valid_kd_verb("demonstrate", level, kd)
            q = replace_question_verb(q, new_v)
            ok, reason = validate_question(q, level)

        # Check Knowledge Dimension verb alignment (Problem 6)
        if ok and knowledge_dimension:
            ok_kd, kd_reason = validate_question_kd(q, level, kd)
            if not ok_kd:
                if attempts < max_retries:
                    ok = False
                    reason = kd_reason
                else:
                    # Final fallback: replace verb with nearest valid verb from same Bloom level & KD
                    old_verb = first_word(q)
                    new_verb = nearest_valid_kd_verb(old_verb, level, kd)
                    q = replace_question_verb(q, new_verb)
                    ok, reason = validate_question(q, level)

        if ok:
            if not cached and not is_truncated_text(q, is_question=True):
                store_cached_response(prompt, client_model, raw, temperature=temp)
            break
        elif attempts < max_retries:
            targeted_verbs = sample_kd_verbs(level, kd, k_verbs, rng)
            if not has_numbers:
                targeted_verbs = [v for v in targeted_verbs if v.lower() not in ("compute", "calculate")]
            targeted_prompt = (
                f"{prompt}\n\nNote: The question must strictly start with one of: {', '.join(targeted_verbs)}. "
                f"Ensure the question is self-contained with no references to unseen material, 'provided', or 'the snippet'."
            )
            raw = client.generate(targeted_prompt, temperature=temp + 0.1, num_predict=num_predict)
            q = clean_output(raw)
            ok, reason = validate_question(q, level)
            if ok:
                has_unseen, unseen_msg = has_unseen_material_reference(q, is_question=True)
                if has_unseen:
                    ok = False
                    reason = unseen_msg
            if ok and has_textbook_reference(q):
                ok = False
                reason = "question contains textbook references"
            if ok and is_truncated_text(q, is_question=True):
                ok = False
                reason = "question is truncated"
            if ok and is_circular_question(q):
                ok = False
                reason = "circular question"
            if ok and not check_compare_entities_in_context(q, context):
                ok = False
                reason = "compared entities not found in context"
            if ok and not has_numbers and first_word(q) in ("compute", "calculate"):
                new_v = nearest_valid_kd_verb("demonstrate", level, kd)
                q = replace_question_verb(q, new_v)
                ok, reason = validate_question(q, level)
            if ok and knowledge_dimension:
                ok_kd, kd_reason = validate_question_kd(q, level, kd)
                if not ok_kd:
                    old_verb = first_word(q)
                    new_verb = nearest_valid_kd_verb(old_verb, level, kd)
                    q = replace_question_verb(q, new_verb)
                    ok, reason = validate_question(q, level)
            attempts += 1
            if ok:
                if not is_truncated_text(q, is_question=True):
                    store_cached_response(prompt, client_model, raw, temperature=temp)
                break

    return {
        "question": q,
        "level": level,
        "verb": first_word(q),
        "marks": marks,
        "knowledge_dimension": kd or "Conceptual",
        "kd": kd or "Conceptual",
        "valid": ok,
        "reason": reason,
        "attempts": attempts,
        "chunk_id": chunk.get("chunk_id"),
        "book_id": chunk.get("book_id"),
        "chapter": chunk.get("chapter"),
        "section": chunk.get("section"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
    }


def generate_answer(
    question: str,
    chunk: dict[str, Any],
    marks: Optional[int],
    client: Any,
    rng: Optional[random.Random] = None,
    max_retries: int = 3,
    check_similarity: Optional[bool] = None,
    force_regenerate: bool = False,
) -> dict[str, Any]:
    """Generate model answer grounded in chunk context.

    Validates non-empty, minimum length for marks, absence of banned meta phrases,
    absence of textbook references, non-truncation, demand coverage, and cosine similarity with chunk text.
    """
    rng = rng or random.Random()
    marks = marks if marks is not None and marks > 0 else 5
    context = " ".join(chunk.get("text", "").split()[:MAX_CONTEXT_WORDS])

    is_mock = getattr(client, "model", None) == "mock" or "Mock" in client.__class__.__name__
    should_check_similarity = (not is_mock) if check_similarity is None else bool(check_similarity)

    last_answer = ""
    last_alternatives: list[str] = []
    reason = "no output"
    ok = False
    ungrounded_sentences: list[str] = []
    demand_issue: Optional[str] = None

    if marks <= 5:
        num_predict = 180
    elif marks <= 8:
        num_predict = 320
    else:
        num_predict = 500

    client_model = getattr(client, "model", "llama3.1:8b")

    for attempt in range(1, max_retries + 1):
        temp = min(0.3 + 0.2 * (attempt - 1), 0.9)
        strict = (attempt > 1)
        prompt = build_answer_prompt(question, context, marks=marks, strict_grounding=strict)

        cached = get_cached_response(prompt, client_model, temperature=temp, force_regenerate=force_regenerate)
        if cached:
            raw = cached
        else:
            raw = client.generate(prompt, temperature=temp, num_predict=num_predict)

        # Clean output fences
        text = re.sub(r"^```\w*|```$", "", (raw or "").strip(), flags=re.M).strip()
        full_answer, alternatives = clean_answer_text(text)
        last_alternatives = alternatives
        last_answer = "\nOR\n".join(alternatives) if len(alternatives) > 1 else full_answer

        # Validation 1: non-empty
        if not last_answer:
            reason = "empty answer"
            continue

        # Validation 2: minimum length for marks
        word_count = len(last_answer.split())
        min_words = min(20, max(5, int(marks * 2)))
        if word_count < min_words:
            reason = f"answer length {word_count} words below minimum {min_words} for {marks} marks"
            continue

        # Validation 3: banned meta-commentary phrases & unseen references (Problem 4 & 5)
        has_unseen, unseen_msg = has_unseen_material_reference(last_answer, is_question=False)
        if has_unseen:
            reason = unseen_msg
            continue
        if has_banned_answer_phrases(last_answer):
            reason = "answer contains banned phrases (meta commentary / apology)"
            continue

        # Validation 4: textbook references
        if has_textbook_reference(last_answer):
            reason = "answer contains textbook references"
            continue

        # Validation 5: truncation check
        if is_truncated_text(last_answer, is_question=False):
            if attempt < max_retries:
                reason = "answer is truncated"
                continue
            else:
                fallback = truncate_to_last_complete_sentence(last_answer)
                if len(fallback.split()) >= min_words:
                    last_answer = fallback
                    last_alternatives = [fallback]
                else:
                    reason = "answer is truncated"
                    continue

        # Validation 6: demand coverage check (Problem 5.2)
        cov_ok, cov_msg = check_demand_coverage(question, last_answer)
        if not cov_ok and attempt < max_retries:
            reason = cov_msg or "missing demanded question content"
            demand_issue = cov_msg
            continue

        # Validation 7: grounded cosine similarity check
        ungrounded_sentences = []
        if should_check_similarity:
            from retrieval.co_mapper import _m
            embedder = _m()
            if embedder is not None:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", last_answer) if s.strip() and len(s.split()) >= 4]
                try:
                    import numpy as np
                    all_texts = [context, last_answer] + sentences
                    all_embs = embedder.encode(all_texts, normalize_embeddings=True)
                    chunk_emb = all_embs[0]
                    ans_emb = all_embs[1]
                    overall_sim = float(np.dot(chunk_emb, ans_emb))
                    if overall_sim < ANSWER_MIN_SIMILARITY:
                        reason = f"similarity {overall_sim:.2f} below threshold {ANSWER_MIN_SIMILARITY}"
                        continue

                    if sentences:
                        sent_embs = all_embs[2:]
                        cos_sims = np.dot(sent_embs, chunk_emb)
                        for s_txt, cos_sim in zip(sentences, cos_sims):
                            if float(cos_sim) < ANSWER_GROUNDING_SENTENCE_SIMILARITY:
                                ungrounded_sentences.append(s_txt)
                        ungrounded_ratio = len(ungrounded_sentences) / len(sentences)
                        if ungrounded_ratio > ANSWER_MAX_UNGROUNDED_SENTENCE_RATIO and attempt < max_retries:
                            reason = f"ungrounded sentences ({len(ungrounded_sentences)}/{len(sentences)})"
                            continue
                except Exception:
                    pass

        ok = True
        reason = "ok"
        if not cached and not is_truncated_text(last_answer, is_question=False):
            store_cached_response(prompt, client_model, raw, temperature=temp)
        break

    res_dict: dict[str, Any] = {
        "answer": last_answer,
        "answer_alternatives": last_alternatives,
        "valid": ok,
        "reason": reason,
    }
    if ungrounded_sentences:
        res_dict["ungrounded_sentences"] = ungrounded_sentences
    if demand_issue and not ok:
        res_dict["demand_issue"] = demand_issue
    return res_dict


def generate_paper(chunks, level_counts, client, seed=None, marks=None, on_progress=None):
    """level_counts: {'L1': 3, 'L4': 2}. Returns {'questions': [...], 'rejected': [...]}."""
    rng = random.Random(seed)
    usable = [c for c in chunks if c.get("word_count", len(c["text"].split())) >= MIN_CONTEXT_WORDS]
    questions, rejected, seen = [], [], []
    for level, n in level_counts.items():
        level = normalize_level(level)
        order = usable[:]
        rng.shuffle(order)
        produced = 0
        for chunk in order * 2:                      # at most two passes over the chunks
            if produced >= n:
                break
            rec = generate_question(chunk, level, client, rng, marks=marks)
            if not rec["valid"] or is_duplicate(rec["question"], seen):
                rejected.append(rec)
                continue
            seen.append(rec["question"])
            rec["id"] = f"Q{len(questions) + 1:03d}"
            questions.append(rec)
            produced += 1
            if on_progress:
                on_progress(rec)
    return {"questions": questions, "rejected": rejected}


def check_question_duplicates(
    questions: list[dict[str, Any]],
    embedding_model: Any,
    threshold: float = QUESTION_SIMILARITY_DUPLICATE_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Compute pairwise cosine similarities across all question texts."""
    valid_qs = [q for q in questions if q.get("valid", True) and not q.get("question", "").startswith("[NEEDS REVIEW")]
    if len(valid_qs) < 2 or embedding_model is None:
        return []
    try:
        texts = [q.get("question", "") for q in valid_qs]
        embs = embedding_model.encode(texts, normalize_embeddings=True)
        import numpy as np
        sim_matrix = np.dot(embs, embs.T)

        dups = []
        for i in range(len(valid_qs)):
            for j in range(i + 1, len(valid_qs)):
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    # Map back to original indices in questions
                    orig_i = questions.index(valid_qs[i])
                    orig_j = questions.index(valid_qs[j])
                    dups.append((orig_i, orig_j, sim))
        return dups
    except Exception:
        return []


def check_question_co_similarity(
    question_text: str,
    co_desc: str,
    embedding_model: Any,
    threshold: float = CO_SIMILARITY_THRESHOLD,
) -> tuple[bool, float]:
    """Compute cosine similarity between generated question and assigned CO description."""
    if not co_desc or embedding_model is None:
        return True, 1.0
    try:
        import numpy as np
        embs = embedding_model.encode([question_text, co_desc], normalize_embeddings=True)
        sim = float(np.dot(embs[0], embs[1]))
        return (sim >= threshold), sim
    except Exception:
        return True, 1.0


def check_paper_identifier_consistency(questions: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Extract backticked or snake_case identifiers across paper and flag conflicting spellings."""
    from difflib import SequenceMatcher
    seen_identifiers: dict[str, int] = {}
    inconsistencies = []

    for q_idx, q in enumerate(questions):
        text = (q.get("question", "") + " " + q.get("answer", ""))
        tokens = set(re.findall(r"`([a-zA-Z0-9_\-]+)`", text))
        tokens.update(re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", text))

        for tok in tokens:
            if len(tok) < 5 or tok.startswith("http"):
                continue
            tok_lower = tok.lower()
            for prev_tok, prev_idx in seen_identifiers.items():
                if prev_idx != q_idx and prev_tok != tok_lower:
                    sm = SequenceMatcher(None, prev_tok, tok_lower)
                    if sm.ratio() >= 0.88 and abs(len(prev_tok) - len(tok_lower)) <= 2:
                        inconsistencies.append((q_idx, f"Inconsistent identifier spelling: '{tok}' vs earlier '{prev_tok}'"))
                        break
            seen_identifiers[tok_lower] = q_idx

    return inconsistencies


def verify_paper_completeness(
    slots: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Check that every blueprint slot has question/answer and OR alternatives mirror each other."""
    # Robust argument order: detect if caller passed (questions, slots) instead of (slots, questions)
    if slots and any("question" in item for item in slots) and not any("question" in item for item in questions):
        slots, questions = questions, slots

    issues = []
    if len(questions) != len(slots):
        issues.append(f"Slot count mismatch: paper has {len(questions)} questions, expected {len(slots)} from blueprint")

    # Check for placeholder / failed slots
    for i, q in enumerate(questions):
        q_text = q.get("question", "")
        a_text = q.get("answer", "")
        q_id = q.get("id") or f"Slot {i+1}"
        if not q_text or q_text.startswith("[NEEDS REVIEW") or q.get("needs_review") or not q.get("valid", True):
            reason = q.get("review_reason") or "generation incomplete"
            issues.append(f"{q_id}: {reason}")
        elif not a_text or a_text.startswith("[NEEDS REVIEW"):
            issues.append(f"{q_id}: missing model answer")

    # Check OR symmetry within units
    by_unit_alt: dict[Any, dict[str, list[dict[str, Any]]]] = {}
    for q in questions:
        u = q.get("unit")
        alt = q.get("alternative_group")
        if u is not None and alt:
            by_unit_alt.setdefault(u, {})
            q_no = q.get("question_no")
            by_unit_alt[u].setdefault(str(q_no), []).append(q)

    for u, q_groups in by_unit_alt.items():
        if len(q_groups) >= 2:
            group_keys = list(q_groups.keys())
            g1_qs = q_groups[group_keys[0]]
            g2_qs = q_groups[group_keys[1]]
            if len(g1_qs) != len(g2_qs):
                issues.append(
                    f"Unit {u}: Alternative options do not mirror each other (Option {group_keys[0]} has {len(g1_qs)} parts, Option {group_keys[1]} has {len(g2_qs)} parts)"
                )
            else:
                g1_marks = [q.get("marks") for q in g1_qs]
                g2_marks = [q.get("marks") for q in g2_qs]
                if g1_marks != g2_marks:
                    issues.append(
                        f"Unit {u}: Sub-question marks do not match between options ({g1_marks} vs {g2_marks})"
                    )

    return len(issues) == 0, issues


def normalize_paper_question_numbers(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure question numbers always start at Q1 for the first unit and run continuously."""
    if not questions:
        return []
    units_order = []
    for q in questions:
        u = q.get("unit")
        if u not in units_order and u is not None:
            units_order.append(u)

    if not units_order:
        q_map: dict[Any, int] = {}
        curr_q = 1
        for q in questions:
            orig_qno = q.get("question_no")
            if orig_qno not in q_map:
                q_map[orig_qno] = curr_q
                curr_q += 1
            new_qno = q_map[orig_qno]
            q["question_no"] = new_qno
            sub = q.get("sub_index")
            if sub:
                letter = chr(96 + int(sub))
                q["id"] = f"{new_qno}({letter})"
            else:
                q["id"] = f"{new_qno}"
        return questions

    curr_q_no = 1
    for u in units_order:
        unit_qs = [q for q in questions if q.get("unit") == u]
        orig_q_nos = []
        for q in unit_qs:
            oq = q.get("question_no")
            if oq not in orig_q_nos:
                orig_q_nos.append(oq)

        for oq in orig_q_nos:
            matching_qs = [q for q in unit_qs if q.get("question_no") == oq]
            has_subs = len(matching_qs) > 1 or any(q.get("sub_index") is not None for q in matching_qs)
            for s_idx, q in enumerate(matching_qs, 1):
                q["question_no"] = curr_q_no
                sub = q.get("sub_index") or (s_idx if has_subs else None)
                if sub:
                    letter = chr(96 + int(sub))
                    q["id"] = f"{curr_q_no}({letter})"
                else:
                    q["id"] = f"{curr_q_no}"
            curr_q_no += 1

    return questions


def regenerate_missing_slots(
    questions: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    client: Any,
    seed: Optional[int] = None,
    unit_sources: Optional[dict[Any, Any]] = None,
    outcomes: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Regenerate ONLY failed/placeholder slots, leaving all existing valid questions untouched."""
    updated_questions = list(questions)
    for idx, q in enumerate(updated_questions):
        if q.get("needs_review") or not q.get("valid", True) or str(q.get("question", "")).startswith("[NEEDS REVIEW"):
            slot_to_run = slots[idx] if idx < len(slots) else q
            regen_res = generate_paper_by_slots(
                chunks=chunks,
                slots=[slot_to_run],
                client=client,
                seed=(seed or 42) + idx * 7,
                unit_sources=unit_sources,
                keep_placeholders=True,
                outcome_descs=outcomes,
                force_regenerate=True,
            )
            if regen_res.get("questions"):
                updated_questions[idx] = regen_res["questions"][0]

    updated_questions = normalize_paper_question_numbers(updated_questions)
    return {"questions": updated_questions}


def generate_paper_by_slots(
    chunks: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    client: Any,
    seed: Optional[int] = None,
    on_progress: Optional[Any] = None,
    unit_sources: Optional[dict[Any, Any]] = None,
    workers: Optional[int] = None,
    force_regenerate: bool = False,
    keep_placeholders: bool = False,
    outcome_descs: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Generate questions and model answers matching blueprint slots.

    Executes concurrently using a ThreadPoolExecutor while preserving exact original slot order.

    Args:
        chunks: List of text chunk dicts.
        slots: Blueprint slot dictionaries.
        client: LLM client (OllamaClient or MockClient).
        seed: Random seed for repeatability.
        on_progress: Optional callback on question accepted.
        unit_sources: Optional mapping {unit: {book_id: [chapters] or None}}.
        workers: Optional max workers count (defaults to config.GENERATION_WORKERS).
        force_regenerate: If True, bypasses generation cache.
        keep_placeholders: If True, keeps failed slots as [NEEDS REVIEW] placeholders in questions.
        outcome_descs: Optional mapping of CO code -> CO description for similarity validation.
    """
    usable = [
        c for c in chunks
        if c.get("word_count", len(c.get("text", "").split())) >= MIN_CONTEXT_WORDS
        and is_content_chunk(c)
    ]
    if not usable or not slots:
        return {"questions": [], "rejected": [], "unfilled": list(slots)}

    num_workers = workers if workers is not None else GENERATION_WORKERS
    num_workers = max(1, min(num_workers, len(slots)))

    seen_lock = threading.Lock()
    group_lock = threading.Lock()
    progress_lock = threading.Lock()

    used_chunks_per_group: dict[str, set[str]] = {}
    used_chunks_per_unit: dict[Any, set[str]] = {}
    seen_questions_in_unit: dict[Any, list[str]] = {}
    seen_questions: list[str] = []

    def _process_single_slot(slot_idx: int, slot: dict[str, Any]) -> dict[str, Any]:
        slot_rng = random.Random(seed + slot_idx * 1000 if seed is not None else None)
        slot_unit = slot.get("unit")
        alt_group = slot.get("alternative_group")

        candidate_chunks = select_chunks_for_unit(usable, unit_sources, slot_unit)
        if not candidate_chunks:
            candidate_chunks = usable
        if not candidate_chunks:
            return {"status": "unfilled", "slot": slot, "rejected": [], "slot_idx": slot_idx}

        # Deterministic slot chunk partitioning (Problem 2.2)
        # Disjoint chunk start for OR alternatives (Option A vs Option B) and sub-questions
        sub_i = slot.get("sub_index", 1) or 1
        q_no = slot.get("question_no", 1) or 1
        slot_offset = ((q_no - 1) * 3 + (sub_i - 1)) if (q_no and sub_i) else slot_idx

        slot_co = slot.get("co")
        if slot_co:
            pool = sorted(
                candidate_chunks,
                key=lambda c: c.get("co_scores", {}).get(slot_co, float("-inf")),
                reverse=True,
            )[:10]
        else:
            pool = candidate_chunks[:10]

        if pool:
            # Shift pool deterministically so alternatives and sub-questions use disjoint chunks
            shift = slot_offset % len(pool)
            pool = pool[shift:] + pool[:shift]

        slot_rng.shuffle(pool)
        slot_kd = slot.get("knowledge_dimension") or slot.get("kd") or "Conceptual"
        slot_rejected = []

        with seen_lock:
            unit_topics = list(seen_questions_in_unit.get(slot_unit, []))

        # Bounded retries: max 2 retries per failing slot (up to 3 attempts total)
        max_slot_tries = min(max(len(pool), 1) * 2, 3)
        for attempt_i in range(max_slot_tries):
            chunk = pool[attempt_i % len(pool)]
            if not is_content_chunk(chunk):
                continue
            try:
                rec = generate_question(
                    chunk,
                    slot["level"],
                    client,
                    slot_rng,
                    marks=slot.get("marks"),
                    knowledge_dimension=slot_kd,
                    used_topics=unit_topics,
                    force_regenerate=force_regenerate,
                )
            except Exception as e:
                rec = {"valid": False, "reason": f"generation exception: {e}"}

            with seen_lock:
                is_dup = is_duplicate(rec.get("question", ""), seen_questions) if rec.get("valid") else False

            if not rec.get("valid") or is_dup:
                slot_rejected.append(rec)
                continue

            try:
                ans_rec = generate_answer(
                    rec["question"],
                    chunk,
                    slot.get("marks"),
                    client,
                    rng=slot_rng,
                    force_regenerate=force_regenerate,
                )
            except Exception as e:
                ans_rec = {"valid": False, "reason": f"answer exception: {e}"}

            if not ans_rec.get("valid"):
                rec["valid"] = False
                rec["reason"] = f"answer generation failed: {ans_rec.get('reason')}"
                slot_rejected.append(rec)
                continue

            rec["answer"] = ans_rec["answer"]
            rec["answer_alternatives"] = ans_rec["answer_alternatives"]
            rec["co"] = slot.get("co", "CO1")
            rec["id"] = slot.get("id") or slot.get("label") or f"Q{slot_idx + 1:03d}"
            rec["unit"] = slot.get("unit")
            rec["question_no"] = slot.get("question_no")
            rec["sub_index"] = slot.get("sub_index")
            rec["marks"] = slot.get("marks")
            rec["alternative_group"] = alt_group
            rec["book_id"] = chunk.get("book_id")
            rec["chapter"] = chunk.get("chapter")
            rec["page"] = chunk.get("page_start")
            rec["knowledge_dimension"] = slot_kd
            rec["kd"] = slot_kd
            rec["review_flags"] = []
            rec["needs_review"] = False

            if ans_rec.get("ungrounded_sentences"):
                rec["ungrounded_sentences"] = ans_rec["ungrounded_sentences"]
                rec["needs_review"] = True
                rec["review_flags"].append(f"ungrounded sentences in model answer ({len(ans_rec['ungrounded_sentences'])})")

            cid = chunk.get("chunk_id")
            if cid:
                with group_lock:
                    if alt_group:
                        used_chunks_per_group.setdefault(alt_group, set()).add(cid)
                    if slot_unit is not None:
                        used_chunks_per_unit.setdefault(slot_unit, set()).add(cid)

            with seen_lock:
                seen_questions.append(rec["question"])
                if slot_unit is not None:
                    seen_questions_in_unit.setdefault(slot_unit, []).append(rec["question"])

            return {"status": "ok", "question": rec, "rejected": slot_rejected, "slot_idx": slot_idx}

        # Try a fresh chunk set outside candidate pool once more before falling back
        outside_pool = [c for c in candidate_chunks if c not in pool and is_content_chunk(c)]
        if outside_pool:
            fresh_chunk = slot_rng.choice(outside_pool)
            try:
                rec = generate_question(
                    fresh_chunk,
                    slot["level"],
                    client,
                    slot_rng,
                    marks=slot.get("marks"),
                    knowledge_dimension=slot_kd,
                    used_topics=unit_topics,
                    force_regenerate=force_regenerate,
                )
                if rec.get("valid"):
                    ans_rec = generate_answer(rec["question"], fresh_chunk, slot.get("marks"), client, rng=slot_rng, force_regenerate=force_regenerate)
                    if ans_rec.get("valid"):
                        rec["answer"] = ans_rec["answer"]
                        rec["answer_alternatives"] = ans_rec["answer_alternatives"]
                        rec["co"] = slot.get("co", "CO1")
                        rec["id"] = slot.get("id") or slot.get("label") or f"Q{slot_idx + 1:03d}"
                        rec["unit"] = slot.get("unit")
                        rec["question_no"] = slot.get("question_no")
                        rec["sub_index"] = slot.get("sub_index")
                        rec["marks"] = slot.get("marks")
                        rec["alternative_group"] = alt_group
                        rec["book_id"] = fresh_chunk.get("book_id")
                        rec["chapter"] = fresh_chunk.get("chapter")
                        rec["page"] = fresh_chunk.get("page_start")
                        rec["knowledge_dimension"] = slot_kd
                        rec["kd"] = slot_kd
                        rec["review_flags"] = []
                        rec["needs_review"] = False
                        return {"status": "ok", "question": rec, "rejected": slot_rejected, "slot_idx": slot_idx}
            except Exception:
                pass

        # All retries failed: build placeholder (Problem 1)
        fail_reason = slot_rejected[-1].get("reason", "Generation retries exhausted") if slot_rejected else "Generation failed"
        placeholder_q = {
            "question": f"[NEEDS REVIEW: {fail_reason}]",
            "answer": f"[NEEDS REVIEW: Model answer generation failed - {fail_reason}]",
            "valid": False,
            "needs_review": True,
            "review_reason": fail_reason,
            "review_flags": [f"generation_failed: {fail_reason}"],
            "marks": slot.get("marks", 0),
            "unit": slot.get("unit"),
            "co": slot.get("co", "CO1"),
            "bloom": slot.get("level") or slot.get("bloom", "L2"),
            "level": slot.get("level") or slot.get("bloom", "L2"),
            "bt_level": slot.get("bt_level") or slot.get("level") or "L2",
            "knowledge_dimension": slot_kd,
            "kd": slot_kd,
            "id": slot.get("id") or slot.get("label") or f"Q{slot_idx + 1:03d}",
            "question_no": slot.get("question_no"),
            "sub_index": slot.get("sub_index"),
            "alternative_group": alt_group,
            "book_id": "",
            "chapter": "",
            "page": "",
        }
        return {"status": "placeholder", "question": placeholder_q, "slot": slot, "rejected": slot_rejected, "slot_idx": slot_idx}

    results_by_index: list[Optional[dict[str, Any]]] = [None] * len(slots)

    if num_workers <= 1 or len(slots) <= 1:
        for idx, slot in enumerate(slots):
            res = _process_single_slot(idx, slot)
            results_by_index[idx] = res
            if on_progress:
                try:
                    on_progress(res.get("question") or slot)
                except Exception:
                    pass
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(_process_single_slot, idx, slot): idx
                for idx, slot in enumerate(slots)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    results_by_index[idx] = res
                    if on_progress:
                        try:
                            on_progress(res.get("question") or slots[idx])
                        except Exception:
                            pass
                except Exception as ex:
                    results_by_index[idx] = {
                        "status": "unfilled",
                        "slot": slots[idx],
                        "rejected": [{"valid": False, "reason": str(ex)}],
                        "slot_idx": idx,
                    }
                    if on_progress:
                        try:
                            on_progress(slots[idx])
                        except Exception:
                            pass

    questions = []
    rejected = []
    unfilled = []
    for item in results_by_index:
        if not item:
            continue
        if item.get("rejected"):
            rejected.extend(item["rejected"])
        if item["status"] == "ok":
            questions.append(item["question"])
        elif item["status"] == "placeholder":
            if keep_placeholders:
                questions.append(item["question"])
            unfilled.append(item.get("slot", item["question"]))
        else:
            unfilled.append(item["slot"])

    # Post-generation embedding checks
    if questions:
        from retrieval.co_mapper import _m
        embedder = _m()

        # 1. Duplicate check (Problem 2.3)
        if len(questions) >= 2 and embedder is not None:
            dups = check_question_duplicates(questions, embedder)
            for orig_i, orig_j, sim in dups:
                q_j = questions[orig_j]
                q_j["needs_review"] = True
                q_j.setdefault("review_flags", []).append(
                    f"duplicate question: cosine similarity {sim:.2f} >= {QUESTION_SIMILARITY_DUPLICATE_THRESHOLD} with {questions[orig_i].get('id', 'earlier slot')}"
                )

        # 2. CO similarity check (Problem 3.2) - Vectorized & Batched
        if outcome_descs and embedder is not None:
            candidate_qs = [
                (idx, q_rec, q_rec.get("co"), outcome_descs.get(q_rec.get("co")))
                for idx, q_rec in enumerate(questions)
                if q_rec.get("co") and outcome_descs.get(q_rec.get("co"))
                and not q_rec.get("needs_review") and q_rec.get("valid", True)
            ]
            if candidate_qs:
                try:
                    import numpy as np
                    unique_cos = list({co_code: co_desc for _, _, co_code, co_desc in candidate_qs}.items())
                    co_keys = [c[0] for c in unique_cos]
                    co_texts = [c[1] for c in unique_cos]
                    q_texts = [c[1].get("question", "") for c in candidate_qs]

                    all_encoded = embedder.encode(co_texts + q_texts, normalize_embeddings=True)
                    co_embs_map = {co_keys[k]: all_encoded[k] for k in range(len(co_keys))}
                    q_embs = all_encoded[len(co_keys):]

                    for k, (orig_idx, q_rec, co_code, _) in enumerate(candidate_qs):
                        co_vec = co_embs_map.get(co_code)
                        if co_vec is not None:
                            sim_val = float(np.dot(q_embs[k], co_vec))
                            if sim_val < CO_SIMILARITY_THRESHOLD:
                                q_rec["needs_review"] = True
                                q_rec.setdefault("review_flags", []).append(
                                    f"CO mismatch - review (similarity {sim_val:.2f} < {CO_SIMILARITY_THRESHOLD})"
                                )
                except Exception:
                    pass

        # 3. Consistency across paper (Problem 5.6)
        inconsistencies = check_paper_identifier_consistency(questions)
        for q_idx, inc_msg in inconsistencies:
            if q_idx < len(questions):
                questions[q_idx].setdefault("review_flags", []).append(inc_msg)

    # Normalize question numbers only if paper starts above Q1 (Problem 1.4)
    if questions and any(q.get("question_no") is not None for q in questions):
        q_nos = [q["question_no"] for q in questions if q.get("question_no") is not None]
        if q_nos and min(q_nos) > 1:
            questions = normalize_paper_question_numbers(questions)

    return {"questions": questions, "rejected": rejected, "unfilled": unfilled}


def generate_full_paper(
    chunks: list[dict[str, Any]],
    blueprint: Any,
    outcomes: dict[str, str],
    unit_sources: Optional[dict[Any, Any]],
    client: Any,
    seed: Optional[int] = None,
    on_progress: Optional[Any] = None,
    workers: Optional[int] = None,
    force_regenerate: bool = False,
    keep_placeholders: bool = True,
) -> dict[str, Any]:
    """Generate a complete exam paper with questions and model answers from blueprint and outcomes."""
    from retrieval.co_mapper import score_chunks_against_outcomes

    # 1. Score chunks against outcomes
    scored_chunks = score_chunks_against_outcomes(chunks, outcomes)

    # 2. Extract slots and report from blueprint
    if isinstance(blueprint, dict):
        slots = blueprint.get("slots", [])
        blueprint_report = blueprint.get("report", {})
    else:
        slots = list(blueprint)
        blueprint_report = {}

    # 3. Generate paper by slots
    result = generate_paper_by_slots(
        chunks=scored_chunks,
        slots=slots,
        client=client,
        seed=seed,
        on_progress=on_progress,
        unit_sources=unit_sources,
        workers=workers,
        force_regenerate=force_regenerate,
        keep_placeholders=keep_placeholders,
        outcome_descs=outcomes if isinstance(outcomes, dict) else None,
    )

    return {
        "questions": result["questions"],
        "rejected": result["rejected"],
        "unfilled": result["unfilled"],
        "blueprint_report": blueprint_report,
    }
