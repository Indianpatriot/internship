"""Step 3c / Issue 4 - question generation with validation, retry and de-duplication."""
import random
import re

from blooms.verb_bank import first_word, normalize_level, sample_verbs, validate_question
from config import MAX_CONTEXT_WORDS, MIN_CONTEXT_WORDS
from generation.prompts import build_prompt


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


def generate_question(chunk, level, client, rng=None, max_retries=3, marks=None, k_verbs=4):
    level = normalize_level(level)
    rng = rng or random.Random()
    context = " ".join(chunk["text"].split()[:MAX_CONTEXT_WORDS])
    q, reason, attempts = "", "no output", 0
    for attempts in range(1, max_retries + 1):
        verbs = sample_verbs(level, k_verbs, rng)
        prompt = build_prompt(context, level, verbs, marks)
        raw = client.generate(prompt, temperature=min(0.3 + 0.2 * (attempts - 1), 0.9))
        q = clean_output(raw)
        ok, reason = validate_question(q, level)
        if ok:
            break
    return {
        "question": q, "level": level, "verb": first_word(q), "marks": marks,
        "valid": ok, "reason": reason, "attempts": attempts,
        "chunk_id": chunk.get("chunk_id"), "chapter": chunk.get("chapter"),
        "section": chunk.get("section"), "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
    }


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
