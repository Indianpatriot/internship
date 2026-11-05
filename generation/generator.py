"""Step 3c / Issue 4 - question and answer generation with validation, retry, de-duplication, and multi-book assembly."""
import random
import re
from typing import Any, Optional

from blooms.revised_taxonomy import normalize_knowledge_dimension
from blooms.verb_bank import first_word, normalize_level, sample_verbs, validate_question
from config import ANSWER_MIN_SIMILARITY, MAX_CONTEXT_WORDS, MIN_CONTEXT_WORDS
from generation.prompts import build_answer_prompt, build_prompt
from retrieval.sources import select_chunks_for_unit


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
):
    level = normalize_level(level)
    rng = rng or random.Random()
    kd = normalize_knowledge_dimension(knowledge_dimension) if knowledge_dimension else None
    context = " ".join(chunk["text"].split()[:MAX_CONTEXT_WORDS])
    q, reason, attempts = "", "no output", 0
    ok = False
    for attempts in range(1, max_retries + 1):
        verbs = sample_verbs(level, k_verbs, rng)
        prompt = build_prompt(context, level, verbs, marks=marks, knowledge_dimension=kd)
        raw = client.generate(prompt, temperature=min(0.3 + 0.2 * (attempts - 1), 0.9))
        q = clean_output(raw)
        ok, reason = validate_question(q, level)
        if ok:
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
) -> dict[str, Any]:
    """Generate model answer grounded in chunk context.

    Splits model output on lines that are exactly OR (case-insensitive) into alternatives.
    Validates non-empty, minimum length for marks, and cosine similarity with chunk text.
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

    for attempt in range(1, max_retries + 1):
        temp = min(0.3 + 0.2 * (attempt - 1), 0.9)
        prompt = build_answer_prompt(question, context, marks=marks)
        raw = client.generate(prompt, temperature=temp)

        # Clean output fences
        text = re.sub(r"^```\w*|```$", "", (raw or "").strip(), flags=re.M).strip()
        lines = [line.strip() for line in text.splitlines()]

        # Split on lines that are exactly "OR" (case-insensitive)
        alternatives: list[str] = []
        cur_buf: list[str] = []
        for line in lines:
            if line.upper() == "OR":
                alt_text = "\n".join(cur_buf).strip()
                if alt_text:
                    alternatives.append(alt_text)
                cur_buf = []
            else:
                cur_buf.append(line)
        if cur_buf:
            alt_text = "\n".join(cur_buf).strip()
            if alt_text:
                alternatives.append(alt_text)

        last_alternatives = alternatives
        last_answer = "\nOR\n".join(alternatives).strip()

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

        # Validation 3: grounded cosine similarity check
        if should_check_similarity:
            from retrieval.co_mapper import _m
            embedder = _m()
            ans_vec = embedder.encode([last_answer], normalize_embeddings=True)
            chunk_vec = embedder.encode([chunk.get("text", "")], normalize_embeddings=True)
            sim = float((ans_vec @ chunk_vec.T)[0, 0])
            if sim < ANSWER_MIN_SIMILARITY:
                reason = f"similarity {sim:.3f} below threshold {ANSWER_MIN_SIMILARITY}"
                continue

        ok = True
        reason = "ok"
        break

    return {
        "answer": last_answer,
        "answer_alternatives": last_alternatives,
        "valid": ok,
        "reason": reason,
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


def generate_paper_by_slots(chunks, slots, client, seed=None, on_progress=None, unit_sources=None):
    """Generate questions and model answers matching blueprint slots.

    Args:
        chunks: List of text chunk dicts.
        slots: Blueprint slot dictionaries.
        client: LLM client (OllamaClient or MockClient).
        seed: Random seed for repeatability.
        on_progress: Optional callback on question accepted.
        unit_sources: Optional mapping {unit: {book_id: [chapters] or None}}.
    """
    rng = random.Random(seed)
    usable = [c for c in chunks if c.get("word_count", len(c["text"].split())) >= MIN_CONTEXT_WORDS]
    questions, rejected, unfilled, seen = [], [], [], []

    # Track chunks used per alternative group to avoid chunk reuse within choice alternatives
    used_chunks_per_group: dict[str, set[str]] = {}

    for slot in slots:
        slot_unit = slot.get("unit")
        alt_group = slot.get("alternative_group")

        # 1. Filter chunks per slot's unit BEFORE ranking by co_scores
        candidate_chunks = select_chunks_for_unit(usable, unit_sources, slot_unit)
        if not candidate_chunks:
            unfilled.append(slot)
            continue

        # 2. Avoid reusing chunk within the same alternative group
        if alt_group:
            used_ids = used_chunks_per_group.get(alt_group, set())
            filtered = [c for c in candidate_chunks if c.get("chunk_id") not in used_ids]
            pool_source = filtered if filtered else candidate_chunks
        else:
            pool_source = candidate_chunks

        slot_co = slot.get("co")
        if slot_co:
            pool = sorted(
                pool_source,
                key=lambda c: c.get("co_scores", {}).get(slot_co, float("-inf")),
                reverse=True,
            )[:10]
        else:
            pool = pool_source[:10]

        rng.shuffle(pool)
        slot_kd = slot.get("knowledge_dimension") or slot.get("kd") or "Conceptual"

        for chunk in pool * 2:
            rec = generate_question(
                chunk,
                slot["level"],
                client,
                rng,
                marks=slot.get("marks"),
                knowledge_dimension=slot_kd,
            )
            if not rec["valid"] or is_duplicate(rec["question"], seen):
                rejected.append(rec)
                continue

            # Generate model answer - question is only accepted if answer is valid
            ans_rec = generate_answer(
                rec["question"],
                chunk,
                slot.get("marks"),
                client,
                rng=rng,
            )
            if not ans_rec["valid"]:
                rec["valid"] = False
                rec["reason"] = f"answer generation failed: {ans_rec['reason']}"
                rejected.append(rec)
                continue

            # Populate provenance and slot linkage fields
            rec["answer"] = ans_rec["answer"]
            rec["answer_alternatives"] = ans_rec["answer_alternatives"]
            rec["co"] = slot.get("co", "CO1")
            rec["id"] = slot.get("id") or slot.get("label") or f"Q{len(questions) + 1:03d}"
            rec["unit"] = slot.get("unit")
            rec["question_no"] = slot.get("question_no")
            rec["sub_index"] = slot.get("sub_index")
            rec["marks"] = slot.get("marks")
            rec["alternative_group"] = alt_group
            rec["book_id"] = chunk.get("book_id")
            rec["chapter"] = chunk.get("chapter")
            rec["page"] = chunk.get("page_start")

            if "part" in slot:
                rec["part"] = slot["part"]
            rec["knowledge_dimension"] = slot_kd
            rec["kd"] = slot_kd

            if alt_group and chunk.get("chunk_id"):
                used_chunks_per_group.setdefault(alt_group, set()).add(chunk["chunk_id"])

            seen.append(rec["question"])
            questions.append(rec)
            if on_progress:
                on_progress(rec)
            break
        else:
            unfilled.append(slot)

    return {"questions": questions, "rejected": rejected, "unfilled": unfilled}


def generate_full_paper(
    chunks: list[dict[str, Any]],
    blueprint: Any,
    outcomes: dict[str, str],
    unit_sources: Optional[dict[Any, Any]],
    client: Any,
    seed: Optional[int] = None,
    on_progress: Optional[Any] = None,
) -> dict[str, Any]:
    """Generate a complete exam paper with questions and model answers from blueprint and outcomes.

    1. Scores chunks against outcomes using semantic embeddings (score_chunks_against_outcomes).
    2. Generates question and answer for every slot using unit-filtered chunks and alternative anti-collision.
    3. Returns {"questions": [QuestionRecord...], "rejected": [...], "unfilled": [...], "blueprint_report": {...}}.
    """
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
    )

    return {
        "questions": result["questions"],
        "rejected": result["rejected"],
        "unfilled": result["unfilled"],
        "blueprint_report": blueprint_report,
    }
