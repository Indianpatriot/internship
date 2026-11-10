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

from blooms.revised_taxonomy import normalize_knowledge_dimension
from blooms.verb_bank import first_word, get_verbs, normalize_level, sample_verbs, validate_question
from config import (
    ANSWER_MIN_SIMILARITY,
    GENERATION_WORKERS,
    MAX_CONTEXT_WORDS,
    MIN_CONTEXT_WORDS,
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

BANNED_ANSWER_PATTERNS = [
    r"\bthe\s+text\s+does\s+not\b",
    r"\bthe\s+provided\s+text\b",
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
    force_regenerate=False,
):
    level = normalize_level(level)
    rng = rng or random.Random()
    kd = normalize_knowledge_dimension(knowledge_dimension) if knowledge_dimension else None
    context = " ".join(chunk["text"].split()[:MAX_CONTEXT_WORDS])
    has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?\b", context))
    q, reason, attempts = "", "no output", 0
    ok = False

    # Problem 4: Raise num_predict for questions to 250-500
    num_predict = max(250, min(500, (marks or 5) * 40))
    client_model = getattr(client, "model", "llama3.1:8b")

    for attempts in range(1, max_retries + 1):
        verbs = sample_verbs(level, k_verbs, rng)
        if not has_numbers:
            verbs = [v for v in verbs if v.lower() not in ("compute", "calculate")]
            if not verbs:
                all_v = get_verbs(level)
                verbs = [v for v in all_v if v.lower() not in ("compute", "calculate")]
                verbs = rng.sample(verbs, min(k_verbs, len(verbs))) if verbs else sample_verbs(level, k_verbs, rng)

        prompt = build_prompt(context, level, verbs, marks=marks, knowledge_dimension=kd)
        temp = min(0.3 + 0.2 * (attempts - 1), 0.9)

        cached = get_cached_response(prompt, client_model, temperature=temp, force_regenerate=force_regenerate)
        if cached:
            raw = cached
        else:
            raw = client.generate(prompt, temperature=temp, num_predict=num_predict)

        q = clean_output(raw)
        ok, reason = validate_question(q, level)
        if ok and has_textbook_reference(q):
            ok = False
            reason = "question contains textbook references"
        if ok and is_truncated_text(q, is_question=True):
            ok = False
            reason = "question is truncated"
        if ok and not has_numbers and first_word(q) in ("compute", "calculate"):
            ok = False
            reason = "compute/calculate verb in non-numerical context"

        if ok:
            if not cached and not is_truncated_text(q, is_question=True):
                store_cached_response(prompt, client_model, raw, temperature=temp)
            break
        elif attempts < max_retries:
            verbs = sample_verbs(level, k_verbs, rng)
            if not has_numbers:
                verbs = [v for v in verbs if v.lower() not in ("compute", "calculate")]
            targeted_prompt = (
                f"{prompt}\n\nNote: The question must strictly start with one of: {', '.join(verbs)}. "
                f"Ensure the first word is one of these verbs and ends with '?'. "
                f"Do not reference textbooks, chapters, or figures, and write a complete self-contained question."
            )
            raw = client.generate(targeted_prompt, temperature=temp + 0.1, num_predict=num_predict)
            q = clean_output(raw)
            ok, reason = validate_question(q, level)
            if ok and has_textbook_reference(q):
                ok = False
                reason = "question contains textbook references"
            if ok and is_truncated_text(q, is_question=True):
                ok = False
                reason = "question is truncated"
            if ok and not has_numbers and first_word(q) in ("compute", "calculate"):
                ok = False
                reason = "compute/calculate verb in non-numerical context"
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
    absence of textbook references, non-truncation, and cosine similarity with chunk text.
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

    # Problem 4: Proportional token caps for answers
    if marks <= 5:
        num_predict = 450
    elif marks <= 8:
        num_predict = 750
    else:
        num_predict = 1000

    client_model = getattr(client, "model", "llama3.1:8b")

    for attempt in range(1, max_retries + 1):
        temp = min(0.3 + 0.2 * (attempt - 1), 0.9)
        prompt = build_answer_prompt(question, context, marks=marks)

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

        # Validation 3: banned meta-commentary phrases (Problem 5)
        if has_banned_answer_phrases(last_answer):
            reason = "answer contains banned phrases (meta commentary / apology)"
            continue

        # Validation 4: textbook references (Problem 6)
        if has_textbook_reference(last_answer):
            reason = "answer contains textbook references"
            continue

        # Validation 5: truncation check (Problem 4)
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

        # Validation 6: grounded cosine similarity check
        if should_check_similarity:
            from retrieval.co_mapper import _m
            embedder = _m()
            ans_vec = embedder.encode([last_answer], normalize_embeddings=True)
            chunk_vec = chunk.get("embedding")
            if chunk_vec is None:
                chunk_vec = embedder.encode([chunk.get("text", "")], normalize_embeddings=True)[0]
                chunk["embedding"] = chunk_vec
            if len(chunk_vec.shape) == 1:
                chunk_vec = chunk_vec.reshape(1, -1)
            sim = float((ans_vec @ chunk_vec.T)[0, 0])
            if sim < ANSWER_MIN_SIMILARITY:
                reason = f"similarity {sim:.3f} below threshold {ANSWER_MIN_SIMILARITY}"
                continue

        ok = True
        reason = "ok"
        if not cached and not is_truncated_text(last_answer, is_question=False):
            store_cached_response(prompt, client_model, raw, temperature=temp)
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


def generate_paper_by_slots(
    chunks: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    client: Any,
    seed: Optional[int] = None,
    on_progress: Optional[Any] = None,
    unit_sources: Optional[dict[Any, Any]] = None,
    workers: Optional[int] = None,
    force_regenerate: bool = False,
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

        if alt_group:
            with group_lock:
                used_ids = set(used_chunks_per_group.get(alt_group, set()))
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

        slot_rng.shuffle(pool)
        slot_kd = slot.get("knowledge_dimension") or slot.get("kd") or "Conceptual"
        slot_rejected = []

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

            if "unit" in slot:
                rec["unit"] = slot["unit"]
            rec["knowledge_dimension"] = slot_kd
            rec["kd"] = slot_kd

            if alt_group and chunk.get("chunk_id"):
                with group_lock:
                    used_chunks_per_group.setdefault(alt_group, set()).add(chunk["chunk_id"])

            with seen_lock:
                seen_questions.append(rec["question"])

            return {"status": "ok", "question": rec, "rejected": slot_rejected, "slot_idx": slot_idx}

        return {"status": "unfilled", "slot": slot, "rejected": slot_rejected, "slot_idx": slot_idx}

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
        else:
            unfilled.append(item["slot"])

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
    )

    return {
        "questions": result["questions"],
        "rejected": result["rejected"],
        "unfilled": result["unfilled"],
        "blueprint_report": blueprint_report,
    }
