"""Syllabus / Scheme parsing and semantic chapter mapping module.

Extracts:
- Units / Modules with numbers (Arabic or Roman), titles, topics, and lecture hours
- Course Outcomes (CO1..COn)
- Textbooks and references
- Fuzzy-matched unit sources and semantic topic-to-chapter embeddings
"""
import difflib
import math
import re
from typing import Any, Optional

import fitz  # PyMuPDF

ROMAN_NUMS = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
    "xi": 11,
    "xii": 12,
}

MANUAL_CONFIDENCE_THRESHOLD = 0.5

UNIT_HEADER_RE = re.compile(
    r"^(?:unit|module|part)\s*[:.\-–—]?\s*([0-9ivxlc]+)\b\s*[:.\-–—]?\s*(.*)$",
    re.I,
)
HOURS_RE = re.compile(r"\(?\b(\d+)\s*(?:hours|hrs|hr|teaching hours|contact hours)\b\)?", re.I)
CO_LINE_RE = re.compile(
    r"^(?:(?:course\s*outcome|learning\s*outcome|course\s*learning\s*outcome|outcome|co|lo|clo)\s*[-_.]?\s*([0-9]+))(?:\s*[:.\-–—)]\s*|\s+)(.*)$",
    re.I,
)
CO_NUMBERED_RE = re.compile(
    r"^(?:\[?\(?([0-9]+)\)?\]?\.?)\s*[:.\-–—)]\s*(.*)$",
    re.I,
)
CO_BULLET_RE = re.compile(r"^[•*–—\-]\s*(.*)$")
TEXTBOOK_HEADER_RE = re.compile(
    r"^(?:text\s*books?|reference\s*books?|references?|prescribed\s*books?|suggested\s*readings?)\b",
    re.I,
)
OUTCOMES_HEADER_RE = re.compile(
    r"^(?:course\s*outcomes?|learning\s*outcomes?|course\s*learning\s*outcomes?)\b",
    re.I,
)


def _parse_unit_number(raw: str) -> Optional[int]:
    """Parse Arabic or Roman numeral into an integer."""
    s = raw.strip().lower()
    if s.isdigit():
        return int(s)
    if s in ROMAN_NUMS:
        return ROMAN_NUMS[s]
    return None


def parse_scheme_text(
    text: str,
    library: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Parse raw syllabus / scheme text to extract units, outcomes, and textbooks.

    Args:
        text: Plain syllabus text (extracted from PDF or pasted).
        library: Optional loaded library catalog for fuzzy-matching textbooks.

    Returns:
        {
            "units": [{"unit": int, "title": str, "topics": list[str], "hours": int|None}, ...],
            "outcomes": {"CO1": str, ...},
            "unit_sources": {unit: {book_id: [chapters] or None}},
            "textbooks": list[str],
            "confidence": float,
            "warnings": list[str]
        }
    """
    warnings: list[str] = []
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]

    units_data: list[dict[str, Any]] = []
    outcomes: dict[str, str] = {}
    textbooks: list[str] = []

    current_section = "GENERAL"  # GENERAL, CO, UNITS, TEXTBOOKS
    current_unit: Optional[dict[str, Any]] = None
    current_unit_lines: list[str] = []
    current_co_code: Optional[str] = None

    def flush_unit():
        nonlocal current_unit, current_unit_lines
        if current_unit is not None:
            combined_topics = []
            for ln in current_unit_lines:
                # Remove hours annotations
                cleaned_ln = HOURS_RE.sub("", ln).strip()
                # Split on commas, semicolons, bullets, and slashes
                tokens = re.split(r"[,;•*|\n]+|\s+-\s+", cleaned_ln)
                for tok in tokens:
                    item = tok.strip(" .()[]:\"'")
                    # Drop trivial single words or stop words
                    if item and len(item) > 2 and not re.match(r"^(etc|e\.g\.|i\.e\.|hours?|\d+)$", item, re.I):
                        combined_topics.append(item)
            current_unit["topics"] = combined_topics
            units_data.append(current_unit)
            current_unit = None
            current_unit_lines = []

    for line in lines:
        # Check Section Transitions
        if OUTCOMES_HEADER_RE.match(line):
            flush_unit()
            current_section = "CO"
            continue
        if TEXTBOOK_HEADER_RE.match(line):
            flush_unit()
            current_section = "TEXTBOOKS"
            continue

        # Check Unit / Module Header
        unit_match = UNIT_HEADER_RE.match(line)
        if unit_match:
            flush_unit()
            current_section = "UNITS"
            num_raw = unit_match.group(1)
            rest = unit_match.group(2).strip()

            unit_num = _parse_unit_number(num_raw)
            if unit_num is None:
                unit_num = len(units_data) + 1

            # Extract hours from title line if present
            hours_val = None
            hrs_match = HOURS_RE.search(rest) or HOURS_RE.search(line)
            if hrs_match:
                hours_val = int(hrs_match.group(1))
                rest = HOURS_RE.sub("", rest).strip(" -:()[]")

            current_unit = {
                "unit": unit_num,
                "title": rest or f"Unit {unit_num}",
                "topics": [],
                "hours": hours_val,
            }
            continue

        # Process Section Contents
        if current_section == "UNITS" and current_unit is not None:
            # Check if this line specifies hours for the current unit
            if current_unit["hours"] is None:
                hrs_match = HOURS_RE.search(line)
                if hrs_match:
                    current_unit["hours"] = int(hrs_match.group(1))
            current_unit_lines.append(line)

        elif current_section == "CO" or CO_LINE_RE.match(line):
            co_match = CO_LINE_RE.match(line)
            num_match = CO_NUMBERED_RE.match(line) if current_section == "CO" else None
            bullet_match = CO_BULLET_RE.match(line) if current_section == "CO" else None
            if co_match:
                code_num = co_match.group(1)
                co_text = co_match.group(2).strip()
                current_co_code = f"CO{code_num}"
                outcomes[current_co_code] = co_text
            elif num_match:
                code_num = num_match.group(1)
                co_text = num_match.group(2).strip()
                current_co_code = f"CO{code_num}"
                outcomes[current_co_code] = co_text
            elif bullet_match:
                code_num = str(len(outcomes) + 1)
                current_co_code = f"CO{code_num}"
                outcomes[current_co_code] = bullet_match.group(1).strip()
            elif current_co_code:
                # Continuation line for the current CO
                outcomes[current_co_code] += " " + line.strip()
            elif current_section == "CO" and line.strip():
                code_num = str(len(outcomes) + 1)
                current_co_code = f"CO{code_num}"
                outcomes[current_co_code] = line.strip()

        elif current_section == "TEXTBOOKS":
            # Strip list numbers or bullets: '1.', '[1]', '-', '*'
            cleaned_tb = re.sub(r"^[\s\d.\[\]•*-]+", "", line).strip()
            if cleaned_tb and len(cleaned_tb) > 5:
                textbooks.append(cleaned_tb)

    flush_unit()

    # Sort units by unit number
    units_data.sort(key=lambda u: u["unit"])

    # Fuzzy match textbooks against library books
    unit_sources: dict[Any, Any] = {}
    matched_books: dict[str, str] = {}
    library_books = library.get("books", {}) if library else {}

    if library_books and textbooks:
        for tb in textbooks:
            tb_lower = tb.lower()
            for bid, binfo in library_books.items():
                title = binfo.get("title", "")
                title_lower = title.lower()
                # Use SequenceMatcher and substring inclusion
                ratio = difflib.SequenceMatcher(None, title_lower, tb_lower).ratio()
                if ratio > 0.5 or title_lower in tb_lower:
                    matched_books[bid] = title

    # Build unit_sources
    for u in units_data:
        u_num = u["unit"]
        # If library books fuzzy matched, check if any book maps
        if matched_books:
            # Check if unit topics/title mentions a specific book
            u_text = (u["title"] + " " + " ".join(u["topics"])).lower()
            assigned = None
            for bid, title in matched_books.items():
                if bid.lower() in u_text or title.lower() in u_text:
                    assigned = bid
                    break
            if assigned:
                unit_sources[u_num] = {assigned: None}
            else:
                # Leave None unless specific match
                unit_sources[u_num] = None
        else:
            unit_sources[u_num] = None

    # Calculate Confidence Score (0.0 to 1.0)
    conf_units = min(0.35, (len(units_data) / 5.0) * 0.35)
    total_topics = sum(len(u.get("topics", [])) for u in units_data)
    conf_topics = min(0.30, (total_topics / 15.0) * 0.30) if units_data else 0.0
    conf_outcomes = min(0.25, (len(outcomes) / 4.0) * 0.25)
    conf_textbooks = 0.10 if textbooks else 0.0

    confidence = round(conf_units + conf_topics + conf_outcomes + conf_textbooks, 2)

    # Compile Warnings
    if not units_data:
        warnings.append("No units or modules found in syllabus scheme.")
    elif total_topics == 0:
        warnings.append("No unit topics could be identified from the text.")

    if not outcomes:
        warnings.append("No course outcomes found in scheme; please enter them manually.")

    if not textbooks:
        warnings.append("No textbooks or references section found in scheme.")

    result = {
        "units": units_data,
        "outcomes": outcomes,
        "unit_sources": unit_sources,
        "textbooks": textbooks,
        "confidence": confidence,
        "warnings": warnings,
    }

    # Pre-fill unit -> book/chapter mapping so the user only has to confirm it.
    if library_books and units_data:
        prefill_unit_sources(result, library)

    # Flag when the UI should drop to manual entry instead of confirm-only.
    result["needs_manual"] = scheme_needs_manual(result)
    if result["needs_manual"]:
        warnings.append(
            "Scheme could not be parsed reliably; please enter units and outcomes manually."
        )
    return result


def scheme_needs_manual(scheme: dict[str, Any]) -> bool:
    """True when parsing found too little for a confirm-only screen."""
    return (
        not scheme.get("units")
        or not scheme.get("outcomes")
        or float(scheme.get("confidence", 0.0)) < MANUAL_CONFIDENCE_THRESHOLD
    )


def manual_scheme_template(n_units: int = 5, n_outcomes: int = 5) -> dict[str, Any]:
    """Blank scheme in the same shape as parse_scheme output, for manual entry."""
    return {
        "units": [
            {"unit": i, "title": f"Unit {i}", "topics": [], "hours": None}
            for i in range(1, n_units + 1)
        ],
        "outcomes": {f"CO{i}": "" for i in range(1, n_outcomes + 1)},
        "unit_sources": {i: None for i in range(1, n_units + 1)},
        "textbooks": [],
        "confidence": 0.0,
        "warnings": [],
        "needs_manual": True,
    }


def merge_manual_entry(
    scheme: Optional[dict[str, Any]],
    units: Optional[list[dict[str, Any]]] = None,
    outcomes: Optional[dict[str, str]] = None,
    unit_sources: Optional[dict[Any, Any]] = None,
) -> dict[str, Any]:
    """Overlay user-entered/confirmed values on a (possibly failed) parse result.

    Anything the user supplies wins; anything omitted keeps the parsed value.
    Empty outcome descriptions are dropped. The result is marked as confirmed.
    """
    base = dict(scheme) if scheme else manual_scheme_template(0, 0)
    if units is not None:
        base["units"] = sorted(units, key=lambda u: u.get("unit", 0))
    if outcomes is not None:
        base["outcomes"] = {k: v for k, v in outcomes.items() if str(v).strip()}
    if unit_sources is not None:
        base["unit_sources"] = unit_sources
    base["needs_manual"] = False
    base["confirmed"] = True
    return base


def parse_scheme(
    pdf_path: str,
    library: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Extract and parse syllabus scheme text from a PDF file.

    Args:
        pdf_path: File path to syllabus PDF.
        library: Optional loaded library catalog.

    Returns:
        Structured scheme dictionary with units, outcomes, unit_sources, textbooks, confidence, and warnings.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text_parts = []
        for page in doc:
            full_text_parts.append(page.get_text())
        doc.close()
        text = "\n".join(full_text_parts)
    except Exception as e:
        failed = manual_scheme_template(0, 0)
        failed["unit_sources"] = {}
        failed["warnings"] = [f"Could not read PDF file: {e}"]
        return failed

    if not text.strip():
        failed = manual_scheme_template(0, 0)
        failed["unit_sources"] = {}
        failed["warnings"] = [
            "No text found in the scheme PDF (scanned image?); please enter it manually."
        ]
        return failed

    try:
        return parse_scheme_text(text, library=library)
    except Exception as e:  # never let a parser bug block the UI
        failed = manual_scheme_template(0, 0)
        failed["unit_sources"] = {}
        failed["warnings"] = [f"Scheme parsing failed: {e}"]
        return failed


def map_topics_to_chapters(
    scheme_units: list[dict[str, Any]],
    library: dict[str, Any],
) -> dict[Any, list[dict[str, Any]]]:
    """Map each syllabus unit to library chapters by semantic embedding similarity.

    Uses the SentenceTransformer embedding model from retrieval.co_mapper.
    Returns per unit the chapters sorted descending by relevance score.
    """
    books = library.get("books", {})
    chunks = library.get("chunks", [])
    if not scheme_units or not books or not chunks:
        return {u.get("unit", i + 1): [] for i, u in enumerate(scheme_units)}

    try:
        from retrieval.co_mapper import _m

        model = _m()
    except Exception:
        # Embedding model unavailable (not installed / offline): keyword overlap.
        return _lexical_chapter_map(scheme_units, books, chunks)

    # Pre-compile chapter texts from chunks
    chapter_entries: list[tuple[str, str, str]] = []  # (book_id, chapter, aggregated_text)
    for bid, binfo in books.items():
        for ch in binfo.get("chapters", []):
            ch_chunks = [
                c.get("text", "")
                for c in chunks
                if c.get("book_id") == bid and c.get("chapter") == ch
            ]
            agg_text = f"{ch}. " + " ".join(ch_chunks[:4])
            chapter_entries.append((bid, ch, agg_text))

    if not chapter_entries:
        return {u.get("unit", i + 1): [] for i, u in enumerate(scheme_units)}

    chapter_texts = [entry[2] for entry in chapter_entries]
    chapter_vecs = model.encode(chapter_texts, normalize_embeddings=True)

    result: dict[Any, list[dict[str, Any]]] = {}

    for u in scheme_units:
        u_num = u.get("unit")
        title = u.get("title", "")
        topics = u.get("topics", [])
        query_text = f"{title}. " + ", ".join(topics)

        query_vec = model.encode([query_text], normalize_embeddings=True)
        sims = (query_vec @ chapter_vecs.T)[0]

        matches = []
        for (bid, ch, _), score in zip(chapter_entries, sims):
            matches.append({
                "book_id": bid,
                "chapter": ch,
                "score": round(float(score), 3),
            })

        matches.sort(key=lambda m: m["score"], reverse=True)
        result[u_num] = matches

    return result


_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "are", "was", "using",
    "use", "its", "their", "of", "in", "on", "to", "a", "an", "or", "by", "is", "as",
    "chapter", "unit", "module", "introduction", "basics", "overview",
}


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z][a-z0-9]{2,}", (text or "").lower()) if t not in _STOPWORDS
    }


def _lexical_chapter_map(
    scheme_units: list[dict[str, Any]],
    books: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[Any, list[dict[str, Any]]]:
    """Keyword-overlap fallback for map_topics_to_chapters (same return shape)."""
    entries: list[tuple[str, str, set[str], set[str]]] = []
    for bid, binfo in books.items():
        for ch in binfo.get("chapters", []):
            body = " ".join(
                c.get("text", "")
                for c in chunks
                if c.get("book_id") == bid and c.get("chapter") == ch
            )
            entries.append((bid, ch, _tokens(ch), _tokens(body)))

    result: dict[Any, list[dict[str, Any]]] = {}
    for i, u in enumerate(scheme_units):
        u_num = u.get("unit", i + 1)
        q = _tokens(u.get("title", "") + " " + " ".join(u.get("topics", [])))
        matches = []
        for bid, ch, title_tok, body_tok in entries:
            if q:
                hit_body = len(q & body_tok) / math.sqrt(len(q))
                hit_title = 2.0 * len(q & title_tok) / math.sqrt(len(q))
                score = (hit_body + hit_title) / 3.0
            else:
                score = 0.0
            matches.append({"book_id": bid, "chapter": ch, "score": round(float(score), 3)})
        matches.sort(key=lambda m: m["score"], reverse=True)
        result[u_num] = matches
    return result


def prefill_unit_sources(
    scheme: dict[str, Any],
    library: dict[str, Any],
    top_k: int = 1,
    min_score: float = 0.0,
) -> dict[Any, Any]:
    """Fill scheme["unit_sources"] with best-guess {book_id: [chapters]} per unit.

    Units already matched to a specific textbook keep that book but still get
    chapters chosen within it. If nothing scores above min_score the page-order
    split from ingestion.library.suggest_unit_sources is used. Mutates and
    returns scheme["unit_sources"]; the user only has to confirm or tweak it.
    """
    from ingestion.library import suggest_unit_sources

    units = scheme.get("units", [])
    if not units or not library.get("books"):
        return scheme.get("unit_sources", {})

    matches = map_topics_to_chapters(units, library)
    fallback = suggest_unit_sources(library, [u["unit"] for u in units])
    existing = scheme.get("unit_sources") or {}
    filled: dict[Any, Any] = {}

    for u in units:
        n = u["unit"]
        ranked = [m for m in matches.get(n, []) if m["score"] > min_score]
        pinned = existing.get(n)
        if pinned:
            ranked = [m for m in ranked if m["book_id"] in pinned]
        if not ranked:
            filled[n] = pinned or fallback.get(n) or None
            continue
        picked: dict[str, list[str]] = {}
        for m in ranked[:top_k]:
            picked.setdefault(m["book_id"], []).append(m["chapter"])
        filled[n] = picked

    scheme["unit_sources"] = filled
    return filled
