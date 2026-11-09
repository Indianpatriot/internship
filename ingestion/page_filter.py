"""Non-content page filtering for textbook ingestion.

Drops front-matter (preface, foreword, acknowledgements, copyright, TOC, list of figures/tables)
and back-matter (index, bibliography, references, glossary, book ads) BEFORE chunking, embedding,
CO mapping, and question generation.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

from config import FILTER_USE_LLM
from ingestion.heading_detect import CHAPTER_RE, NUM_RE, detect_headings

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class PageText:
    """Represents the extracted text and metadata of a single PDF page."""

    def __init__(
        self,
        page_number: Optional[int] = None,
        text: str = "",
        lines: Optional[list[dict[str, Any]]] = None,
        page: Optional[int] = None,
    ):
        p = page if page is not None else page_number
        self.page_number = 1 if p is None else int(p)
        self.text = text or ""
        self.lines = list(lines) if lines else []

    @property
    def page(self) -> int:
        return self.page_number

    @page.setter
    def page(self, val: int) -> None:
        self.page_number = int(val)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __getitem__(self, item: str) -> Any:
        if item in ("page", "page_number"):
            return self.page_number
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item) or item in ("page", "page_number", "text", "lines")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PageText):
            return self.page_number == other.page_number and self.text == other.text
        return False

    def __repr__(self) -> str:
        words = len(self.text.split())
        snippet = self.text[:40].replace("\n", " ").strip()
        return f"PageText(page={self.page_number}, words={words}, snippet={snippet!r})"


class SkippedPage(tuple):
    """(page, reason) tuple supporting tuple unpacking, item indexing, and attribute access."""

    def __new__(cls, page: PageText, reason: str):
        return super().__new__(cls, (page, reason))

    @property
    def page(self) -> PageText:
        return self[0]

    @property
    def reason(self) -> str:
        return self[1]

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item == "page":
                return self[0]
            if item == "reason":
                return self[1]
            raise KeyError(item)
        return super().__getitem__(item)

    def __repr__(self) -> str:
        p_num = self.page.page_number if hasattr(self.page, "page_number") else getattr(self.page, "page", "?")
        return f"SkippedPage(page={p_num}, reason={self.reason!r})"



@dataclass
class FilterResult:
    """Container for the output of filter_pages."""

    kept_pages: list[PageText]
    skipped_pages: list[SkippedPage]
    content_start_page: Optional[int] = None
    content_end_page: Optional[int] = None

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)


# ---------------------------------------------------------------------------
# Marker Patterns
# ---------------------------------------------------------------------------

# Front-matter keyword markers
FRONT_MATTER_MARKERS = [
    (r"\b(isbn(?:-1[03])?:?\s*[0-9-x]+)\b", "isbn_metadata"),
    (r"\b(all\s+rights\s+reserved)\b", "all_rights_reserved"),
    (r"\b(copyright\s*(?:©|\(c\)|20\d\d|19\d\d)?)\b", "copyright_notice"),
    (r"\b(acknowledgements?)\b", "acknowledgements"),
    (r"\b(preface)\b", "preface"),
    (r"\b(foreword)\b", "foreword"),
    (r"\b(dedication|dedicated\s+to)\b", "dedication"),
    (r"\b(table\s+of\s+contents|brief\s+contents|contents)\b", "table_of_contents"),
    (r"\b(list\s+of\s+figures|list\s+of\s+tables|list\s+of\s+illustrations)\b", "list_of_figures_or_tables"),
    (r"\b(about\s+the\s+authors?|author\s+biograph(?:y|ies))\b", "about_the_author"),
]

# Back-matter keyword markers
BACK_MATTER_MARKERS = [
    (r"\b(subject\s+index|author\s+index|general\s+index|\bindex\b)\b", "index"),
    (r"\b(bibliograph(?:y|ies))\b", "bibliography"),
    (r"\b(references|suggested\s+readings?|further\s+readings?)\b", "references"),
    (r"\b(glossar(?:y|ies))\b", "glossary"),
    (r"\b(answers\s+to\s+(?:selected\s+)?(?:exercises|problems))\b", "answers_to_exercises"),
]

# Chapter-end / content protection markers (Safety Rule 3)
CONTENT_PROTECTION_PATTERNS = [
    r"\b(chapter\s+summary|summary\s+of\s+chapter|key\s+points|chapter\s+review|review\s+questions)\b",
    r"\b(exercises?|problems?|practice\s+problems?|programming\s+exercises?|discussion\s+questions?)\b",
    r"\b(case\s+study|learning\s+objectives|chapter\s+objectives)\b",
]

# Structural TOC / Index indicators
DOT_LEADER_RE = re.compile(r"(\.{2,}|\b)\s*\d+$")
INDEX_ENTRY_RE = re.compile(r"^[\w\s\-\(\)]+,\s*\d+(?:\s*,\s*\d+)*")


# ---------------------------------------------------------------------------
# Structural & Pattern Detection Helpers
# ---------------------------------------------------------------------------


def _looks_like_toc(lines: list[str]) -> bool:
    """Return True if lines exhibit Table of Contents pattern (dot leaders or ending page numbers)."""
    if len(lines) < 4:
        return False
    ending_in_page = sum(1 for ln in lines if DOT_LEADER_RE.search(ln))
    return (ending_in_page / len(lines)) >= 0.40


def _looks_like_index(lines: list[str]) -> bool:
    """Return True if lines exhibit Index pattern (comma-separated page numbers or short alphabetical entries)."""
    if len(lines) < 4:
        return False
    index_matches = sum(
        1
        for ln in lines
        if INDEX_ENTRY_RE.search(ln) or re.match(r"^[A-Z]$", ln) or (len(ln.split()) <= 4 and re.search(r",\s*\d+", ln))
    )
    return (index_matches / len(lines)) >= 0.35


def _is_chapter_content(text: str, lines: Optional[list[str]] = None) -> bool:
    """Safety check: True if page contains chapter content, exercises, or summaries that must be kept."""
    t_lower = text.lower()
    raw_lines = lines or [l.strip() for l in text.splitlines() if l.strip()]

    # If it is clearly a Table of Contents or Index, it is NOT chapter content
    if _looks_like_toc(raw_lines) or re.search(r"\b(table\s+of\s+contents|brief\s+contents)\b", t_lower):
        return False
    if _looks_like_index(raw_lines) and re.search(r"\b(subject\s+index|author\s+index|\bindex\b)\b", t_lower):
        return False

    # Chapter / unit / module indicator anywhere in text
    if re.search(r"\b(chapter|unit|module|part)\s+([0-9ivxlc]+|[a-z]+)\b", t_lower):
        return True
    if NUM_RE.search(text) and len(text.split()) > 25:
        return True

    # Chapter-end summaries, key points, exercises
    for pat in CONTENT_PROTECTION_PATTERNS:
        if re.search(pat, t_lower):
            return True

    return False



def _find_chapter_boundaries(
    pages: list[PageText],
    toc_info: Any = None,
) -> tuple[Optional[int], Optional[int]]:
    """Determine (first_chapter_page, back_matter_start_page) from TOC or detected headings."""
    items: list[tuple[str, int]] = []

    # Non-content titles to exclude when searching for first chapter
    non_content_titles = {
        "preface",
        "foreword",
        "acknowledgement",
        "acknowledgements",
        "dedication",
        "contents",
        "table of contents",
        "brief contents",
        "about the author",
        "about the authors",
        "list of figures",
        "list of tables",
        "half title",
        "title page",
        "copyright",
    }

    def is_front_marker(t: str) -> bool:
        t_clean = re.sub(r"[^a-z0-9\s]", "", t.lower()).strip()
        if any(t_clean == nc or t_clean.startswith(nc + " ") or t_clean.endswith(" " + nc) for nc in non_content_titles):
            return True
        for pat, _ in FRONT_MATTER_MARKERS:
            if re.search(pat, t, re.I):
                return True
        return False

    # 1. Inspect toc_info if provided
    if toc_info:
        for entry in toc_info:
            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                title, page = entry[1], entry[2]
                items.append((str(title).strip(), int(page)))
            elif isinstance(entry, dict):
                title = entry.get("title", "")
                page = entry.get("page", 1)
                items.append((str(title).strip(), int(page)))

    # 2. If no items from toc_info, inspect lines and page text
    if not items:
        for p in pages:
            p_text = p.text.strip()
            # If the page header is front matter, skip searching chapter headings on it
            raw_lines = [l["text"].strip() for l in p.lines] if p.lines else [l.strip() for l in p_text.splitlines() if l.strip()]
            if is_front_marker(p_text[:120]) or _looks_like_toc(raw_lines):
                continue

            for l in p.lines:
                t = l.get("text", "").strip()
                if CHAPTER_RE.match(t) and len(t.split()) <= 12:
                    items.append((t, p.page_number))
                elif l.get("bold") and NUM_RE.match(t) and len(t.split()) <= 10:
                    items.append((t, p.page_number))

        if not items:
            for p in pages:
                p_text = p.text.strip()
                raw_lines = [l["text"].strip() for l in p.lines] if p.lines else [l.strip() for l in p_text.splitlines() if l.strip()]
                if is_front_marker(p_text[:120]) or _looks_like_toc(raw_lines):
                    continue
                found_for_page = False
                for ln in raw_lines:
                    if CHAPTER_RE.match(ln) and len(ln.split()) <= 12:
                        items.append((ln, p.page_number))
                        found_for_page = True
                        break
                    m = re.match(r"^(?:chapter|unit|module|part)\s+[0-9ivxlc]+[:\.]?\s*([^.\n]+)", ln, re.I)
                    if m and len(m.group(0).split()) <= 15:
                        items.append((m.group(0).strip(), p.page_number))
                        found_for_page = True
                        break
                if not found_for_page:
                    m = re.search(r"\b(?:chapter|unit|module|part)\s+[0-9ivxlc]+[:\.]?\s*([^.\n]+)", p_text[:250], re.I)
                    if m:
                        title = m.group(0).strip()
                        if len(title.split()) <= 15:
                            items.append((title, p.page_number))


    first_chapter_page: Optional[int] = None
    last_chapter_page: Optional[int] = None
    back_matter_start_page: Optional[int] = None

    for title, page_no in items:
        t_clean = re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()
        is_front = is_front_marker(title)
        is_back = any(re.search(pat, t_clean) for pat, _ in BACK_MATTER_MARKERS)

        if not is_front and not is_back:
            if CHAPTER_RE.match(title) or NUM_RE.match(title) or len(title.split()) <= 12:
                if first_chapter_page is None or page_no < first_chapter_page:
                    first_chapter_page = page_no
                if last_chapter_page is None or page_no > last_chapter_page:
                    last_chapter_page = page_no

    # Scan for back matter after last chapter
    for p in pages:
        p_text = p.text.strip()
        if last_chapter_page is not None and p.page_number < last_chapter_page:
            continue
        p_head = p_text[:150].lower()
        for pat, _ in BACK_MATTER_MARKERS:
            if re.search(pat, p_head):
                if back_matter_start_page is None or p.page_number < back_matter_start_page:
                    back_matter_start_page = p.page_number
                break

    return first_chapter_page, back_matter_start_page



# ---------------------------------------------------------------------------
# LLM Fallback (Requirement 2d)
# ---------------------------------------------------------------------------

CLASSIFIER_PROMPT = """You are an expert textbook analyzer.
Classify each of the following textbook pages as either "KEEP" or "SKIP".

Rules:
- "KEEP": Real chapter content, technical explanations, diagrams, chapter summaries, key points, exercises, problems.
- "SKIP": Front matter (preface, foreword, acknowledgements, copyright, table of contents) or back matter (index, bibliography, glossary, book ads).
- Safety rule: When uncertain, classify as "KEEP". Never drop real chapter content, chapter-end summaries, or exercises.

Pages to classify:
{pages_block}

Respond with one line per page in the format:
PAGE <page_no>: <KEEP|SKIP> - <brief reason>
"""


def _classify_with_llm(
    uncertain_pages: list[PageText],
    client: Any = None,
) -> dict[int, tuple[bool, str]]:
    """Batch up to 10 pages per call to classify uncertain pages using LLM.

    Returns mapping: {page_number: (should_skip, reason)}.
    Defaults to (False, 'kept_by_safety_rule') on any error or missing entry.
    """
    if not uncertain_pages:
        return {}

    # Initialize client if not provided
    llm = client
    if llm is None:
        try:
            from generation.ollama_client import OllamaClient

            llm = OllamaClient()
            if not llm.is_up():
                from generation.mock_client import MockClient

                llm = MockClient()
        except Exception:
            from generation.mock_client import MockClient

            llm = MockClient()

    results: dict[int, tuple[bool, str]] = {}
    batch_size = 10

    for i in range(0, len(uncertain_pages), batch_size):
        batch = uncertain_pages[i : i + batch_size]
        blocks = []
        for p in batch:
            snippet = " ".join(p.text.split()[:120])
            blocks.append(f"--- Page {p.page_number} ---\n{snippet}")
        prompt = CLASSIFIER_PROMPT.format(pages_block="\n\n".join(blocks))

        try:
            resp = llm.generate(prompt)
            for line in resp.splitlines():
                m = re.match(r"^\s*PAGE\s*(\d+)\s*:\s*(KEEP|SKIP)\s*[-:]?\s*(.*)$", line, re.I)
                if m:
                    p_num = int(m.group(1))
                    decision = m.group(2).upper()
                    reason = m.group(3).strip() or "classified_by_llm"
                    results[p_num] = (decision == "SKIP", reason)
        except Exception:
            # Fallback on failure: default to KEEP
            pass

    # Ensure every uncertain page has an entry (defaulting to KEEP)
    for p in uncertain_pages:
        if p.page_number not in results:
            results[p.page_number] = (False, "kept_by_safety_rule")

    return results


# ---------------------------------------------------------------------------
# Core Filter Implementation
# ---------------------------------------------------------------------------


def filter_pages(
    pages: list[PageText] | list[dict[str, Any]],
    toc_info: Any = None,
    client: Any = None,
    use_llm: Optional[bool] = None,
) -> FilterResult:
    """Filter non-content front-matter and back-matter pages from a textbook.

    Layered detection:
      1. Rule-based checks (word count < 30, keyword markers, TOC/Index structure).
      2. Position-aware: front-matter rules applied only to first ~15%; back-matter rules to last ~15%.
      3. Chapter boundaries: everything before the first real chapter is front matter;
         everything starting at back-matter (Index/Bibliography) after the last chapter is back matter.
      4. LLM fallback (optional, controlled by FILTER_USE_LLM): classifies uncertain pages.
      5. Safety rule: When uncertain, KEEP the page. Real chapter content, summaries,
         and exercises are NEVER dropped.

    Returns:
        FilterResult(kept_pages, skipped_pages, content_start_page, content_end_page)
    """
    if not pages:
        return FilterResult(kept_pages=[], skipped_pages=[], content_start_page=None, content_end_page=None)

    # Normalize pages to PageText instances
    normalized_pages: list[PageText] = []
    for item in pages:
        if isinstance(item, PageText):
            normalized_pages.append(item)
        elif isinstance(item, dict):
            normalized_pages.append(
                PageText(
                    page_number=item.get("page_number", item.get("page", 1)),
                    text=item.get("text", ""),
                    lines=item.get("lines", []),
                )
            )
        else:
            p_no = getattr(item, "page_number", getattr(item, "page", 1))
            t = getattr(item, "text", "")
            lns = getattr(item, "lines", [])
            normalized_pages.append(PageText(page_number=p_no, text=t, lines=lns))

    total_pages = len(normalized_pages)

    # Determine position-aware zones (~15% front, ~15% back)
    front_limit_idx = max(1, int(math.ceil(total_pages * 0.15)))
    back_limit_idx = total_pages - max(1, int(math.ceil(total_pages * 0.15)))

    # Determine chapter and TOC boundaries
    first_chapter_page, back_matter_start_page = _find_chapter_boundaries(normalized_pages, toc_info=toc_info)

    kept: list[PageText] = []
    skipped: list[SkippedPage] = []
    uncertain: list[PageText] = []

    for idx, page in enumerate(normalized_pages):
        p_num = page.page_number
        text = page.text.strip()
        word_count = len(text.split())

        # Extract lines for structural checks
        raw_lines = [l["text"].strip() for l in page.lines] if page.lines else [l.strip() for l in text.splitlines() if l.strip()]

        # Position flags
        in_front_zone = (idx < front_limit_idx) or (first_chapter_page is not None and p_num < first_chapter_page)
        in_back_zone = (idx >= back_limit_idx) or (back_matter_start_page is not None and p_num >= back_matter_start_page)
        in_middle_zone = not in_front_zone and not in_back_zone

        # -------------------------------------------------------------------
        # Rule 3 / Safety Guard: NEVER drop chapter content or exercises
        # -------------------------------------------------------------------
        if _is_chapter_content(text, raw_lines):
            kept.append(page)
            continue

        # -------------------------------------------------------------------
        # Requirement 2c: Explicit boundary before first chapter or after back matter
        # -------------------------------------------------------------------
        if first_chapter_page is not None and p_num < first_chapter_page:
            # Everything before the first real chapter is front matter
            reason = "before_first_chapter"
            for pat, r_name in FRONT_MATTER_MARKERS:
                if re.search(pat, text, re.I):
                    reason = f"front_matter_marker ({r_name})"
                    break
            if _looks_like_toc(raw_lines):
                reason = "table_of_contents_structure"
            elif word_count < 30:
                reason = f"insufficient_words (< 30 words: {word_count})"
            skipped.append(SkippedPage(page=page, reason=reason))
            continue

        if back_matter_start_page is not None and p_num >= back_matter_start_page:
            # Everything starting at back matter is back matter
            reason = "back_matter"
            for pat, r_name in BACK_MATTER_MARKERS:
                if re.search(pat, text, re.I):
                    reason = f"back_matter_marker ({r_name})"
                    break
            if _looks_like_index(raw_lines):
                reason = "index_structure"
            elif word_count < 30:
                reason = f"insufficient_words (< 30 words: {word_count})"
            skipped.append(SkippedPage(page=page, reason=reason))
            continue

        # -------------------------------------------------------------------
        # Requirement 2b: Position-aware filtering
        # Never skip a middle page because of a keyword match!
        # -------------------------------------------------------------------
        if in_middle_zone:
            # In the middle zone, keep pages unless completely empty (< 5 words)
            if word_count < 5:
                skipped.append(SkippedPage(page=page, reason=f"empty_page (< 5 words: {word_count})"))
            else:
                kept.append(page)
            continue

        # -------------------------------------------------------------------
        # Front-matter zone rules (first ~15%)
        # -------------------------------------------------------------------
        if in_front_zone:
            # Low word count check
            if word_count < 30:
                skipped.append(SkippedPage(page=page, reason=f"insufficient_words (< 30 words: {word_count})"))
                continue

            # Front-matter keyword check
            matched_fm: Optional[str] = None
            for pat, r_name in FRONT_MATTER_MARKERS:
                if re.search(pat, text, re.I):
                    matched_fm = r_name
                    break
            if matched_fm:
                skipped.append(SkippedPage(page=page, reason=f"front_matter_marker ({matched_fm})"))
                continue

            # Table of contents pattern check
            if _looks_like_toc(raw_lines):
                skipped.append(SkippedPage(page=page, reason="table_of_contents_structure"))
                continue

            # If none matched, mark as uncertain in front-matter zone
            uncertain.append(page)
            continue

        # -------------------------------------------------------------------
        # Back-matter zone rules (last ~15%)
        # -------------------------------------------------------------------
        if in_back_zone:
            # Low word count check
            if word_count < 30:
                skipped.append(SkippedPage(page=page, reason=f"insufficient_words (< 30 words: {word_count})"))
                continue

            # Back-matter keyword check
            matched_bm: Optional[str] = None
            for pat, r_name in BACK_MATTER_MARKERS:
                if re.search(pat, text, re.I):
                    matched_bm = r_name
                    break
            if matched_bm:
                skipped.append(SkippedPage(page=page, reason=f"back_matter_marker ({matched_bm})"))
                continue

            # Index pattern check
            if _looks_like_index(raw_lines):
                skipped.append(SkippedPage(page=page, reason="index_structure"))
                continue

            # If none matched, mark as uncertain in back-matter zone
            uncertain.append(page)
            continue

    # -----------------------------------------------------------------------
    # Requirement 2d: LLM Fallback for uncertain pages
    # -----------------------------------------------------------------------
    enable_llm = use_llm if use_llm is not None else FILTER_USE_LLM
    if uncertain:
        if enable_llm:
            llm_decisions = _classify_with_llm(uncertain, client=client)
            for page in uncertain:
                should_skip, reason = llm_decisions.get(page.page_number, (False, "kept_by_safety_rule"))
                if should_skip:
                    skipped.append(SkippedPage(page=page, reason=f"llm_classified_skip: {reason}"))
                else:
                    kept.append(page)
        else:
            # Safety rule: when uncertain and no LLM, KEEP
            for page in uncertain:
                kept.append(page)

    # Sort kept pages by page_number
    kept.sort(key=lambda p: p.page_number)
    skipped.sort(key=lambda item: item.page.page_number)

    # If all pages were filtered out (extreme safety edge-case), restore all pages
    if not kept and normalized_pages:
        kept = list(normalized_pages)
        skipped = []

    content_start = kept[0].page_number if kept else None
    content_end = kept[-1].page_number if kept else None

    return FilterResult(
        kept_pages=kept,
        skipped_pages=skipped,
        content_start_page=content_start,
        content_end_page=content_end,
    )


# ---------------------------------------------------------------------------
# Utility Conversion Helpers
# ---------------------------------------------------------------------------


def lines_to_pages(lines: list[dict[str, Any]], n_pages: Optional[int] = None) -> list[PageText]:
    """Convert line records from pdf_extract into PageText objects."""
    lines_by_page: dict[int, list[dict[str, Any]]] = {}
    for l in lines:
        p = l.get("page", 1)
        lines_by_page.setdefault(p, []).append(l)

    max_p = max(lines_by_page.keys()) if lines_by_page else 0
    total = max(max_p, n_pages or 0)

    pages: list[PageText] = []
    for p in range(1, total + 1):
        p_lines = lines_by_page.get(p, [])
        text = " ".join(l["text"] for l in p_lines)
        pages.append(PageText(page_number=p, text=text, lines=p_lines))

    return pages
