"""Chapter and section filtering for question paper generation.

Filters chunks before outcome scoring so question generation strictly respects
user-selected textbook chapters or page ranges.
"""
from typing import Any, Optional


def get_chapter_metadata(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract unique chapters in book order (by earliest page_start) with chunk counts and page ranges.

    Returns:
        List of dicts:
        [{
            "chapter": str,
            "chunk_count": int,
            "page_start": int,
            "page_end": int,
            "sections": list[str],
            "display_label": str
        }, ...]
    """
    if not chunks:
        return []

    from ingestion.heading_detect import is_non_content_title

    distinct_books = {c.get("book_id") or c.get("book_title") for c in chunks if c.get("book_id") or c.get("book_title")}
    multiple_books = len(distinct_books) > 1

    chapter_dict: dict[Any, dict[str, Any]] = {}
    for c in chunks:
        chap = c.get("chapter")
        if not chap or chap == "Document" or is_non_content_title(chap):
            continue

        p_start = c.get("page_start", 0)
        p_end = c.get("page_end", p_start)
        sec = c.get("section", "")
        book_label = c.get("book_title") or c.get("display_label") or c.get("book_id") or ""

        if multiple_books and book_label and not str(chap).startswith(f"{book_label} -"):
            display_chap = f"{book_label} - {chap}"
        else:
            display_chap = chap

        dict_key = (book_label, chap) if multiple_books else chap

        if dict_key not in chapter_dict:
            chapter_dict[dict_key] = {
                "chapter": display_chap,
                "raw_chapter": chap,
                "book_id": c.get("book_id", ""),
                "book_title": book_label,
                "chunk_count": 0,
                "page_start": p_start,
                "page_end": p_end,
                "sections": set(),
            }

        entry = chapter_dict[dict_key]
        entry["chunk_count"] += 1
        entry["page_start"] = min(entry["page_start"], p_start)
        entry["page_end"] = max(entry["page_end"], p_end)
        if sec:
            entry["sections"].add(sec)

    # Sort strictly by book order (page_start)
    if multiple_books:
        sorted_chapters = sorted(
            chapter_dict.values(),
            key=lambda x: (x["book_title"], x["page_start"], x["chapter"])
        )
    else:
        sorted_chapters = sorted(
            chapter_dict.values(),
            key=lambda x: (x["page_start"], x["chapter"])
        )

    result = []
    for entry in sorted_chapters:
        p_s = entry["page_start"]
        p_e = entry["page_end"]
        page_str = f"p. {p_s}" if p_s == p_e else f"pp. {p_s}–{p_e}"
        result.append({
            "chapter": entry["chapter"],
            "raw_chapter": entry.get("raw_chapter", entry["chapter"]),
            "book_id": entry.get("book_id", ""),
            "book_title": entry.get("book_title", ""),
            "chunk_count": entry["chunk_count"],
            "page_start": p_s,
            "page_end": p_e,
            "sections": sorted(list(entry["sections"])),
            "display_label": f"{entry['chapter']} ({entry['chunk_count']} chunks, {page_str})",
        })

    return result


def filter_chunks(
    chunks: list[dict[str, Any]],
    selected_chapters: Optional[list[str]] = None,
    selected_sections: Optional[list[str]] = None,
    page_range: Optional[tuple[int, int]] = None,
) -> list[dict[str, Any]]:
    """Filter chunks by selected chapters, sections, or page range.

    If selected_chapters is passed as an empty list [], returns [].
    If selected_chapters is None, all chapters are included unless page_range is provided.
    """
    if not chunks:
        return []

    # If selected_chapters was explicitly provided as empty list, nothing matches
    if selected_chapters is not None and len(selected_chapters) == 0:
        return []

    filtered = []
    chap_set = set(selected_chapters) if selected_chapters is not None else None
    sec_set = set(selected_sections) if selected_sections is not None else None

    p_min, p_max = page_range if page_range else (None, None)

    for c in chunks:
        chap = c.get("chapter")
        book_label = c.get("book_title") or c.get("book_id") or ""
        disambiguated = f"{book_label} - {chap}" if (book_label and not str(chap).startswith(f"{book_label} -")) else chap

        # Chapter filter
        if chap_set is not None:
            if disambiguated in chap_set:
                pass
            elif not any(" - " in s for s in chap_set) and chap in chap_set:
                pass
            elif chap in chap_set:
                pass
            else:
                continue

        # Section filter (if specified)
        if sec_set is not None and c.get("section") and c.get("section") not in sec_set:
            continue

        # Page range filter
        c_start = c.get("page_start", 0)
        c_end = c.get("page_end", c_start)
        if p_min is not None and c_end < p_min:
            continue
        if p_max is not None and c_start > p_max:
            continue

        filtered.append(c)

    # Maintain original book order (sorted by page_start)
    filtered.sort(key=lambda x: (x.get("page_start", 0), x.get("chunk_id", "")))
    return filtered


def get_filter_summary(
    filtered_chunks: list[dict[str, Any]],
    selected_chapters: Optional[list[str]] = None,
) -> str:
    """Produce human-readable live summary: '3 chapters selected, 41 chunks, pages 52–118'."""
    if not filtered_chunks:
        return "No chunks match the current selection."

    n_chunks = len(filtered_chunks)
    chaps = {c.get("chapter") for c in filtered_chunks if c.get("chapter")}
    n_chaps = len(chaps) if selected_chapters is None else len(selected_chapters)

    p_starts = [c.get("page_start", 0) for c in filtered_chunks if "page_start" in c]
    p_ends = [c.get("page_end", 0) for c in filtered_chunks if "page_end" in c]

    min_p = min(p_starts) if p_starts else 1
    max_p = max(p_ends) if p_ends else min_p

    page_str = f"page {min_p}" if min_p == max_p else f"pages {min_p}–{max_p}"
    chap_noun = "chapter" if n_chaps == 1 else "chapters"
    chunk_noun = "chunk" if n_chunks == 1 else "chunks"

    return f"{n_chaps} {chap_noun} selected, {n_chunks} {chunk_noun}, {page_str}"

