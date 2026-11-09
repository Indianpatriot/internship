"""Multi-book retrieval and unit-to-textbook mapping."""
import logging
from typing import Any, Optional

from ingestion.heading_detect import is_non_content_title

logger = logging.getLogger(__name__)


def is_content_chunk(c: dict[str, Any]) -> bool:
    """Check whether a chunk belongs to real content rather than front/back matter."""
    chap = c.get("chapter", "")
    sec = c.get("section", "")
    if is_non_content_title(chap) or is_non_content_title(sec):
        return False
    return True


def select_chunks_for_unit(
    chunks: list[dict[str, Any]],
    unit_sources: Optional[dict[Any, Any]],
    unit: Optional[Any],
) -> list[dict[str, Any]]:
    """Filter chunks to only those matching the textbooks and chapters mapped to a unit.

    Also guards against any non-content chunks (preface, index, TOC, about this book, etc.).

    Args:
        chunks: List of chunk dicts (with 'book_id' and 'chapter').
        unit_sources: Mapping of {unit: {book_id: [chapters] or None for all chapters}}.
        unit: Unit identifier (e.g. 1, 2, "Unit 1", or None).

    Returns:
        Filtered list of chunks.
    """
    # Guard against non-content chunks
    valid_chunks = []
    for c in chunks:
        if is_content_chunk(c):
            valid_chunks.append(c)
        else:
            logger.warning(
                "Filtered out non-content chunk: chap=%r, sec=%r, page=%r",
                c.get("chapter"),
                c.get("section"),
                c.get("page_start"),
            )

    if not unit_sources or unit is None:
        return valid_chunks

    if unit not in unit_sources:
        if isinstance(unit, int) and str(unit) in unit_sources:
            sources_for_unit = unit_sources[str(unit)]
        elif isinstance(unit, str) and unit.isdigit() and int(unit) in unit_sources:
            sources_for_unit = unit_sources[int(unit)]
        else:
            # A missing unit means all valid chunks
            return valid_chunks
    else:
        sources_for_unit = unit_sources[unit]

    if not sources_for_unit:
        return valid_chunks

    filtered: list[dict[str, Any]] = []
    for c in valid_chunks:
        c_book = c.get("book_id")
        if c_book in sources_for_unit:
            allowed_chapters = sources_for_unit[c_book]
            if allowed_chapters is None:
                filtered.append(c)
            else:
                c_chap = c.get("chapter")
                allowed_str = [str(ch).strip().lower() for ch in allowed_chapters if ch is not None]
                c_chap_str = str(c_chap).strip().lower() if c_chap is not None else ""
                c_title_str = str(c.get("book_title", "")).strip().lower()
                c_book_str = str(c_book).strip().lower() if c_book is not None else ""
                matched = (
                    c_chap in allowed_chapters
                    or c_chap_str in allowed_str
                    or any(
                        achap == f"{c_book_str} - {c_chap_str}"
                        or achap == f"{c_title_str} - {c_chap_str}"
                        or achap.endswith(f"- {c_chap_str}")
                        for achap in allowed_str
                    )
                )
                if matched:
                    filtered.append(c)

    return filtered
