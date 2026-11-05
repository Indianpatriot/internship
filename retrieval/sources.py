"""Multi-book retrieval and unit-to-textbook mapping."""
from typing import Any, Optional


def select_chunks_for_unit(
    chunks: list[dict[str, Any]],
    unit_sources: Optional[dict[Any, Any]],
    unit: Optional[Any],
) -> list[dict[str, Any]]:
    """Filter chunks to only those matching the textbooks and chapters mapped to a unit.

    Args:
        chunks: List of chunk dicts (with 'book_id' and 'chapter').
        unit_sources: Mapping of {unit: {book_id: [chapters] or None for all chapters}}.
        unit: Unit identifier (e.g. 1, 2, "Unit 1", or None).

    Returns:
        Filtered list of chunks. If unit_sources is None or unit is missing, returns all chunks.
    """
    if not unit_sources or unit is None:
        return chunks

    if unit not in unit_sources:
        if isinstance(unit, int) and str(unit) in unit_sources:
            sources_for_unit = unit_sources[str(unit)]
        elif isinstance(unit, str) and unit.isdigit() and int(unit) in unit_sources:
            sources_for_unit = unit_sources[int(unit)]
        else:
            # A missing unit means all chunks
            return chunks
    else:
        sources_for_unit = unit_sources[unit]

    if not sources_for_unit:
        return chunks

    filtered: list[dict[str, Any]] = []
    for c in chunks:
        c_book = c.get("book_id")
        if c_book in sources_for_unit:
            allowed_chapters = sources_for_unit[c_book]
            if allowed_chapters is None:
                filtered.append(c)
            else:
                c_chap = c.get("chapter")
                allowed_str = [str(ch).strip().lower() for ch in allowed_chapters if ch is not None]
                c_chap_str = str(c_chap).strip().lower() if c_chap is not None else ""
                if c_chap in allowed_chapters or c_chap_str in allowed_str:
                    filtered.append(c)

    return filtered
