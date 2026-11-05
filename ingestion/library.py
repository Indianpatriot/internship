"""Multi-book ingestion, library cataloging, and caching module.

Handles:
- Per-book ingestion with SHA256 disk caching under .cache/ingestion/
- Book ID disambiguation for duplicate file stems
- Chapter listing with chunk counts and page order
- Page range block fallback when no chapter headings are detected
- Unit-to-source mapping validation and automatic default suggestions
"""
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from config import MIN_CONTEXT_WORDS
from ingestion.pipeline import ingest
from retrieval.sources import select_chunks_for_unit

CACHE_DIR = Path(".cache/ingestion")


def _compute_file_hash(path: str) -> str:
    """Compute truncated SHA256 hex digest of file contents for cache validation."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sanitize_book_id(name: str) -> str:
    """Normalize filename stem into a clean slug."""
    stem = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return stem or "book"


def _apply_page_range_fallback(chunks: list[dict[str, Any]], block_size: int = 20) -> None:
    """Fallback when no headings are found: label chunks by page range blocks (e.g. 'Pages 1-20')."""
    if not chunks:
        return
    # Check if all chunks lack meaningful chapter names
    has_real_chapters = any(
        c.get("chapter") and c.get("chapter") not in ("Document", "", None)
        for c in chunks
    )
    if not has_real_chapters:
        for c in chunks:
            p_start = c.get("page_start", 1)
            b_idx = (p_start - 1) // block_size
            b_start = b_idx * block_size + 1
            b_end = b_start + block_size - 1
            c["chapter"] = f"Pages {b_start}-{b_end}"


def add_book(
    pdf_path: str,
    book_id: Optional[str] = None,
    title: Optional[str] = None,
    cache_dir: Optional[Path | str] = None,
) -> list[dict[str, Any]]:
    """Ingest a textbook PDF, caching results on disk by file hash.

    Args:
        pdf_path: Path to textbook PDF.
        book_id: Unique book identifier. Defaults to sanitized PDF file stem.
        title: Human-readable book title. Defaults to title-cased filename stem.
        cache_dir: Optional custom cache directory.

    Returns:
        List of chunks with book_id, book_title, unique chunk_id, and chapter metadata.
    """
    pdf_path_obj = Path(pdf_path)
    file_hash = _compute_file_hash(str(pdf_path_obj))
    bid = book_id or _sanitize_book_id(pdf_path_obj.stem)
    btitle = title or pdf_path_obj.stem.replace("-", " ").replace("_", " ").title()

    cdir = Path(cache_dir) if cache_dir else CACHE_DIR
    cache_file = cdir / f"{bid}_{file_hash}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_chunks = json.load(f)
            # Ensure metadata consistency
            for c in cached_chunks:
                c["book_id"] = bid
                c["book_title"] = btitle
            return cached_chunks
        except Exception:
            # On corrupted cache, fall through to re-ingest
            pass

    # Run base ingestion pipeline
    chunks = ingest(str(pdf_path_obj), book_id=bid)

    # Check and apply page range fallback if no chapters detected
    _apply_page_range_fallback(chunks)

    # Ensure chunk_ids and book titles
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{bid}_{i:04d}"
        c["book_id"] = bid
        c["book_title"] = btitle

    # Save to disk cache
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
    except Exception:
        # Cache write failures should not block execution
        pass

    return chunks


def load_library(paths: list[str | Path], cache_dir: Optional[Path | str] = None) -> dict[str, Any]:
    """Load multiple textbooks, disambiguating IDs and compiling catalog metadata.

    Args:
        paths: List of PDF file paths.
        cache_dir: Optional custom cache directory.

    Returns:
        {"books": {book_id: {...}}, "chunks": [all chunks]}
    """
    books: dict[str, dict[str, Any]] = {}
    all_chunks: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}

    for p in paths:
        path_obj = Path(p)
        base_id = _sanitize_book_id(path_obj.stem)

        # Disambiguate duplicate stems with numeric suffix
        if base_id not in seen_ids:
            seen_ids[base_id] = 1
            assigned_id = base_id
        else:
            seen_ids[base_id] += 1
            assigned_id = f"{base_id}_{seen_ids[base_id]}"

        title = path_obj.stem.replace("-", " ").replace("_", " ").title()
        chunks = add_book(str(path_obj), book_id=assigned_id, title=title, cache_dir=cache_dir)

        # Collect ordered unique chapters
        ordered_chapters: list[str] = []
        for c in chunks:
            ch = c.get("chapter")
            if ch and ch not in ordered_chapters:
                ordered_chapters.append(ch)

        min_page = min((c.get("page_start", 1) for c in chunks), default=1)
        max_page = max((c.get("page_end", 1) for c in chunks), default=1)

        books[assigned_id] = {
            "title": title,
            "path": str(path_obj),
            "chapters": ordered_chapters,
            "n_chunks": len(chunks),
            "page_range": (min_page, max_page),
        }
        all_chunks.extend(chunks)

    return {
        "books": books,
        "chunks": all_chunks,
    }


def list_chapters(library: dict[str, Any], book_id: str) -> list[dict[str, Any]]:
    """Return chapters of a book in page order with chunk counts and page bounds."""
    books = library.get("books", {})
    if book_id not in books:
        return []

    book_info = books[book_id]
    ordered_chapters = list(book_info.get("chapters", []))
    book_chunks = [c for c in library.get("chunks", []) if c.get("book_id") == book_id]

    chapter_stats: dict[str, dict[str, Any]] = {}
    for c in book_chunks:
        ch = c.get("chapter", "Document")
        if ch not in chapter_stats:
            chapter_stats[ch] = {
                "chapter": ch,
                "chunks": 0,
                "chunk_count": 0,
                "page_start": c.get("page_start", 1),
                "page_end": c.get("page_end", 1),
            }
        stats = chapter_stats[ch]
        stats["chunks"] += 1
        stats["chunk_count"] += 1
        stats["page_start"] = min(stats["page_start"], c.get("page_start", 1))
        stats["page_end"] = max(stats["page_end"], c.get("page_end", 1))

    result: list[dict[str, Any]] = []
    for ch in ordered_chapters:
        if ch in chapter_stats:
            result.append(chapter_stats[ch])
    for ch, stats in chapter_stats.items():
        if ch not in ordered_chapters:
            result.append(stats)

    return result


def validate_unit_sources(
    library: dict[str, Any],
    unit_sources: Optional[dict[Any, Any]],
) -> list[str]:
    """Validate a unit-to-textbook mapping against the loaded library catalog.

    Checks:
    - Unknown book_id
    - Chapter not found in that book
    - Unit with zero usable chunks

    Returns:
        List of human-readable problem descriptions (empty if valid).
    """
    problems: list[str] = []
    if not unit_sources:
        return problems

    books = library.get("books", {})
    chunks = library.get("chunks", [])

    for unit, sources in unit_sources.items():
        if sources is None:
            continue

        if isinstance(sources, dict):
            for bid, allowed_chapters in sources.items():
                if bid not in books:
                    problems.append(f"Unit {unit}: Unknown book_id '{bid}'.")
                else:
                    book_info = books[bid]
                    book_chaps = book_info.get("chapters", [])
                    if allowed_chapters is not None:
                        for ch in allowed_chapters:
                            matched = (
                                ch in book_chaps
                                or str(ch).lower() in [str(c).lower() for c in book_chaps]
                            )
                            if not matched:
                                problems.append(
                                    f"Unit {unit}: Chapter '{ch}' not found in book '{bid}'."
                                )

        # Verify usable chunk count for this unit
        filtered = select_chunks_for_unit(chunks, unit_sources, unit)
        usable = [
            c
            for c in filtered
            if c.get("word_count", len(c.get("text", "").split())) >= MIN_CONTEXT_WORDS
        ]
        if not usable:
            problems.append(
                f"Unit {unit}: Zero usable chunks available with current source mapping."
            )

    return problems


def suggest_unit_sources(library: dict[str, Any], units: Any) -> dict[Any, dict[str, Any]]:
    """Generate best-effort default unit sources splitting chapters across units in page order.

    Args:
        library: Loaded library catalog dictionary.
        units: Number of units (int) or list of unit identifiers.

    Returns:
        Mapping of {unit: {book_id: [chapters]}}.
    """
    if isinstance(units, int):
        unit_list = list(range(1, units + 1))
    else:
        unit_list = list(units)

    if not unit_list:
        return {}

    books = library.get("books", {})
    if not books:
        return {u: {} for u in unit_list}

    # Collect all (book_id, chapter) pairs in library book order
    all_pairs: list[tuple[str, str]] = []
    for bid, binfo in books.items():
        for ch in binfo.get("chapters", []):
            all_pairs.append((bid, ch))

    unit_sources: dict[Any, dict[str, Any]] = {u: {} for u in unit_list}
    n_units = len(unit_list)

    if not all_pairs:
        # If no chapters detected, map all books with None
        for u in unit_list:
            unit_sources[u] = {bid: None for bid in books}
        return unit_sources

    total_pairs = len(all_pairs)
    base = total_pairs // n_units
    rem = total_pairs % n_units

    idx = 0
    for i, u in enumerate(unit_list):
        count = base + (1 if i < rem else 0)
        u_pairs = all_pairs[idx : idx + count]
        idx += count
        if u_pairs:
            for bid, ch in u_pairs:
                unit_sources[u].setdefault(bid, []).append(ch)
        else:
            # If fewer chapters than units, cycle through existing pairs
            fallback_bid, fallback_ch = all_pairs[i % total_pairs]
            unit_sources[u].setdefault(fallback_bid, []).append(fallback_ch)

    return unit_sources
