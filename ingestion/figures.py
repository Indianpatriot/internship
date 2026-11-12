"""Figure extraction module for textbook PDFs.

Extracts both raster embedded images and vector diagram regions (flowcharts,
architecture diagrams) using PyMuPDF, captures captions and surrounding text,
filters out page headers/footers, logos and decoration junk, and saves PNGs
under .cache/figures/<book_id>/<file_hash>/.
"""
import hashlib
import json
import os
import re
from typing import Any, Optional

import fitz  # PyMuPDF

import config
from contracts import Figure

CAPTION_PATTERN = re.compile(
    r"^\s*(Figure|Fig\.?)\s*([0-9]+[A-Za-z0-9.\-–—]*)\s*[:.\-–—]?\s*(.*)$",
    re.IGNORECASE,
)
MAX_SURROUNDING_WORDS = 150
MIN_FIG_WIDTH = 120
MIN_FIG_HEIGHT = 80
MIN_PAGE_AREA_RATIO = 0.015
MAX_REPEATED_RATIO = 0.30


def _compute_file_hash(pdf_path: str) -> str:
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def _merge_rects(rects: list[fitz.Rect], pad: float = 6.0) -> list[fitz.Rect]:
    """Merge overlapping or nearby bounding boxes."""
    if not rects:
        return []
    # Work on copies expanded by pad
    merged: list[fitz.Rect] = []
    for r in rects:
        expanded = fitz.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)
        placed = False
        for i, existing in enumerate(merged):
            if existing.intersects(expanded):
                merged[i] = existing | expanded
                placed = True
                break
        if not placed:
            merged.append(expanded)

    # Consolidate iteratively
    changed = True
    while changed:
        changed = False
        new_merged: list[fitz.Rect] = []
        while merged:
            current = merged.pop(0)
            combine_idx = -1
            for j, candidate in enumerate(new_merged):
                if candidate.intersects(current):
                    combine_idx = j
                    break
            if combine_idx >= 0:
                new_merged[combine_idx] = new_merged[combine_idx] | current
                changed = True
            else:
                new_merged.append(current)
        merged = new_merged

    # Unpad slightly while keeping within reasonable limits
    return [fitz.Rect(r.x0 + pad * 0.5, r.y0 + pad * 0.5, r.x1 - pad * 0.5, r.y1 - pad * 0.5) for r in merged]


def _find_caption_and_nearby(page: fitz.Page, fig_rect: fitz.Rect, text_blocks: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Find caption below or above the figure, and extract surrounding context text."""
    caption = ""
    figure_label = ""
    surrounding_words: list[str] = []

    # Sort blocks vertically
    # A caption block is typically within 40 pts below or 25 pts above the figure
    best_caption_dist = 9999.0
    for b in text_blocks:
        b_rect = fitz.Rect(b["bbox"])
        text = b.get("text", "").strip()
        if not text:
            continue

        # Check for caption matching
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            first_line = lines[0]
            m = CAPTION_PATTERN.match(first_line)
            if m:
                # Check vertical proximity
                # Below: b_rect.y0 >= fig_rect.y1 - 5 and b_rect.y0 <= fig_rect.y1 + 50
                # Above: b_rect.y1 <= fig_rect.y0 + 5 and b_rect.y1 >= fig_rect.y0 - 40
                dist_below = b_rect.y0 - fig_rect.y1
                dist_above = fig_rect.y0 - b_rect.y1
                if 0 <= dist_below <= 60 and dist_below < best_caption_dist:
                    best_caption_dist = dist_below
                    figure_label = f"{m.group(1)} {m.group(2)}".strip()
                    caption = text
                elif 0 <= dist_above <= 40 and dist_above < best_caption_dist:
                    best_caption_dist = dist_above
                    figure_label = f"{m.group(1)} {m.group(2)}".strip()
                    caption = text

        # Surrounding text collects words near the figure
        if fig_rect.intersects(b_rect) or abs(b_rect.y0 - fig_rect.y1) < 120 or abs(fig_rect.y0 - b_rect.y1) < 120:
            for w in text.split():
                if len(surrounding_words) < MAX_SURROUNDING_WORDS:
                    surrounding_words.append(w)

    nearby_text = " ".join(surrounding_words)
    return caption, figure_label, nearby_text


def extract_figures(pdf_path: str, book_id: str = "book") -> list[Figure]:
    """Extract figures from a PDF with caching on disk."""
    if not getattr(config, "EXTRACT_FIGURES", True):
        return []

    if not os.path.exists(pdf_path):
        return []

    file_hash = _compute_file_hash(pdf_path)
    cache_dir = os.path.join(".cache", "figures", book_id, file_hash)
    meta_path = os.path.join(cache_dir, "figures_meta.json")

    # Disk cache hit
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            # Verify images exist
            if all(os.path.exists(fig.get("image_path", "")) for fig in cached_data):
                return cached_data
        except Exception:
            pass

    os.makedirs(cache_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        return []

    # First pass: collect image hashes / xrefs across pages to identify repeated logos / headers
    img_xref_counts: dict[int, int] = {}
    for pno in range(total_pages):
        page = doc[pno]
        for img_info in page.get_images():
            xref = img_info[0]
            img_xref_counts[xref] = img_xref_counts.get(xref, 0) + 1

    repeated_xrefs = {
        xref for xref, count in img_xref_counts.items()
        if (count / float(total_pages)) > MAX_REPEATED_RATIO and count >= 2
    }

    figures: list[Figure] = []
    fig_counter = 1

    for pno in range(total_pages):
        page = doc[pno]
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        header_cutoff = page_rect.y0 + page_rect.height * 0.08
        footer_cutoff = page_rect.y1 - page_rect.height * 0.08

        # Extract text blocks for caption & surrounding text detection
        raw_blocks = page.get_text("blocks")
        text_blocks: list[dict[str, Any]] = [
            {"bbox": (b[0], b[1], b[2], b[3]), "text": b[4]}
            for b in raw_blocks if len(b) >= 5
        ]

        # 1. Embedded raster images
        img_rects: list[tuple[fitz.Rect, str]] = []
        try:
            image_list = page.get_image_info(xrefs=True)
            for info in image_list:
                xref = info.get("xref", 0)
                if xref in repeated_xrefs:
                    continue
                bbox = fitz.Rect(info.get("bbox", (0, 0, 0, 0)))
                # Filter small images or headers/footers
                w, h = bbox.width, bbox.height
                if w < MIN_FIG_WIDTH or h < MIN_FIG_HEIGHT:
                    continue
                if (w * h) < (page_area * MIN_PAGE_AREA_RATIO):
                    continue
                if bbox.y1 <= header_cutoff or bbox.y0 >= footer_cutoff:
                    continue
                img_rects.append((bbox, "raster"))
        except Exception:
            pass

        # 2. Vector diagrams (from drawing paths)
        vector_rects: list[fitz.Rect] = []
        try:
            drawings = page.get_drawings()
            # If page has drawings, group bounding rects
            drawing_rects = [d["rect"] for d in drawings if d["rect"].width > 4 and d["rect"].height > 4]
            # Exclude full page background rects
            valid_drawings = [
                r for r in drawing_rects
                if (r.width * r.height) < (page_area * 0.90) and r.y1 > header_cutoff and r.y0 < footer_cutoff
            ]
            if len(valid_drawings) >= 1:  # Vector diagrams consist of strokes/boxes/lines
                merged_drawings = _merge_rects(valid_drawings, pad=12.0)
                for mr in merged_drawings:
                    # Clip to page bounds
                    clipped = mr & page_rect
                    if clipped.width >= MIN_FIG_WIDTH and clipped.height >= MIN_FIG_HEIGHT:
                        if (clipped.width * clipped.height) >= (page_area * MIN_PAGE_AREA_RATIO):
                            # Ensure it does not duplicate a raster rect
                            if not any(clipped.intersects(r[0]) for r in img_rects):
                                vector_rects.append(clipped)
        except Exception:
            pass

        all_page_candidates: list[tuple[fitz.Rect, str]] = img_rects + [(r, "vector") for r in vector_rects]

        for bbox, kind in all_page_candidates:
            # Crop bounds
            clip_box = fitz.Rect(
                max(0, bbox.x0 - 4),
                max(0, bbox.y0 - 4),
                min(page_rect.width, bbox.x1 + 4),
                min(page_rect.height, bbox.y1 + 4),
            )
            w_px = int(clip_box.width * (config.FIGURE_DPI / 72.0))
            h_px = int(clip_box.height * (config.FIGURE_DPI / 72.0))

            # Scale if larger than max width
            zoom = config.FIGURE_DPI / 72.0
            if w_px > config.FIGURE_MAX_WIDTH_PX:
                zoom = (config.FIGURE_MAX_WIDTH_PX / clip_box.width)

            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(clip=clip_box, matrix=matrix)

            out_filename = f"p{pno + 1}_f{fig_counter}.png"
            out_path = os.path.join(cache_dir, out_filename)
            pix.save(out_path)

            caption, fig_label, nearby = _find_caption_and_nearby(page, clip_box, text_blocks)
            fig_record: Figure = {
                "figure_id": f"fig_{book_id}_{pno + 1}_{fig_counter}",
                "book_id": book_id,
                "page": pno + 1,
                "bbox": (round(clip_box.x0, 1), round(clip_box.y0, 1), round(clip_box.x1, 1), round(clip_box.y1, 1)),
                "image_path": out_path,
                "caption": caption,
                "figure_label": fig_label,
                "nearby_text": nearby,
                "chapter": None,
                "section": None,
                "kind": kind,
                "width": pix.width,
                "height": pix.height,
            }
            figures.append(fig_record)
            fig_counter += 1

    doc.close()

    # Save metadata cache
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(figures, f, indent=2)
    except Exception:
        pass

    return figures


def link_figures_to_chunks(figures: list[Figure], chunks: list[dict[str, Any]]) -> None:
    """Link extracted figures to chunks by page overlap, setting figure_ids and chapter/section."""
    for fig in figures:
        p = fig["page"]
        # Find chunk(s) covering this page
        matching_chunks = [
            c for c in chunks
            if c.get("page_start", 0) <= p <= c.get("page_end", 999999)
        ]
        if matching_chunks:
            primary_chunk = matching_chunks[0]
            fig["chapter"] = primary_chunk.get("chapter")
            fig["section"] = primary_chunk.get("section")
            for c in matching_chunks:
                f_list = c.setdefault("figure_ids", [])
                if fig["figure_id"] not in f_list:
                    f_list.append(fig["figure_id"])
