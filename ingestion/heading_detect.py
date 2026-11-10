"""Step 1.2 - chapter / section heading detection (TOC first, heuristics fallback)."""
import re
from collections import Counter
from typing import Any, Optional

CHAPTER_RE = re.compile(r"^(chapter|unit|module|part)\s+([0-9ivxlc]+|[a-z]+)\b", re.I)
NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+[A-Za-z(]")

NON_CONTENT_TITLES = {
    "preface", "foreword", "acknowledgement", "acknowledgements",
    "acknowledgment", "acknowledgments", "table of contents", "contents",
    "brief contents", "list of figures", "list of tables", "list of illustrations",
    "index", "subject index", "author index", "general index", "bibliography",
    "references", "suggested readings", "further readings", "glossary",
    "dedication", "dedicated to", "copyright", "all rights reserved",
    "about the author", "about the authors", "about the technical editor",
    "about the technical editors", "about the reviewer", "about the reviewers",
    "about the technical reviewers", "about this book", "about the cover",
    "about the cover illustration", "welcome", "to get the most out of this book",
    "conventions used", "get in touch", "reviews", "other books you may enjoy",
    "assessments", "answers", "answers to exercises", "answers to problems",
    "appendix", "appendices", "title page", "half title"
}

NON_CONTENT_PATTERNS = [
    r"\babout\s+this\s+book\b",
    r"\babout\s+the\s+cover(?:\s+illustration)?\b",
    r"\babout\s+the\s+authors?\b",
    r"\babout\s+the\s+technical\s+(?:editor|reviewer)s?\b",
    r"\babout\s+the\s+reviewers?\b",
    r"\bwelcome\b",
    r"\bpreface\b",
    r"\bforeword\b",
    r"\backnowledg[e]?ments?\b",
    r"\bdedication\b",
    r"\bdedicated\s+to\b",
    r"\btable\s+of\s+contents\b",
    r"\bbrief\s+contents\b",
    r"\bcontents\b",
    r"\blist\s+of\s+(?:figures|tables|illustrations)\b",
    r"\bto\s+get\s+the\s+most\s+out\s+of\s+this\s+book\b",
    r"\bconventions\s+used\b",
    r"\bget\s+in\s+touch\b",
    r"\breviews\b",
    r"\bother\s+books\s+you\s+may\s+enjoy\b",
    r"\bassessments?\b",
    r"\banswers?\b",
    r"\banswers\s+to\s+(?:exercises|problems)\b",
    r"\bindex\b",
    r"\bbibliograph(?:y|ies)\b",
    r"\breferences\b",
    r"\bsuggested\s+readings?\b",
    r"\bfurther\s+readings?\b",
    r"\bglossar(?:y|ies)\b",
    r"\bcopyright\b",
    r"\ball\s+rights\s+reserved\b",
    r"\btitle\s+page\b",
    r"\bhalf\s+title\b",
    r"\bappendix\b",
    r"\bappendices\b",
]
NON_CONTENT_REGEX = re.compile("|".join(NON_CONTENT_PATTERNS), re.I)


def normalize_chapter_title(t: str) -> str:
    """Normalize chapter title: strip bullets (•), dot leaders, extra whitespace,
    and unify 'Chapter N:' vs 'Chapter N,' formats."""
    s = str(t or "").strip()
    s = re.sub(r"^[\s•*–—\-]+", "", s).strip()
    s = re.sub(r"\.{2,}\s*\d+$", "", s).strip()
    s = re.sub(r"\s{3,}\d+$", "", s).strip()
    s = re.sub(r"^(chapter\s+[0-9ivxlc]+)\s*[:,\-–—.]\s*", r"\1: ", s, flags=re.I)
    s = re.sub(r"[\s,:;\-]+$", "", s).strip()
    return re.sub(r"\s+", " ", s)


def _clean(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def is_non_content_title(t: str) -> bool:
    if not t:
        return False
    norm = normalize_chapter_title(t)
    clean_t = _clean(norm)
    if clean_t in NON_CONTENT_TITLES:
        return True
    if NON_CONTENT_REGEX.search(norm):
        return True
    for nc in (
        "preface", "foreword", "acknowledgement", "acknowledgment",
        "table of contents", "contents", "index", "bibliography",
        "references", "glossary", "dedication", "copyright",
        "about this book", "about the cover"
    ):
        if clean_t == nc or clean_t.startswith(nc + " ") or clean_t.endswith(" " + nc):
            return True
    return False


def body_size(lines) -> float:
    c = Counter()
    for l in lines:
        c[l["size"]] += len(l["text"])
    return c.most_common(1)[0][0] if c else 11.0


def _is_chapter(t):
    if is_non_content_title(t):
        return False
    return bool(CHAPTER_RE.match(t)) and len(t.split()) <= 12


def _is_numbered(t):
    if is_non_content_title(t):
        return False
    return bool(NUM_RE.match(t)) and len(t.split()) <= 14


def _looks_like_heading(l, body):
    t = l["text"].strip()
    words = len(t.split())
    if words == 0 or len(t) > 120 or re.fullmatch(r"[\W\d_]+", t):
        return False
    # Discard TOC lines that end in dot leaders or trailing page numbers
    if re.search(r"\.{2,}\s*\d+$", t) or (re.search(r"\s{3,}\d+$", t) and words >= 3):
        return False
    if is_non_content_title(t):
        return False
    if _is_chapter(t):
        return True
    ends_sentence = t.endswith((".", ",", ";"))
    if _is_numbered(t) and (l["bold"] or l["size"] >= body + 0.5):
        return True
    if l["size"] >= body + 1.0 and words <= 20 and not ends_sentence:
        return True
    if l["bold"] and words <= 12 and not ends_sentence:
        return True
    return False


def _level_from_text(t):
    if is_non_content_title(t):
        return None
    if _is_chapter(t):
        return 1
    m = NUM_RE.match(t)
    if m:
        return min(m.group(1).count(".") + 1, 4)
    return None


def validate_headings_list(heads: list[dict[str, Any]], n_pages: Optional[int] = None) -> bool:
    """Validate each layer's output:
    - At least 1 explicit chapter or at least 2 headings
    - Chapter start pages must be non-decreasing: p >= prev_p
    - Within the page range (1 <= p <= n_pages)
    - Not duplicated (page, title)
    - Not front/back-matter titles
    """
    if not heads:
        return False
    if len(heads) < 2:
        if not any(_is_chapter(h.get("title", "")) for h in heads):
            return False
    prev_p = 0
    seen = set()
    for h in heads:
        t = h.get("title", "")
        p = h.get("page", 1)
        if is_non_content_title(t):
            return False
        if n_pages and (p < 1 or p > n_pages):
            return False
        if p < prev_p:
            return False
        prev_p = p
        key = (p, _clean(t))
        if key in seen:
            return False
        seen.add(key)
    return True


def _toc_headings(lines, toc):
    clean_toc = []
    for entry in toc:
        if len(entry) >= 3:
            lvl, t, p_target = entry[0], entry[1], entry[2]
            norm_t = normalize_chapter_title(t)
            if norm_t and not is_non_content_title(norm_t):
                clean_toc.append((lvl, norm_t, p_target))

    found = []
    for lvl, t, p_target in clean_toc:
        best_match = None
        for i, l in enumerate(lines):
            if abs(l["page"] - p_target) <= 1:
                cl_t = _clean(t)
                cl_l = _clean(l["text"])
                if cl_t and cl_l and (cl_t == cl_l or cl_t in cl_l or cl_l in cl_t):
                    best_match = (i, l)
                    break
        if best_match:
            i, l = best_match
            found.append({
                "line_idx": i,
                "last_idx": i,
                "level": min(lvl, 4),
                "title": t,
                "page": l["page"],
            })
        else:
            page_lines = [(i, l) for i, l in enumerate(lines) if l["page"] == p_target]
            if page_lines:
                i, l = page_lines[0]
                found.append({
                    "line_idx": i,
                    "last_idx": i,
                    "level": min(lvl, 4),
                    "title": t,
                    "page": p_target,
                })

    found.sort(key=lambda h: (h["page"], h["line_idx"]))
    return found


def _detect_layer_1_outline(lines, toc, n_pages=None):
    if not toc or len(toc) < 2:
        return None
    lvl1 = [e for e in toc if len(e) >= 3 and e[0] == 1 and not is_non_content_title(e[1])]
    lvl2 = [e for e in toc if len(e) >= 3 and e[0] == 2 and not is_non_content_title(e[1])]

    is_lvl1_parts = lvl1 and all(re.match(r"^(?:part)\b", e[1].strip(), re.I) for e in lvl1)
    if is_lvl1_parts and len(lvl2) >= 2:
        target_toc = lvl2
    elif len(lvl1) >= 2:
        target_toc = lvl1
    else:
        target_toc = [e for e in toc if len(e) >= 3 and not is_non_content_title(e[1])]

    found = _toc_headings(lines, target_toc)
    clean_found = [h for h in found if not is_non_content_title(h["title"])]
    if validate_headings_list(clean_found, n_pages):
        for h in clean_found:
            h["method"] = "pdf_outline"
            h["confidence"] = 0.95
        return clean_found
    return None


def _detect_layer_2_toc_pages(lines, n_pages=None):
    if not lines:
        return None
    toc_pages = set()
    for l in lines:
        t = l["text"].strip()
        p = l["page"]
        if is_non_content_title(t) and ("content" in t.lower() or "table of contents" in t.lower()):
            toc_pages.add(p)
        elif re.search(r"\.{2,}\s*\d+$", t) or re.search(r"\s{4,}\d+$", t):
            toc_pages.add(p)

    if not toc_pages:
        return None

    parsed_entries = []
    TOC_ENTRY_RE = re.compile(r"^([^\.\d].*?)(?:\.{2,}|\s{3,})\s*(\d+)$")
    for l in lines:
        if l["page"] not in toc_pages:
            continue
        t = l["text"].strip()
        m = TOC_ENTRY_RE.match(t)
        if m:
            entry_title = normalize_chapter_title(m.group(1).strip())
            try:
                target_p = int(m.group(2).strip())
            except ValueError:
                continue
            if entry_title and not is_non_content_title(entry_title):
                parsed_entries.append((entry_title, target_p))

    if len(parsed_entries) < 2:
        return None

    found = []
    for entry_title, target_p in parsed_entries:
        cl_entry = _clean(entry_title)
        best_match = None
        candidates = [
            (i, l) for i, l in enumerate(lines)
            if abs(l["page"] - target_p) <= 2 and cl_entry and (cl_entry in _clean(l["text"]) or _clean(l["text"]) in cl_entry)
        ]
        if candidates:
            best_match = candidates[0]
        else:
            exact_page = [(i, l) for i, l in enumerate(lines) if l["page"] == target_p and _looks_like_heading(l, body_size(lines))]
            if exact_page:
                best_match = exact_page[0]

        if best_match:
            i, l = best_match
            found.append({
                "line_idx": i,
                "last_idx": i,
                "level": 1,
                "title": entry_title,
                "page": l["page"],
            })

    found.sort(key=lambda h: (h["page"], h["line_idx"]))
    dedup = []
    seen_p = set()
    for h in found:
        if h["page"] not in seen_p:
            seen_p.add(h["page"])
            dedup.append(h)

    if validate_headings_list(dedup, n_pages):
        for h in dedup:
            h["method"] = "toc_pages"
            h["confidence"] = 0.85
        return dedup
    return None


def _detect_layer_3_body_headings(lines, n_pages=None):
    body = body_size(lines)
    cand = [i for i, l in enumerate(lines) if _looks_like_heading(l, body)]
    sizes = sorted({lines[i]["size"] for i in cand}, reverse=True)
    size_rank = {s: r + 1 for r, s in enumerate(sizes)}

    heads = []
    for i in cand:
        l = lines[i]
        t = l["text"]
        if is_non_content_title(t):
            continue
        explicit = _level_from_text(t)
        prev = heads[-1] if heads else None
        if (prev and prev["last_idx"] == i - 1 and prev["page"] == l["page"]
                and explicit is None):
            prev["title"] += " " + t
            prev["last_idx"] = i
            continue
        level = explicit or min(size_rank[l["size"]], 4)
        heads.append({"line_idx": i, "last_idx": i, "level": level,
                      "title": t, "page": l["page"]})

    clean_heads = [h for h in heads if not is_non_content_title(h["title"])]
    is_only_parts = clean_heads and all(re.match(r"^part\b", h["title"].strip(), re.I) for h in clean_heads)
    if is_only_parts:
        finer = [
            {"line_idx": i, "last_idx": i, "level": 2, "title": lines[i]["text"], "page": lines[i]["page"]}
            for i, l in enumerate(lines)
            if _is_numbered(l["text"]) and not is_non_content_title(l["text"])
        ]
        if validate_headings_list(finer, n_pages):
            for h in finer:
                h["method"] = "body_headings"
                h["confidence"] = 0.70
            return finer

    if validate_headings_list(clean_heads, n_pages):
        for h in clean_heads:
            h["method"] = "body_headings"
            h["confidence"] = 0.70
        return clean_heads
    return None


def _detect_layer_4_semantic_windows(lines, n_pages=None):
    if not lines:
        return []
    kept_pages = sorted({l["page"] for l in lines})
    if not kept_pages:
        return []
    total_pages = len(kept_pages)

    min_sec = max(2, min(10, total_pages // 5)) if total_pages >= 10 else max(1, total_pages // 3)
    max_sec = max(min_sec + 2, min(25, total_pages // 2)) if total_pages >= 10 else total_pages

    window_boundaries = []
    try:
        from retrieval.co_mapper import _m
        model = _m()
        blocks = []
        block_pages = []
        for i in range(0, total_pages, min_sec):
            pgs = kept_pages[i:i + min_sec]
            txt = " ".join(l["text"] for l in lines if l["page"] in pgs)
            if txt.strip():
                blocks.append(txt[:500])
                block_pages.append(pgs[0])
        if len(blocks) >= 2:
            vecs = model.encode(blocks, normalize_embeddings=True)
            for j in range(len(vecs) - 1):
                sim = float(vecs[j] @ vecs[j + 1])
                if sim < 0.65:
                    window_boundaries.append(block_pages[j + 1])
    except Exception:
        pass

    boundaries = [p for p in window_boundaries if p > kept_pages[0]]
    if not boundaries:
        step = max(min_sec, min(max_sec, total_pages // 5 or 1))
        boundaries = [kept_pages[idx] for idx in range(step, total_pages, step)]

    split_points = [kept_pages[0]] + [b for b in boundaries if b <= kept_pages[-1]]
    if split_points[-1] != kept_pages[-1]:
        split_points.append(kept_pages[-1] + 1)
    else:
        split_points[-1] = kept_pages[-1] + 1

    heads = []
    body = body_size(lines)
    for idx in range(len(split_points) - 1):
        p_start = split_points[idx]
        p_end = split_points[idx + 1] - 1
        if p_end < p_start:
            p_end = p_start

        rep_line = None
        for l in lines:
            if l["page"] == p_start:
                t = l["text"].strip()
                words = len(t.split())
                if 2 <= words <= 10 and not t.endswith((".", ",", ";")) and not is_non_content_title(t):
                    if l.get("bold") or l.get("size", 0) >= body + 0.5:
                        rep_line = t
                        break

        title = f"Pages {p_start}-{p_end}: {rep_line}" if rep_line else f"Pages {p_start}-{p_end}"
        first_line_idx = next((i for i, l in enumerate(lines) if l["page"] == p_start), 0)
        heads.append({
            "line_idx": first_line_idx,
            "last_idx": first_line_idx,
            "level": 1,
            "title": title,
            "page": p_start,
            "method": "semantic_page_windows",
            "confidence": 0.40,
        })

    return heads


def detect_headings(lines, toc=None, n_pages=None):
    """Return [{line_idx,last_idx,level,title,page,method,confidence}] in document order."""
    if n_pages is None and lines:
        n_pages = max(l["page"] for l in lines)

    # Layer 1: PDF bookmarks/outline
    h1 = _detect_layer_1_outline(lines, toc, n_pages)
    if h1:
        return h1

    # Layer 2: TOC pages parsed from body
    h2 = _detect_layer_2_toc_pages(lines, n_pages)
    if h2:
        return h2

    # Layer 3: Font-size/bold + regex headings in body
    h3 = _detect_layer_3_body_headings(lines, n_pages)
    if h3:
        return h3

    # Layer 4: Semantic page-range pseudo-sections
    return _detect_layer_4_semantic_windows(lines, n_pages)
