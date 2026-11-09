"""Step 1.2 - chapter / section heading detection (TOC first, heuristics fallback)."""
import re
from collections import Counter

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


def detect_headings(lines, toc=None):
    """Return [{line_idx,last_idx,level,title,page}] in document order."""
    if toc and len(toc) >= 3:
        found = _toc_headings(lines, toc)
        if len(found) >= 2:
            return [h for h in found if not is_non_content_title(h["title"])]

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
        # merge multi-line headings: "Chapter 1" + "Introduction"
        if (prev and prev["last_idx"] == i - 1 and prev["page"] == l["page"]
                and explicit is None):
            prev["title"] += " " + t
            prev["last_idx"] = i
            continue
        level = explicit or min(size_rank[l["size"]], 4)
        heads.append({"line_idx": i, "last_idx": i, "level": level,
                      "title": t, "page": l["page"]})
    return [h for h in heads if not is_non_content_title(h["title"])]
