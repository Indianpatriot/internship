"""Step 1.2 - chapter / section heading detection (TOC first, heuristics fallback)."""
import re
from collections import Counter

CHAPTER_RE = re.compile(r"^(chapter|unit|module|part)\s+([0-9ivxlc]+|[a-z]+)\b", re.I)
NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+[A-Za-z(]")


def _clean(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def body_size(lines) -> float:
    c = Counter()
    for l in lines:
        c[l["size"]] += len(l["text"])
    return c.most_common(1)[0][0] if c else 11.0


def _is_chapter(t):
    return bool(CHAPTER_RE.match(t)) and len(t.split()) <= 12


def _is_numbered(t):
    return bool(NUM_RE.match(t)) and len(t.split()) <= 14


def _looks_like_heading(l, body):
    t = l["text"].strip()
    words = len(t.split())
    if words == 0 or len(t) > 120 or re.fullmatch(r"[\W\d_]+", t):
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
    if _is_chapter(t):
        return 1
    m = NUM_RE.match(t)
    if m:
        return min(m.group(1).count(".") + 1, 4)
    return None


def _toc_headings(lines, toc):
    titles = {_clean(t): lvl for lvl, t, _ in toc if _clean(t)}
    found = []
    for i, l in enumerate(lines):
        lvl = titles.get(_clean(l["text"]))
        if lvl:
            found.append({"line_idx": i, "last_idx": i, "level": min(lvl, 4),
                          "title": l["text"], "page": l["page"]})
    return found


def detect_headings(lines, toc=None):
    """Return [{line_idx,last_idx,level,title,page}] in document order."""
    if toc and len(toc) >= 3:
        found = _toc_headings(lines, toc)
        if len(found) >= 2:
            return found

    body = body_size(lines)
    cand = [i for i, l in enumerate(lines) if _looks_like_heading(l, body)]
    sizes = sorted({lines[i]["size"] for i in cand}, reverse=True)
    size_rank = {s: r + 1 for r, s in enumerate(sizes)}

    heads = []
    for i in cand:
        l = lines[i]
        t = l["text"]
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
    return heads
