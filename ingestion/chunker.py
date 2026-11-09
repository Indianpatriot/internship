"""Step 1.3 - section-wise chunking with overlap and metadata."""
from config import MAX_CHUNK_WORDS, MIN_CHUNK_WORDS, OVERLAP_WORDS
from ingestion.heading_detect import is_non_content_title, normalize_chapter_title


def join_lines(texts):
    out = ""
    for t in texts:
        if out.endswith("-") and t[:1].islower():
            out = out[:-1] + t            # de-hyphenate "learn-" + "ing"
        else:
            out = (out + " " + t).strip()
    return out


def _wc(unit):
    return len(unit["text"].split())


def _split_units(units, max_words, overlap_words, min_words):
    """Split one section's lines into word-bounded groups with overlap."""
    groups, cur, wc, n_new = [], [], 0, 0
    for u in units:
        cur.append(u)
        n_new += 1
        wc += _wc(u)
        ends = u["text"].rstrip().endswith((".", "!", "?", ":", ";"))
        if (wc >= max_words and ends) or wc >= int(max_words * 1.4):
            groups.append(cur)
            ov, ow = [], 0
            for x in reversed(cur):
                if ow + _wc(x) > overlap_words:
                    break
                ov.insert(0, x)
                ow += _wc(x)
            cur, wc, n_new = ov, ow, 0
    if n_new:
        new_words = sum(_wc(u) for u in cur[-n_new:])
        if groups and new_words < min_words:
            groups[-1].extend(cur[-n_new:])       # merge tiny tail
        elif sum(_wc(u) for u in cur) >= min_words:
            groups.append(cur)
    return groups


def build_chunks(lines, headings, max_words=MAX_CHUNK_WORDS,
                 overlap_words=OVERLAP_WORDS, min_words=MIN_CHUNK_WORDS):
    # Filter out any non-content headings
    clean_headings = []
    for h in headings:
        t = normalize_chapter_title(h.get("title", ""))
        if not is_non_content_title(t):
            h_copy = dict(h)
            h_copy["title"] = t
            clean_headings.append(h_copy)

    starts = {h["line_idx"]: h for h in clean_headings}
    skip = set()
    for h in clean_headings:
        skip.update(range(h["line_idx"], h["last_idx"] + 1))

    # Identify real level-1 chapter boundaries with start pages
    chap_starts = []
    seen_chap_pages = set()
    for h in clean_headings:
        if h.get("level") == 1:
            p = h.get("page", 1)
            if p not in seen_chap_pages:
                seen_chap_pages.add(p)
                chap_starts.append({"title": h["title"], "page": p, "line_idx": h["line_idx"]})
    chap_starts.sort(key=lambda x: (x["page"], x["line_idx"]))

    raw_chunks, buf = [], []
    cur_section = ""

    def flush():
        for g in _split_units(buf, max_words, overlap_words, min_words):
            text = join_lines([u["text"] for u in g])
            p_start = min(u["page"] for u in g)
            p_end = max(u["page"] for u in g)

            # Assign chapter by page range
            assigned_chap = ""
            if chap_starts:
                matching = [c for c in chap_starts if c["page"] <= p_start]
                if matching:
                    assigned_chap = matching[-1]["title"]

            raw_chunks.append({
                "chapter": assigned_chap or "Document",
                "section": cur_section,
                "page_start": p_start,
                "page_end": p_end,
                "text": text,
                "word_count": len(text.split()),
            })
        buf.clear()

    for i, l in enumerate(lines):
        if i in starts:
            flush()
            h = starts[i]
            if h.get("level") != 1:
                cur_section = h["title"]
            else:
                cur_section = ""
        if i in skip:
            continue
        buf.append(l)
    flush()

    # Filter out chunks that fell into non-content sections
    final_chunks = []
    for c in raw_chunks:
        ch = c.get("chapter", "")
        sec = c.get("section", "")
        if is_non_content_title(ch) or is_non_content_title(sec):
            continue
        final_chunks.append(c)

    return final_chunks
