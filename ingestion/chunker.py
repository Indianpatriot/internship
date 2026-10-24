"""Step 1.3 - section-wise chunking with overlap and metadata."""
from config import MAX_CHUNK_WORDS, MIN_CHUNK_WORDS, OVERLAP_WORDS


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
    starts = {h["line_idx"]: h for h in headings}
    skip = set()
    for h in headings:
        skip.update(range(h["line_idx"], h["last_idx"] + 1))

    chunks, buf = [], []
    chapter = section = ""

    def flush():
        for g in _split_units(buf, max_words, overlap_words, min_words):
            text = join_lines([u["text"] for u in g])
            chunks.append({
                "chapter": chapter or "Document",
                "section": section,
                "page_start": min(u["page"] for u in g),
                "page_end": max(u["page"] for u in g),
                "text": text,
                "word_count": len(text.split()),
            })
        buf.clear()

    for i, l in enumerate(lines):
        if i in starts:
            flush()
            h = starts[i]
            if h["level"] == 1:
                chapter, section = h["title"], ""
            else:
                section = h["title"]
        if i in skip:
            continue
        buf.append(l)
    flush()
    return chunks
