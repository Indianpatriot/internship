"""Step 1.1 - text extraction with PyMuPDF (keeps font size / bold / position)."""
import re
from collections import Counter

import fitz  # PyMuPDF

BOLD_FLAG = 16
PAGE_NUM_RE = re.compile(r"^(page\s*)?\d+(\s*(/|of)\s*\d+)?$", re.I)


def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))


def _line_record(line, page_no, page_h):
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return None
    text = re.sub(r"\s+", " ", "".join(s["text"] for s in spans)).strip()
    if not text:
        return None
    chars = Counter()
    bold_chars = 0
    for s in spans:
        n = len(s["text"].strip())
        chars[round(s["size"], 1)] += n
        if (s["flags"] & BOLD_FLAG) or "bold" in s["font"].lower():
            bold_chars += n
    total = sum(chars.values())
    return {
        "page": page_no,
        "text": text,
        "size": chars.most_common(1)[0][0],
        "bold": bold_chars > total / 2,
        "y0": line["bbox"][1],
        "y1": line["bbox"][3],
        "page_h": page_h,
    }


def strip_margins(lines, n_pages, margin=0.08, repeat_ratio=0.3):
    """Drop running headers/footers and page numbers found in top/bottom margins."""

    def in_margin(l):
        return l["y0"] < l["page_h"] * margin or l["y1"] > l["page_h"] * (1 - margin)

    cnt = Counter(_norm(l["text"]) for l in lines if in_margin(l))
    threshold = max(2, int(repeat_ratio * n_pages))
    kept = []
    for l in lines:
        if in_margin(l):
            if PAGE_NUM_RE.match(l["text"].strip()):
                continue
            if n_pages >= 3 and cnt[_norm(l["text"])] >= threshold:
                continue
        kept.append(l)
    return kept


def load_pdf(pdf_path):
    """Return (lines, toc). Each line: page, text, size, bold, y0, y1."""
    doc = fitz.open(pdf_path)
    lines = []
    for page_no, page in enumerate(doc, start=1):
        h = page.rect.height
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block.get("type") != 0:
                continue
            for ln in block["lines"]:
                rec = _line_record(ln, page_no, h)
                if rec:
                    lines.append(rec)
    toc = doc.get_toc()
    n_pages = len(doc)
    doc.close()
    return strip_margins(lines, n_pages), toc
