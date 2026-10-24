"""End-to-end: PDF(s) -> chunks.json.   python -m ingestion.pipeline --pdf book.pdf"""
import argparse
import json
import os
import re

from config import MAX_CHUNK_WORDS, MIN_CHUNK_WORDS, OVERLAP_WORDS
from ingestion.chunker import build_chunks
from ingestion.heading_detect import detect_headings
from ingestion.pdf_extract import load_pdf


def ingest(pdf_path, max_words=MAX_CHUNK_WORDS, overlap_words=OVERLAP_WORDS,
           min_words=MIN_CHUNK_WORDS):
    lines, toc = load_pdf(pdf_path)
    heads = detect_headings(lines, toc)
    chunks = build_chunks(lines, heads, max_words, overlap_words, min_words)
    stem = re.sub(r"[^a-z0-9]+", "-", os.path.splitext(os.path.basename(pdf_path))[0].lower()).strip("-")
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{stem}_{i:04d}"
        c["source"] = os.path.basename(pdf_path)
    return chunks


def save_chunks(chunks, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="PDF -> section-wise chunks")
    ap.add_argument("--pdf", nargs="+", required=True)
    ap.add_argument("--out", default="data/processed/chunks.json")
    ap.add_argument("--max-words", type=int, default=MAX_CHUNK_WORDS)
    ap.add_argument("--overlap", type=int, default=OVERLAP_WORDS)
    a = ap.parse_args(argv)
    chunks = []
    for p in a.pdf:
        chunks += ingest(p, a.max_words, a.overlap)
    save_chunks(chunks, a.out)
    chapters = {c["chapter"] for c in chunks}
    print(f"{len(chunks)} chunks, {len(chapters)} chapters -> {a.out}")


if __name__ == "__main__":
    main()
