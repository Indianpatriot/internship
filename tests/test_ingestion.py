from ingestion.chunker import join_lines
from ingestion.heading_detect import detect_headings
from ingestion.pdf_extract import load_pdf
from ingestion.pipeline import ingest


def test_headers_footers_removed(sample_pdf):
    lines, _ = load_pdf(sample_pdf)
    texts = " ".join(l["text"] for l in lines)
    assert "Sample Textbook" not in texts
    assert "Page 1" not in texts


def test_headings_detected(sample_pdf):
    lines, toc = load_pdf(sample_pdf)
    titles = [h["title"] for h in detect_headings(lines, toc)]
    assert any("Introduction" in t for t in titles)
    assert any("What is Learning" in t for t in titles)
    assert any("Chapter 2" in t for t in titles)
    levels = {h["title"]: h["level"] for h in detect_headings(lines, toc)}
    assert levels["1.1 What is Learning"] == 2


def test_chunks_have_metadata_and_bounds(sample_pdf):
    chunks = ingest(sample_pdf, max_words=80, overlap_words=15, min_words=20)
    assert len(chunks) > 3
    assert {"chunk_id", "chapter", "section", "page_start", "page_end", "text"} <= chunks[0].keys()
    assert all(c["word_count"] <= 80 * 1.4 + 40 for c in chunks)
    assert all(c["page_start"] <= c["page_end"] for c in chunks)
    assert chunks[0]["chapter"].startswith("Chapter 1")
    assert chunks[0]["section"] == "1.1 What is Learning"
    assert any(c["chapter"].startswith("Chapter 2") for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_dehyphenate():
    assert join_lines(["machine learn-", "ing is fun"]) == "machine learning is fun"
