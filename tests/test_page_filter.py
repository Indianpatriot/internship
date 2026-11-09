"""Tests for non-content page filter in ingestion pipeline."""
import fitz
import pytest

from config import FILTER_USE_LLM
from generation.mock_client import MockClient
from ingestion.page_filter import (
    FilterResult,
    PageText,
    SkippedPage,
    _looks_like_index,
    _looks_like_toc,
    filter_pages,
    lines_to_pages,
)
from ingestion.pipeline import ingest, ingest_with_report


def test_page_text_and_skipped_page_contracts():
    """Verify PageText, SkippedPage, and FilterResult behavior and indexing."""
    p1 = PageText(page_number=1, text="Sample intro text with several technical terms.", lines=[{"text": "Sample intro"}])
    assert p1.page == 1
    assert p1.page_number == 1
    assert p1["page"] == 1
    assert p1["text"].startswith("Sample")

    # Alternate init using page=...
    p2 = PageText(page=2, text="Second page text")
    assert p2.page == 2
    assert p2.word_count == 3

    # SkippedPage behaves as tuple and has properties
    sp = SkippedPage(page=p1, reason="too_few_words")
    page_out, reason_out = sp
    assert page_out == p1
    assert reason_out == "too_few_words"
    assert sp.page == p1
    assert sp.reason == "too_few_words"
    assert sp["page"] == p1
    assert sp["reason"] == "too_few_words"
    assert sp[0] == p1
    assert sp[1] == "too_few_words"


def test_rule_based_low_word_count_in_margins():
    """Pages with fewer than 30 words in front or back margins are skipped."""
    pages = [
        PageText(1, "Title Page. Author Name."),  # 4 words in front matter
        PageText(2, "Dedicated to my parents."),    # 4 words in front matter
        PageText(3, "Chapter 1: Foundational Algorithms and Computational Complexity " + "data structures " * 30),
        PageText(4, "Chapter 2: Graph Theory and Network Optimizations " + "edges vertices " * 30),
        PageText(5, "Blank end page."),            # 3 words in back matter
    ]
    res = filter_pages(pages)
    skipped_nums = {sp.page.page_number for sp in res.skipped_pages}
    assert 1 in skipped_nums
    assert 2 in skipped_nums
    assert 5 in skipped_nums
    assert 3 in [p.page_number for p in res.kept_pages]
    assert 4 in [p.page_number for p in res.kept_pages]
    assert res.content_start_page == 3
    assert res.content_end_page == 4


def test_rule_based_front_matter_markers():
    """Markers such as ISBN, copyright, preface, TOC are skipped in front zone."""
    pages = [
        PageText(1, "ISBN 978-0-123456-78-9. All rights reserved. Copyright © 2024 Academic Press. " * 5),
        PageText(2, "Preface. This textbook provides a comprehensive exploration of core algorithms. " * 5),
        PageText(3, "Table of Contents. Chapter 1 Introduction. Chapter 2 Sorting. Chapter 3 Trees. " * 5),
        PageText(4, "Acknowledgements. The authors thank their students and university faculty. " * 5),
        PageText(5, "Chapter 1: Introduction to Data Science and Machine Learning Algorithms. " + "supervised models " * 30),
        PageText(6, "Chapter 2: Optimization Methods and Gradient Descent Principles. " + "convex functions " * 30),
    ]
    res = filter_pages(pages)
    kept_nums = [p.page_number for p in res.kept_pages]
    assert 1 not in kept_nums
    assert 2 not in kept_nums
    assert 3 not in kept_nums
    assert 4 not in kept_nums
    assert 5 in kept_nums
    assert 6 in kept_nums


def test_rule_based_back_matter_markers():
    """Back matter markers like Index, Bibliography, References, Glossary are skipped in back zone."""
    # 20 pages total, so last ~3 pages are in the back zone
    pages = [
        PageText(i, f"Chapter {i}: Detailed technical exploration of topic {i}. " + "mathematical formulations " * 20)
        for i in range(1, 18)
    ]
    # Add back matter pages
    pages.append(PageText(18, "Glossary of Technical Definitions and Standard Nomenclature. " * 6))
    pages.append(PageText(19, "Bibliography and Further Reading for Computer Science Students. " * 6))
    pages.append(PageText(20, "Subject Index. Key topics, algorithmic implementations, and citations. " * 6))

    res = filter_pages(pages)
    skipped_nums = {sp.page.page_number for sp in res.skipped_pages}
    assert 18 in skipped_nums
    assert 19 in skipped_nums
    assert 20 in skipped_nums
    assert res.content_start_page == 1
    assert res.content_end_page == 17


def test_position_aware_never_skips_middle_pages_on_keyword_match():
    """A page in the middle of a book citing 'references' or mentioning 'copyright' is NEVER dropped."""
    pages = []
    # 20 pages total
    for i in range(1, 21):
        if i == 10:
            # Middle page discussing copyright law or citing internal references
            text = (
                "Section 10.3 Legal Aspects: Software licensing, copyright compliance, and open source intellectual property. "
                "Students must cite all academic references accurately in their documentation. "
                "Furthermore, algorithm complexity bounds establish lower limits on computation. "
            ) * 3
        else:
            text = f"Chapter {i}: Advanced concepts and engineering applications. " + "system analysis " * 20
        pages.append(PageText(i, text))

    res = filter_pages(pages)
    kept_nums = {p.page_number for p in res.kept_pages}
    # Middle page 10 MUST be kept!
    assert 10 in kept_nums


def test_safety_rule_preserves_chapter_summaries_and_exercises():
    """Chapter-end summaries, key points, and exercises MUST be kept even near end."""
    pages = [
        PageText(1, "Chapter 1: Foundational Systems. " + "core principles " * 30),
        PageText(2, "Chapter 2: Network Architectures. " + "packet switching " * 30),
        # Page 3 near the end contains chapter summary and exercises
        PageText(3, "Chapter Summary and Key Points:\n1. All nodes maintain routing state.\n"
                    "Exercises and Problems:\n1. Calculate the shortest path using Dijkstra's algorithm.\n"
                    "2. Prove convergence of distance vector protocols.\n" + "exercise problems " * 10),
    ]
    res = filter_pages(pages)
    kept_nums = {p.page_number for p in res.kept_pages}
    assert 3 in kept_nums


def test_structural_toc_and_index_detection():
    """TOC dot leaders and Index line structures are recognized accurately."""
    toc_lines = [
        "1.1 Introduction ...................................... 1",
        "1.2 Asymptotic Notation ............................... 8",
        "1.3 Recurrence Relations ............................. 15",
        "1.4 Divide and Conquer ............................... 24",
        "2.1 Sorting Basics ................................... 35",
    ]
    assert _looks_like_toc(toc_lines)

    index_lines = [
        "A",
        "Adjacency list, 45, 68",
        "Adjacency matrix, 46, 70",
        "Algorithm, 1, 12, 105",
        "B",
        "Binary tree, 88, 92, 110",
        "Breadth-first search, 150",
    ]
    assert _looks_like_index(index_lines)


def test_toc_chapter_boundary_reuse():
    """Everything before first real chapter is front matter; after last chapter is back matter."""
    pages = [
        PageText(1, "Half Title: Algorithms"),
        PageText(2, "Published by Standard Press 2024. All rights reserved."),
        PageText(3, "Preface to the Third Edition. " + "introductory notes " * 20),
        PageText(4, "Chapter 1: Foundations of Algorithms. " + "runtime analysis " * 20),
        PageText(5, "Chapter 2: Sorting and Searching. " + "divide and conquer " * 20),
        PageText(6, "Index of Terms and Citations. " + "search keys " * 20),
    ]
    toc = [
        [1, "Preface", 3],
        [1, "Chapter 1: Foundations", 4],
        [1, "Chapter 2: Sorting", 5],
        [1, "Index", 6],
    ]
    res = filter_pages(pages, toc_info=toc)
    kept_nums = [p.page_number for p in res.kept_pages]
    assert kept_nums == [4, 5]
    assert res.content_start_page == 4
    assert res.content_end_page == 5


def test_llm_fallback_uncertain_pages():
    """When FILTER_USE_LLM=False, uncertain pages are kept; when True, mock client classifies."""
    uncertain_page = PageText(
        1,
        "Notes for the reader regarding mathematical notation and conventions used throughout this volume. " * 5,
    )
    # 10 pages total, so page 1 is in the front-matter zone (~15%) without an explicit chapter heading
    pages = [uncertain_page] + [
        PageText(i, f"Algorithmic formulations, data structures, and system complexity section {i}. " * 10)
        for i in range(2, 11)
    ]

    # Off by default: safety rule keeps uncertain page
    res_off = filter_pages(pages, use_llm=False)
    assert 1 in [p.page_number for p in res_off.kept_pages]

    # When LLM enabled with a mock classifier client
    class CustomMockClassifier:
        def generate(self, prompt):
            # Tell LLM to skip page 1 as front matter
            return "PAGE 1: SKIP - notation and conventions front matter"

    res_on = filter_pages(pages, use_llm=True, client=CustomMockClassifier())
    assert 1 not in [p.page_number for p in res_on.kept_pages]
    skipped_reasons = {sp.page.page_number: sp.reason for sp in res_on.skipped_pages}
    assert "notation and conventions" in skipped_reasons[1]



def test_end_to_end_pdf_ingestion_drops_front_and_back_matter(tmp_path):
    """End-to-end PDF test: front and back matter dropped before chunking."""
    pdf_path = tmp_path / "textbook_with_margins.pdf"
    doc = fitz.open()

    # Page 1: Title & Copyright
    p1 = doc.new_page()
    p1.insert_text((72, 100), "Modern Algorithm Design", fontsize=18)
    p1.insert_text((72, 140), "ISBN 978-0-123456-78-9. All rights reserved. Copyright 2024.", fontsize=10)

    # Page 2: Table of Contents
    p2 = doc.new_page()
    p2.insert_text((72, 80), "Table of Contents", fontsize=16)
    p2.insert_text((72, 120), "Chapter 1 Introduction ...................... 3", fontsize=10)
    p2.insert_text((72, 140), "Chapter 2 Sorting ........................... 4", fontsize=10)

    # Page 3: Real Chapter 1 content
    p3 = doc.new_page()
    p3.insert_text((72, 80), "Chapter 1 Introduction", fontsize=18)
    p3.insert_text((72, 110), "1.1 Algorithmic Paradigms", fontsize=14)
    body3 = (
        "An algorithm is a well-defined computational procedure that takes input values and produces output. "
        "Sorting algorithms rearrange an array of elements in monotonic non-decreasing order. "
        "Divide and conquer algorithms partition problems into sub-problems solved recursively. "
        "Merge sort achieves an asymptotic runtime bound of O(n log n) in all operational cases."
    ) * 3
    p3.insert_textbox(fitz.Rect(72, 130, 520, 750), body3, fontsize=10)

    # Page 4: Exercises & Chapter 2
    p4 = doc.new_page()
    p4.insert_text((72, 80), "Chapter 2 Sorting and Searching", fontsize=18)
    body4 = (
        "Quick sort partitions arrays around a pivot element with average time complexity of O(n log n). "
        "Heap sort organizes items into a balanced binary max-heap data structure to extract maxima. "
        "Exercises and Review Questions:\n"
        "1. Analyze the worst-case space complexity of recursive quicksort implementations.\n"
        "2. Prove the comparison-based sorting lower bound using a decision tree model."
    ) * 3
    p4.insert_textbox(fitz.Rect(72, 110, 520, 750), body4, fontsize=10)

    # Page 5: Index
    p5 = doc.new_page()
    p5.insert_text((72, 80), "Index", fontsize=18)
    p5.insert_text((72, 110), "Binary search, 12, 15", fontsize=10)
    p5.insert_text((72, 130), "Divide and conquer, 3, 20", fontsize=10)
    p5.insert_text((72, 150), "Merge sort, 3, 4", fontsize=10)
    p5.insert_text((72, 170), "Quick sort, 4, 18", fontsize=10)

    doc.save(str(pdf_path))
    doc.close()

    # Ingest with report
    chunks, fres = ingest_with_report(str(pdf_path))

    # Pages 1, 2, and 5 must be skipped
    skipped_pages = [sp.page.page_number for sp in fres.skipped_pages]
    assert 1 in skipped_pages
    assert 2 in skipped_pages
    assert 5 in skipped_pages

    # Pages 3 and 4 must be kept
    kept_pages = [p.page_number for p in fres.kept_pages]
    assert kept_pages == [3, 4]
    assert fres.content_start_page == 3
    assert fres.content_end_page == 4

    # All generated chunks must come only from kept pages
    assert len(chunks) > 0
    assert all(c["page_start"] in (3, 4) and c["page_end"] in (3, 4) for c in chunks)
    assert not any("Modern Algorithm Design" in c["text"] for c in chunks)
