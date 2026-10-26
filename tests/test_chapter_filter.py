"""Step 6 - Tests for chapter and section filtering."""
from chapter_filter import filter_chunks, get_chapter_metadata, get_filter_summary
from generation.mock_client import MockClient
from generation.generator import generate_paper_by_slots


def _sample_chunks():
    return [
        {
            "chunk_id": "c1",
            "chapter": "Chapter 1",
            "section": "1.1 Intro",
            "page_start": 5,
            "page_end": 7,
            "text": "Machine learning algorithms parse data and learn patterns to make intelligent decisions.",
            "word_count": 60,
            "co_scores": {"CO1": 0.85, "CO2": 0.20},
        },
        {
            "chunk_id": "c2",
            "chapter": "Chapter 2",
            "section": "2.1 Regression",
            "page_start": 20,
            "page_end": 22,
            "text": "Linear and logistic regression model continuous target values and discrete classification probabilities.",
            "word_count": 65,
            "co_scores": {"CO1": 0.30, "CO2": 0.90},
        },
        {
            "chunk_id": "c3",
            "chapter": "Chapter 2",
            "section": "2.2 Evaluation",
            "page_start": 23,
            "page_end": 25,
            "text": "Evaluation metrics include precision, recall, F1 score, and ROC-AUC curve analysis.",
            "word_count": 70,
            "co_scores": {"CO1": 0.25, "CO2": 0.88},
        },
        {
            "chunk_id": "c4",
            "chapter": "Chapter 3",
            "section": "3.1 Neural Nets",
            "page_start": 40,
            "page_end": 45,
            "text": "Neural networks utilize multilayer perceptrons, backpropagation gradients, and activation functions.",
            "word_count": 75,
            "co_scores": {"CO1": 0.40, "CO2": 0.75},
        },
    ]


def test_substep_6_1_select_single_chapter():
    """Substep 6.1: Select 'Chapter 2' and assert every returned chunk has chapter == 'Chapter 2'."""
    chunks = _sample_chunks()
    filtered = filter_chunks(chunks, selected_chapters=["Chapter 2"])
    assert len(filtered) == 2
    for c in filtered:
        assert c["chapter"] == "Chapter 2"


def test_substep_6_2_empty_selection_and_page_order():
    """Substep 6.2: Assert that an empty selection returns an empty list, and chapters come back in page order."""
    chunks = _sample_chunks()

    # Empty selection returns empty list
    empty_result = filter_chunks(chunks, selected_chapters=[])
    assert empty_result == []

    # Verify chapters come back in book order (by page_start)
    meta = get_chapter_metadata(chunks)
    assert len(meta) == 3
    assert [m["chapter"] for m in meta] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert meta[0]["page_start"] < meta[1]["page_start"] < meta[2]["page_start"]

    # Filter with mixed chapter order should preserve book order in result
    filtered = filter_chunks(chunks, selected_chapters=["Chapter 3", "Chapter 1"])
    assert len(filtered) == 2
    assert filtered[0]["chapter"] == "Chapter 1"
    assert filtered[1]["chapter"] == "Chapter 3"


def test_substep_6_3_generate_paper_respects_chapter_filter():
    """Substep 6.3: Run generate_paper_by_slots on filtered list; all question chapters are within selection."""
    chunks = _sample_chunks()
    selected = ["Chapter 2"]
    filtered = filter_chunks(chunks, selected_chapters=selected)

    slots = [
        {"id": "1", "part": "Part A", "co": "CO2", "level": "L1", "marks": 2},
        {"id": "2", "part": "Part A", "co": "CO2", "level": "L2", "marks": 2},
    ]
    client = MockClient()
    result = generate_paper_by_slots(filtered, slots, client=client, seed=42)

    assert len(result["questions"]) > 0
    for q in result["questions"]:
        assert q["chapter"] in selected
        assert q["chapter"] == "Chapter 2"


def test_edge_case_page_range_fallback():
    """Test filtering by page range when chapters are not defined."""
    chunks = [
        {"chunk_id": "c1", "chapter": None, "page_start": 1, "page_end": 5, "text": "Page 1 to 5 content"},
        {"chunk_id": "c2", "chapter": None, "page_start": 6, "page_end": 10, "text": "Page 6 to 10 content"},
        {"chunk_id": "c3", "chapter": None, "page_start": 11, "page_end": 15, "text": "Page 11 to 15 content"},
    ]
    filtered = filter_chunks(chunks, page_range=(5, 9))
    assert len(filtered) == 2
    assert filtered[0]["chunk_id"] == "c1"
    assert filtered[1]["chunk_id"] == "c2"


def test_filter_summary_message():
    """Test live summary string formatting."""
    chunks = _sample_chunks()
    summary = get_filter_summary(chunks[:3], selected_chapters=["Chapter 1", "Chapter 2"])
    assert "2 chapters selected" in summary
    assert "3 chunks" in summary
    assert "pages 5–25" in summary

