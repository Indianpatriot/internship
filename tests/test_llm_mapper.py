"""Tests for LLM-based Textbook Mapping auto-fill, manual preservation, and error handling."""
import pytest
from generation.mock_client import MockClient
from retrieval.llm_mapper import (
    auto_map_units_with_llm,
    build_mapping_prompt,
    extract_available_chapters,
    parse_mapping_response,
)


@pytest.fixture
def sample_library():
    return {
        "books": {
            "book_devops": {
                "title": "Learning DevOps and Containers",
                "display_label": "DevOps Guide",
                "chapters": ["Chapter 1: Container Fundamentals", "Chapter 2: Kubernetes Orchestration"],
            },
            "book_db": {
                "title": "Database Systems Concepts",
                "display_label": "Database Systems",
                "chapters": ["Chapter 1: Relational Algebra and SQL", "Chapter 2: Transaction Management"],
            },
        },
        "chunks": [
            {
                "book_id": "book_devops",
                "chapter": "Chapter 1: Container Fundamentals",
                "text": "Containers isolate applications using cgroups, namespaces, and lightweight Linux primitives.",
            },
            {
                "book_id": "book_db",
                "chapter": "Chapter 1: Relational Algebra and SQL",
                "text": "Relational algebra defines operations on relations including select, project, join, and union.",
            },
        ],
    }


def test_autofill_unmapped_units_only(sample_library):
    """Test that auto-fill only processes unmapped units and leaves already-mapped units untouched."""
    units = [
        {"unit": 1, "title": "Container Isolation", "topics": ["cgroups", "namespaces", "containers"]},
        {"unit": 2, "title": "Database Queries", "topics": ["relational algebra", "sql", "select"]},
        {"unit": 3, "title": "Cloud Deployment", "topics": ["kubernetes", "orchestration"]},
    ]

    client = MockClient()
    # Unit 1 is already mapped; only Units 2 and 3 should be auto-filled
    manually_mapped = {1}

    result = auto_map_units_with_llm(
        units=units,
        library=sample_library,
        client=client,
        min_confidence=0.50,
        manually_mapped_units=manually_mapped,
    )

    # Unit 1 was in manually_mapped, so it must not be in result
    assert 1 not in result

    # Units 2 and 3 should be mapped
    assert 2 in result
    assert result[2]["status"] == "auto_suggested"
    assert result[2]["confidence"] >= 0.50
    assert "book_db" in result[2]["chapters"]
    assert "Chapter 1: Relational Algebra and SQL" in result[2]["chapters"]["book_db"]

    assert 3 in result
    assert result[3]["status"] == "auto_suggested"
    assert "book_devops" in result[3]["chapters"]


def test_manual_mappings_preserved(sample_library):
    """Test that manual mappings are never overwritten even if auto-fill is triggered."""
    units = [
        {"unit": 1, "title": "Container Isolation", "topics": ["containers"]},
        {"unit": 2, "title": "Database Queries", "topics": ["sql"]},
    ]

    client = MockClient()
    # User manually mapped both units
    manually_mapped = {1, 2}

    result = auto_map_units_with_llm(
        units=units,
        library=sample_library,
        client=client,
        min_confidence=0.50,
        manually_mapped_units=manually_mapped,
    )

    # Empty result because all units are manually mapped
    assert result == {}


def test_invalid_model_output_rejected(sample_library):
    """Test that invalid chapter IDs, non-existent chapters, or malformed JSON are rejected cleanly."""
    available_chapters = extract_available_chapters(sample_library)

    # 1. Invalid chapter ID (e.g. C999 or non-existent chapter name)
    raw_invalid_ids = """
    {
      "1": {
        "chapter_ids": ["C999", "Fake Chapter That Does Not Exist"],
        "confidence": 0.95
      }
    }
    """
    res1 = parse_mapping_response(raw_invalid_ids, available_chapters, min_confidence=0.50)
    assert res1[1]["status"] == "unmapped"
    assert res1[1]["chapters"] == {}

    # 2. Malformed JSON
    raw_malformed = "I recommend mapping Unit 1 to Chapter 1, but I am not returning JSON."
    with pytest.raises(ValueError):
        parse_mapping_response(raw_malformed, available_chapters, min_confidence=0.50)

    # 3. Valid JSON with one real chapter and one invalid chapter -> real kept, invalid dropped
    raw_mixed = """
    {
      "2": {
        "chapter_ids": ["C1", "C999"],
        "confidence": 0.90
      }
    }
    """
    res3 = parse_mapping_response(raw_mixed, available_chapters, min_confidence=0.50)
    assert res3[2]["status"] == "auto_suggested"
    # Only the valid chapter for C1 (book_devops / Chapter 1) is retained
    all_resolved = [ch for chaps in res3[2]["chapters"].values() for ch in chaps]
    assert len(all_resolved) == 1
    assert "Chapter 1: Container Fundamentals" in all_resolved


def test_fallback_path_on_client_error(sample_library):
    """Test graceful fallback to unmapped status when the LLM client throws an exception."""
    class FailingClient:
        model = "failing_model"

        def generate(self, *args, **kwargs):
            raise ConnectionError("Ollama server connection refused at http://localhost:11434")

    units = [
        {"unit": 1, "title": "Databases", "topics": ["sql"]},
    ]

    result = auto_map_units_with_llm(
        units=units,
        library=sample_library,
        client=FailingClient(),
        min_confidence=0.50,
    )

    assert 1 in result
    assert result[1]["status"] == "unmapped"
    assert result[1]["chapters"] == {}
    assert "error" in result[1]["warning"].lower() or "connection" in result[1]["warning"].lower()


def test_low_confidence_leaves_unit_unmapped(sample_library):
    """Test that a unit below minimum confidence threshold remains unmapped."""
    available_chapters = extract_available_chapters(sample_library)
    raw_low_conf = """
    {
      "1": {
        "chapter_ids": ["C1"],
        "confidence": 0.35
      }
    }
    """
    # min_confidence is 0.50, response has 0.35
    res = parse_mapping_response(raw_low_conf, available_chapters, min_confidence=0.50)
    assert res[1]["status"] == "unmapped"
    assert res[1]["chapters"] == {}
    assert "low confidence" in res[1]["warning"].lower()

