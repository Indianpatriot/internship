"""Unit tests for Revised Bloom's Taxonomy (2D Cognitive & Knowledge Dimension framework)."""
from blooms.revised_taxonomy import (
    COGNITIVE_LEVELS,
    COGNITIVE_LEVEL_NAMES,
    KNOWLEDGE_DIMENSIONS,
    TAXONOMY_MATRIX_2D,
    compute_2d_taxonomy_matrix,
    get_taxonomy_cell,
    normalize_knowledge_dimension,
)
from blooms.verb_bank import validate_question, verb_level
from blueprint import calculate_blueprint_stats, get_50_mark_blueprint
from export_utils import create_docx_exam_paper, create_pdf_exam_paper
from generation.generator import generate_paper_by_slots
from generation.mock_client import MockClient


def test_revised_taxonomy_dimensions_and_matrix_structure():
    """Verify all 4 Knowledge Dimensions, 6 Cognitive Levels, and 24 matrix cells exist."""
    assert len(KNOWLEDGE_DIMENSIONS) == 4
    assert set(KNOWLEDGE_DIMENSIONS) == {"Factual", "Conceptual", "Procedural", "Metacognitive"}

    assert len(COGNITIVE_LEVELS) == 6
    assert set(COGNITIVE_LEVELS) == {"L1", "L2", "L3", "L4", "L5", "L6"}
    assert COGNITIVE_LEVEL_NAMES["L6"] == "Create"

    assert len(TAXONOMY_MATRIX_2D) == 24
    for kd in KNOWLEDGE_DIMENSIONS:
        for lvl in COGNITIVE_LEVELS:
            assert (kd, lvl) in TAXONOMY_MATRIX_2D
            cell = TAXONOMY_MATRIX_2D[(kd, lvl)]
            assert "verb" in cell and "task" in cell and "focus" in cell


def test_image_exemplar_verbs_in_verb_bank():
    """Verify exemplar verbs from the teacher's reference diagram are registered and validate."""
    # L2 Procedural
    assert verb_level("clarify") == "L2"
    # L3 Factual & Conceptual
    assert verb_level("respond") == "L3"
    assert verb_level("provide") == "L3"
    # L4 Procedural
    assert verb_level("integrate") == "L4"
    # L5 Factual, Conceptual & Metacognitive
    assert verb_level("check") == "L5"
    assert verb_level("determine") == "L5"
    assert verb_level("reflect") == "L5"
    # L6 Factual & Conceptual
    assert verb_level("generate") == "L6"
    assert verb_level("assemble") == "L6"

    # Validate generated questions
    assert validate_question("Clarify the database normalization rules?", "L2")[0]
    assert validate_question("Provide a solution for graph cycle detection?", "L3")[0]
    assert validate_question("Integrate security policies into the deployment pipeline?", "L4")[0]
    assert validate_question("Determine the runtime complexity of Dijkstra's algorithm?", "L5")[0]
    assert validate_question("Generate an optimal caching scheme for web servers?", "L6")[0]


def test_compute_2d_taxonomy_matrix():
    """Test generating 4x6 question distribution matrix from blueprint slots."""
    slots = [
        {"level": "L1", "knowledge_dimension": "Factual"},
        {"level": "L1", "knowledge_dimension": "Conceptual"},
        {"level": "L2", "knowledge_dimension": "Procedural"},
        {"level": "L6", "knowledge_dimension": "Procedural"},
    ]
    matrix = compute_2d_taxonomy_matrix(slots)
    assert matrix["Factual"]["L1"] == 1
    assert matrix["Conceptual"]["L1"] == 1
    assert matrix["Procedural"]["L2"] == 1
    assert matrix["Procedural"]["L6"] == 1
    assert matrix["Metacognitive"]["L1"] == 0

    total_counted = sum(matrix[kd][lvl] for kd in KNOWLEDGE_DIMENSIONS for lvl in COGNITIVE_LEVELS)
    assert total_counted == len(slots)


def test_50_mark_blueprint_with_revised_blooms():
    """Verify 50-mark blueprint covers Cognitive L1-L6 and Knowledge dimensions."""
    slots = get_50_mark_blueprint(["CO1", "CO2", "CO3"])
    stats = calculate_blueprint_stats(slots)

    assert stats["total_marks"] == 50
    assert stats["is_standard_50"] is True
    # Cognitive Process coverage
    assert "L1" in stats["bt_marks"]
    assert "L2" in stats["bt_marks"]
    assert "L3" in stats["bt_marks"]
    assert "L4" in stats["bt_marks"]
    assert "L5" in stats["bt_marks"]
    assert "L6" in stats["bt_marks"]

    # Knowledge Dimension coverage
    assert stats["kd_counts"]["Factual"] > 0
    assert stats["kd_counts"]["Conceptual"] > 0
    assert stats["kd_counts"]["Procedural"] > 0
    assert stats["kd_counts"]["Metacognitive"] > 0

    total_kd_marks = sum(stats["kd_marks"].values())
    assert total_kd_marks == 50


def test_docx_and_pdf_export_with_knowledge_dimension():
    """Verify DOCX and PDF export include Knowledge Dimension columns and reference."""
    questions = [
        {
            "id": "1",
            "part": "Part A",
            "question": "List the primary properties of binary search trees?",
            "level": "L1",
            "knowledge_dimension": "Factual",
            "co": "CO1",
            "marks": 2,
            "chapter": "Chapter 1",
            "page_start": 4,
        },
        {
            "id": "6(a)",
            "part": "Part B",
            "question": "Design an optimal indexing strategy for query execution?",
            "level": "L6",
            "knowledge_dimension": "Procedural",
            "co": "CO2",
            "marks": 5,
            "chapter": "Chapter 2",
            "page_start": 18,
        },
    ]
    outcomes = {"CO1": "Understand trees", "CO2": "Design queries"}

    docx_bytes = create_docx_exam_paper(questions, outcomes=outcomes, include_knowledge_dim=True)
    assert isinstance(docx_bytes, bytes)
    assert docx_bytes.startswith(b"PK\x03\x04")

    pdf_bytes = create_pdf_exam_paper(questions, outcomes=outcomes, include_knowledge_dim=True)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_paper_preserves_knowledge_dimension():
    """Ensure generation pipeline preserves knowledge dimension metadata on questions."""
    chunks = [
        {
            "chunk_id": "c1",
            "chapter": "Chapter 1",
            "text": "Binary search trees organize keys hierarchically such that left descendants are less than root.",
            "word_count": 50,
            "co_scores": {"CO1": 0.95},
        }
    ]
    slots = [
        {
            "id": "1",
            "part": "Part A",
            "co": "CO1",
            "level": "L1",
            "knowledge_dimension": "Factual",
            "marks": 2,
        }
    ]
    client = MockClient()
    result = generate_paper_by_slots(chunks, slots, client=client, seed=42)
    assert len(result["questions"]) == 1
    q = result["questions"][0]
    assert q["knowledge_dimension"] == "Factual"
    assert q["kd"] == "Factual"
    assert q["level"] == "L1"
