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
            "unit": 1,
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
            "unit": 2,
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
            "unit": 1,
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


# -----------------------------------------------------------------------------
# PART F: Dynamic 2D KD classification, stem corrections, and CO verification
# -----------------------------------------------------------------------------

def test_infer_knowledge_dimension():
    """Verify 2D BT x KD classification of questions and answers."""
    from blooms.revised_taxonomy import infer_knowledge_dimension

    # Procedural questions
    assert infer_knowledge_dimension("Modify the code in the Dockerfile to enable multi-stage caching?") == "Procedural"
    assert infer_knowledge_dimension("Calculate the maximum throughput across cluster worker nodes?") == "Procedural"
    assert infer_knowledge_dimension("Outline the steps to configure persistent volume claims?") == "Procedural"
    assert infer_knowledge_dimension("Describe the steps to deploy a rolling update in Kubernetes?") == "Procedural"

    # Conceptual questions
    assert infer_knowledge_dimension("Distinguish between managed and unmanaged pods in Kubernetes?") == "Conceptual"
    assert infer_knowledge_dimension("Paraphrase the core principles of namespace isolation?") == "Conceptual"
    assert infer_knowledge_dimension("Compare StatefulSets and Deployments in Kubernetes clusters?") == "Conceptual"
    assert infer_knowledge_dimension("Explain how container overlay networks function?") == "Conceptual"

    # Metacognitive questions
    assert infer_knowledge_dimension("Assess and justify the choice of Helm charts over plain manifests?") == "Metacognitive"
    assert infer_knowledge_dimension("Evaluate the strategic trade-offs of microservice vs monolithic architectures?") == "Metacognitive"
    assert infer_knowledge_dimension("Choose between StatefulSets and Deployments and defend the choice?") == "Metacognitive"

    # Factual questions
    assert infer_knowledge_dimension("State the definition of a container image layer?") == "Factual"
    assert infer_knowledge_dimension("List the core components of the Kubernetes control plane?") == "Factual"
    assert infer_knowledge_dimension("Define the term daemonset?") == "Factual"


def test_problem6_kd_evidence_regressions():
    """Verify stem rewriting and correction for all 7 evidence items from Problem 6."""
    from blooms.verb_bank import correct_stem_verb_for_kd, is_verb_allowed_for_kd

    # 7(b): modify code tagged Conceptual -> rewrite to Explain how to adapt
    q_7b = "Modify code in Dockerfile to optimize layer caching?"
    corr_7b, was_corr_7b, _ = correct_stem_verb_for_kd(q_7b, level="L3", kd="Conceptual")
    assert was_corr_7b is True
    assert corr_7b.startswith("Explain how to adapt")

    # 8(b): calculate tagged Conceptual -> rewrite to Explain the factors determining
    q_8b = "Calculate the network throughput of the Kubernetes cluster pods?"
    corr_8b, was_corr_8b, _ = correct_stem_verb_for_kd(q_8b, level="L3", kd="Conceptual")
    assert was_corr_8b is True
    assert corr_8b.startswith("Explain the factors determining")

    # 5(b) & 6(b): distinguish/contrast tagged Procedural -> rewrite to Differentiate (L4) or Outline steps (L2)
    q_5b = "Distinguish between managed and unmanaged pods?"
    corr_5b, was_corr_5b, _ = correct_stem_verb_for_kd(q_5b, level="L4", kd="Procedural")
    assert was_corr_5b is True
    assert corr_5b.startswith("Differentiate between")

    q_6b = "Contrast bridge networking with host networking?"
    corr_6b, was_corr_6b, _ = correct_stem_verb_for_kd(q_6b, level="L4", kd="Procedural")
    assert was_corr_6b is True
    assert corr_6b.startswith("Differentiate bridge")

    # 5(c): paraphrase tagged Factual -> rewrite to State
    q_5c = "Paraphrase the definition of container virtualization?"
    corr_5c, was_corr_5c, _ = correct_stem_verb_for_kd(q_5c, level="L2", kd="Factual")
    assert was_corr_5c is True
    assert corr_5c.startswith("State")

    # 2(a): describe the steps tagged Conceptual -> rewrite to Explain the process of
    q_2a = "Describe the steps of pod initialization sequence?"
    corr_2a, was_corr_2a, _ = correct_stem_verb_for_kd(q_2a, level="L2", kd="Conceptual")
    assert was_corr_2a is True
    assert corr_2a.startswith("Explain the process of")

    # 4(a): compare tagged Metacognitive -> rewrite to Evaluate and justify
    q_4a = "Compare Helm and Kustomize for continuous deployment?"
    corr_4a, was_corr_4a, _ = correct_stem_verb_for_kd(q_4a, level="L4", kd="Metacognitive")
    assert was_corr_4a is True
    assert corr_4a.startswith("Evaluate and justify the trade-offs between")

    # 4(b): assess and justify tagged Procedural -> rewrite to Validate procedural efficiency
    q_4b = "Assess and justify the use of StatefulSets for clustered databases?"
    corr_4b, was_corr_4b, _ = correct_stem_verb_for_kd(q_4b, level="L5", kd="Procedural")
    assert was_corr_4b is True
    assert corr_4b.startswith("Validate the procedural efficiency of")


def test_allowed_kds_by_level_constraint():
    """Verify allowed KDs by Bloom's level for deficit balancing."""
    from blooms.revised_taxonomy import ALLOWED_KDS_BY_LEVEL

    # L1 must only have Factual and Conceptual
    assert ALLOWED_KDS_BY_LEVEL["L1"] == ["Factual", "Conceptual"]
    assert "Metacognitive" not in ALLOWED_KDS_BY_LEVEL["L1"]
    assert "Procedural" not in ALLOWED_KDS_BY_LEVEL["L1"]

    # Higher levels allow Metacognitive
    assert "Metacognitive" in ALLOWED_KDS_BY_LEVEL["L5"]
    assert "Metacognitive" in ALLOWED_KDS_BY_LEVEL["L6"]


def test_verify_question_co_alignment_keyword_and_qa_flag():
    """Verify that questions misaligned with their assigned CO are flagged with CO_MISMATCH."""
    from export_utils import extract_qa_flags_cell
    from generation.generator import verify_question_co_alignment

    outcomes = {
        "CO1": "Understand container virtualization fundamentals, Docker daemon, and image management.",
        "CO2": "Deploy and manage container orchestration clusters using Kubernetes, pods, services, and deployments.",
    }

    # Question clearly about Kubernetes pods/services, but assigned CO1 (Docker fundamentals)
    questions = [
        {
            "id": "1",
            "question": "Deploy and manage container orchestration clusters with Kubernetes pods and services?",
            "answer": "Point 1 (3M): Deploy clusters. Point 2 (2M): Manage services.",
            "co": "CO1",
            "valid": True,
            "review_flags": [],
            "qa_flags": [],
        },
        {
            "id": "2",
            "question": "Explain Docker daemon architecture and container image management?",
            "answer": "Point 1 (3M): Docker daemon. Point 2 (2M): Image layers.",
            "co": "CO1",
            "valid": True,
            "review_flags": [],
            "qa_flags": [],
        },
    ]

    mismatches = verify_question_co_alignment(questions, outcomes, embedder=None)
    assert len(mismatches) == 1
    assert mismatches[0][0] == 0  # Question 1 was flagged
    assert "CO mismatch" in mismatches[0][1]

    # Verify QA flag extraction
    questions[0]["review_flags"].append(mismatches[0][1])
    questions[0]["qa_flags"].append("CO_MISMATCH")
    flag_cell = extract_qa_flags_cell(questions[0])
    assert "CO_MISMATCH" in flag_cell


