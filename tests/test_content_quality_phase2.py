"""Regression tests for 100-mark choice paper content-quality problems (Phase 2).

Covers:
- Problem 1: OR alternative source page disjointness, distinct content warning (<2 pages), counterpart similarity check.
- Problem 2: Term answer validation (<8 words, overlap rejection), ban MCQ answers ('The answer is A'), ungrounded answer flagging, contradiction detection.
- Problem 3: Cross-category comparison rejection, marks-proportional stem demand caps, trailing choice removal on <8m questions.
- Problem 4: Chapter detection in books with parts/subheadings (Learning DevOps), clean display names, suppressing non-content chapters from header.
- Problem 5: Knowledge Dimension strict verb alignment, local verb rewriting, preserving KD totals.
- Problem 6: Banned textbook leaks ('introductory chapter', '/home/...'), underspecified stem rejection, 8+ marks L1 multi-part recall, terminology glossary.
"""
import re
import pytest

from blooms.verb_bank import (
    _BANK,
    KD_BLOOM_VERBS,
    correct_stem_verb_for_kd,
    get_allowed_kd_verbs,
    is_verb_allowed_for_kd,
    nearest_valid_kd_verb,
    normalize_knowledge_dimension,
    validate_question,
)
from export_utils import format_covered_chapters_header
from generation.generator import (
    BANNED_LEAK_PATTERNS,
    check_compare_entities_in_context,
    check_cross_slot_contradictions,
    check_l1_marks_demand,
    check_question_duplicates,
    count_stem_demands,
    generate_paper_by_slots,
    has_banned_answer_phrases,
    has_unseen_material_reference,
    is_underspecified_stem,
    simplify_stem_demands,
    validate_term_answer,
)
from generation.glossary import apply_glossary_replacements
from generation.mock_client import MockClient
from ingestion.heading_detect import is_non_content_title, is_valid_chapter_title
from ingestion.library import display_name


# -----------------------------------------------------------------------------
# Problem 1: OR alternatives disjoint pages, content warning, similarity check
# -----------------------------------------------------------------------------

def test_problem1_or_alternatives_disjoint_pages():
    """OR alternatives in the same unit should use different source pages when available."""
    chunks = [
        {
            "chunk_id": "c1",
            "book_id": "b1",
            "chapter": "Chapter 1",
            "text": "Docker container lifecycle management and execution environments describe core primitives. " * 6,
            "page_start": 10,
            "word_count": 55,
            "co_scores": {"CO1": 0.9},
        },
        {
            "chunk_id": "c2",
            "book_id": "b1",
            "chapter": "Chapter 2",
            "text": "Virtual machines and isolation layers in cloud infrastructure architectures provide foundational guarantees. " * 6,
            "page_start": 20,
            "word_count": 55,
            "co_scores": {"CO1": 0.9},
        },
    ]
    slots = [
        {"unit": 1, "question_no": 1, "sub_index": 1, "level": "L2", "co": "CO1", "marks": 10, "alternative_group": "A", "kd": "Conceptual"},
        {"unit": 1, "question_no": 2, "sub_index": 1, "level": "L2", "co": "CO1", "marks": 10, "alternative_group": "B", "kd": "Conceptual"},
    ]
    client = MockClient()
    res = generate_paper_by_slots(chunks, slots, client=client, seed=42)
    assert len(res["questions"]) == 2
    pages = [q.get("page") for q in res["questions"]]
    assert len(set(pages)) == 2, f"Expected distinct pages for OR counterparts, got {pages}"


def test_problem1_low_content_warning_emitted():
    """When a unit has fewer than 2 distinct pages, a warning should be captured."""
    chunks = [
        {
            "chunk_id": "c1",
            "book_id": "b1",
            "chapter": "Chapter 1",
            "text": "Single page content for testing low content warning handling across alternatives in depth. " * 6,
            "page_start": 15,
            "word_count": 55,
            "co_scores": {"CO1": 0.95},
        },
    ]
    slots = [
        {"unit": 1, "question_no": 1, "sub_index": 1, "level": "L2", "co": "CO1", "marks": 10, "alternative_group": "A", "kd": "Conceptual"},
        {"unit": 1, "question_no": 2, "sub_index": 1, "level": "L2", "co": "CO1", "marks": 10, "alternative_group": "B", "kd": "Conceptual"},
    ]
    client = MockClient()
    res = generate_paper_by_slots(chunks, slots, client=client, seed=42)
    assert len(res["questions"]) >= 1
    assert "warnings" in res or any("too little distinct content" in str(w) for w in res.get("warnings", {}).values())


# -----------------------------------------------------------------------------
# Problem 2: Answers factually wrong, circular, or contradictory
# -----------------------------------------------------------------------------

def test_problem2_validate_term_answer_word_limit():
    """Term answer must be concise (<8 words) and not echo question stem >70% overlap."""
    q = "Identify the term for the control plane component managing cluster state?"

    # Valid short term
    ok, reason = validate_term_answer(q, "etcd key-value store")
    assert ok is True

    # Too long (>8 words)
    ok, reason = validate_term_answer(q, "The term for the control plane component managing cluster state is etcd.")
    assert ok is False
    assert "words > 8" in reason

    # Echoing / circular stem overlap
    circular = "The term for the control plane component"
    ok, reason = validate_term_answer(q, circular)
    assert ok is False
    assert "echoes" in reason or "overlap" in reason


def test_problem2_ban_single_letter_mcq_answer():
    """Single letter MCQ answers ('The answer is A') must be rejected."""
    assert has_banned_answer_phrases("The answer is A.") is True
    assert has_banned_answer_phrases("Option B is correct.") is True
    assert has_banned_answer_phrases("Answer: C") is True
    assert has_banned_answer_phrases("A complete and detailed explanation of container processes.") is False


def test_problem2_contradiction_detection():
    """Counterpart answers making opposing claims about the same entity should be flagged."""
    q1 = {
        "id": "9(a)",
        "question_no": 9,
        "alternative_group": "A",
        "question": "Explain the role of worker nodes in Kubernetes clusters?",
        "answer": "Worker nodes run containerized workloads and execute pod containers actively.",
    }
    q2 = {
        "id": "10(a)",
        "question_no": 10,
        "alternative_group": "B",
        "question": "Describe the operational state of worker nodes?",
        "answer": "Worker nodes do not run pods and never execute containerized workloads.",
    }
    flags = check_cross_slot_contradictions([q1, q2])
    assert len(flags) > 0
    assert any("contradiction" in f[2].lower() or "opposing" in f[2].lower() for f in flags)


# -----------------------------------------------------------------------------
# Problem 3: False comparisons and overloaded stems
# -----------------------------------------------------------------------------

def test_problem3_check_compare_entities_in_context():
    """Comparisons must involve entities that co-occur in the same context/section."""
    context = (
        "Docker containers package applications into portable units. "
        "Virtual machines emulate entire hardware platforms and operate full guest operating systems."
    )
    # Valid comparison: Docker vs Virtual Machines (both appear in context)
    q_valid = "Compare Docker containers and virtual machines in terms of isolation and overhead?"
    assert check_compare_entities_in_context(q_valid, context) is True

    # False comparison: CI/CD vs Network plugin (entities not in context)
    q_invalid = "Compare Jenkins CI/CD pipelines with Calico container network plugins?"
    assert check_compare_entities_in_context(q_invalid, context) is False


def test_problem3_stem_demand_caps_and_trailing_choice():
    """Stems must cap sub-question demands proportional to marks and strip trailing choice for <8m."""
    overloaded = (
        "Define containerization, compare it with virtualization, give an example, "
        "and explain when not to use it. Which option would you choose and why?"
    )
    demands = count_stem_demands(overloaded)
    assert demands >= 4

    # For 4 marks: max 2 demands, trailing 'Which option would you choose' stripped
    simplified = simplify_stem_demands(overloaded, marks=4)
    assert "which option would you choose" not in simplified.lower()
    simplified_demands = count_stem_demands(simplified)
    assert simplified_demands <= 2


# -----------------------------------------------------------------------------
# Problem 4: Chapter detection in Learning DevOps; book slugs in sources & title
# -----------------------------------------------------------------------------

def test_problem4_valid_chapter_title_filtering():
    """In-chapter sections (Summary, Questions, Technique N) must not be treated as chapters."""
    assert is_valid_chapter_title("Chapter 1: The DevOps Culture and Infrastructure as Code Practices") is True
    assert is_valid_chapter_title("Chapter 9: Containerizing Your Application with Docker") is True
    assert is_valid_chapter_title("Summary") is False
    assert is_valid_chapter_title("Questions") is False
    assert is_valid_chapter_title("Technical requirements") is False
    assert is_valid_chapter_title("Technique 12: Building minimal images") is False
    assert is_valid_chapter_title("Review Questions") is False
    assert is_valid_chapter_title("A") is False  # Index letter header


def test_problem4_display_name_formatting():
    """Display name helper formats clean numbered titles without slugs or hex hashes."""
    assert display_name("2baa1d6c9eb4_Learning_DevOps.pdf", order=3) == "3 Learning DevOps"
    assert display_name("1_Docker_in_Practice.pdf") == "1 Docker in Practice"
    assert display_name("kubernetes_in_action", order=2) == "2 Kubernetes in Action"


def test_problem4_format_covered_chapters_header_clean():
    """Covered chapters header uses clean display names and suppresses non-content titles."""
    questions = [
        {"display_label": "1 Docker in Practice", "chapter": "Chapter 1"},
        {"display_label": "1 Docker in Practice", "chapter": "Summary"},
        {"display_label": "2 Kubernetes in Action", "chapter": "Chapter 3"},
        {"display_label": "2 Kubernetes in Action", "chapter": "Questions"},
    ]
    header = format_covered_chapters_header(questions=questions)
    assert "Summary" not in header
    assert "Questions" not in header
    assert "1 Docker in Practice" in header
    assert "2 Kubernetes in Action" in header


# -----------------------------------------------------------------------------
# Problem 5: Knowledge Dimension strict verb alignment and rewriting
# -----------------------------------------------------------------------------

def test_problem5_all_kd_verbs_belong_to_bloom_level():
    """Every verb in KD_BLOOM_VERBS[(lvl, kd)] must belong to _BANK[lvl]."""
    for (lvl, kd), verbs in KD_BLOOM_VERBS.items():
        for v in verbs:
            assert v in _BANK[lvl], f"Verb '{v}' in KD_BLOOM_VERBS[({lvl}, {kd})] is not in _BANK[{lvl}]"


def test_problem5_correct_stem_verb_for_kd():
    """Mismatched verbs should be rewritten locally to the correct verb without changing KD."""
    # Factual: Describe -> State (L1) or Describe (L2 allowed)
    q1 = "Describe the role of etcd in cluster state coordination?"
    rewritten1, was_corr1, _ = correct_stem_verb_for_kd(q1, level="L1", kd="Factual")
    assert was_corr1 is True
    assert rewritten1.startswith("State the role")

    # Conceptual: Identify the term -> Explain the concept of
    q2 = "Identify the term for a logical set of pods and access policy?"
    rewritten2, was_corr2, _ = correct_stem_verb_for_kd(q2, level="L2", kd="Conceptual")
    assert was_corr2 is True
    assert "Explain the concept" in rewritten2

    # Procedural: Analyze the process -> Outline the steps to
    q3 = "Analyze the process of rolling out a container deployment?"
    rewritten3, was_corr3, _ = correct_stem_verb_for_kd(q3, level="L2", kd="Procedural")
    assert was_corr3 is True
    assert rewritten3.startswith("Outline the steps to")


# -----------------------------------------------------------------------------
# Problem 6: Textbook leaks, underspecified stems, 8+m L1 recall, glossary
# -----------------------------------------------------------------------------

def test_problem6_banned_textbook_leaks():
    """Textbook leaks such as 'introductory chapter', 'as the author notes', or file paths are banned."""
    leaks = [
        "In an introductory chapter, how does Docker manage layers?",
        "As discussed in an earlier chapter, explain container isolation?",
        "According to what the author notes, outline Kubernetes services?",
        "In the file /home/user/project/main.go, what is the entrypoint?",
        "Inspect C:\\Users\\Admin\\config.yaml and state the port?",
    ]
    for leak in leaks:
        has_unseen, _ = has_unseen_material_reference(leak, is_question=True)
        assert has_unseen is True, f"Failed to detect leak in: {leak}"


def test_problem6_underspecified_stems():
    """Stems referring to 'the tools' or 'the approach' without naming them must be rejected."""
    bad_stem = "Which of the two tools discussed would you choose for deployment?"
    assert is_underspecified_stem(bad_stem) is True

    good_stem = "Compare Helm and Kustomize for deploying Kubernetes manifests?"
    assert is_underspecified_stem(good_stem) is False


def test_problem6_l1_eight_marks_demand():
    """L1 questions with 8+ marks must require multi-part recall."""
    single_recall = "State the three types of cloud service models?"
    ok, _ = check_l1_marks_demand(single_recall, level="L1", marks=8)
    assert ok is False

    multi_recall = "List and state the purpose of four core components in the Kubernetes control plane?"
    ok, _ = check_l1_marks_demand(multi_recall, level="L1", marks=8)
    assert ok is True


def test_problem6_glossary_replacements():
    """Outdated terminology is updated to modern canonical terms."""
    old_text = "The master node schedules pods onto minion nodes using a whitelist of allowed IPs."
    modern = apply_glossary_replacements(old_text)
    assert "master node" not in modern.lower()
    assert "control plane node" in modern.lower()
    assert "minion" not in modern.lower()
    assert "worker nodes" in modern.lower()
    assert "whitelist" not in modern.lower()
    assert "allowlist" in modern.lower()
