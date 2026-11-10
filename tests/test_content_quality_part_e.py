"""Tests for PART E: Content Quality & Academic Standard Verification.

Covers:
1. Token hallucination extraction & context verification (`check_hallucinated_tokens`).
2. Stem linting (`lint_question_stem`): deictic phrases, presuppositions, premise-in-stem, stray quotes/typos, two-part stems < 6M.
3. Deprecated currency watchlist (`check_currency_watchlist`).
4. Known technical traps (`validate_known_technical_facts`).
5. Zero shared pages between OR twins and paper-wide page deduplication.
"""
import pytest
from generation.generator import (
    check_currency_watchlist,
    check_hallucinated_tokens,
    extract_technical_tokens,
    generate_paper_by_slots,
    lint_question_stem,
    validate_known_technical_facts,
)
from generation.mock_client import MockClient


# -----------------------------------------------------------------------------
# 1. Hallucinated Token Extraction & Context Grounding
# -----------------------------------------------------------------------------

def test_extract_technical_tokens():
    answer = (
        "Run `docker run` with `--privileged` and `-d` flags. Use `apt-get` inside the container "
        "and verify with `kubectl get pods`."
    )
    tokens = extract_technical_tokens(answer)
    assert "--privileged" in tokens
    assert "-d" in tokens
    assert "docker run" in tokens
    assert "apt-get" in tokens
    assert "kubectl get" in tokens


def test_check_hallucinated_tokens_flagging():
    context = "The container runs in isolated mode. Use standard commands to inspect state."
    answer = "Use `docker run` with `--privileged` flag to bypass isolation."
    ungrounded = check_hallucinated_tokens(answer, context)
    assert "--privileged" in ungrounded
    assert "docker run" in ungrounded

    grounded_context = "Execute docker run with --privileged to grant container full capabilities."
    ungrounded_clean = check_hallucinated_tokens(answer, grounded_context)
    assert len(ungrounded_clean) == 0


# -----------------------------------------------------------------------------
# 2. Stem Linter (Academic Paper Standards)
# -----------------------------------------------------------------------------

def test_lint_question_stem_deictic():
    stems_with_deictic = [
        "In this chapter, explain how container namespaces work?",
        "As seen above, outline the steps of image layer caching?",
        "At the time of writing, what was the status of Docker Swarm?",
        "According to the author, describe pod scheduling?",
        "Based on the provided passage, define cgroups?",
    ]
    for stem in stems_with_deictic:
        ok, reason = lint_question_stem(stem, marks=5)
        assert not ok, f"Expected rejection for deictic stem '{stem}', got reason: {reason}"
        assert "book-deictic" in reason


def test_lint_question_stem_presupposition():
    stems_with_presupposition = [
        "Justify your choice of Kubernetes over Docker Swarm for enterprise microservices?",
        "Why did you choose containerization instead of virtual machines?",
        "Which would you prefer between monolithic and microservice architecture?",
        "Explain your decision regarding statefulset volume claims?",
    ]
    for stem in stems_with_presupposition:
        ok, reason = lint_question_stem(stem, marks=10)
        assert not ok, f"Expected rejection for presupposition stem '{stem}', got reason: {reason}"
        assert "presupposition" in reason


def test_lint_question_stem_premise():
    bad_stem = "Given that microservices always fail under load, why should organizations adopt them?"
    ok, reason = lint_question_stem(bad_stem, marks=10)
    assert not ok
    assert "premise" in reason


def test_lint_question_stem_punctuation_and_quotes():
    assert not lint_question_stem('"Explain container isolation?"', marks=5)[0]
    assert not lint_question_stem("What is a container??", marks=5)[0]
    assert not lint_question_stem("Outline the steps..", marks=5)[0]


def test_lint_question_stem_twopart_under_six_marks():
    compound_stem = "Explain Docker container networking and compare bridge with host modes?"
    ok, reason = lint_question_stem(compound_stem, marks=4)
    assert not ok
    assert "not permitted under 6 marks" in reason

    # Permitted when >= 6 marks
    ok_6m, _ = lint_question_stem(compound_stem, marks=6)
    assert ok_6m


# -----------------------------------------------------------------------------
# 3. Currency Watchlist
# -----------------------------------------------------------------------------

def test_check_currency_watchlist():
    deprecated_texts = [
        ("Deploy a pod using rkt runtime", "rkt"),
        ("Templates generated using ksonnet for Kubernetes", "ksonnet"),
        ("CI/CD pipeline using fabric8 tooling", "fabric8"),
        ("Provisioning host nodes using docker machine", "docker machine"),
        ("Connect web container to db using docker links", "docker links"),
        ("Kubernetes node using dockershim bridge", "dockershim"),
        ("Manage pod replicas via ReplicationController manifest", "replicationcontroller"),
    ]
    for text, expected_term in deprecated_texts:
        found = check_currency_watchlist(text)
        assert expected_term in found, f"Failed to detect {expected_term} in '{text}'"


# -----------------------------------------------------------------------------
# 4. Known Technical Traps
# -----------------------------------------------------------------------------

def test_validate_known_technical_facts():
    # Trap 1: docker export config loss
    ok1, reason1 = validate_known_technical_facts(
        question="What does docker export do?",
        answer="docker export preserves config and entrypoint metadata while saving the filesystem.",
    )
    assert not ok1
    assert "loses metadata" in reason1

    # Trap 2: docker build creates Dockerfile
    ok2, reason2 = validate_known_technical_facts(
        question="How does docker build work?",
        answer="docker build creates a Dockerfile from the container state.",
    )
    assert not ok2
    assert "does NOT create" in reason2

    # Trap 3: apt-get on Alpine
    ok3, reason3 = validate_known_technical_facts(
        question="How to install packages in an Alpine container?",
        answer="In an Alpine Linux container, run apt-get update to install dependencies.",
    )
    assert not ok3
    assert "Alpine Linux uses apk" in reason3

    # Trap 4: restartPolicy on ReplicationController
    ok4, reason4 = validate_known_technical_facts(
        question="Configure ReplicationController",
        answer="Set restartPolicy: OnFailure in the ReplicationController spec.",
    )
    assert not ok4
    assert "ReplicationController only supports restartPolicy: Always" in reason4


# -----------------------------------------------------------------------------
# 5. Paper-wide OR Twin Page Disjointness
# -----------------------------------------------------------------------------

def test_or_twin_zero_shared_pages():
    chunks = [
        {
            "chunk_id": "c1",
            "book_id": "b1",
            "chapter": "Chapter 1",
            "text": "Docker storage drivers and volume management primitives overview. " * 6,
            "page_start": 10,
            "word_count": 60,
            "co_scores": {"CO1": 0.9},
        },
        {
            "chunk_id": "c2",
            "book_id": "b1",
            "chapter": "Chapter 1",
            "text": "Kubernetes persistent volumes and persistent volume claims lifecycle. " * 6,
            "page_start": 25,
            "word_count": 60,
            "co_scores": {"CO1": 0.9},
        },
    ]
    slots = [
        {"unit": 1, "question_no": 1, "sub_index": 1, "level": "L2", "co": "CO1", "marks": 10, "alternative_group": "A", "kd": "Conceptual"},
        {"unit": 1, "question_no": 2, "sub_index": 1, "level": "L2", "co": "CO1", "marks": 10, "alternative_group": "B", "kd": "Conceptual"},
    ]
    res = generate_paper_by_slots(chunks, slots, client=MockClient(), seed=42)
    assert len(res["questions"]) == 2
    pages = [q.get("page") for q in res["questions"]]
    assert len(set(pages)) == 2
    assert pages[0] != pages[1]

