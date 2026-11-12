"""Figure retrieval and ranking module for exam questions.

Scores candidate textbook figures against questions and scheme diagram criteria
using semantic embeddings, proximity boosts (same book, adjacent pages, same chapter),
and caption/label detection, ensuring honest reporting when no suitable figure exists.
"""
import re
from typing import Any, Optional

import numpy as np

import config
from contracts import Figure

DIAGRAM_KEYWORDS = {
    "diagram",
    "draw",
    "illustrate",
    "represent",
    "architecture",
    "flowchart",
    "layered",
    "layering",
    "topology",
    "model",
    "lifecycle",
}


def question_needs_diagram(question_text: str, scheme: Optional[dict[str, Any]] = None) -> bool:
    """Check if a question or its marking scheme explicitly requests or benefits from a diagram."""
    q_lower = (question_text or "").lower()
    words = set(re.findall(r"\b[a-z]{3,}\b", q_lower))
    if any(k in words for k in DIAGRAM_KEYWORDS):
        return True

    if scheme and isinstance(scheme, dict):
        for crit in scheme.get("criteria", []):
            if crit.get("kind") == "diagram":
                return True
    return False


def find_figures_for_question(
    question_record: dict[str, Any],
    source_chunk: Optional[dict[str, Any]],
    scheme: Optional[dict[str, Any]],
    figures: list[Figure],
    embedder: Any,
    max_figures: Optional[int] = None,
    min_score: Optional[float] = None,
    used_figure_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Retrieve and rank textbook figures for a question.

    Args:
        question_record: Question dictionary containing 'question', 'book_id', 'page', etc.
        source_chunk: Text chunk from which the question was generated.
        scheme: Marking scheme dictionary (optional).
        figures: Full list of extracted textbook figures.
        embedder: SentenceTransformer embedding model.
        max_figures: Maximum figures to return (defaults to config.FIGURE_MAX_PER_QUESTION).
        min_score: Minimum cosine similarity threshold (defaults to config.FIGURE_MIN_SCORE).
        used_figure_ids: Set of figure IDs already assigned to previous questions.

    Returns:
        List of matched figure dictionaries: [{figure_id, image_path, caption, book_id, book_title, page}].
    """
    if not figures:
        return []

    q_text = question_record.get("question", "")
    if not question_needs_diagram(q_text, scheme):
        return []

    limit = max_figures if max_figures is not None else getattr(config, "FIGURE_MAX_PER_QUESTION", 2)
    threshold = min_score if min_score is not None else getattr(config, "FIGURE_MIN_SCORE", 0.45)

    # Compile query from question and diagram expected criteria
    query_parts = [q_text]
    if scheme and isinstance(scheme, dict):
        for crit in scheme.get("criteria", []):
            if crit.get("kind") == "diagram":
                if crit.get("expected"):
                    query_parts.append(crit["expected"])
                if crit.get("criterion"):
                    query_parts.append(crit["criterion"])

    query_str = " ".join(query_parts).strip()
    if not query_str:
        return []

    # Encode query
    query_vec = embedder.encode([query_str], normalize_embeddings=True)[0]

    # Candidate figure descriptions
    fig_texts = [
        f"{fig.get('caption', '')}. {fig.get('nearby_text', '')}".strip() or "Figure diagram"
        for fig in figures
    ]
    fig_vecs = embedder.encode(fig_texts, normalize_embeddings=True)

    sims = np.dot(fig_vecs, query_vec)

    # Reference book and page from source chunk or question record
    ref_book_id = (source_chunk.get("book_id") if source_chunk else None) or question_record.get("book_id")
    ref_page = (source_chunk.get("page_start") if source_chunk else None) or question_record.get("page")
    ref_chapter = (source_chunk.get("chapter") if source_chunk else None) or question_record.get("chapter")

    scored_candidates = []
    for i, fig in enumerate(figures):
        base_score = float(sims[i])
        # Base semantic similarity must satisfy the threshold so proximity does not overpower mismatch
        if base_score < threshold:
            continue

        boost = 0.0
        # Boost 1: Same book
        if ref_book_id and fig.get("book_id") == ref_book_id:
            boost += 0.08

        # Boost 2: Page within or adjacent to source chunk (+/- 2 pages)
        fig_p = fig.get("page", 0)
        if ref_page and fig_p > 0:
            p_dist = abs(fig_p - ref_page)
            if p_dist == 0:
                boost += 0.12
            elif p_dist <= 2:
                boost += 0.06

        # Boost 3: Same chapter
        if ref_chapter and fig.get("chapter") and fig.get("chapter") == ref_chapter:
            boost += 0.06

        # Boost 4: Caption with a clear figure label
        if fig.get("figure_label"):
            boost += 0.05

        # Cap proximity/label boost at +0.25
        score = base_score + min(boost, 0.25)

        # Penalize already used figures across unrelated questions
        fid = fig.get("figure_id", "")
        if used_figure_ids and fid in used_figure_ids:
            score -= 0.15

        if score >= threshold:
            scored_candidates.append((score, fig))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    results = []
    book_title = (source_chunk.get("book_title") if source_chunk else None) or question_record.get("book_title", "Textbook")
    for score, fig in scored_candidates[:limit]:
        fid = fig["figure_id"]
        if used_figure_ids is not None:
            used_figure_ids.add(fid)
        results.append({
            "figure_id": fid,
            "image_path": fig.get("image_path", ""),
            "caption": fig.get("caption", "").strip() or fig.get("figure_label", "Diagram"),
            "book_id": fig.get("book_id", "book"),
            "book_title": book_title,
            "page": fig.get("page", 1),
            "score": round(score, 3),
        })

    return results


def attach_figures_to_scheme(
    scheme: dict[str, Any],
    figures_matched: list[dict[str, Any]],
) -> None:
    """Attach matched figures to the diagram criterion or reference_figures."""
    if not figures_matched:
        return

    criteria = scheme.get("criteria", [])
    attached = False
    for crit in criteria:
        if crit.get("kind") == "diagram":
            crit["figures"] = list(figures_matched)
            attached = True
            break

    if not attached:
        scheme["reference_figures"] = list(figures_matched)
