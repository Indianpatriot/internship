"""Course Outcome (CO/LO) semantic mapping using SentenceTransformer embeddings."""

_model = None


def _m():
    """Lazy-load the SentenceTransformer embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model


def score_chunks_against_outcomes(chunks, outcomes: dict[str, str]):
    """Embed outcome descriptions and chunk texts to compute cosine similarities.

    Sets chunk["co_scores"] = {code: cosine_similarity} for every chunk and every outcome.
    Does not assign a single CO per chunk.

    Args:
        chunks: list of chunk dicts (must contain 'text').
        outcomes: mapping of outcome code to description (e.g. {'CO1': '...', 'LO1': '...'}).

    Returns:
        The modified list of chunks with 'co_scores' populated.
    """
    if not chunks:
        return chunks
    if not outcomes:
        for c in chunks:
            c["co_scores"] = {}
        return chunks

    codes = list(outcomes.keys())
    model = _m()
    o_vecs = model.encode(list(outcomes.values()), normalize_embeddings=True)
    c_vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    sims = c_vecs @ o_vecs.T
    for c, row in zip(chunks, sims):
        c["co_scores"] = {code: float(s) for code, s in zip(codes, row)}
    return chunks