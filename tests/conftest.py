import fitz
import pytest

A = ["kmeans", "svm", "knn", "perceptron", "bayes", "boosting", "pca", "lasso", "ridge", "forest", "tree", "kernel"]
B = ["loss", "margin", "weight", "bias", "entropy", "gradient", "variance", "cluster", "feature", "threshold"]
C = ["noisy", "sparse", "labelled", "streaming", "batch", "random", "sampled"]


def make_body(seed, n=36):
    return " ".join(
        f"The {A[(i + seed) % len(A)]} method tunes the {B[(i * 3 + seed) % len(B)]} "
        f"parameter using {C[(i * 5 + seed) % len(C)]} feedback signals."
        for i in range(n))



def _heading(page, y, text, size):
    page.insert_text((72, y), text, fontsize=size, fontname="hebo")
    return y + size + 22


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "Sample Book.pdf"
    doc = fitz.open()
    for p in range(3):
        page = doc.new_page()
        page.insert_text((72, 30), "Sample Textbook", fontsize=9)
        page.insert_text((72, 820), f"Page {p + 1}", fontsize=9)
        y = 110
        if p == 0:
            y = _heading(page, y, "Chapter 1 Introduction", 20)
            y = _heading(page, y, "1.1 What is Learning", 14)
        elif p == 1:
            y = _heading(page, y, "1.2 Types of Learning", 14)
        else:
            y = _heading(page, y, "Chapter 2 Evaluation", 20)
        page.insert_textbox(fitz.Rect(72, y, 520, 780), make_body(p * 7), fontsize=11, fontname="helv")
    doc.save(path)
    return str(path)


class _FakeEmbedder:
    """Deterministic bag-of-words embedder used when sentence-transformers is unavailable."""

    DIM = 256

    def encode(self, texts, normalize_embeddings=True, **_):
        import re
        import zlib

        import numpy as np

        out = np.zeros((len(texts), self.DIM), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % self.DIM] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms == 0, 1.0, norms)
        return out


@pytest.fixture(autouse=True)
def _offline_embedder(monkeypatch):
    """Use a fake embedder only if the real model package is not installed."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        import retrieval.co_mapper as co_mapper

        monkeypatch.setattr(co_mapper, "_m", lambda: _FakeEmbedder())
