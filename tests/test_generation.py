import json

from evaluation.evaluate import evaluate, write_review_sheet
from generation.generator import clean_output, generate_paper, generate_question
from generation.mock_client import MockClient
from generation.prompts import build_prompt
from ingestion.pipeline import ingest


class BadThenGood:
    def __init__(self):
        self.n = 0

    def generate(self, prompt, temperature=None, num_predict=120):
        self.n += 1
        return "Here is a question:\nWhat is overfitting?" if self.n == 1 else "Q1. Define overfitting in machine learning models?"


def test_prompt_contains_constraints():
    p = build_prompt("some context here", "L3", ["apply", "solve"], marks=5)
    assert "apply, solve" in p and "ONLY" in p and "some context here" in p and "paragraph" in p


def test_clean_output():
    assert clean_output('Sure, here is one:\n1. "Define X in context?"') == "Define X in context?"
    assert clean_output("```\nQuestion: Explain Y clearly please?\n```") == "Explain Y clearly please?"


def test_retry_until_valid():
    chunk = {"text": "word " * 60, "chunk_id": "c1"}
    rec = generate_question(chunk, "L1", BadThenGood(), max_retries=3)
    assert rec["valid"] and rec["attempts"] == 2 and rec["verb"] == "define"


def test_end_to_end_with_mock(sample_pdf, tmp_path):
    chunks = ingest(sample_pdf, max_words=80, overlap_words=15, min_words=20)
    out = generate_paper(chunks, {"L1": 2, "L3": 2, "L6": 2}, MockClient(), seed=1)
    qs = out["questions"]
    assert len(qs) >= 4
    assert all(q["valid"] for q in qs)
    assert {q["level"] for q in qs} <= {"L1", "L3", "L6"}
    report, rows = evaluate(qs, chunks)
    assert report["verb_compliance_pct"] == 100.0
    write_review_sheet(rows, tmp_path / "sheet.csv")
    assert (tmp_path / "sheet.csv").read_text().startswith("id,level,question")
    json.dumps(out)
