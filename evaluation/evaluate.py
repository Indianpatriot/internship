"""Quality report + manual review sheet.
   python -m evaluation.evaluate --questions data/outputs/questions.json --chunks data/processed/chunks.json"""
import argparse
import csv
import json
import os
import re
from collections import Counter

from blooms.verb_bank import validate_question
from generation.generator import _jaccard

STOP = set("""what which when where who whom whose does this that these those with from into about
between their there them they then than have been were will would could should your yours
explain describe define list state name discuss compare analyze main idea using based""".split())


def groundedness(question, chunk_text):
    """Proxy: share of the question's content words that appear in the source chunk."""
    words = [w for w in re.findall(r"[a-z]+", question.lower()) if len(w) > 3 and w not in STOP]
    if not words:
        return 0.0
    src = set(re.findall(r"[a-z]+", chunk_text.lower()))
    return sum(w in src for w in words) / len(words)


def evaluate(questions, chunks, dup_threshold=0.8, low_ground=0.5):
    by_id = {c["chunk_id"]: c for c in chunks}
    rows, dups = [], []
    for i, q in enumerate(questions):
        ok, reason = validate_question(q["question"], q["level"])
        chunk = by_id.get(q.get("chunk_id"), {})
        g = groundedness(q["question"], chunk.get("text", ""))
        rows.append({"id": q.get("id", f"Q{i + 1:03d}"), "level": q["level"], "question": q["question"],
                     "verb_ok": ok, "reason": reason, "grounded_proxy": round(g, 2),
                     "low_grounding": g < low_ground, "chunk": chunk})
        for j in range(i):
            if _jaccard(q["question"], questions[j]["question"]) >= dup_threshold:
                dups.append((questions[j].get("id"), q.get("id")))
    n = len(rows) or 1
    return {
        "total": len(rows),
        "verb_compliance_pct": round(100 * sum(r["verb_ok"] for r in rows) / n, 1),
        "avg_grounded_proxy": round(sum(r["grounded_proxy"] for r in rows) / n, 2),
        "low_grounding_ids": [r["id"] for r in rows if r["low_grounding"]],
        "duplicate_pairs": dups,
        "per_level": dict(sorted(Counter(r["level"] for r in rows).items())),
        "avg_words": round(sum(len(r["question"].split()) for r in rows) / n, 1),
    }, rows


def write_review_sheet(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "level", "question", "verb_ok", "grounded_proxy", "page",
                    "source_excerpt", "human_grounded(Y/N)", "human_notes"])
        for r in rows:
            c = r["chunk"]
            w.writerow([r["id"], r["level"], r["question"], r["verb_ok"], r["grounded_proxy"],
                        c.get("page_start", ""), c.get("text", "")[:300], "", ""])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/outputs/questions.json")
    ap.add_argument("--chunks", default="data/processed/chunks.json")
    ap.add_argument("--out-dir", default="data/outputs")
    a = ap.parse_args(argv)
    with open(a.questions, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with open(a.chunks, encoding="utf-8") as f:
        chunks = json.load(f)
    report, rows = evaluate(questions, chunks)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    write_review_sheet(rows, os.path.join(a.out_dir, "review_sheet.csv"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
