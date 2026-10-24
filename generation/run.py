"""CLI.  python -m generation.run --pdf book.pdf --levels L1,L2,L4 --n 5
        python -m generation.run --chunks data/processed/chunks.json --levels all --n 3 --mock"""
import argparse
import json
import os
import sys

from blooms.verb_bank import LEVELS, normalize_level
from config import OLLAMA_MODEL, TEMPERATURE
from generation.generator import generate_paper
from generation.mock_client import MockClient
from generation.ollama_client import OllamaClient
from ingestion.pipeline import ingest, save_chunks


def parse_levels(s):
    if s.strip().lower() == "all":
        return list(LEVELS)
    return [normalize_level(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bloom's-level question generator")
    ap.add_argument("--pdf", nargs="+", help="PDF(s) to ingest")
    ap.add_argument("--chunks", help="existing chunks.json (skips ingestion)")
    ap.add_argument("--levels", default="all", help="e.g. L1,L2,L4 or all")
    ap.add_argument("--n", type=int, default=3, help="questions per level")
    ap.add_argument("--marks", type=int, choices=[2, 5, 10])
    ap.add_argument("--model", default=OLLAMA_MODEL)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="data/outputs/questions.json")
    ap.add_argument("--mock", action="store_true", help="use offline mock LLM")
    a = ap.parse_args(argv)

    if a.chunks:
        with open(a.chunks, encoding="utf-8") as f:
            chunks = json.load(f)
    elif a.pdf:
        chunks = [c for p in a.pdf for c in ingest(p)]
        save_chunks(chunks, "data/processed/chunks.json")
    else:
        ap.error("give --pdf or --chunks")

    if a.mock:
        client = MockClient()
    else:
        client = OllamaClient(model=a.model, temperature=a.temperature)
        if not client.is_up():
            sys.exit("Ollama is not running. Start it with: ollama serve")
        if not client.has_model():
            sys.exit(f"Model not found. Run: ollama pull {a.model}")

    counts = {lvl: a.n for lvl in parse_levels(a.levels)}
    result = generate_paper(chunks, counts, client, seed=a.seed, marks=a.marks,
                            on_progress=lambda r: print(f"[{r['id']}] {r['level']}: {r['question']}"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n{len(result['questions'])} questions, {len(result['rejected'])} rejected -> {a.out}")


if __name__ == "__main__":
    main()
