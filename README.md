# RAG Question Generator - Ingestion & Generation

Bloom's-level (L1-L6) question generation from PDF textbooks using a local LLM (Ollama).

## Setup
```bash
pip install -r requirements.txt
ollama serve                 # separate terminal
ollama pull llama3.1:8b      # or set OLLAMA_MODEL
```

## Usage
```bash
python -m ingestion.pipeline --pdf data/raw/book.pdf --out data/processed/chunks.json
python -m generation.run --chunks data/processed/chunks.json --levels L1,L2,L4 --n 5 --seed 42
python -m generation.run --pdf data/raw/book.pdf --levels all --n 3 --marks 5
python -m generation.run --chunks data/processed/chunks.json --mock      # no LLM needed
python -m evaluation.evaluate                                            # report + review_sheet.csv
python -m blooms.learn_from_paper data/raw/sample_paper.pdf              # mine verbs from sample paper
pytest -q
```

## Modules
| Module | Purpose |
|---|---|
| `ingestion/pdf_extract.py` | PyMuPDF text + font info, header/footer removal |
| `ingestion/heading_detect.py` | TOC-first, font/regex heading detection, chapter > section hierarchy |
| `ingestion/chunker.py` | Section-wise chunks with overlap and page metadata |
| `blooms/verb_bank.py`, `verbs.json` | L1-L6 verbs, sampling, question validator |
| `generation/ollama_client.py` | Ollama REST client (retry, timeout) |
| `generation/prompts.py` | Prompt templates per Bloom level and marks |
| `generation/generator.py` | Generate, validate, retry, de-duplicate |
| `evaluation/evaluate.py` | Verb compliance, duplicates, groundedness proxy, manual review CSV |

## chunks.json schema
`chunk_id, source, chapter, section, page_start, page_end, text, word_count`

## Generator interface (for retrieval teammate)
`generate_question(chunk: dict, level: str, client, marks=None) -> dict`
