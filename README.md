# RAG Question Generator - Ingestion, Retrieval & Generation

Outcome-aware and Bloom's-level (L1-L5) exam question paper generation from PDF textbooks using local embeddings (`BAAI/bge-small-en-v1.5`) and a local LLM (Ollama or offline mock client). Zero API cost.

## Setup & Run Instructions

Python 3.12 is recommended.

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
python -m pytest -v

# 5. Launch the Streamlit web application
streamlit run app.py
```

### Optional: Ollama Setup (for live LLM generation)
If you want to use a live local LLM instead of the built-in offline mock client:
```bash
ollama serve                 # run in a separate terminal
ollama pull llama3.1:8b      # or set OLLAMA_MODEL in config.py
```

### LLM Generation & Performance Configuration
To maximize generation reliability and speed, question slots are generated with strict concurrency controls, stream inactivity timeouts, per-slot call budgets, and incremental progress checkpointing.

#### Recommended Ollama Environment Variables (for 4 GB VRAM / Laptop GPUs like RTX 3050):
Because `llama3.1:8b` (~4.9 GB) exceeds 4 GB VRAM and must split layers across GPU and CPU (often ~3-6 tok/s), configure Ollama with memory-efficient flags before launching `ollama serve`:

**Linux / macOS (Bash):**
```bash
export OLLAMA_NUM_PARALLEL=1         # CRITICAL on 4 GB VRAM: prevents KV-cache duplication that forces model layers to CPU
export OLLAMA_MAX_LOADED_MODELS=1    # Prevents simultaneous model memory footprint
export OLLAMA_FLASH_ATTENTION=1      # Enables Flash Attention to speed up prompt evaluation and save VRAM
export OLLAMA_KV_CACHE_TYPE=q8_0     # Quantizes KV cache to 8-bit, fitting context into GPU memory
export OLLAMA_KEEP_ALIVE=30m         # Retains model residency in VRAM across requests
ollama serve
```

**Windows (PowerShell):**
```powershell
$env:OLLAMA_NUM_PARALLEL="1"
$env:OLLAMA_MAX_LOADED_MODELS="1"
$env:OLLAMA_FLASH_ATTENTION="1"
$env:OLLAMA_KV_CACHE_TYPE="q8_0"
$env:OLLAMA_KEEP_ALIVE="30m"
ollama serve
```

> **Note on Concurrency**: Setting `OLLAMA_NUM_PARALLEL > 1` on 4 GB VRAM duplicates the KV cache across parallel contexts, pushing layers from GPU to system RAM and causing server queueing and timeouts. The generation pipeline clamps Ollama workers to `1` by default (configurable via `ALLOW_PARALLEL_OLLAMA`). For fast generation on 4 GB VRAM, smaller 3B models (`llama3.2:3b` or `qwen2.5:3b-instruct`) offload 100% of layers into GPU memory at ~15-25 tok/s.

Key performance configurations available via environment variables or `config.py`:
- `ALLOW_PARALLEL_OLLAMA`: Set `True` only when Ollama is provisioned on high-VRAM hardware (default `False`, clamping to 1 worker).
- `OLLAMA_INACTIVITY_TIMEOUT`: Max seconds without streamed tokens before timing out (default `60.0`s).
- `MAX_SLOT_CALL_BUDGET`: Per-slot LLM invocation ceiling (default `6`).
- `GENERATION_WORKERS`: Worker threads for MockClient or parallel mode (default `3`).
- `OLLAMA_KEEP_ALIVE`: Model residency duration to avoid reloads (default `"30m"`).
- `OLLAMA_NUM_CTX`: Static context window matching Ollama server options (default `2048`).
- `QUESTION_NUM_PREDICT`: Right-sized token cap for single-sentence questions (default `120`).
- `ENABLE_GENERATION_CACHE`: In-memory & disk caching under `.cache/generation/` keyed by prompt hash (default `true`).
- `EMBEDDING_BATCH_SIZE`: Batch size for SentenceTransformer embedding operations (default `32`).


## CLI Usage
```bash
# Ingest PDF into chunks
python -m ingestion.pipeline --pdf data/samples/sample_book.pdf --out data/processed/chunks.json

# Generate questions using CLI
python -m generation.run --chunks data/processed/chunks.json --levels L1,L2,L4 --n 5 --seed 42
python -m generation.run --pdf data/samples/sample_book.pdf --mock

# Run evaluation & verb mining
python -m evaluation.evaluate
python -m blooms.learn_from_paper data/samples/sample_book.pdf
```

## Modules
| Module | Purpose |
|---|---|
| `app.py` | Streamlit web application with dynamic CO/LO management, slot blueprint, and CSV download |
| `retrieval/co_mapper.py` | SentenceTransformer (`BAAI/bge-small-en-v1.5`) semantic scoring of chunks against Course Outcomes |
| `ingestion/pdf_extract.py` | PyMuPDF text + font info, header/footer removal |
| `ingestion/heading_detect.py` | TOC-first, font/regex heading detection, chapter > section hierarchy |
| `ingestion/chunker.py` | Section-wise chunks with overlap and page metadata |
| `blooms/verb_bank.py`, `verbs.json` | L1-L6 verbs, sampling, question validator |
| `generation/ollama_client.py` | Ollama REST client (retry, timeout) |
| `generation/mock_client.py` | Offline deterministic mock client for testing and demos |
| `generation/prompts.py` | Prompt templates per Bloom level and marks |
| `generation/generator.py` | Slot-based paper generation (`generate_paper_by_slots`), validation, retry, de-duplication |
| `evaluation/evaluate.py` | Verb compliance, duplicates, groundedness proxy, manual review CSV |

## chunks.json schema
`chunk_id, source, chapter, section, page_start, page_end, text, word_count, co_scores`
