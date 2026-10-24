"""Central config. Override with environment variables."""
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))

# Chunking
MAX_CHUNK_WORDS = int(os.getenv("MAX_CHUNK_WORDS", "350"))
OVERLAP_WORDS = int(os.getenv("OVERLAP_WORDS", "40"))
MIN_CHUNK_WORDS = int(os.getenv("MIN_CHUNK_WORDS", "25"))

# Generation
MIN_CONTEXT_WORDS = 40      # chunks shorter than this are not used for questions
MAX_CONTEXT_WORDS = 450     # context truncated to this before prompting
