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
ANSWER_MIN_SIMILARITY = float(os.getenv("ANSWER_MIN_SIMILARITY", "0.35"))
QUESTION_NUM_PREDICT = int(os.getenv("QUESTION_NUM_PREDICT", "120"))
GENERATE_QA_TOGETHER = os.getenv("GENERATE_QA_TOGETHER", "false").lower() in ("1", "true", "yes")

# Quality & Validation Thresholds (Problems 2, 3, 5)
MIN_PAGE_CONTENT_WORDS = int(os.getenv("MIN_PAGE_CONTENT_WORDS", "150"))  # exclude divider/thin pages
QUESTION_SIMILARITY_DUPLICATE_THRESHOLD = float(os.getenv("QUESTION_SIMILARITY_DUPLICATE_THRESHOLD", "0.85"))
CO_SIMILARITY_THRESHOLD = float(os.getenv("CO_SIMILARITY_THRESHOLD", "0.40"))
CO_MAPPING_WARNING_THRESHOLD = float(os.getenv("CO_MAPPING_WARNING_THRESHOLD", "0.25"))
ANSWER_GROUNDING_SENTENCE_SIMILARITY = float(os.getenv("ANSWER_GROUNDING_SENTENCE_SIMILARITY", "0.45"))
ANSWER_MAX_UNGROUNDED_SENTENCE_RATIO = float(os.getenv("ANSWER_MAX_UNGROUNDED_SENTENCE_RATIO", "0.35"))

# Course Outcomes constraints
MIN_COS = 4
MAX_COS = 6

# Page filtering
FILTER_USE_LLM = os.getenv("FILTER_USE_LLM", "false").lower() in ("1", "true", "yes")

# Performance & Parallel Generation
GENERATION_WORKERS = int(os.getenv("GENERATION_WORKERS", "3"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_NUM_PARALLEL = int(os.getenv("OLLAMA_NUM_PARALLEL", "4"))
ENABLE_GENERATION_CACHE = os.getenv("ENABLE_GENERATION_CACHE", "true").lower() in ("1", "true", "yes")

# Embeddings & Retrieval Performance
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# Topic to Chapter Mapping Confidence Thresholds
MAPPING_MIN_CONFIDENCE = float(os.getenv("MAPPING_MIN_CONFIDENCE", "0.45"))
MAPPING_SUGGEST_CONFIDENCE = float(os.getenv("MAPPING_SUGGEST_CONFIDENCE", "0.25"))

# Marking Scheme configuration
ALLOW_HALF_MARKS: bool = os.getenv("ALLOW_HALF_MARKS", "false").lower() in ("1", "true", "yes")

# Figure Extraction & Retrieval Configuration
EXTRACT_FIGURES: bool = os.getenv("EXTRACT_FIGURES", "true").lower() in ("1", "true", "yes")
GEN_WORKERS: int = int(os.getenv("GEN_WORKERS", str(GENERATION_WORKERS)))
FIGURE_MIN_SCORE: float = float(os.getenv("FIGURE_MIN_SCORE", "0.45"))
FIGURE_MAX_PER_QUESTION: int = int(os.getenv("FIGURE_MAX_PER_QUESTION", "2"))
FIGURE_MAX_WIDTH_PX: int = int(os.getenv("FIGURE_MAX_WIDTH_PX", "1400"))
FIGURE_DPI: int = int(os.getenv("FIGURE_DPI", "150"))




