"""Diagnostic reason codes, failure mapping, and structured logging for question paper generation."""
from enum import Enum
import logging
from typing import Any, Optional

logger = logging.getLogger("exam_generator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [slot:%(slot_id)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ReasonCode(str, Enum):
    OLLAMA_TIMEOUT = "OLLAMA_TIMEOUT"
    OLLAMA_CONN = "OLLAMA_CONN"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    JSON_PARSE = "JSON_PARSE"
    BLOOM_VERB_FAIL = "BLOOM_VERB_FAIL"
    GROUNDING_FAIL = "GROUNDING_FAIL"
    DUPLICATE = "DUPLICATE"
    NO_CHUNKS = "NO_CHUNKS"
    STEM_LINT = "STEM_LINT"
    ANSWER_RUBRIC = "ANSWER_RUBRIC"
    UNGROUNDED_TOKEN = "UNGROUNDED_TOKEN"
    CO_MISMATCH = "CO_MISMATCH"
    OUTDATED = "OUTDATED"
    SCHEME_VALIDATION = "SCHEME_VALIDATION"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    UNKNOWN_EXCEPTION = "UNKNOWN_EXCEPTION"


REASON_HINTS: dict[ReasonCode, str] = {
    ReasonCode.OLLAMA_TIMEOUT: "Ollama timed out. Lower worker count to 1-2, increase timeout, or reduce context length.",
    ReasonCode.OLLAMA_CONN: "Cannot connect to Ollama. Ensure 'ollama serve' is running on the configured host.",
    ReasonCode.MODEL_NOT_LOADED: "Model is not loaded or available. Run 'ollama pull <model>' or run a preflight check.",
    ReasonCode.EMPTY_RESPONSE: "LLM returned an empty response. Verify model temperature and prompt parameters.",
    ReasonCode.JSON_PARSE: "Model failed to output parseable structured JSON. Try regenerating with a fresh seed.",
    ReasonCode.BLOOM_VERB_FAIL: "Question verb does not match the assigned Bloom's Taxonomy level and Knowledge Dimension.",
    ReasonCode.GROUNDING_FAIL: "Model answer failed factual grounding check against textbook text. Provide richer chapter context.",
    ReasonCode.DUPLICATE: "Question duplicates an existing question in the paper. Select more chapters to expand topic variety.",
    ReasonCode.NO_CHUNKS: "No text chunks found for this Course Outcome and Unit in selected chapters. Select more chapters.",
    ReasonCode.STEM_LINT: "Question stem failed academic quality linting (e.g. premise in stem, deictic words, stray syntax).",
    ReasonCode.ANSWER_RUBRIC: "Model answer failed rubric criteria (points do not sum to slot marks or criteria missing).",
    ReasonCode.UNGROUNDED_TOKEN: "Answer references specific commands, tools, or flags not present in source chunks.",
    ReasonCode.CO_MISMATCH: "Question content does not align strongly with the assigned Course Outcome description.",
    ReasonCode.OUTDATED: "Question references deprecated or legacy technologies from the currency watchlist.",
    ReasonCode.SCHEME_VALIDATION: "Marking scheme criteria failed sum validation (criteria marks must sum to question marks).",
    ReasonCode.BUDGET_EXCEEDED: "Slot exceeded maximum allowed LLM call budget. Check chunk content quality or raise budget.",
    ReasonCode.UNKNOWN_EXCEPTION: "An unexpected exception occurred during slot processing. Review server logs.",
}


def log_slot_attempt(
    slot_id: str,
    attempt: int,
    status: str,
    latency: float = 0.0,
    tokens: int = 0,
    exception_type: Optional[str] = None,
    exception_msg: Optional[str] = None,
    http_status: Optional[int] = None,
    failed_validator: Optional[str] = None,
    reason_code: Optional[str] = None,
) -> None:
    """Log structured diagnostics per slot, per attempt."""
    extra = {"slot_id": slot_id}
    msg_parts = [
        f"attempt={attempt}",
        f"status={status}",
        f"latency={latency:.2f}s",
        f"tokens={tokens}",
    ]
    if http_status:
        msg_parts.append(f"http_status={http_status}")
    if failed_validator:
        msg_parts.append(f"validator={failed_validator}")
    if reason_code:
        msg_parts.append(f"reason_code={reason_code}")
    if exception_type or exception_msg:
        msg_parts.append(f"error={exception_type}: {exception_msg}")

    log_line = " | ".join(msg_parts)
    if status in ("ok", "success"):
        logger.info(log_line, extra=extra)
    elif status == "retry":
        logger.warning(log_line, extra=extra)
    else:
        logger.error(log_line, extra=extra)


def classify_exception_to_reason_code(exc: Exception) -> tuple[ReasonCode, str, str]:
    """Map any exception to a standard ReasonCode, diagnostic detail, and actionable hint."""
    exc_type = type(exc).__name__
    exc_str = str(exc)
    exc_lower = exc_str.lower()

    if "timeout" in exc_lower or "timed out" in exc_lower:
        code = ReasonCode.OLLAMA_TIMEOUT
        detail = f"Request timed out: {exc_str}"
    elif "connection" in exc_lower or "failed to establish" in exc_lower or "refused" in exc_lower:
        code = ReasonCode.OLLAMA_CONN
        detail = f"Connection error: {exc_str}"
    elif "not found" in exc_lower or "model" in exc_lower and ("pull" in exc_lower or "load" in exc_lower):
        code = ReasonCode.MODEL_NOT_LOADED
        detail = f"Model unavailable: {exc_str}"
    elif "json" in exc_lower or "expecting value" in exc_lower:
        code = ReasonCode.JSON_PARSE
        detail = f"JSON parsing failed: {exc_str}"
    elif "empty" in exc_lower or "no response" in exc_lower:
        code = ReasonCode.EMPTY_RESPONSE
        detail = "Model returned empty text"
    else:
        code = ReasonCode.UNKNOWN_EXCEPTION
        detail = f"{exc_type}: {exc_str}"

    hint = REASON_HINTS.get(code, REASON_HINTS[ReasonCode.UNKNOWN_EXCEPTION])
    return code, detail, hint

