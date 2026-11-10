import pytest
from generation.diagnostics import ReasonCode, classify_exception_to_reason_code
from generation.mock_client import MockClient
from generation.ollama_client import (
    OllamaClient,
    OllamaTimeoutError,
    OllamaConnectionError,
    OllamaEmptyResponseError,
)
from generation.generator import (
    generate_paper_by_slots,
    regenerate_missing_slots,
    verify_paper_completeness,
)


def _sample_chunk(text="Docker container virtualization provides isolated environments. "):
    return {
        "text": text * 30,
        "word_count": 200,
        "co_scores": {"CO1": 0.8},
        "book_id": "b1",
        "chapter": "Chapter 1 Container Overview",
        "page_start": 10,
    }


def test_classify_exceptions():
    assert classify_exception_to_reason_code(OllamaTimeoutError("timed out"))[0] == ReasonCode.OLLAMA_TIMEOUT.value
    assert classify_exception_to_reason_code(OllamaConnectionError("conn refused"))[0] == ReasonCode.OLLAMA_CONN.value
    assert classify_exception_to_reason_code(OllamaEmptyResponseError("empty"))[0] == ReasonCode.EMPTY_RESPONSE.value
    assert classify_exception_to_reason_code(ValueError("JSON parse error"))[0] == ReasonCode.JSON_PARSE.value
    assert classify_exception_to_reason_code(RuntimeError("something else"))[0] == ReasonCode.UNKNOWN_EXCEPTION.value


def test_ollama_preflight_offline():
    client = OllamaClient(host="http://127.0.0.1:9999", model="nonexistent_model", timeout=1)
    ok, msg, err = client.preflight_check()
    assert not ok
    assert msg == ReasonCode.OLLAMA_CONN.value
    assert any(term in err.lower() for term in ["failed to connect", "refused", "not reachable", "error"])


def test_mock_client_failure_injection_timeout():
    mock = MockClient(fail_mode="timeout")
    with pytest.raises(OllamaTimeoutError):
        mock.generate("test prompt")


def test_mock_client_failure_injection_connection():
    mock = MockClient(fail_mode="connection")
    with pytest.raises(OllamaConnectionError):
        mock.generate("test prompt")


def test_mock_client_failure_injection_empty():
    mock = MockClient(fail_mode="empty")
    res = mock.generate("test prompt")
    assert res == ""


def test_mock_client_failure_injection_bad_json():
    mock = MockClient(fail_mode="bad_json")
    with pytest.raises(ValueError):
        mock.generate("test prompt")


def test_generation_maps_timeout_to_reason_code():
    slots = [
        {"id": "Q001", "marks": 5, "unit": 1, "co": "CO1", "level": "L2", "knowledge_dimension": "Conceptual"}
    ]
    chunks = [_sample_chunk()]
    mock = MockClient(fail_mode="timeout")
    res = generate_paper_by_slots(chunks, slots, client=mock, keep_placeholders=True)
    assert len(res["questions"]) == 1
    q = res["questions"][0]
    assert q["needs_review"] is True
    assert q.get("reason_code") == ReasonCode.OLLAMA_TIMEOUT.value or "OLLAMA_TIMEOUT" in q.get("review_reason", "")


def test_generation_maps_connection_error_to_reason_code():
    slots = [
        {"id": "Q001", "marks": 5, "unit": 1, "co": "CO1", "level": "L2", "knowledge_dimension": "Conceptual"}
    ]
    chunks = [_sample_chunk()]
    mock = MockClient(fail_mode="connection")
    res = generate_paper_by_slots(chunks, slots, client=mock, keep_placeholders=True)
    assert len(res["questions"]) == 1
    q = res["questions"][0]
    assert q["needs_review"] is True
    assert q.get("reason_code") == ReasonCode.OLLAMA_CONN.value or "OLLAMA_CONN" in q.get("review_reason", "")


def test_generation_maps_bad_json_to_reason_code():
    slots = [
        {"id": "Q001", "marks": 5, "unit": 1, "co": "CO1", "level": "L2", "knowledge_dimension": "Conceptual"}
    ]
    chunks = [_sample_chunk()]
    mock = MockClient(fail_mode="bad_json")
    res = generate_paper_by_slots(chunks, slots, client=mock, keep_placeholders=True)
    assert len(res["questions"]) == 1
    q = res["questions"][0]
    assert q["needs_review"] is True
    assert q.get("reason_code") == ReasonCode.JSON_PARSE.value or "JSON_PARSE" in q.get("review_reason", "")


def test_verify_paper_completeness_includes_reason_codes():
    slots = [
        {"id": "Q001", "marks": 5, "unit": 1, "co": "CO1", "level": "L2", "knowledge_dimension": "Conceptual"}
    ]
    chunks = [_sample_chunk()]
    mock = MockClient(fail_mode="timeout")
    res = generate_paper_by_slots(chunks, slots, client=mock, keep_placeholders=True)
    is_complete, issues = verify_paper_completeness(slots, res["questions"])
    assert not is_complete
    assert len(issues) >= 1
    assert any("OLLAMA_TIMEOUT" in iss for iss in issues)
