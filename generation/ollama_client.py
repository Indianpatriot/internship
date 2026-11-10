"""Thin Ollama REST client with adaptive timeout, connection pooling, keep-alive, and transport retries."""
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

from config import (
    LLM_TIMEOUT,
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    TEMPERATURE,
)


class OllamaError(RuntimeError):
    def __init__(self, message: str, reason_code: str = "UNKNOWN_EXCEPTION", http_status: Optional[int] = None):
        super().__init__(message)
        self.reason_code = reason_code
        self.http_status = http_status


class OllamaTimeoutError(OllamaError):
    def __init__(self, message: str):
        super().__init__(message, reason_code="OLLAMA_TIMEOUT")


class OllamaConnectionError(OllamaError):
    def __init__(self, message: str):
        super().__init__(message, reason_code="OLLAMA_CONN")


class OllamaEmptyResponseError(OllamaError):
    def __init__(self, message: str):
        super().__init__(message, reason_code="EMPTY_RESPONSE")


def get_num_predict_for_marks(marks: Optional[int], is_answer: bool = False) -> int:
    """Calculate token budget capped by marks to avoid VRAM bloat and slow inference."""
    m = marks if marks is not None and marks > 0 else 5
    if is_answer:
        if m <= 3:
            return 120
        elif m <= 5:
            return 220
        elif m <= 8:
            return 320
        else:
            return 450
    else:
        if m <= 3:
            return 70
        elif m <= 6:
            return 110
        else:
            return 160


class OllamaClient:
    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        temperature: float = TEMPERATURE,
        timeout: int = LLM_TIMEOUT,
        retries: int = 3,
        keep_alive: str = OLLAMA_KEEP_ALIVE,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.keep_alive = keep_alive
        self.last_metadata: dict = {}
        self.call_history: list[dict] = []
        self.consecutive_timeouts: int = 0
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def is_up(self) -> bool:
        try:
            return self.session.get(f"{self.host}/api/tags", timeout=5).ok
        except requests.RequestException:
            return False

    def has_model(self) -> bool:
        try:
            tags = self.session.get(f"{self.host}/api/tags", timeout=5).json().get("models", [])
        except (requests.RequestException, ValueError):
            return False
        names = {m.get("name", "") for m in tags}
        return self.model in names or f"{self.model}:latest" in names or any(m.startswith(self.model) for m in names)

    def preflight_check(self) -> tuple[bool, str, str]:
        """Preflight check: verifies reachability and model availability with warm-up.

        Returns:
            (is_ok, reason_code, message)
        """
        try:
            r = self.session.get(f"{self.host}/api/tags", timeout=5)
            if not r.ok:
                return False, "OLLAMA_CONN", f"Ollama HTTP {r.status_code} at {self.host}"
            models = [m.get("name", "") for m in r.json().get("models", [])]
            has_m = any(self.model in m or m.startswith(self.model) for m in models)
            if not has_m:
                return False, "MODEL_NOT_LOADED", f"Model '{self.model}' not found in Ollama ({len(models)} models available). Run: ollama pull {self.model}"
            # Quick warm-up ping
            warm = self.session.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "ping",
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {"num_predict": 1},
                },
                timeout=15,
            )
            if not warm.ok:
                return False, "MODEL_NOT_LOADED", f"Model warm-up failed with HTTP {warm.status_code}"
            return True, "OK", f"Ollama is reachable and model '{self.model}' is ready."
        except requests.exceptions.Timeout:
            return False, "OLLAMA_TIMEOUT", f"Connection to {self.host} timed out during preflight."
        except requests.RequestException as e:
            return False, "OLLAMA_CONN", f"Cannot connect to Ollama at {self.host}: {e}"

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: int = 120,
        format: Optional[str] = None,
        num_ctx: int = OLLAMA_NUM_CTX,
    ) -> str:
        """Call Ollama generate endpoint with adaptive timeout and transport retry isolation."""
        # Scale timeout dynamically with expected output tokens
        adaptive_timeout = max(self.timeout, int(25 + num_predict * 0.25))

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }
        if format:
            payload["format"] = format

        last_error = None
        current_num_predict = num_predict

        for attempt in range(self.retries):
            try:
                payload["options"]["num_predict"] = current_num_predict
                t0 = time.time()
                r = self.session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=adaptive_timeout,
                )
                latency = time.time() - t0

                if r.status_code == 404:
                    raise OllamaError(f"Model '{self.model}' not found. Run: ollama pull {self.model}", reason_code="MODEL_NOT_LOADED", http_status=404)
                r.raise_for_status()

                data = r.json()
                resp = data.get("response", "")
                if not resp and not data.get("done_reason"):
                    raise OllamaEmptyResponseError("Ollama returned an empty response")

                meta = {
                    "load_duration": data.get("load_duration", 0),
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0),
                    "total_duration": data.get("total_duration", 0),
                    "done_reason": data.get("done_reason", ""),
                    "latency": latency,
                }
                if meta["eval_duration"] and meta["eval_count"]:
                    meta["tokens_per_second"] = meta["eval_count"] / (meta["eval_duration"] / 1e9)
                else:
                    meta["tokens_per_second"] = 0.0

                self.last_metadata = meta
                self.call_history.append(meta)
                self.consecutive_timeouts = 0  # reset on success

                if data.get("done_reason") == "length" and attempt < self.retries - 1:
                    current_num_predict = int(current_num_predict * 1.4)
                    continue

                return resp
            except requests.exceptions.Timeout as e:
                self.consecutive_timeouts += 1
                last_error = OllamaTimeoutError(f"Ollama request timed out after {adaptive_timeout}s on attempt {attempt+1}/{self.retries}")
                time.sleep(min(1.5 ** attempt, 4.0))
            except requests.exceptions.ConnectionError as e:
                last_error = OllamaConnectionError(f"Ollama connection error at {self.host}: {e}")
                time.sleep(min(1.5 ** attempt, 4.0))
            except (OllamaError, requests.RequestException, ValueError) as e:
                last_error = e
                time.sleep(min(1.5 ** attempt, 4.0))

        if isinstance(last_error, OllamaError):
            raise last_error
        raise OllamaError(f"Ollama call failed after {self.retries} tries: {last_error}")
