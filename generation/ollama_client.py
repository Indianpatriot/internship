"""Step 3a - thin Ollama REST client with retry/timeout, connection pooling, and keep-alive."""
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
    pass


class OllamaClient:
    def __init__(self, model=OLLAMA_MODEL, host=OLLAMA_HOST, temperature=TEMPERATURE,
                 timeout=LLM_TIMEOUT, retries=3, keep_alive=OLLAMA_KEEP_ALIVE):
        self.model, self.host = model, host.rstrip("/")
        self.temperature, self.timeout, self.retries = temperature, timeout, retries
        self.keep_alive = keep_alive
        self.last_metadata: dict = {}
        self.call_history: list[dict] = []
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
        return self.model in names or f"{self.model}:latest" in names

    def generate(self, prompt: str, temperature: Optional[float] = None, num_predict: int = 120,
                 format: Optional[str] = None, num_ctx: int = OLLAMA_NUM_CTX) -> str:
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

        last = None
        current_num_predict = num_predict
        for attempt in range(self.retries):
            try:
                payload["options"]["num_predict"] = current_num_predict
                r = self.session.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
                if r.status_code == 404:
                    raise OllamaError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
                r.raise_for_status()
                data = r.json()
                resp = data.get("response", "")
                meta = {
                    "load_duration": data.get("load_duration", 0),
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0),
                    "total_duration": data.get("total_duration", 0),
                    "done_reason": data.get("done_reason", ""),
                }
                if meta["eval_duration"] and meta["eval_count"]:
                    meta["tokens_per_second"] = meta["eval_count"] / (meta["eval_duration"] / 1e9)
                else:
                    meta["tokens_per_second"] = 0.0
                self.last_metadata = meta
                self.call_history.append(meta)

                if data.get("done_reason") == "length" and attempt < self.retries - 1:
                    current_num_predict = int(current_num_predict * 1.5)
                    continue
                return resp
            except OllamaError:
                raise
            except (requests.RequestException, ValueError) as e:
                last = e
                time.sleep(2 ** attempt)
        raise OllamaError(f"Ollama request failed after {self.retries} tries: {last}")
