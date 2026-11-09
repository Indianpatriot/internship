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
        for attempt in range(self.retries):
            try:
                r = self.session.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
                if r.status_code == 404:
                    raise OllamaError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
                r.raise_for_status()
                return r.json().get("response", "")
            except OllamaError:
                raise
            except (requests.RequestException, ValueError) as e:
                last = e
                time.sleep(2 ** attempt)
        raise OllamaError(f"Ollama request failed after {self.retries} tries: {last}")
