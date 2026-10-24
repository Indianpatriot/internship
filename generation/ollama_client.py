"""Step 3a - thin Ollama REST client with retry/timeout."""
import time

import requests

from config import LLM_TIMEOUT, OLLAMA_HOST, OLLAMA_MODEL, TEMPERATURE


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model=OLLAMA_MODEL, host=OLLAMA_HOST, temperature=TEMPERATURE,
                 timeout=LLM_TIMEOUT, retries=3):
        self.model, self.host = model, host.rstrip("/")
        self.temperature, self.timeout, self.retries = temperature, timeout, retries

    def is_up(self) -> bool:
        try:
            return requests.get(f"{self.host}/api/tags", timeout=5).ok
        except requests.RequestException:
            return False

    def has_model(self) -> bool:
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=5).json().get("models", [])
        except (requests.RequestException, ValueError):
            return False
        names = {m.get("name", "") for m in tags}
        return self.model in names or f"{self.model}:latest" in names

    def generate(self, prompt, temperature=None, num_predict=120) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature if temperature is None else temperature,
                        "num_predict": num_predict},
        }
        last = None
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
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
