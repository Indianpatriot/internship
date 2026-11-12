"""Thin Ollama REST client with adaptive streaming timeout, connection pooling, keep-alive, and transport retries."""
import json
import logging
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

import config
from config import (
    LLM_TIMEOUT,
    OLLAMA_HOST,
    OLLAMA_INACTIVITY_TIMEOUT,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    TEMPERATURE,
)

logger = logging.getLogger("exam_generator")


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
        self.retries = min(retries, 2)
        self.keep_alive = keep_alive
        self.last_metadata: dict = {}
        self.call_history: list[dict] = []
        self.consecutive_timeouts: int = 0
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.inactivity_timeout: float = getattr(config, "OLLAMA_INACTIVITY_TIMEOUT", 60.0)
        self.measured_tok_s: float = 5.0

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

    def detailed_preflight_check(self, slot_count: int = 10) -> dict:
        """Perform comprehensive preflight: reachability, model residency, GPU offload %,
        host RAM/swap via psutil, warm-up token velocity, and estimated total runtime.
        """
        import psutil
        status = {
            "ok": True,
            "reason_code": "OK",
            "message": "",
            "gpu_offload_pct": None,
            "free_ram_mb": None,
            "swap_percent": None,
            "measured_tok_s": self.measured_tok_s,
            "est_minutes": None,
            "warnings": [],
            "suggest_small_model": False,
        }

        # 1. Host memory and swap check via psutil
        try:
            ram = psutil.virtual_memory()
            swap = psutil.swap_memory()
            free_ram_mb = int(ram.available / (1024 * 1024))
            swap_pct = float(swap.percent)
            status["free_ram_mb"] = free_ram_mb
            status["swap_percent"] = swap_pct

            if free_ram_mb < 800:
                status["warnings"].append(
                    f"Low host RAM ({free_ram_mb} MB available). LLM inference may trigger disk thrashing."
                )
            if swap_pct > 80.0:
                status["warnings"].append(
                    f"High swap usage ({swap_pct:.1f}%). High memory pressure detected."
                )
        except Exception:
            pass

        # 2. Ollama tags & reachability
        try:
            r = self.session.get(f"{self.host}/api/tags", timeout=5)
            if not r.ok:
                status["ok"] = False
                status["reason_code"] = "OLLAMA_CONN"
                status["message"] = f"Cannot reach Ollama at {self.host} (HTTP {r.status_code})"
                return status
            models = [m.get("name", "") for m in r.json().get("models", [])]
            if not any(self.model in m or m.startswith(self.model) for m in models):
                status["ok"] = False
                status["reason_code"] = "MODEL_NOT_LOADED"
                status["message"] = f"Model '{self.model}' not found in Ollama. Run: ollama pull {self.model}"
                return status
        except Exception as e:
            status["ok"] = False
            status["reason_code"] = "OLLAMA_CONN"
            status["message"] = f"Cannot connect to Ollama at {self.host}: {e}"
            return status

        # 3. Model residency & GPU offload check via /api/ps
        try:
            ps_res = self.session.get(f"{self.host}/api/ps", timeout=5)
            if ps_res.ok:
                running_models = ps_res.json().get("models", [])
                for rm in running_models:
                    if self.model in rm.get("name", "") or rm.get("name", "").startswith(self.model):
                        total_sz = rm.get("size", 0)
                        vram_sz = rm.get("size_vram", 0)
                        if total_sz > 0:
                            pct = round((vram_sz / total_sz) * 100.0, 1)
                            status["gpu_offload_pct"] = pct
                            if pct < 100.0:
                                status["warnings"].append(
                                    f"Model '{self.model}' is only {pct}% offloaded to GPU. Remaining layers execute on CPU at reduced tok/s."
                                )
                        break
        except Exception:
            pass

        # 4. Warm-up ping to measure actual tok/s
        try:
            warm = self.session.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "Hello",
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {"num_predict": 10},
                },
                timeout=20,
            )
            if warm.ok:
                w_data = warm.json()
                e_cnt = w_data.get("eval_count", 0)
                e_dur = w_data.get("eval_duration", 0)
                if e_dur > 0 and e_cnt > 0:
                    measured = round(e_cnt / (e_dur / 1e9), 2)
                    self.measured_tok_s = measured
                    status["measured_tok_s"] = measured
        except Exception:
            pass

        # 5. Runtime estimation
        calls_per_slot = 2
        tokens_per_call = 180
        effective_speed = max(self.measured_tok_s, 0.5)
        est_total_secs = (slot_count * calls_per_slot * tokens_per_call) / effective_speed
        est_min = round(est_total_secs / 60.0, 1)
        status["est_minutes"] = est_min

        if est_min > 90.0:
            status["warnings"].append(
                f"Estimated paper generation time is {est_min:.0f} minutes! This exceeds 90 minutes. "
                "We strongly recommend switching to a 3B model (e.g. llama3.2:3b or qwen2.5:3b-instruct) that fully fits 4 GB VRAM."
            )
            status["suggest_small_model"] = True

        status["message"] = f"Ollama ready ({self.model} at ~{self.measured_tok_s:.1f} tok/s, estimated total: {est_min:.0f}m)"
        return status

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: int = 120,
        format: Optional[str] = None,
        num_ctx: int = OLLAMA_NUM_CTX,
    ) -> str:
        """Call Ollama generate endpoint with streaming, inactivity timeout, dynamic hard cap, and tok/s tracking."""
        # Calculate hard cap scaled by num_predict / measured tok/s so slow generations are not killed
        effective_speed = max(self.measured_tok_s, 0.5)
        expected_eval_secs = num_predict / effective_speed
        hard_cap = max(float(self.timeout), expected_eval_secs * 2.5 + 45.0)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
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
            payload["options"]["num_predict"] = current_num_predict
            start_time = time.time()
            last_chunk_time = start_time
            text_chunks: list[str] = []
            final_data: dict = {}
            r = None

            try:
                # Connection timeout: 10s, read/inactivity timeout: inactivity_timeout
                try:
                    r = self.session.post(
                        f"{self.host}/api/generate",
                        json=payload,
                        stream=True,
                        timeout=(10.0, self.inactivity_timeout),
                    )
                except TypeError:
                    # In case mock session does not accept stream or tuple timeout
                    r = self.session.post(
                        f"{self.host}/api/generate",
                        json=payload,
                    )

                if r.status_code == 404:
                    raise OllamaError(
                        f"Model '{self.model}' not found. Run: ollama pull {self.model}",
                        reason_code="MODEL_NOT_LOADED",
                        http_status=404,
                    )
                r.raise_for_status()

                if hasattr(r, "iter_lines"):
                    for raw_line in r.iter_lines(decode_unicode=True):
                        now = time.time()
                        if (now - start_time) > hard_cap:
                            raise OllamaTimeoutError(
                                f"Generation exceeded hard cap of {hard_cap:.1f}s (budget: {current_num_predict} tokens)"
                            )
                        if (now - last_chunk_time) > self.inactivity_timeout:
                            raise OllamaTimeoutError(
                                f"No tokens received from Ollama for {self.inactivity_timeout:.1f}s"
                            )
                        if not raw_line:
                            continue
                        last_chunk_time = now
                        try:
                            chunk = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue

                        resp_piece = chunk.get("response", "")
                        if resp_piece:
                            text_chunks.append(resp_piece)

                        if chunk.get("done"):
                            final_data = chunk
                            break
                elif hasattr(r, "json"):
                    data = r.json()
                    resp_piece = data.get("response", "")
                    if resp_piece:
                        text_chunks.append(resp_piece)
                    final_data = data

            except requests.exceptions.ReadTimeout:
                self.consecutive_timeouts += 1
                last_error = OllamaTimeoutError(
                    f"No tokens received from Ollama for {self.inactivity_timeout:.1f}s (read timeout on attempt {attempt+1}/{self.retries})"
                )
                time.sleep(min(1.5 ** attempt, 4.0))
                continue
            except requests.exceptions.Timeout as e:
                self.consecutive_timeouts += 1
                last_error = OllamaTimeoutError(
                    f"Connection to Ollama timed out on attempt {attempt+1}/{self.retries}: {e}"
                )
                time.sleep(min(1.5 ** attempt, 4.0))
                continue
            except requests.exceptions.ConnectionError as e:
                last_error = OllamaConnectionError(f"Ollama connection error at {self.host}: {e}")
                time.sleep(min(1.5 ** attempt, 4.0))
                continue
            except (OllamaError, requests.RequestException, ValueError) as e:
                last_error = e
                time.sleep(min(1.5 ** attempt, 4.0))
                continue
            finally:
                # Ensure server request stream and socket are cleanly closed / aborted
                if r is not None:
                    try:
                        r.close()
                    except Exception:
                        pass

            latency = time.time() - start_time
            resp = "".join(text_chunks)
            if not resp and not final_data.get("done_reason"):
                last_error = OllamaEmptyResponseError("Ollama returned an empty response")
                time.sleep(min(1.5 ** attempt, 4.0))
                continue

            eval_count = final_data.get("eval_count", 0)
            eval_duration = final_data.get("eval_duration", 0)
            tok_s = 0.0
            if eval_duration > 0 and eval_count > 0:
                tok_s = eval_count / (eval_duration / 1e9)
                self.measured_tok_s = round(0.7 * tok_s + 0.3 * self.measured_tok_s, 2)

            logger.info(
                f"[OLLAMA] call complete: model={self.model} tokens={eval_count} "
                f"duration={eval_duration/1e9:.2f}s speed={tok_s:.2f} tok/s "
                f"(rolling: {self.measured_tok_s:.2f} tok/s) latency={latency:.2f}s"
            )

            meta = {
                "load_duration": final_data.get("load_duration", 0),
                "prompt_eval_count": final_data.get("prompt_eval_count", 0),
                "prompt_eval_duration": final_data.get("prompt_eval_duration", 0),
                "eval_count": eval_count,
                "eval_duration": eval_duration,
                "total_duration": final_data.get("total_duration", 0),
                "done_reason": final_data.get("done_reason", ""),
                "latency": latency,
                "tokens_per_second": tok_s,
                "rolling_tokens_per_second": self.measured_tok_s,
            }
            self.last_metadata = meta
            self.call_history.append(meta)
            self.consecutive_timeouts = 0

            if final_data.get("done_reason") == "length" and attempt < self.retries - 1:
                current_num_predict = int(current_num_predict * 1.4)
                continue

            return resp

        if isinstance(last_error, OllamaError):
            raise last_error
        raise OllamaError(f"Ollama call failed after {self.retries} tries: {last_error}")
