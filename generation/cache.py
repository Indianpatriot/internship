"""Disk and in-memory cache for LLM generation calls."""
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from config import ENABLE_GENERATION_CACHE

CACHE_DIR = Path(".cache/generation")
_MEM_CACHE: dict[str, str] = {}


def _make_key(prompt: str, model: str, temperature: Optional[float] = None, seed: Optional[Any] = None) -> str:
    payload = f"m:{model}|t:{temperature}|s:{seed}|p:{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached_response(
    prompt: str,
    model: str,
    temperature: Optional[float] = None,
    seed: Optional[Any] = None,
    force_regenerate: bool = False,
) -> Optional[str]:
    """Retrieve cached LLM response if available and caching is enabled."""
    if not ENABLE_GENERATION_CACHE or force_regenerate:
        return None

    key = _make_key(prompt, model, temperature, seed)
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]

    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            resp = data.get("response", "")
            _MEM_CACHE[key] = resp
            return resp
        except Exception:
            return None

    return None


def store_cached_response(
    prompt: str,
    model: str,
    response: str,
    temperature: Optional[float] = None,
    seed: Optional[Any] = None,
) -> None:
    """Store LLM response in in-memory and disk cache."""
    if not ENABLE_GENERATION_CACHE or not response:
        return

    key = _make_key(prompt, model, temperature, seed)
    _MEM_CACHE[key] = response

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "response": response,
            }, f, ensure_ascii=False)
    except Exception:
        pass


def clear_generation_cache() -> None:
    """Clear memory and disk generation cache."""
    global _MEM_CACHE
    _MEM_CACHE.clear()
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.json"):
            try:
                p.unlink()
            except OSError:
                pass

