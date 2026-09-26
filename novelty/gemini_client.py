"""Gemini wrappers for embeddings and text generation.

Embeddings prefer ``text-embedding-004`` and fall back to ``gemini-embedding-001``,
which is the model this API actually serves. Vectors are cached on disk by
model and text so a submission is embedded once.

Generation prefers ``gemini-2.5-flash`` and falls forward to current flash
models when that one is not served. Calls retry on rate limits and 503s.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_CACHE_PATH = ROOT / ".cache" / "embeddings.json"

EMBED_MODELS = ("text-embedding-004", "gemini-embedding-001")
GENERATE_MODELS = (
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
)
EMBED_DIMENSIONS = 768

_unavailable_models: set[str] = set()
_active_embed_model: str | None = None
_active_generate_model: str | None = None


class GeminiError(RuntimeError):
    """Raised when a Gemini request fails after retries."""


def embed_texts(texts: list[str], api_key: str | None = None) -> np.ndarray:
    """Embed each text, reading and writing the local vector cache."""
    if not texts:
        return np.zeros((0, EMBED_DIMENSIONS), dtype=np.float64)
    key = api_key or _require_api_key()
    cache = _read_cache()
    resolved: list[np.ndarray | None] = [None] * len(texts)
    missing_positions: list[int] = []
    missing_texts: list[str] = []
    for index, text in enumerate(texts):
        cached = _cache_get(cache, text)
        if cached is None:
            missing_positions.append(index)
            missing_texts.append(text)
        else:
            resolved[index] = cached
    if missing_texts:
        fresh = _embed_uncached(missing_texts, key)
        model = active_embed_model()
        for position, text, vector in zip(missing_positions, missing_texts, fresh):
            resolved[position] = vector
            _cache_put(cache, model, text, vector)
        _write_cache(cache)
    return np.vstack(resolved)


def active_embed_model() -> str:
    return _active_embed_model or EMBED_MODELS[-1]


def active_generate_model() -> str:
    return _active_generate_model or GENERATE_MODELS[0]


def generate_text(prompt: str, api_key: str | None = None, max_output_tokens: int = 2048) -> str:
    """Return the text of one Gemini generation, with retries."""
    key = api_key or _require_api_key()
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            "responseMimeType": "application/json",
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    last_error = "no model attempted"
    global _active_generate_model
    for model in GENERATE_MODELS:
        if model in _unavailable_models:
            continue
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for attempt in range(6):
            request = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                _active_generate_model = model
                return _candidate_text(payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = f"{model} failed ({exc.code}): {detail}"
                if exc.code == 404:
                    _unavailable_models.add(model)
                    break
                if exc.code in (429, 503) and attempt < 5:
                    delay = _retry_delay(detail) if exc.code == 429 else 15.0
                    print(f"{model} returned {exc.code}; waiting {delay:.0f}s")
                    time.sleep(delay)
                    continue
                if exc.code == 400 and "thinkingConfig" in body.get("generationConfig", {}):
                    del body["generationConfig"]["thinkingConfig"]
                    continue
                raise GeminiError(last_error) from exc
    raise GeminiError(last_error)


def _embed_uncached(texts: list[str], api_key: str) -> list[np.ndarray]:
    global _active_embed_model
    last_error = "no embedding model attempted"
    for model in EMBED_MODELS:
        if model in _unavailable_models:
            continue
        try:
            vectors = _batch_embed(model, texts, api_key)
        except GeminiError as exc:
            last_error = str(exc)
            if "404" in last_error or "not found" in last_error.lower():
                _unavailable_models.add(model)
                continue
            raise
        _active_embed_model = model
        return vectors
    raise GeminiError(last_error)


def _batch_embed(model: str, texts: list[str], api_key: str) -> list[np.ndarray]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), 32):
        chunk = texts[start : start + 32]
        body = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": EMBED_DIMENSIONS,
                }
                for text in chunk
            ]
        }
        payload = _post_json(url, body, api_key)
        rows = payload.get("embeddings")
        if not isinstance(rows, list) or len(rows) != len(chunk):
            raise GeminiError(f"{model} returned an unexpected embedding payload")
        for row in rows:
            values = row.get("values") if isinstance(row, dict) else None
            if not values:
                raise GeminiError(f"{model} returned an empty embedding")
            vectors.append(np.asarray(values, dtype=np.float64))
    return vectors


def _post_json(url: str, body: dict, api_key: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GeminiError(f"Gemini request failed ({exc.code}): {detail}") from exc


def _candidate_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise GeminiError(f"Gemini returned no candidates: {payload}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not text.strip():
        raise GeminiError(f"Gemini returned empty text: {payload}")
    return text


def _retry_delay(detail: str) -> float:
    match = re.search(r"retry in ([0-9.]+)s", detail, flags=re.IGNORECASE)
    if match:
        return max(float(match.group(1)), 1.0) + 1.0
    return 15.0


def _require_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise GeminiError("Set GEMINI_API_KEY before calling Gemini.")
    return key


def _text_key(model: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model}:{EMBED_DIMENSIONS}:{digest}"


def _read_cache() -> dict:
    if not EMBEDDING_CACHE_PATH.exists():
        return {"vectors": {}}
    return json.loads(EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))


def _write_cache(cache: dict) -> None:
    EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def _cache_get(cache: dict, text: str) -> np.ndarray | None:
    vectors = cache.get("vectors", {})
    for model in EMBED_MODELS:
        row = vectors.get(_text_key(model, text))
        if row:
            return np.asarray(row, dtype=np.float64)
    return None


def _cache_put(cache: dict, model: str, text: str, vector: np.ndarray) -> None:
    cache.setdefault("vectors", {})[_text_key(model, text)] = [float(value) for value in vector]
