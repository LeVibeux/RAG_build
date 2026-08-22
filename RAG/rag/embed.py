"""Ollama embeddings + generation (local HTTP)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    pass


def _post(host: str, path: str, payload: dict, timeout: float = 120.0) -> dict:
    url = host.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise OllamaError(f"Ollama unreachable at {url}: {exc}") from exc
    if not isinstance(body, dict):
        raise OllamaError(f"unexpected Ollama response at {url}")
    return body


def embed_texts(host: str, model: str, texts: list[str], batch: int = 16) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        body = _post(host, "/api/embed", {"model": model, "input": chunk}, timeout=180.0)
        embs = body.get("embeddings")
        if not embs or len(embs) != len(chunk):
            raise OllamaError(f"unexpected embed response keys={list(body)}")
        vectors.extend(embs)
    return vectors


def generate(
    host: str,
    model: str,
    prompt: str,
    timeout: float = 90.0,
    *,
    json_schema: dict | None = None,
    think: bool | None = None,
    num_predict: int | None = None,
) -> str:
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if json_schema is not None:
        payload["format"] = json_schema
    if think is not None:
        payload["think"] = think
    if num_predict is not None:
        payload["options"] = {"temperature": 0, "num_predict": num_predict}
    body = _post(
        host,
        "/api/generate",
        payload,
        timeout=timeout,
    )
    return (body.get("response") or "").strip()
