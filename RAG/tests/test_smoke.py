from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest


RAG_ROOT = Path(__file__).resolve().parents[1]
WORKDIR = RAG_ROOT.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
OLLAMA_HOST = "http://127.0.0.1:11434"


def _embedding_model_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return False
    names = {str(model.get("name", "")).split(":", 1)[0] for model in payload.get("models", [])}
    return "embeddinggemma" in names


@pytest.mark.ollama
def test_ingest_and_search_fixture_with_local_ollama(tmp_path):
    if not _embedding_model_available():
        pytest.skip("local Ollama embeddinggemma model is unavailable")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "embed: embeddinggemma\n"
        "rewrite: qwen3.5:0.8b\n"
        f"ollama_host: {OLLAMA_HOST}\n"
        "rewrite_enabled: true\n"
        "hyde: true\n"
        "store_dir: indexes\n"
        f"docs_path: {FIXTURES}\n"
        "chunk: parent_child\n"
        "child_tokens: 256\n"
        "parent_tokens: 1024\n"
        "overlap_tokens: 32\n",
        encoding="utf-8",
    )

    ingest = subprocess.run(
        [sys.executable, str(RAG_ROOT / "ingest.py"), "--config", str(config_path)],
        cwd=WORKDIR,
        text=True,
        capture_output=True,
        check=False,
        timeout=240,
    )
    assert ingest.returncode == 0, ingest.stdout + ingest.stderr
    ingest_payload = json.loads(ingest.stdout)
    assert ingest_payload["ok"] is True
    assert ingest_payload["files"] >= 1

    search = subprocess.run(
        [
            sys.executable,
            str(RAG_ROOT / "search.py"),
            "--config",
            str(config_path),
            "--no-rewrite",
            "clause d'agrément",
        ],
        cwd=WORKDIR,
        text=True,
        capture_output=True,
        check=False,
        timeout=240,
    )
    assert search.returncode == 0, search.stdout + search.stderr
    search_payload = json.loads(search.stdout)
    assert search_payload["ok"] is True
    assert search_payload["hits"]
    assert search_payload["hits"][0]["path"]
    assert search_payload["hits"][0]["text"]
