from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


RAG_ROOT = Path(__file__).resolve().parents[1]
WORKDIR = RAG_ROOT.parent


def _json_stdout(proc: subprocess.CompletedProcess[str]) -> dict:
    assert proc.stdout.count("\n") == 1
    return json.loads(proc.stdout)


def test_search_missing_query_is_one_json_object():
    proc = subprocess.run(
        [sys.executable, str(RAG_ROOT / "search.py")],
        cwd=WORKDIR,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert _json_stdout(proc) == {"ok": False, "error": "missing query"}


def test_ingest_empty_folder_is_one_json_object(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(RAG_ROOT / "ingest.py"), "--path", str(tmp_path)],
        cwd=WORKDIR,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    payload = _json_stdout(proc)
    assert payload["ok"] is False
    assert payload["error"].startswith("no documents in ")


def test_unknown_flag_is_rejected_as_json():
    proc = subprocess.run(
        [sys.executable, str(RAG_ROOT / "search.py"), "question", "--unknown-option"],
        cwd=WORKDIR,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    payload = _json_stdout(proc)
    assert payload["ok"] is False
    assert "unrecognized arguments: --unknown-option" in payload["error"]
