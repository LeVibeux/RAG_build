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


def test_list_collections_reports_configured_and_indexed_state(tmp_path):
    rag_dir = tmp_path / "RAG"
    (rag_dir / "indexes").mkdir(parents=True)
    config = rag_dir / "config.yaml"
    config.write_text(
        "store_dir: indexes\ncollections:\n  default:\n    path: ../docs\n  notes:\n    path: ../notes\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(RAG_ROOT))
    from rag.store import Store

    Store(rag_dir / "indexes" / "notes.sqlite").close()

    proc = subprocess.run(
        [sys.executable, str(RAG_ROOT / "search.py"), "--list-collections", "--config", str(config)],
        cwd=WORKDIR,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = _json_stdout(proc)
    by_name = {c["name"]: c for c in payload["collections"]}
    assert set(by_name) == {"default", "notes"}
    assert by_name["default"]["indexed"] is False
    assert by_name["notes"]["indexed"] is True
    assert by_name["notes"]["documents"] == 0
    assert by_name["notes"]["docs_path"] == str((tmp_path / "notes").resolve())
