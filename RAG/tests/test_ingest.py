"""Incremental ingestion, end to end, with a fake embedder (no Ollama)."""

from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

import ingest


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(
        "# Alpha\n\nPremier bloc sur la clause d’agrément.\n\n# Beta\n\nSecond bloc comptable.\n",
        encoding="utf-8",
    )
    (docs / "b.md").write_text("# Gamma\n\nTroisième document.\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"docs_path: {docs}\nstore_dir: indexes\nchunk: parent_child\n", encoding="utf-8"
    )

    calls: list[list[str]] = []

    def fake_embed(host, model, texts, batch=16, workers=1):
        calls.append(list(texts))
        return [list(hashlib.sha256(t.encode()).digest()[:4]) for t in texts]

    monkeypatch.setattr(ingest, "embed_texts", fake_embed)
    return {"docs": docs, "config": config, "calls": calls, "monkeypatch": monkeypatch}


def _run(env, capsys, *extra) -> dict:
    env["monkeypatch"].setattr(
        sys, "argv", ["ingest.py", "--config", str(env["config"]), *extra]
    )
    assert ingest.main() == 0
    return json.loads(capsys.readouterr().out)


def test_second_run_skips_unchanged_files_without_embedding(workdir, capsys):
    first = _run(workdir, capsys)
    assert (first["files"], first["skipped"], first["reused"]) == (2, 0, 0)
    assert first["embedded"] == first["chunks"] == 3
    n_calls = len(workdir["calls"])

    second = _run(workdir, capsys)

    assert (second["files"], second["skipped"], second["embedded"]) == (0, 2, 0)
    assert second["stats"] == first["stats"]
    assert len(workdir["calls"]) == n_calls


def test_touched_but_identical_file_is_skipped_via_sha(workdir, capsys):
    _run(workdir, capsys)
    path = workdir["docs"] / "a.md"
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))

    result = _run(workdir, capsys)

    assert (result["files"], result["skipped"], result["embedded"]) == (0, 2, 0)


def test_editing_one_section_only_embeds_the_changed_chunk(workdir, capsys):
    _run(workdir, capsys)
    (workdir["docs"] / "a.md").write_text(
        "# Alpha\n\nPremier bloc sur la clause d’agrément.\n\n# Beta\n\nBloc comptable révisé.\n",
        encoding="utf-8",
    )

    result = _run(workdir, capsys)

    assert (result["files"], result["skipped"]) == (1, 1)
    assert (result["embedded"], result["reused"]) == (1, 1)
    assert workdir["calls"][-1] == ["# Beta Bloc comptable révisé."]


def test_changing_chunk_settings_reingests_everything(workdir, capsys):
    _run(workdir, capsys)
    workdir["config"].write_text(
        workdir["config"].read_text(encoding="utf-8").replace("parent_child", "section"),
        encoding="utf-8",
    )

    result = _run(workdir, capsys)

    assert (result["files"], result["skipped"]) == (2, 0)


def test_force_reingests_but_still_reuses_cached_vectors(workdir, capsys):
    _run(workdir, capsys)
    n_calls = len(workdir["calls"])

    result = _run(workdir, capsys, "--force")

    assert (result["files"], result["skipped"]) == (2, 0)
    assert (result["embedded"], result["reused"]) == (0, 3)
    assert len(workdir["calls"]) == n_calls


def test_deleted_file_is_removed_from_the_index(workdir, capsys):
    _run(workdir, capsys)
    (workdir["docs"] / "b.md").unlink()

    result = _run(workdir, capsys)

    assert result["removed"] == 1
    assert result["stats"]["documents"] == 1


def _run_raw(env, capsys, *extra) -> tuple[int, dict]:
    env["monkeypatch"].setattr(
        sys, "argv", ["ingest.py", "--config", str(env["config"]), *extra]
    )
    code = ingest.main()
    return code, json.loads(capsys.readouterr().out)


def test_emptied_folder_prunes_its_documents(workdir, capsys):
    _run(workdir, capsys)
    for f in workdir["docs"].iterdir():
        f.unlink()

    code, result = _run_raw(workdir, capsys)

    assert code == 2
    assert result["error"].startswith("no documents in ")
    assert result["removed"] == 2


def test_ingesting_another_folder_keeps_documents_from_the_first(workdir, capsys, tmp_path):
    first = _run(workdir, capsys)
    other = tmp_path / "docs2"
    other.mkdir()
    (other / "c.md").write_text("# Delta\n\nAutre dossier.\n", encoding="utf-8")

    result = _run(workdir, capsys, "--path", str(other))

    assert result["removed"] == 0
    assert result["stats"]["documents"] == first["stats"]["documents"] + 1

    empty = tmp_path / "vide"
    empty.mkdir()
    code, result = _run_raw(workdir, capsys, "--path", str(empty))
    assert (code, result["removed"]) == (2, 0)


def test_missing_folder_is_an_error_and_leaves_the_index_intact(workdir, capsys, tmp_path):
    first = _run(workdir, capsys)

    code, result = _run_raw(workdir, capsys, "--path", str(tmp_path / "absent"))

    assert code == 2
    assert result["error"].startswith("docs folder not found")
    assert _run(workdir, capsys)["stats"] == first["stats"]
