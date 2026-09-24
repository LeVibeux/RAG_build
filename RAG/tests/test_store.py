from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from rag.store import Store, embed_key


def _row(text: str, vector: list[float], *, kind: str = "child") -> dict:
    return {
        "text": text,
        "parent_text": f"Contexte: {text}",
        "locator": "Contrat",
        "page": None,
        "kind": kind,
        "embedding": vector,
    }


def test_replace_document_fts_and_vectors(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        inserted = store.replace_document(
            collection="default",
            path="../docs/contrat.md",
            title="contrat",
            sha="first",
            rows=[
                _row("La clause d’agrément protège les associés.", [1.0, 0.0]),
                _row("Une section traite du calendrier fiscal.", [0.0, 1.0], kind="section"),
            ],
        )

        assert inserted == 2
        assert store.stats("default") == {"documents": 1, "chunks": 2}
        matrix, ids = store.all_vectors("default")
        np.testing.assert_allclose(matrix, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        assert len(ids) == 2
        assert store.fts("default", "clause d'agrément", 5)
        assert store.fts("default", "calendrier fiscal", 5)

        store.replace_document(
            collection="default",
            path="../docs/contrat.md",
            title="contrat",
            sha="second",
            rows=[_row("Le nouveau texte concerne uniquement les dividendes.", [0.5, 0.5])],
        )

        assert store.stats("default") == {"documents": 1, "chunks": 1}
        assert store.fts("default", "clause agrément", 5) == []
        assert store.fts("default", "nouveau dividendes", 5)
    finally:
        store.close()


def test_document_state_tracks_sha_mtime_size_and_params(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        assert store.document_state("../docs/contrat.md") is None

        store.replace_document(
            collection="default",
            path="../docs/contrat.md",
            title="contrat",
            sha="abc123",
            rows=[_row("La clause d’agrément protège les associés.", [1.0, 0.0])],
            mtime_ns=111,
            size=42,
            params='{"embed": "m"}',
        )
        assert store.document_state("../docs/contrat.md") == {
            "sha": "abc123",
            "mtime_ns": 111,
            "size": 42,
            "params": '{"embed": "m"}',
        }

        store.touch_document("../docs/contrat.md", 222, 43)
        state = store.document_state("../docs/contrat.md")
        assert (state["sha"], state["mtime_ns"], state["size"]) == ("abc123", 222, 43)
        assert store.document_state("../docs/absent.md") is None
    finally:
        store.close()


def test_prune_missing_removes_deleted_documents_and_their_chunks(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        store.replace_document(
            collection="default",
            path="../docs/keep.md",
            title="keep",
            sha="sha-keep",
            rows=[_row("Texte conservé.", [1.0, 0.0])],
        )
        store.replace_document(
            collection="default",
            path="../docs/stale.md",
            title="stale",
            sha="sha-stale",
            rows=[_row("Texte supprimé du dossier docs.", [0.0, 1.0])],
        )
        assert store.stats("default") == {"documents": 2, "chunks": 2}

        removed = store.prune_missing("default", {"../docs/keep.md"})

        assert removed == 1
        assert store.stats("default") == {"documents": 1, "chunks": 1}
        assert store.document_state("../docs/stale.md") is None
        assert store.fts("default", "supprimé", 5) == []
    finally:
        store.close()


def test_fts_ignores_only_punctuation_and_one_letter_tokens(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        store.replace_document(
            collection="default",
            path="../docs/note.md",
            title="note",
            sha="sha",
            rows=[_row("L'été québécois est agréable.", [1.0, 0.0])],
        )

        assert store.fts("default", "l'été québécois !", 5)
        assert store.fts("default", "!!!", 5) == []
    finally:
        store.close()


def test_fts_matches_natural_language_questions(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        store.replace_document(
            collection="default",
            path="../docs/contrat.md",
            title="contrat",
            sha="sha",
            rows=[
                _row("La clause d’agrément impose l’accord préalable des associés.", [1.0, 0.0]),
                _row("La clôture annuelle intervient le 31 décembre.", [0.0, 1.0]),
            ],
        )

        hits = store.fts("default", "que dit la clause d agrément sur les associés", 5)

        assert hits
        assert hits[0][0] == 1  # the chunk matching the most rare terms ranks first
    finally:
        store.close()


def test_cached_embeddings_reuses_vectors_by_content_key(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        key = embed_key("embeddinggemma", "Texte partagé.")
        row = _row("Texte partagé.", [0.25, 0.75])
        row["embed_key"] = key
        store.replace_document(
            collection="default", path="../docs/a.md", title="a", sha="a", rows=[row]
        )

        cached = store.cached_embeddings({key, "absent"})

        assert set(cached) == {key}
        np.testing.assert_allclose(cached[key], [0.25, 0.75])
        assert embed_key("autre-modele", "Texte partagé.") != key
    finally:
        store.close()


def test_chunks_by_ids_returns_text_and_document_path(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        store.replace_document(
            collection="default",
            path="../docs/a.md",
            title="a",
            sha="a",
            rows=[_row("Texte A.", [1.0, 0.0])],
        )
        _, ids = store.all_vectors("default")

        rows = store.chunks_by_ids(ids)

        assert rows[ids[0]]["path"] == "../docs/a.md"
        assert rows[ids[0]]["parent_text"] == "Contexte: Texte A."
    finally:
        store.close()


def test_index_built_before_new_columns_is_migrated_in_place(tmp_path):
    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE,
            title TEXT, sha TEXT, collection TEXT NOT NULL);
        CREATE TABLE chunks (id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            collection TEXT NOT NULL, text TEXT NOT NULL, parent_text TEXT NOT NULL,
            locator TEXT, page INTEGER, kind TEXT NOT NULL, embedding BLOB NOT NULL);
        INSERT INTO documents VALUES (1, '../docs/old.md', 'old', 'sha-old', 'default');
        """
    )
    con.execute(
        "INSERT INTO chunks VALUES (1, 1, 'default', 'ancien', 'ancien', NULL, NULL, 'child', ?)",
        (np.asarray([1.0, 0.0], dtype=np.float32).tobytes(),),
    )
    con.commit()
    con.close()

    store = Store(db)
    try:
        state = store.document_state("../docs/old.md")
        assert state == {"sha": "sha-old", "mtime_ns": None, "size": None, "params": None}
        assert store.stats("default") == {"documents": 1, "chunks": 1}
    finally:
        store.close()


def test_mixed_embedding_sizes_raise_a_clear_error(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        store.replace_document(
            collection="default", path="a", title="a", sha="a", rows=[_row("a", [1.0, 0.0])]
        )
        store.replace_document(
            collection="default", path="b", title="b", sha="b", rows=[_row("b", [1.0, 0.0, 0.0])]
        )
        with pytest.raises(ValueError, match="re-run ingest.py"):
            store.all_vectors("default")
    finally:
        store.close()


def test_prune_missing_under_only_touches_that_folder(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        for path in ("../docs/a.md", "../docs2/b.md"):
            store.replace_document(
                collection="default", path=path, title="t", sha="s", rows=[_row("x", [1.0])]
            )

        assert store.prune_missing("default", set(), under="../docs") == 1
        assert store.document_state("../docs2/b.md") is not None
    finally:
        store.close()
