from __future__ import annotations

import numpy as np

from rag.store import Store


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
        matrix, rows = store.all_vectors("default")
        np.testing.assert_allclose(matrix, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        assert len(rows) == 2
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


def test_document_sha_enables_skip_on_unchanged_reingest(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        assert store.document_sha("../docs/contrat.md") is None

        store.replace_document(
            collection="default",
            path="../docs/contrat.md",
            title="contrat",
            sha="abc123",
            rows=[_row("La clause d’agrément protège les associés.", [1.0, 0.0])],
        )

        assert store.document_sha("../docs/contrat.md") == "abc123"
        assert store.document_sha("../docs/absent.md") is None
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
        assert store.document_sha("../docs/stale.md") is None
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
