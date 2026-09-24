from __future__ import annotations

import rag.retrieve as retrieve
from rag.store import Store


def _row(text: str, parent: str, vector: list[float], locator: str, kind: str = "child") -> dict:
    return {
        "text": text,
        "parent_text": parent,
        "locator": locator,
        "page": 4,
        "kind": kind,
        "embedding": vector,
    }


def test_hybrid_search_uses_requested_embedder_and_returns_parent(monkeypatch, tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        store.replace_document(
            collection="default",
            path="../docs/contrat.md",
            title="contrat",
            sha="sha",
            rows=[
                _row(
                    "clause agrément associés",
                    "Parent complet contenant le contexte de la clause d’agrément.",
                    [1.0, 0.0],
                    "Agrément",
                ),
                _row(
                    "calendrier clôture annuelle",
                    "Section comptable complète.",
                    [0.0, 1.0],
                    "Comptabilité",
                    kind="section",
                ),
            ],
        )
        calls = []

        def fake_embed(host, model, texts):
            calls.append((host, model, texts))
            return [[1.0, 0.0] for _ in texts]

        monkeypatch.setattr(retrieve, "embed_texts", fake_embed)

        hits = retrieve.search(
            store,
            host="http://127.0.0.1:11434",
            embed_model="embeddinggemma",
            collection="default",
            queries=["clause agrément"],
            hyde_text="Une cession soumise à l'accord des associés.",
            k=2,
        )

        assert calls == [
            (
                "http://127.0.0.1:11434",
                "embeddinggemma",
                ["clause agrément", "Une cession soumise à l'accord des associés."],
            )
        ]
        assert hits[0]["text"] == "Parent complet contenant le contexte de la clause d’agrément."
        assert hits[0]["path"] == "../docs/contrat.md"
        assert hits[0]["page"] == 4
        assert hits[0]["locator"] == "Agrément"
        assert hits[0]["collection"] == "default"
        assert hits[0]["score"] == 1.0
    finally:
        store.close()


def test_hybrid_search_resolves_paths_across_several_documents(monkeypatch, tmp_path):
    store = Store(tmp_path / "index.sqlite")
    try:
        store.replace_document(
            collection="default",
            path="../docs/a.md",
            title="a",
            sha="sha-a",
            rows=[_row("texte a", "Parent A", [1.0, 0.0], "A")],
        )
        store.replace_document(
            collection="default",
            path="../docs/b.md",
            title="b",
            sha="sha-b",
            rows=[_row("texte b", "Parent B", [0.0, 1.0], "B")],
        )

        monkeypatch.setattr(
            retrieve, "embed_texts", lambda host, model, texts: [[1.0, 1.0] for _ in texts]
        )

        hits = retrieve.search(
            store,
            host="http://127.0.0.1:11434",
            embed_model="embeddinggemma",
            collection="default",
            queries=["texte"],
            k=2,
        )

        assert {h["path"] for h in hits} == {"../docs/a.md", "../docs/b.md"}
    finally:
        store.close()
