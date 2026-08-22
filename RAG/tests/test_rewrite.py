from __future__ import annotations

import pytest

import rag.rewrite as rewrite
from rag.embed import OllamaError


def test_rewrite_keeps_original_caps_rephrases_and_returns_hyde(monkeypatch):
    raw = """```json
{"queries": ["Reformulation une", "Question originale", "Reformulation deux", "Reformulation trois", "En trop"], "hyde": "Réponse hypothétique."}
```"""
    monkeypatch.setattr(rewrite, "generate", lambda *args, **kwargs: raw)

    result = rewrite.rewrite_query("http://127.0.0.1:11434", "qwen3.5:0.8b", "Question originale")

    assert result == {
        "queries": [
            "Question originale",
            "Reformulation une",
            "Reformulation deux",
            "Reformulation trois",
        ],
        "hyde": "Réponse hypothétique.",
    }


def test_rewrite_can_disable_hyde(monkeypatch):
    monkeypatch.setattr(
        rewrite,
        "generate",
        lambda *args, **kwargs: '{"queries": ["Une", "Deux"], "hyde": "Ignoré"}',
    )

    result = rewrite.rewrite_query("host", "model", "Originale", hyde=False)

    assert result == {"queries": ["Originale", "Une", "Deux"], "hyde": ""}


def test_rewrite_falls_back_when_ollama_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise OllamaError("down")

    monkeypatch.setattr(rewrite, "generate", fail)

    assert rewrite.rewrite_query("host", "model", " Originale ") == {
        "queries": ["Originale"],
        "hyde": "",
    }


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '{"queries": "not-a-list", "hyde": "texte"}',
        '{"queries": [], "hyde": "texte"}',
        '{"queries": ["Une seule"], "hyde": "texte"}',
        '{"queries": ["Une", "Deux"]}',
    ],
)
def test_rewrite_falls_back_on_invalid_json_or_schema(monkeypatch, raw):
    monkeypatch.setattr(rewrite, "generate", lambda *args, **kwargs: raw)

    assert rewrite.rewrite_query("host", "model", "Originale") == {
        "queries": ["Originale"],
        "hyde": "",
    }
