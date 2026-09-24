from __future__ import annotations

import pytest

from rag.chunk import chunk_pages, estimate_tokens


def test_parent_child_indexes_children_and_returns_larger_parent():
    body = " ".join(f"mot{i}" for i in range(500))
    chunks = chunk_pages(
        [(7, f"# Clause importante\n\n{body}")],
        strategy="parent_child",
        child_tokens=40,
        parent_tokens=160,
        overlap_tokens=8,
    )

    assert len(chunks) > 1
    assert all(chunk.kind == "child" for chunk in chunks)
    assert all(chunk.locator == "Clause importante" for chunk in chunks)
    assert all(chunk.page == 7 for chunk in chunks)
    assert all(estimate_tokens(chunk.text) <= 40 for chunk in chunks)
    assert all(estimate_tokens(chunk.parent_text) <= 160 for chunk in chunks)
    assert any(len(chunk.text) < len(chunk.parent_text) for chunk in chunks)


def test_section_uses_markdown_and_article_locators():
    text = """Titre souligné
===============
Premier bloc.

Article 12
Deuxième bloc.
"""

    chunks = chunk_pages([(3, text)], strategy="section")

    assert [chunk.locator for chunk in chunks] == ["Titre souligné", "Article 12"]
    assert all(chunk.kind == "section" for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)


def test_window_has_no_locator_and_propagates_pdf_page():
    chunks = chunk_pages(
        [(11, "un deux trois quatre cinq six sept huit neuf dix")],
        strategy="window",
        child_tokens=5,
        parent_tokens=20,
        overlap_tokens=1,
    )

    assert len(chunks) > 1
    assert all(chunk.locator is None for chunk in chunks)
    assert all(chunk.page == 11 for chunk in chunks)
    assert all(chunk.text == chunk.parent_text for chunk in chunks)


def test_empty_text_produces_no_chunks():
    assert chunk_pages([(None, "  \n"), (2, "")]) == []


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown chunk strategy"):
        chunk_pages([(None, "texte")], strategy="semantic")


def test_section_strategy_ignores_the_unused_overlap_bound():
    # `section` never slides a window, so overlap_tokens >= child_tokens must
    # not be rejected here even though it would be invalid for parent_child/window.
    chunks = chunk_pages(
        [(None, "# Titre\n\nUn petit bloc de texte.")],
        strategy="section",
        child_tokens=10,
        parent_tokens=10,
        overlap_tokens=50,
    )
    assert len(chunks) == 1


def test_parent_child_still_rejects_overlap_not_smaller_than_child():
    with pytest.raises(ValueError, match="overlap_tokens must be smaller"):
        chunk_pages([(None, "texte")], strategy="parent_child", overlap_tokens=256)
