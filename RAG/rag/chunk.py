"""Generic, dependency-free chunk strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words * 1.3)) if text.strip() else 0


@dataclass
class Chunk:
    text: str
    parent_text: str
    locator: str | None
    page: int | None
    kind: str  # child | section | window


def _words_for_tokens(tokens: int) -> int:
    """Convert the kit's approximate token targets to conservative word counts."""
    return max(1, int(tokens / 1.3))


def _window(text: str, size_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    size = _words_for_tokens(size_tokens)
    overlap = min(_words_for_tokens(overlap_tokens), size - 1) if overlap_tokens else 0
    step = max(1, size - overlap)
    out: list[str] = []
    i = 0
    while i < len(words):
        piece = words[i : i + size]
        out.append(" ".join(piece))
        if i + size >= len(words):
            break
        i += step
    return out


def _sections(text: str) -> list[tuple[str, str]]:
    """Return (locator, body) pairs. Locator may be empty."""
    lines = text.splitlines()
    heads: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if re.match(r"^#{1,6}\s+\S", line):
            heads.append((i, re.sub(r"^#{1,6}\s+", "", line).strip()))
        elif re.match(r"^Article\s+\S", line, re.I):
            heads.append((i, line.strip()))
        elif (
            line.strip()
            and i + 1 < len(lines)
            and re.match(r"^[=-]{3,}\s*$", lines[i + 1])
        ):
            heads.append((i, line.strip()))
    if not heads:
        return [("", text.strip())] if text.strip() else []

    sections: list[tuple[str, str]] = []
    for idx, (start, title) in enumerate(heads):
        end = heads[idx + 1][0] if idx + 1 < len(heads) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if body:
            sections.append((title, body))
    preface = "\n".join(lines[: heads[0][0]]).strip()
    if preface:
        sections.insert(0, ("", preface))
    return sections


def chunk_pages(
    pages: list[tuple[int | None, str]],
    strategy: str = "parent_child",
    child_tokens: int = 256,
    parent_tokens: int = 1024,
    overlap_tokens: int = 32,
) -> list[Chunk]:
    strategy = (strategy or "parent_child").lower()
    if strategy not in {"parent_child", "section", "window"}:
        raise ValueError(f"unknown chunk strategy: {strategy}")
    if min(child_tokens, parent_tokens) <= 0 or overlap_tokens < 0:
        raise ValueError("chunk token sizes must be positive and overlap non-negative")
    if overlap_tokens >= child_tokens:
        raise ValueError("overlap_tokens must be smaller than child_tokens")
    if strategy == "parent_child" and child_tokens > parent_tokens:
        raise ValueError("child_tokens must not exceed parent_tokens")

    out: list[Chunk] = []
    for page, raw in pages:
        text = (raw or "").strip()
        if not text:
            continue
        if strategy == "window":
            for piece in _window(text, child_tokens, overlap_tokens):
                out.append(Chunk(piece, piece, None, page, "window"))
            continue

        sections = _sections(text) if strategy in {"section", "parent_child"} else [("", text)]
        if strategy == "section":
            for loc, body in sections:
                out.append(Chunk(body, body, loc or None, page, "section"))
            continue

        # parent_child
        for loc, body in sections:
            parents = _window(body, parent_tokens, overlap_tokens)
            for parent in parents:
                children = _window(parent, child_tokens, overlap_tokens) or [parent]
                for child in children:
                    out.append(Chunk(child, parent, loc or None, page, "child"))
    return out
