"""Load .md / .txt / .pdf into plain text pages."""

from __future__ import annotations

from pathlib import Path

TEXT_EXT = {".md", ".txt", ".rst", ".org"}
PDF_EXT = {".pdf"}


def iter_source_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in TEXT_EXT | PDF_EXT:
            files.append(p)
    return files


def load_pages(path: Path) -> list[tuple[int | None, str]]:
    suf = path.suffix.lower()
    if suf in TEXT_EXT:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [(None, text)]
    if suf in PDF_EXT:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: list[tuple[int | None, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            pages.append((i, raw))
        return pages
    raise ValueError(f"unsupported file type: {path}")
