#!/usr/bin/env python3
"""Ingest documents into the local SQLite index."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_VENV = HERE / ".venv"
if _VENV.is_dir() and Path(sys.prefix).resolve() != _VENV.resolve():
    for _cand in (_VENV / "bin" / "python", _VENV / "bin" / "python3"):
        if _cand.exists():
            os.execv(str(_cand), [str(_cand), *sys.argv])

sys.path.insert(0, str(HERE))

from rag.chunk import chunk_pages  # noqa: E402
from rag.config import collection_cfg, load_config, resolve  # noqa: E402
from rag.embed import embed_texts  # noqa: E402
from rag.store import Store, embed_key, file_sha  # noqa: E402
from rag.textio import iter_source_files, load_pages  # noqa: E402


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def main() -> int:
    p = JsonArgumentParser(description="Ingest files into the local RAG index")
    p.add_argument("--path", help="Override docs folder")
    p.add_argument("--collection", default="default")
    p.add_argument("--config", default=None)
    p.add_argument(
        "--force", action="store_true", help="Re-chunk every file even if unchanged"
    )
    args = p.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    col = collection_cfg(cfg, args.collection)
    docs_dir = Path(args.path).resolve() if args.path else resolve(cfg, col["path"])
    db_path = resolve(cfg, cfg.get("store_dir", "indexes")) / f"{col['name']}.sqlite"
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    model = cfg.get("embed", "embeddinggemma")
    batch = int(cfg.get("embed_batch", 16))
    workers = int(cfg.get("embed_workers", 1))
    # Changing the embed model or chunking must invalidate "unchanged file" skips.
    params = json.dumps(
        {
            "embed": model,
            "chunk": col["chunk"],
            "child_tokens": col["child_tokens"],
            "parent_tokens": col["parent_tokens"],
            "overlap_tokens": col["overlap_tokens"],
        },
        sort_keys=True,
    )

    files = iter_source_files(docs_dir)
    if not files:
        emit({"ok": False, "error": f"no documents in {docs_dir}"})
        return 2

    store = Store(db_path)
    ingested = 0
    skipped = 0
    chunks_n = 0
    embedded = 0
    reused = 0
    keep_paths: set[str] = set()
    try:
        for path in files:
            rel = os.path.relpath(path, start=HERE)
            keep_paths.add(rel)
            st = path.stat()
            state = store.document_state(rel)
            same_params = bool(state) and state["params"] == params and not args.force
            if (
                same_params
                and state["mtime_ns"] == st.st_mtime_ns
                and state["size"] == st.st_size
            ):
                skipped += 1
                continue
            sha = file_sha(path)
            if same_params and state["sha"] == sha:
                store.touch_document(rel, st.st_mtime_ns, st.st_size)
                skipped += 1
                continue

            pages = load_pages(path)
            chunks = chunk_pages(
                pages,
                strategy=col["chunk"],
                child_tokens=col["child_tokens"],
                parent_tokens=col["parent_tokens"],
                overlap_tokens=col["overlap_tokens"],
            )
            if not chunks:
                continue
            keys = [embed_key(model, c.text) for c in chunks]
            vectors = store.cached_embeddings(set(keys))
            missing = {k: c.text for k, c in zip(keys, chunks) if k not in vectors}
            reused += sum(1 for k in keys if k in vectors)
            if missing:
                fresh = embed_texts(
                    host, model, list(missing.values()), batch=batch, workers=workers
                )
                vectors.update(zip(missing.keys(), fresh))
                embedded += len(missing)
            rows = [
                {
                    "text": c.text,
                    "parent_text": c.parent_text,
                    "locator": c.locator,
                    "page": c.page,
                    "kind": c.kind,
                    "embedding": vectors[k],
                    "embed_key": k,
                }
                for k, c in zip(keys, chunks)
            ]
            chunks_n += store.replace_document(
                collection=col["name"],
                path=rel,
                title=path.stem,
                sha=sha,
                rows=rows,
                mtime_ns=st.st_mtime_ns,
                size=st.st_size,
                params=params,
            )
            ingested += 1
        removed = store.prune_missing(col["name"], keep_paths)
        stats = store.stats(col["name"])
    finally:
        store.close()

    if not ingested and not skipped:
        emit({"ok": False, "error": f"no extractable text in {docs_dir}"})
        return 2

    emit(
        {
            "ok": True,
            "collection": col["name"],
            "docs_dir": str(docs_dir),
            "files": ingested,
            "skipped": skipped,
            "removed": removed,
            "chunks": chunks_n,
            "embedded": embedded,
            "reused": reused,
            "index": str(db_path),
            "stats": stats,
        }
    )
    return 0


def cli() -> int:
    try:
        return main()
    except Exception as exc:  # CLI boundary: stdout remains one JSON object.
        emit({"ok": False, "error": str(exc) or type(exc).__name__})
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
