#!/usr/bin/env python3
"""Search the local index and print one JSON object on stdout."""

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

from rag.config import collection_cfg, load_config, resolve  # noqa: E402
from rag.retrieve import search as retrieve  # noqa: E402
from rag.rewrite import rewrite_query  # noqa: E402
from rag.store import Store  # noqa: E402


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _index_dir(cfg: dict) -> Path:
    return resolve(cfg, cfg.get("store_dir", "indexes"))


def list_collections(cfg: dict) -> list[dict]:
    index_dir = _index_dir(cfg)
    configured = set((cfg.get("collections") or {}).keys())
    on_disk = {p.stem for p in index_dir.glob("*.sqlite")} if index_dir.is_dir() else set()
    out = []
    for name in sorted(configured | on_disk):
        db_path = index_dir / f"{name}.sqlite"
        entry = {
            "name": name,
            "configured": name in configured,
            "docs_path": str(resolve(cfg, collection_cfg(cfg, name)["path"])),
            "index": str(db_path),
            "indexed": db_path.exists(),
            "documents": 0,
            "chunks": 0,
        }
        if db_path.exists():
            store = Store(db_path)
            try:
                entry.update(store.stats(name))
            finally:
                store.close()
        out.append(entry)
    return out


def main() -> int:
    p = JsonArgumentParser(description="Search the local RAG index")
    p.add_argument("query", nargs="?", help="Question")
    p.add_argument("--query", dest="query_opt", help="Question (named form)")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--collection", default="default")
    p.add_argument("--no-rewrite", action="store_true")
    p.add_argument("--config", default=None)
    p.add_argument(
        "--list-collections", action="store_true", help="List collections and index sizes"
    )
    args = p.parse_args()

    if args.list_collections:
        cfg = load_config(Path(args.config) if args.config else None)
        emit({"ok": True, "collections": list_collections(cfg)})
        return 0

    question = (args.query_opt or args.query or "").strip()
    if not question:
        emit({"ok": False, "error": "missing query"})
        return 2
    if args.k <= 0:
        emit({"ok": False, "error": "k must be greater than zero"})
        return 2

    cfg = load_config(Path(args.config) if args.config else None)
    col = collection_cfg(cfg, args.collection)
    db_path = _index_dir(cfg) / f"{col['name']}.sqlite"
    if not db_path.exists():
        emit({"ok": False, "error": "empty index — run ingest.py first", "index": str(db_path)})
        return 2

    host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    embed_model = cfg.get("embed", "embeddinggemma")
    rewrite_model = cfg.get("rewrite", "qwen3.5:0.8b")
    do_rewrite = bool(cfg.get("rewrite_enabled", True)) and not args.no_rewrite
    hyde_on = bool(cfg.get("hyde", True)) and do_rewrite

    store = Store(db_path)
    try:
        if store.stats(col["name"])["chunks"] == 0:
            emit({"ok": False, "error": "empty index — run ingest.py first", "index": str(db_path)})
            return 2

        rewritten = [question]
        hyde_text = ""
        if do_rewrite:
            rw = rewrite_query(host, rewrite_model, question, hyde=hyde_on)
            rewritten = rw["queries"]
            hyde_text = rw.get("hyde") or ""

        hits = retrieve(
            store,
            host=host,
            embed_model=embed_model,
            collection=col["name"],
            queries=rewritten,
            hyde_text=hyde_text,
            k=args.k,
            dense_weight=float(cfg.get("dense_weight", 1.0)),
            bm25_weight=float(cfg.get("bm25_weight", 1.0)),
            rrf_k=int(cfg.get("rrf_k", 60)),
        )
    finally:
        store.close()

    payload = {
        "ok": True,
        "query": question,
        "rewritten": rewritten,
        "hyde": hyde_text,
        "collection": col["name"],
        "hits": [
            {
                "text": h["text"],
                "score": round(h["score"], 4),
                "path": h["path"],
                "page": h["page"],
                "locator": h["locator"],
                "collection": h["collection"],
            }
            for h in hits
        ],
    }
    emit(payload)
    return 0 if hits else 1


def cli() -> int:
    try:
        return main()
    except Exception as exc:  # CLI boundary: stdout remains one JSON object.
        emit({"ok": False, "error": str(exc) or type(exc).__name__})
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
