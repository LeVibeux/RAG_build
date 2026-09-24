"""SQLite store: dense vectors + FTS5 BM25. No Docker, no extra DB."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    title TEXT,
    sha TEXT,
    collection TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection TEXT NOT NULL,
    text TEXT NOT NULL,
    parent_text TEXT NOT NULL,
    locator TEXT,
    page INTEGER,
    kind TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_coll ON chunks(collection);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.con = _connect(db_path)

    def close(self) -> None:
        self.con.close()

    def replace_document(
        self,
        *,
        collection: str,
        path: str,
        title: str,
        sha: str,
        rows: list[dict],
    ) -> int:
        with self.con:
            cur = self.con.cursor()
            cur.execute("SELECT id FROM documents WHERE path = ?", (path,))
            old = cur.fetchone()
            if old:
                cur.execute("DELETE FROM documents WHERE id = ?", (old["id"],))
            cur.execute(
                "INSERT INTO documents(path, title, sha, collection) VALUES (?,?,?,?)",
                (path, title, sha, collection),
            )
            doc_id = int(cur.lastrowid or 0)
            for row in rows:
                vec = np.asarray(row["embedding"], dtype=np.float32)
                cur.execute(
                    """INSERT INTO chunks(
                        doc_id, collection, text, parent_text, locator, page, kind, embedding
                    ) VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        doc_id,
                        collection,
                        row["text"],
                        row["parent_text"],
                        row.get("locator"),
                        row.get("page"),
                        row.get("kind", "child"),
                        vec.tobytes(),
                    ),
                )
        return len(rows)

    def document_sha(self, path: str) -> str | None:
        row = self.con.execute(
            "SELECT sha FROM documents WHERE path = ?", (path,)
        ).fetchone()
        return row["sha"] if row else None

    def prune_missing(self, collection: str, keep_paths: set[str]) -> int:
        """Delete documents (and their chunks, via cascade) no longer present on disk."""
        rows = self.con.execute(
            "SELECT id, path FROM documents WHERE collection = ?", (collection,)
        ).fetchall()
        stale_ids = [r["id"] for r in rows if r["path"] not in keep_paths]
        if not stale_ids:
            return 0
        with self.con:
            self.con.executemany(
                "DELETE FROM documents WHERE id = ?", [(i,) for i in stale_ids]
            )
        return len(stale_ids)

    def stats(self, collection: str | None = None) -> dict:
        if collection:
            n_docs = self.con.execute(
                "SELECT COUNT(*) FROM documents WHERE collection=?", (collection,)
            ).fetchone()[0]
            n_chunks = self.con.execute(
                "SELECT COUNT(*) FROM chunks WHERE collection=?", (collection,)
            ).fetchone()[0]
        else:
            n_docs = self.con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            n_chunks = self.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": int(n_docs), "chunks": int(n_chunks)}

    def all_vectors(self, collection: str) -> tuple[np.ndarray, list[sqlite3.Row]]:
        rows = list(
            self.con.execute(
                "SELECT * FROM chunks WHERE collection=?",
                (collection,),
            )
        )
        if not rows:
            return np.zeros((0, 1), dtype=np.float32), []
        mat = np.stack(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        )
        return mat, rows

    def fts(self, collection: str, query: str, k: int) -> list[tuple[int, float]]:
        # FTS5 bm25: lower is better → convert to a descending score
        tokens = [t for t in re.findall(r"[^\W_]+", query, flags=re.UNICODE) if len(t) >= 2]
        q = " AND ".join(f'"{token}"' for token in tokens)
        if not q:
            return []
        try:
            cur = self.con.execute(
                """
                SELECT c.id, bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ? AND c.collection = ?
                ORDER BY rank
                LIMIT ?
                """,
                (q, collection, k),
            )
        except sqlite3.OperationalError:
            return []
        out = []
        for row in cur:
            rank = float(row["rank"])
            out.append((int(row["id"]), -rank))
        return out
