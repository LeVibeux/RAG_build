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
    collection TEXT NOT NULL,
    mtime_ns INTEGER,
    size INTEGER,
    params TEXT
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
    embedding BLOB NOT NULL,
    embed_key TEXT
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

# Columns added after the first release; indexes built before them get upgraded in place.
_MIGRATIONS = {
    "documents": {"mtime_ns": "INTEGER", "size": "INTEGER", "params": "TEXT"},
    "chunks": {"embed_key": "TEXT"},
}

# Stay under SQLITE_MAX_VARIABLE_NUMBER on old SQLite builds (999).
_IN_BATCH = 500


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    for table, columns in _MIGRATIONS.items():
        present = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
        for name, sql_type in columns.items():
            if name not in present:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    con.execute("CREATE INDEX IF NOT EXISTS chunks_embed_key ON chunks(embed_key)")
    con.commit()
    return con


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def embed_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()


def _batches(items: list, size: int = _IN_BATCH):
    for i in range(0, len(items), size):
        yield items[i : i + size]


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
        mtime_ns: int | None = None,
        size: int | None = None,
        params: str | None = None,
    ) -> int:
        with self.con:
            cur = self.con.cursor()
            cur.execute("SELECT id FROM documents WHERE path = ?", (path,))
            old = cur.fetchone()
            if old:
                cur.execute("DELETE FROM documents WHERE id = ?", (old["id"],))
            cur.execute(
                """INSERT INTO documents(path, title, sha, collection, mtime_ns, size, params)
                VALUES (?,?,?,?,?,?,?)""",
                (path, title, sha, collection, mtime_ns, size, params),
            )
            doc_id = int(cur.lastrowid or 0)
            for row in rows:
                vec = np.asarray(row["embedding"], dtype=np.float32)
                cur.execute(
                    """INSERT INTO chunks(
                        doc_id, collection, text, parent_text, locator, page, kind,
                        embedding, embed_key
                    ) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        doc_id,
                        collection,
                        row["text"],
                        row["parent_text"],
                        row.get("locator"),
                        row.get("page"),
                        row.get("kind", "child"),
                        vec.tobytes(),
                        row.get("embed_key"),
                    ),
                )
        return len(rows)

    def document_state(self, path: str) -> dict | None:
        row = self.con.execute(
            "SELECT sha, mtime_ns, size, params FROM documents WHERE path = ?", (path,)
        ).fetchone()
        return dict(row) if row else None

    def touch_document(self, path: str, mtime_ns: int, size: int) -> None:
        with self.con:
            self.con.execute(
                "UPDATE documents SET mtime_ns = ?, size = ? WHERE path = ?",
                (mtime_ns, size, path),
            )

    def cached_embeddings(self, keys: set[str]) -> dict[str, np.ndarray]:
        """Vectors already stored for these content keys (any document)."""
        out: dict[str, np.ndarray] = {}
        for group in _batches(sorted(keys)):
            placeholders = ",".join("?" * len(group))
            for r in self.con.execute(
                f"SELECT embed_key, embedding FROM chunks WHERE embed_key IN ({placeholders})",
                group,
            ):
                out.setdefault(r["embed_key"], np.frombuffer(r["embedding"], dtype=np.float32))
        return out

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

    def all_vectors(self, collection: str) -> tuple[np.ndarray, list[int]]:
        """Only ids + vectors: texts are fetched later for the few ranked hits."""
        ids: list[int] = []
        blobs: list[bytes] = []
        for chunk_id, blob in self.con.execute(
            "SELECT id, embedding FROM chunks WHERE collection=?", (collection,)
        ):
            ids.append(int(chunk_id))
            blobs.append(blob)
        if not ids:
            return np.zeros((0, 1), dtype=np.float32), []
        if len({len(b) for b in blobs}) != 1:
            raise ValueError("index mixes embedding sizes — re-run ingest.py")
        mat = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(ids), -1)
        return mat, ids

    def chunks_by_ids(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        out: dict[int, sqlite3.Row] = {}
        for group in _batches(list(ids)):
            placeholders = ",".join("?" * len(group))
            for r in self.con.execute(
                f"""SELECT c.id, c.text, c.parent_text, c.locator, c.page, d.path
                FROM chunks c JOIN documents d ON d.id = c.doc_id
                WHERE c.id IN ({placeholders})""",
                group,
            ):
                out[int(r["id"])] = r
        return out

    def fts(self, collection: str, query: str, k: int) -> list[tuple[int, float]]:
        # OR, not AND: a natural-language question almost never has every word in one
        # chunk; bm25's IDF already down-weights common words.
        tokens = [t for t in re.findall(r"[^\W_]+", query, flags=re.UNICODE) if len(t) >= 2]
        q = " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))
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
        # FTS5 bm25: lower is better → convert to a descending score
        return [(int(row["id"]), -float(row["rank"])) for row in cur]
