"""Hybrid dense + FTS retrieval with reciprocal-rank fusion."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .embed import embed_texts
from .store import Store


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


def _rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rank_lists:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] += 1.0 / (k + rank)
    return dict(scores)


def search(
    store: Store,
    *,
    host: str,
    embed_model: str,
    collection: str,
    queries: list[str],
    hyde_text: str = "",
    k: int = 8,
    pool: int = 24,
) -> list[dict]:
    mat, rows = store.all_vectors(collection)
    if not rows:
        return []

    q_texts = [q for q in queries if q.strip()]
    if hyde_text:
        q_texts.append(hyde_text)
    if not q_texts:
        return []
    q_vecs = np.asarray(embed_texts(host, embed_model, q_texts), dtype=np.float32)
    docs = _l2_normalize(mat)
    qn = _l2_normalize(q_vecs)
    sim = qn @ docs.T  # (nq, n)

    id_to_row = {int(r["id"]): r for r in rows}
    dense_lists = [
        [int(rows[i]["id"]) for i in np.argsort(-query_scores)[:pool]]
        for query_scores in sim
    ]
    fts_hits: list[list[int]] = []
    for q in queries:
        pairs = store.fts(collection, q, pool)
        fts_hits.append([cid for cid, _ in pairs])

    fused = _rrf([*dense_lists, *[hits for hits in fts_hits if hits]])
    ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:pool]
    max_score = ranked[0][1] if ranked else 1.0
    hits = []
    for cid, score in ranked:
        row = id_to_row.get(cid)
        if row is None:
            continue
        hits.append(
            {
                "id": cid,
                "text": row["parent_text"] or row["text"],
                "child": row["text"],
                "score": float(score / max_score),
                "path": _doc_path(store, int(row["doc_id"])),
                "page": row["page"],
                "locator": row["locator"],
                "collection": collection,
            }
        )

    return hits[:k]


def _doc_path(store: Store, doc_id: int) -> str:
    row = store.con.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()
    return row["path"] if row else ""
