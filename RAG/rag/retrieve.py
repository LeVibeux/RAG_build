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


def _rrf(
    rank_lists: list[list[int]], weights: list[float] | None = None, k: int = 60
) -> dict[int, float]:
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rank_lists, weights):
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] += weight / (k + rank)
    return dict(scores)


def _top_indices(scores: np.ndarray, n: int) -> np.ndarray:
    if n >= scores.shape[0]:
        return np.argsort(-scores)
    top = np.argpartition(-scores, n)[:n]
    return top[np.argsort(-scores[top])]


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
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[dict]:
    mat, ids = store.all_vectors(collection)
    if not ids:
        return []

    q_texts = [q for q in queries if q.strip()]
    if hyde_text:
        q_texts.append(hyde_text)
    if not q_texts:
        return []
    q_vecs = np.asarray(embed_texts(host, embed_model, q_texts), dtype=np.float32)
    if q_vecs.shape[1] != mat.shape[1]:
        raise ValueError(
            f"query embedding size {q_vecs.shape[1]} != index size {mat.shape[1]}: "
            "the index was built with another embed model — re-run ingest.py"
        )
    sim = _l2_normalize(q_vecs) @ _l2_normalize(mat).T  # (nq, n)

    dense_lists = [[ids[i] for i in _top_indices(row, pool)] for row in sim]
    fts_lists = [
        hits for hits in ([cid for cid, _ in store.fts(collection, q, pool)] for q in queries) if hits
    ]

    fused = _rrf(
        [*dense_lists, *fts_lists],
        [dense_weight] * len(dense_lists) + [bm25_weight] * len(fts_lists),
        k=rrf_k,
    )
    ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
    if not ranked:
        return []
    max_score = ranked[0][1] or 1.0
    rows = store.chunks_by_ids([cid for cid, _ in ranked])
    hits = []
    for cid, score in ranked:
        row = rows.get(cid)
        if row is None:
            continue
        hits.append(
            {
                "id": cid,
                "text": row["parent_text"] or row["text"],
                "child": row["text"],
                "score": float(score / max_score),
                "path": row["path"],
                "page": row["page"],
                "locator": row["locator"],
                "collection": collection,
            }
        )
    return hits
