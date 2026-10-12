"""Dense retrieval over the pooled corpus (the study's single retriever).

    retriever.search(queries: list[str], k: int) -> list[list[str]]   # ranked doc_ids
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .corpus import doc_text


class DenseRetriever:
    """Bi-encoder: query and passage are embedded separately; score = cosine similarity.

    e5 models were trained with 'query: ' / 'passage: ' prefixes, so we keep them.
    """

    def __init__(self, corpus: list[dict], model_name: str = "intfloat/e5-base-v2",
                 device: str | None = None, batch_size: int = 128):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.batch_size = batch_size
        self.doc_emb = self.model.encode(
            ["passage: " + doc_text(d) for d in corpus],
            batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def scores(self, queries: Sequence[str]) -> np.ndarray:
        q = self.model.encode(["query: " + x for x in queries], batch_size=self.batch_size,
                              normalize_embeddings=True, convert_to_numpy=True)
        return q.astype(np.float32) @ self.doc_emb.T

    def search(self, queries, k=10):
        return _topk(self.scores(queries), self.doc_ids, k)


def _topk(scores: np.ndarray, doc_ids: list[str], k: int) -> list[list[str]]:
    out = []
    ids = np.asarray(doc_ids)
    kk = min(k, scores.shape[1])
    for s in scores:
        top = np.argpartition(-s, kk - 1)[:kk]
        top = top[np.argsort(-s[top], kind="stable")]
        out.append(ids[top].tolist())
    return out
