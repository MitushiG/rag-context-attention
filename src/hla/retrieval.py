"""Sparse, dense and hybrid retrievers over the pooled corpus.

All retrievers share one interface:

    retriever.search(queries: list[str], k: int, allowed: list[set[str]] | None)
        -> list[list[str]]   # ranked doc_ids per query

``allowed`` optionally restricts each query to a subset of doc ids (used in H3, where
RAG must retrieve from the same padded pool the long-context condition sees).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .corpus import doc_text


class BM25Retriever:
    """Lexical retrieval: scores exact term overlap, weighted by rarity (IDF) and
    saturated by term frequency and document length."""

    def __init__(self, corpus: list[dict]):
        import bm25s

        self.doc_ids = [d["doc_id"] for d in corpus]
        self._bm25s = bm25s
        self.index = bm25s.BM25()
        self.index.index(bm25s.tokenize([doc_text(d) for d in corpus], stopwords="en", show_progress=False))

    def scores(self, queries: Sequence[str]) -> np.ndarray:
        """Full score matrix (n_queries x n_docs)."""
        toks = self._bm25s.tokenize(list(queries), stopwords="en", return_ids=False, show_progress=False)
        out = np.zeros((len(queries), len(self.doc_ids)), dtype=np.float32)
        for i, q in enumerate(toks):
            q = [t for t in q if t in self.index.vocab_dict]  # unseen terms score 0
            if q:
                out[i] = self.index.get_scores(q)
        return out

    def search(self, queries, k=10, allowed=None):
        return _topk(self.scores(queries), self.doc_ids, k, allowed)


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

    def search(self, queries, k=10, allowed=None):
        return _topk(self.scores(queries), self.doc_ids, k, allowed)


class HybridRetriever:
    """Reciprocal rank fusion: score(d) = sum_r 1 / (rrf_k + rank_r(d)).

    RRF only uses ranks, so BM25 and cosine scores never need to be put on one scale.
    """

    def __init__(self, retrievers: list, rrf_k: int = 60, depth: int = 100):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth

    def search(self, queries, k=10, allowed=None):
        runs = [r.search(queries, k=self.depth, allowed=allowed) for r in self.retrievers]
        out = []
        for qi in range(len(queries)):
            fused: dict[str, float] = {}
            for run in runs:
                for rank, d in enumerate(run[qi], start=1):
                    fused[d] = fused.get(d, 0.0) + 1.0 / (self.rrf_k + rank)
            out.append(sorted(fused, key=fused.get, reverse=True)[:k])
        return out


class CrossEncoderReranker:
    """Reads query and passage *together* (full cross-attention), so it is more precise
    than a bi-encoder but too slow to run over the whole corpus: use on top candidates."""

    def __init__(self, corpus: list[dict], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.docs = {d["doc_id"]: d for d in corpus}

    def rerank(self, queries: Sequence[str], candidates: list[list[str]], k: int = 10):
        out = []
        for q, cands in zip(queries, candidates):
            s = self.model.predict([(q, doc_text(self.docs[d])) for d in cands])
            order = np.argsort(-np.asarray(s))
            out.append([cands[i] for i in order[:k]])
        return out


def _topk(scores: np.ndarray, doc_ids: list[str], k: int, allowed=None) -> list[list[str]]:
    out = []
    ids = np.asarray(doc_ids)
    for i in range(scores.shape[0]):
        s = scores[i]
        if allowed is not None:
            mask = np.isin(ids, list(allowed[i]))
            s = np.where(mask, s, -np.inf)
        kk = min(k, int(np.isfinite(s).sum()))
        top = np.argpartition(-s, kk - 1)[:kk] if kk > 0 else np.array([], int)
        top = top[np.argsort(-s[top])]
        out.append(ids[top].tolist())
    return out
