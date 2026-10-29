"""Hybrid retrieval: dense (Chroma) + sparse (BM25), fused with reciprocal rank
fusion, then optionally reordered by a cross-encoder.

The approach is the one wine-sommelier-rag uses. Dense embeddings match
paraphrase ("blowing up late" finds the wall card); BM25 matches exact terms
("dew point", "hyponatremia", "W80+"). RRF merges the two rankings without
needing their scores on a common scale, and the cross-encoder reads the query
and each candidate together to rescore a short list.

The corpus here is 46 chunks, so BM25 is built in memory on first use and the
cross-encoder can afford to score every fused candidate. `src/retrieval_eval.py`
measures whether any of this beats plain dense search on this corpus.

Modes: "dense", "bm25", "hybrid", "dense+rerank", "hybrid+rerank".
"""

from __future__ import annotations

import re
from functools import lru_cache

from .config import CHROMA_DIR, COLLECTION, RERANK_MODEL
from .embedder import embed_query
from .retriever import Passage

MODES = ("dense", "bm25", "hybrid", "dense+rerank", "hybrid+rerank")
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Corpus:
    def __init__(self, ids, docs, metas):
        from rank_bm25 import BM25Okapi

        self.ids, self.docs, self.metas = ids, docs, metas
        self._pos = {i: n for n, i in enumerate(ids)}
        self.bm25 = BM25Okapi([_tokenize(d) for d in docs])

    def passage(self, _id: str, score: float = 0.0) -> Passage:
        n = self._pos[_id]
        m = self.metas[n]
        return Passage(self.docs[n], m.get("title", ""), m.get("source", ""),
                       m.get("card", ""), score)

    def bm25_ranking(self, query: str, top: int) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [self.ids[i] for i in order[:top] if scores[i] > 0]


def _collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(COLLECTION)
    except Exception as exc:
        raise SystemExit(
            "No knowledge index found. Build it first: python -m src.ingest") from exc


@lru_cache(maxsize=1)
def load_corpus() -> Corpus:
    got = _collection().get(include=["documents", "metadatas"])
    return Corpus(got["ids"], got["documents"], got["metadatas"])


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, _id in enumerate(ranking):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


class HybridRetriever:
    """Same `search(query, k)` interface as `Retriever`, with a selectable mode."""

    def __init__(self, mode: str = "hybrid+rerank", pool: int = 20):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.pool = mode, pool
        self.coll = _collection()
        self.corpus = load_corpus()

    def _dense(self, query: str, top: int) -> list[str]:
        res = self.coll.query(query_embeddings=[embed_query(query)], n_results=top)
        return res["ids"][0]

    def ranking(self, query: str) -> list[tuple[str, float]]:
        """Full ordered list of (chunk id, score) for the configured mode."""
        if self.mode.startswith("dense"):
            cands = self._dense(query, self.pool)
        elif self.mode == "bm25":
            cands = self.corpus.bm25_ranking(query, self.pool)
        else:
            cands = reciprocal_rank_fusion(
                [self._dense(query, self.pool), self.corpus.bm25_ranking(query, self.pool)]
            )[: self.pool]
        if not self.mode.endswith("rerank"):
            return [(c, 1.0 / (r + 1)) for r, c in enumerate(cands)]
        scores = _reranker().predict([(query, self.corpus.passage(c).text) for c in cands])
        return sorted(zip(cands, map(float, scores)), key=lambda t: t[1], reverse=True)

    def search(self, query: str, k: int = 5) -> list[Passage]:
        return [self.corpus.passage(c, s) for c, s in self.ranking(query)[:k]]
