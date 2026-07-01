"""Semantic retrieval over the marathon knowledge base."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CHROMA_DIR, COLLECTION
from .embedder import embed_query


@dataclass
class Passage:
    text: str
    title: str
    source: str
    card: str
    distance: float


class Retriever:
    def __init__(self):
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.coll = client.get_collection(COLLECTION)
        except Exception as exc:
            raise SystemExit(
                "No knowledge index found. Build it first: python -m src.ingest"
            ) from exc

    def search(self, query: str, k: int = 5) -> list[Passage]:
        res = self.coll.query(query_embeddings=[embed_query(query)], n_results=k)
        out = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(
                Passage(
                    text=doc,
                    title=meta.get("title", ""),
                    source=meta.get("source", ""),
                    card=meta.get("card", ""),
                    distance=dist,
                )
            )
        return out
