"""Local sentence-transformers embedder (free, offline, no API key)."""

from __future__ import annotations

from functools import lru_cache

from .config import EMBED_MODEL, embed_device


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device=embed_device())


def embed_documents(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    vecs = _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    vec = _model().encode(
        [text], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    return vec.tolist()
