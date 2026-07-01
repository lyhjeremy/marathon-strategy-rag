"""Build the marathon-knowledge vector index from bundled markdown cards.

Each file in knowledge/ is a "knowledge card" with a `# Title` and a
`> Source: ...` attribution line. Cards are split into overlapping chunks,
embedded locally, and stored in Chroma with their title + source as metadata so
every retrieved passage stays citable.

Usage:  python -m src.ingest
"""

from __future__ import annotations

import re

from .config import CHROMA_DIR, COLLECTION, KNOWLEDGE_DIR
from .embedder import embed_documents


def _parse_card(text: str) -> tuple[str, str, str]:
    """Return (title, source, body) from a knowledge-card markdown string."""
    title, source, body_lines = "Untitled", "Unknown", []
    for line in text.splitlines():
        if line.startswith("# ") and title == "Untitled":
            title = line[2:].strip()
        elif line.lstrip().startswith("> Source:"):
            source = line.split("Source:", 1)[1].strip()
        else:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    return title, source, body


def _chunk(body: str, target_words: int = 120, overlap: int = 30) -> list[str]:
    """Split a card body into overlapping word-windows, respecting paragraphs."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks, buf, count = [], [], 0
    for para in paras:
        words = para.split()
        if count + len(words) > target_words and buf:
            chunks.append(" ".join(buf))
            # carry an overlap window into the next chunk for context continuity
            tail = " ".join(buf).split()[-overlap:]
            buf, count = tail.copy(), len(tail)
        buf.extend(words)
        count += len(words)
    if buf:
        chunks.append(" ".join(buf))
    return chunks or [body]


def build() -> int:
    import chromadb

    cards = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not cards:
        raise SystemExit(f"No knowledge cards found in {KNOWLEDGE_DIR}.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    ids, docs, metas = [], [], []
    for card in cards:
        title, source, body = _parse_card(card.read_text(encoding="utf-8"))
        for i, chunk in enumerate(_chunk(body)):
            ids.append(f"{card.stem}-{i}")
            # Prepend the title so a chunk is self-describing to the retriever.
            docs.append(f"{title}. {chunk}")
            metas.append({"title": title, "source": source, "card": card.stem})

    print(f"Indexing {len(docs)} chunks from {len(cards)} knowledge cards…")
    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embed_documents(docs))
    print(f"Done. Collection '{COLLECTION}' holds {coll.count()} chunks.")
    return coll.count()


if __name__ == "__main__":
    build()
