"""Record what the retriever returns for the seven gold questions.

Chunks every knowledge card exactly as src/ingest.py does, embeds the chunks and
the gold questions with the same local model, and writes the cosine similarity of
every question against every chunk to eval/retrieval_demo.json. The write-up page
reads that file to show the top-k passages per question. Chroma is not needed:
the embeddings are normalized, so cosine similarity is a dot product, and with
this few chunks the ranking matches an exact search.

Usage:  python -m scripts.export_retrieval
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np

from src.config import EMBED_MODEL, KNOWLEDGE_DIR, ROOT
from src.embedder import embed_documents, embed_query
from src.eval_v2 import GOLD
from src.ingest import _chunk, _parse_card


def main() -> None:
    chunks = []
    for card in sorted(KNOWLEDGE_DIR.glob("*.md")):
        title, source, body = _parse_card(card.read_text(encoding="utf-8"))
        for i, chunk in enumerate(_chunk(body)):
            chunks.append({"id": f"{card.stem}-{i}", "card": card.stem, "title": title,
                           "text": f"{title}. {chunk}"})
    doc_vecs = np.array(embed_documents([c["text"] for c in chunks]))
    questions = []
    for q in GOLD:
        sims = doc_vecs @ np.array(embed_query(q))
        questions.append({"q": q, "sims": [round(float(s), 4) for s in sims]})
    out = {"model": EMBED_MODEL, "generated": date.today().isoformat(),
           "chunks": chunks, "questions": questions}
    path = ROOT / "eval" / "retrieval_demo.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(chunks)} chunks, {len(questions)} questions)")


if __name__ == "__main__":
    main()
