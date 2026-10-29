"""Shared configuration and paths."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"          # bundled markdown knowledge base
# The vector index is a rebuildable cache, so it lives outside the repo (and outside
# any synced folder). Override with MSR_DATA_DIR.
DATA_DIR = Path(os.environ.get(
    "MSR_DATA_DIR", Path.home() / "data-cache" / "marathon-strategy-rag"))
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION = "marathon_knowledge"

# Small, fast, CPU/MPS-friendly embedding model (384-dim). No API key required.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Cross-encoder used to rerank the fused shortlist (same model as wine-sommelier-rag).
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def embed_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
