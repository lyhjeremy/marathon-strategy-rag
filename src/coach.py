"""The RAG chain: retrieve marathon knowledge, then answer as a coach with citations."""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LLM
from .retriever import Passage, Retriever

SYSTEM = (
    "You are an experienced, evidence-minded marathon coach. Answer the runner's "
    "question using ONLY the retrieved knowledge passages provided — do not rely "
    "on outside facts or invent statistics. Cite the passages you use by their "
    "[n] index. Be direct and practical: give a clear recommendation, the reason "
    "behind it, and a concrete action. If the passages don't cover the question, "
    "say what you can and note the gap."
)

PROMPT = """A runner asks:
"{query}"

Retrieved knowledge (use ONLY these):

{context}

Give practical coaching advice grounded in these passages, citing each fact you
use as [n]. Keep it focused and actionable."""


@dataclass
class Answer:
    text: str
    sources: list[Passage]


def _format_context(passages: list[Passage]) -> str:
    lines = []
    for i, p in enumerate(passages, 1):
        lines.append(f"[{i}] ({p.title} — source: {p.source})\n{p.text}")
    return "\n\n".join(lines)


class Coach:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLM()

    def ask(self, query: str, k: int = 5) -> Answer:
        passages = self.retriever.search(query, k=k)
        if not passages:
            return Answer("I don't have knowledge on that yet.", [])
        prompt = PROMPT.format(query=query, context=_format_context(passages))
        return Answer(self.llm.complete(prompt, system=SYSTEM), passages)
