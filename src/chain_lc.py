"""The same grounded coach, expressed as a LangChain **LCEL** pipeline.

`coach.py` wires retrieval → prompt → LLM by hand. This module builds the
identical flow declaratively with LangChain Runnables, so the retrieval, prompt
and model become composable pieces:

    {context: retrieve|format, query: passthrough} | prompt | llm

Same knowledge base, same grounded/cited system prompt, same answers — it just
demonstrates the framework most RAG systems in production are built on. Select it
from the CLI with ``--engine lc``.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from .coach import PROMPT, SYSTEM, Answer, _format_context
from .llm import LLM
from .retriever import Retriever


class LCCoach:
    """Drop-in replacement for ``Coach`` backed by an LCEL chain."""

    def __init__(self, k: int = 5):
        self.k = k
        self.retriever = Retriever()
        self.llm = LLM()
        self._last: list = []  # passages from the most recent retrieval, for citing
        self.chain = self._build()

    def _context(self, query: str) -> str:
        self._last = self.retriever.search(query, k=self.k)
        return _format_context(self._last)

    def _build(self):
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", PROMPT)])
        # A ChatPromptValue → plain text → our Claude wrapper. Keeping the LLM as a
        # RunnableLambda lets the CLI backend (no API key) slot into LCEL unchanged.
        llm = RunnableLambda(lambda pv: self.llm.complete(pv.to_string()))
        return (
            RunnableParallel(
                context=RunnableLambda(self._context),
                query=RunnablePassthrough(),
            )
            | prompt
            | llm
        )

    def ask(self, query: str, k: int | None = None) -> Answer:
        if k is not None and k != self.k:
            self.k = k
        text = self.chain.invoke(query)
        return Answer(text, self._last)
