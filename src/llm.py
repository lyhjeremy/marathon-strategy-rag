"""Thin LLM wrapper with two interchangeable backends.

Backend selection (automatic, in priority order):

1. ``cli``  – shells out to the local ``claude`` binary (Claude Code / Agent SDK).
   Uses your Claude subscription auth, so it needs **no API key and costs nothing
   per token**. This is the default and the path used to build/test this repo.

2. ``api``  – uses the official ``anthropic`` Python SDK. Activated automatically
   when ``ANTHROPIC_API_KEY`` is set, so anyone who clones the repo can run it
   without the Claude CLI installed.

Override the choice with ``LLM_BACKEND=cli`` or ``LLM_BACKEND=api``.
"""

from __future__ import annotations

import os
import shutil
import subprocess


DEFAULT_CLI_MODEL = os.environ.get("CLAUDE_MODEL", "")          # "" -> CLI default
DEFAULT_API_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")


class LLMError(RuntimeError):
    """Raised when the underlying LLM call fails."""


def _choose_backend() -> str:
    forced = os.environ.get("LLM_BACKEND", "").strip().lower()
    if forced in {"cli", "api"}:
        return forced
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    if shutil.which("claude"):
        return "cli"
    raise LLMError(
        "No LLM backend available. Either install the Claude CLI "
        "(https://claude.com/claude-code) or set ANTHROPIC_API_KEY."
    )


class LLM:
    """Minimal text-completion interface used across the project."""

    def __init__(self, backend: str | None = None, timeout: int = 180):
        self.backend = backend or _choose_backend()
        self.timeout = timeout

    def complete(self, prompt: str, system: str | None = None) -> str:
        if self.backend == "cli":
            return self._complete_cli(prompt, system)
        return self._complete_api(prompt, system)

    # --- CLI backend (Claude subscription, no API key) -----------------------
    def _complete_cli(self, prompt: str, system: str | None) -> str:
        cmd = ["claude", "-p"]
        if system:
            cmd += ["--append-system-prompt", system]
        if DEFAULT_CLI_MODEL:
            cmd += ["--model", DEFAULT_CLI_MODEL]
        try:
            # Prompt is piped via stdin so long RAG contexts never hit argv limits.
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:  # claude not on PATH
            raise LLMError("`claude` CLI not found on PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"claude CLI timed out after {self.timeout}s.") from exc
        if result.returncode != 0:
            raise LLMError(f"claude CLI failed: {result.stderr.strip()}")
        return result.stdout.strip()

    # --- API backend (official SDK) ------------------------------------------
    def _complete_api(self, prompt: str, system: str | None) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("pip install anthropic to use the API backend.") from exc
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=DEFAULT_API_MODEL,
            max_tokens=1500,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text").strip()


if __name__ == "__main__":  # tiny smoke test
    llm = LLM()
    print(f"[backend={llm.backend}]")
    print(llm.complete("Reply with exactly one word: PONG"))
