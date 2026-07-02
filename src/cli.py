"""Command-line interface for the Marathon Strategy RAG coach.

    python -m src.cli ask "should I try to negative split my first marathon?"
    python -m src.cli chat
"""

from __future__ import annotations

import argparse

from .coach import Coach


def _print(ans) -> None:
    print("\n" + ans.text.strip() + "\n")
    if ans.sources:
        print("— Sources —")
        seen = set()
        for i, p in enumerate(ans.sources, 1):
            key = (p.title, p.source)
            marker = "" if key in seen else f"  ({p.source})"
            seen.add(key)
            print(f"  [{i}] {p.title}{marker}")
    print()


def _make_coach(args):
    if getattr(args, "engine", "native") == "lc":
        from .chain_lc import LCCoach  # LangChain LCEL implementation

        return LCCoach(k=args.k)
    return Coach()


def cmd_ask(args) -> None:
    _print(_make_coach(args).ask(args.query, k=args.k))


def cmd_chat(args) -> None:
    coach = _make_coach(args)
    print("🏃 Marathon Coach — ask a race-strategy question (Ctrl-C to quit).")
    try:
        while True:
            q = input("\nyou › ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            _print(coach.ask(q, k=args.k))
    except (KeyboardInterrupt, EOFError):
        print("\nRun strong! 🏁")


def main() -> None:
    ap = argparse.ArgumentParser(description="Marathon Strategy RAG coach")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="One-shot question.")
    a.add_argument("query")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--engine", choices=["native", "lc"], default="native",
                   help="native hand-rolled chain (default) or LangChain LCEL ('lc').")
    a.set_defaults(func=cmd_ask)

    c = sub.add_parser("chat", help="Interactive session.")
    c.add_argument("-k", type=int, default=5)
    c.add_argument("--engine", choices=["native", "lc"], default="native",
                   help="native hand-rolled chain (default) or LangChain LCEL ('lc').")
    c.set_defaults(func=cmd_chat)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
