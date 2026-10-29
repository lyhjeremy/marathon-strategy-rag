"""Deterministic post-check: does every number in an answer have a source?

Receipt Auditor's `assert_numbers_present` checks that a phrased answer kept the
numbers it was given. This runs the other way: it pulls every number out of the
coach's answer and looks for it in the evidence the answer was allowed to use,
which is the question itself, the tool results, and the passages the answer
cites. A number with no match is flagged as unsourced.

Matching rules, kept loose enough that ordinary rounding passes:

* A number matches a source value within half a unit of its last digit, or within
  2.5% of the source value, whichever is larger ("about 31 minutes" for 30.6).
* Clock times are read both ways: "3:30" is 210 (minutes) and "4:59" is 299
  (seconds); "47:30" is also 47.5 minutes. Source times get the same treatment.
* Source values also match after x100 / /100 (fraction and percent) and x60 / /60
  (minutes and seconds, hours and minutes).
* Digits inside a tool result's field names count as sources ("lo95_min" backs
  "the 95% range"), and so do the tool descriptions the coach was given (they
  define the wall as 35-40 km pace more than 25% slower than 10-20 km pace).
* Citation markers ([2], [T1]), list numbering and a few fixed race distances
  (42.195 km, 26.2 mi, 21.1 km, 13.1 mi) are ignored.

The check cannot tell a correct derived number (a sum the model did itself) from
an invented one; both get flagged. That is deliberate: the coach is told to get
every number from a tool, and it has a calculator for sums.
"""

from __future__ import annotations

import json
import re

_CITE = re.compile(r"\[(?:T?\d+)(?:\s*[,;]\s*T?\d+)*\]")
_LISTNUM = re.compile(r"(?m)^\s*(?:\*\*)?\d+[.)](?=\s|\*\*)")  # "1. " and "**1.**"
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:])")
_NUM = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)")
_FINAL_KEYS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*=")
FIXED = {42.195, 42.2, 26.2, 21.1, 21.0975, 13.1}


def _time_values(m) -> list[float]:
    a, b, c = int(m.group(1)), int(m.group(2)), m.group(3)
    if c is not None:                       # h:mm:ss
        mins = a * 60 + b + int(c) / 60
        return [mins, mins * 60]
    return [a * 60 + b, a + b / 60]         # h:mm as minutes / m:ss as seconds; m:ss as minutes


def extract(text: str, *, answer: bool = False) -> list[tuple[str, list[float], int]]:
    """Return (token, candidate values, decimals) for every number in `text`."""
    if answer:
        text = _CITE.sub(" ", text)
        text = _LISTNUM.sub(" ", text)
        text = re.sub(r"(?m)^(\s*FINAL:)(.*)$",
                      lambda m: m.group(1) + _FINAL_KEYS.sub(" ", m.group(2)), text)
    out = []
    for m in _TIME.finditer(text):
        out.append((m.group(0), _time_values(m), 0))
    text = _TIME.sub(" ", text)
    for m in _NUM.finditer(text):
        tok = m.group(1).rstrip(",")
        try:
            v = float(tok.replace(",", ""))
        except ValueError:
            continue
        dec = len(tok.split(".")[1]) if "." in tok else 0
        out.append((tok, [v], dec))
    return out


def source_values(*texts) -> set[float]:
    vals: set[float] = set(FIXED)
    for t in texts:
        if not isinstance(t, str):
            t = json.dumps(t, ensure_ascii=False)
            # digits inside field names count too: "lo95_min" is a 95% bound
            t += " " + " ".join(re.findall(r'(?<=[A-Za-z_])(\d+)(?=[A-Za-z_]*":)', t))
        for _, cands, _ in extract(t):
            for v in cands:
                vals.update({v, v * 100, v / 100, v * 60, v / 60})
    return vals


def unsourced(answer: str, sources: set[float]) -> list[str]:
    flagged = []
    for tok, cands, dec in extract(answer, answer=True):
        half = 0.5 * 10 ** (-dec)
        ok = any(abs(c - s) <= max(half, 0.025 * abs(s)) + 1e-9
                 for c in cands for s in sources)
        if not ok:
            flagged.append(tok)
    return flagged


def check(answer: str, question: str, evidence: list) -> dict:
    """Flag numbers in `answer` that appear in neither the question nor `evidence`."""
    flags = unsourced(answer, source_values(question, *evidence))
    n = len(extract(answer, answer=True))
    return {"numbers": n, "unsourced": flags, "fired": bool(flags)}
