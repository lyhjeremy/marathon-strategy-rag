"""pace_table: split targets for a goal time, run evenly or with a planned fade.

Two strategies:

* ``even``: one pace for all 42.195 km.
* ``fade``: a positive split sized from the negative-split card, which reports how
  much slower the second half was than the first for Boston 2014 finishers: about
  +5% for sub-3:00 runners, +22% for runners over 4:00, and +14% at the median.
  The card gives no figure for the 3:00 to 4:00 band, so that band uses the
  overall median. ``fade:<pct>`` sets the fade directly (for example ``fade:8``).
  The fade table answers "if I slow down like a typical runner at this level, how
  fast must the first half be to still make the goal?". The cards recommend even
  pacing; the fade table is a planning aid, not advice to positive-split.

Each half is run at a constant pace (a two-segment approximation).
"""

from __future__ import annotations

import re

from .card_facts import cite, value

MARATHON_KM = 42.195
HALF_KM = MARATHON_KM / 2
KM_PER_MILE = 1.609344
MARKS_KM = [5, 10, 15, 20, HALF_KM, 25, 30, 35, 40, MARATHON_KM]


def parse_time(t) -> float:
    """'3:30' or '3:30:00' (h:mm[:ss]) or a number of minutes -> minutes."""
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = [float(p) for p in s.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 60 + parts[1] + parts[2] / 60
    raise ValueError(f"cannot parse time {t!r}; use h:mm, h:mm:ss or minutes")


def fmt_hms(minutes: float) -> str:
    s = int(round(minutes * 60))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fmt_pace(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 60}:{s % 60:02d}"


def _fade_for(goal_min: float) -> tuple[float, dict]:
    if goal_min < 180:
        name = "fade_pct_sub3"
    elif goal_min > 240:
        name = "fade_pct_over4"
    else:
        name = "fade_pct_median"
    return value(name), cite(name)


def pace_table(goal_time, strategy: str = "even") -> dict:
    goal = parse_time(goal_time)
    if not 120 <= goal <= 480:
        raise ValueError("goal_time should be between 2:00 and 8:00")
    strategy = (strategy or "even").strip().lower()
    source = {"card": "even-pacing-strategy",
              "quote": "Holding a steady effort and pace throughout"}
    if strategy == "even":
        fade = 0.0
    elif strategy == "fade":
        fade, source = _fade_for(goal)
    elif strategy.startswith("fade:"):
        fade = float(strategy.split(":", 1)[1].rstrip("%"))
        source = {"card": None, "quote": "fade percent set by the caller"}
        if not 0 <= fade <= 40:
            raise ValueError("fade percent should be between 0 and 40")
    else:
        raise ValueError("strategy must be 'even', 'fade' or 'fade:<pct>'")

    first = goal / (2 + fade / 100)          # second half = first * (1 + fade)
    second = goal - first
    p1 = first * 60 / HALF_KM                # seconds per km
    p2 = second * 60 / HALF_KM

    splits = []
    for km in MARKS_KM:
        t = km * p1 / 60 if km <= HALF_KM else first + (km - HALF_KM) * p2 / 60
        splits.append({"km": round(km, 3), "elapsed_min": round(t, 2), "elapsed": fmt_hms(t)})

    return {
        "goal_min": round(goal, 2),
        "goal": fmt_hms(goal),
        "strategy": strategy,
        "fade_pct": fade,
        "fade_source": source,
        "first_half_min": round(first, 2),
        "second_half_min": round(second, 2),
        "first_half": fmt_hms(first),
        "second_half": fmt_hms(second),
        "pace_first_half_sec_per_km": round(p1, 1),
        "pace_second_half_sec_per_km": round(p2, 1),
        "pace_first_half_per_km": fmt_pace(p1),
        "pace_second_half_per_km": fmt_pace(p2),
        "pace_first_half_per_mile": fmt_pace(p1 * KM_PER_MILE),
        "pace_second_half_per_mile": fmt_pace(p2 * KM_PER_MILE),
        "pace_first_half_sec_per_mile": round(p1 * KM_PER_MILE, 1),
        "pace_second_half_sec_per_mile": round(p2 * KM_PER_MILE, 1),
        "splits": splits,
    }
