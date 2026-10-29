"""
heat_model.py: expected Boston Marathon slowdown from race-day weather.

Standard library and NumPy only. Loads model/heat_model.json next to this file, which
model_fit.py writes. Public function:

    heat_effect(finish_min, temp_f, dewpoint_f=None, headwind_mph=None) -> dict

finish_min is the runner's time in baseline conditions (55 F or cooler, no wind along the
course). temp_f is the race-day temperature averaged over the hours on course. The return
value is the expected extra time and a 90% interval from a bootstrap over race years:

    {"expected_slowdown_min", "lo_min", "hi_min", "percentile_used",
     "baseline_temp_f", "model_version"}

percentile_used is the approximate place in a Boston field (percent of finishers faster)
whose coefficients were used; finish times between two anchor series are interpolated.
dewpoint_f is accepted for interface stability. Dew point, heat index and humidity did not
beat dry-bulb temperature in cross-validation, so the current model does not use it.
headwind_mph is the wind component against the runner along the course (negative for a
tailwind); None means no wind effect.
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_HERE, "model", "heat_model.json")
_M = None


def load_model(path=_PATH):
    global _M
    if _M is None or path != _PATH:
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        m["_typ"] = np.array([a["typical_min"] for a in m["anchors"]])
        m["_beta"] = np.array([a["beta"] for a in m["anchors"]])
        k = min(len(a["draws"]) for a in m["anchors"])
        m["_draws"] = np.array([a["draws"][:k] for a in m["anchors"]])          # anchors x draws
        m["_pct"] = np.array([a["field_pct"] for a in m["anchors"]])
        m["_wind"] = np.array(m["wind"]["draws_pct_per_mph_headwind"][:k]) / 100.0
        if path != _PATH:
            return m
        _M = m
    return _M


def _weights(m, finish_min):
    t = m["_typ"]
    if finish_min <= t[0]:
        return 0, 0, 0.0
    if finish_min >= t[-1]:
        n = len(t) - 1
        return n, n, 0.0
    j = int(np.searchsorted(t, finish_min))
    i = j - 1
    return i, j, float((finish_min - t[i]) / (t[j] - t[i]))


def heat_effect(finish_min: float, temp_f: float, dewpoint_f: float | None = None,
                headwind_mph: float | None = None) -> dict:
    m = load_model()
    finish_min = float(finish_min)
    if not (60.0 <= finish_min <= 600.0):
        raise ValueError("finish_min should be a marathon time in minutes (60 to 600)")
    i, j, u = _weights(m, finish_min)
    excess = max(float(temp_f) - m["knot_f"], 0.0)
    beta = (1 - u) * m["_beta"][i] + u * m["_beta"][j]
    bdraw = (1 - u) * m["_draws"][i] + u * m["_draws"][j]
    log_eff = beta * excess
    log_draw = bdraw * excess
    if headwind_mph is not None:
        w = float(headwind_mph)
        log_eff += m["wind"]["pct_per_mph_headwind"] / 100.0 * w
        log_draw = log_draw + m["_wind"] * w
    eff = finish_min * (np.exp(log_eff) - 1.0)
    draws = finish_min * (np.exp(log_draw) - 1.0)
    lo, hi = np.percentile(draws, [5, 95])
    pct = (1 - u) * m["_pct"][i] + u * m["_pct"][j]
    return {
        "expected_slowdown_min": round(float(eff), 2),
        "lo_min": round(float(min(lo, eff)), 2),
        "hi_min": round(float(max(hi, eff)), 2),
        "percentile_used": round(float(pct), 1),
        "baseline_temp_f": float(m["baseline_temp_f"]),
        "model_version": m["model_version"],
    }


if __name__ == "__main__":
    for args in [(180, 50), (226.8, 70), (226.8, 81), (240, 75, None, 10), (150, 80), (300, 85), (210, 60, 40, -12)]:
        print(args, heat_effect(*args))
