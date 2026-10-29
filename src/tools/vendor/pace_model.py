"""
pace_model.py - in-race Boston finish-time, wall-risk and pacing-type model.

Standard library and NumPy only. Loads model/pace_model.json from next to this
file (written by train_model.py). The browser tool on the site reads the same
JSON and mirrors these functions in JavaScript.

    from pace_model import predict_finish
    predict_finish({5: 22.0, 10: 44.5, 21: 95.0}, "F", 34)

`splits` maps km marks already passed (5, 10, 15, 20, 21 for the half, 25, 30,
35, 40) to cumulative elapsed minutes. Missing marks in between are filled by
straight-line interpolation of elapsed time against distance, the same rule
the training data used for 2014's missing 15 km split.

Returns median_min, lo80_min, hi80_min, lo95_min, hi95_min (finish-time
interval bounds in minutes), wall_risk (probability that the runner's 35 to
40 km pace ends up more than 25% slower than their 10 to 20 km pace),
pace_type (nearest pacing type on the segments seen so far), n_train and
model_version.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "model", "pace_model.json")
HALF = 21.0975
FULL = 42.195
CHECKPOINTS = [5.0, 10.0, 15.0, 20.0, HALF, 25.0, 30.0, 35.0, 40.0]
# pacing-type segments: the eight Boston 2014 segments (no 15 km split that year)
TYPE_SEGS = [(0, 5), (5, 10), (10, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, FULL)]
BASE_FEATURES = ["f_avg", "f_recent", "f_start", "f_avg2", "f_avg_recent", "f_avg_start"]
DEMO_FEATURES = {"g": ["female"], "a": ["age10", "age10sq"]}
VARIANTS = ["none", "g", "a", "ga"]

_MODEL = None


def load_model(path=JSON_PATH):
    global _MODEL
    if _MODEL is None or path != JSON_PATH:
        with open(path) as f:
            m = json.load(f)
        if path != JSON_PATH:
            return m
        _MODEL = m
    return _MODEL


def checkpoint_key(km):
    """Snap a km mark to a checkpoint: 21, 21.1 and 21.0975 all mean the half."""
    km = float(km)
    if abs(km - HALF) < 0.2:
        return HALF
    for c in CHECKPOINTS:
        if abs(km - c) < 1e-6:
            return c
    raise ValueError(f"unknown split mark {km!r}; use 5, 10, 15, 20, 21, 25, 30, 35 or 40")


def interp_at(km, t, x):
    """Elapsed time at distances x from known (km, t) points, (0, 0) included."""
    return np.interp(x, np.concatenate([[0.0], km]), np.concatenate([[0.0], t]))


def features_matrix(T, km, c, female=None, age=None):
    """Vectorized features at checkpoint c.

    T: (n, k) cumulative minutes at distances km (k,), all <= c, NaN allowed for
    interior marks (interpolated per row). Returns dict of feature arrays plus
    the elapsed time at c.
    """
    T = np.asarray(T, dtype=float)
    km = np.asarray(km, dtype=float)
    n = T.shape[0]
    need = np.array([5.0, min(10.0, c), max(c - 5.0, 0.0), c])
    Q = np.empty((n, len(need)))
    full = np.all(np.isfinite(T), axis=1)
    if full.any():
        grid = np.concatenate([[0.0], km])
        Tg = np.hstack([np.zeros((full.sum(), 1)), T[full]])
        # vectorized linear interpolation on a shared grid
        idx = np.clip(np.searchsorted(grid, need, side="right") - 1, 0, len(grid) - 2)
        x0, x1 = grid[idx], grid[idx + 1]
        w = (need - x0) / (x1 - x0)
        Q[full] = Tg[:, idx] * (1 - w) + Tg[:, idx + 1] * w
    for i in np.where(~full)[0]:
        ok = np.isfinite(T[i])
        Q[i] = interp_at(km[ok], T[i, ok], need)
    t5, tearly, tprev, tc = Q.T
    avg = tc / c
    start = t5 / 5.0
    recent = (tc - tprev) / (c - max(c - 5.0, 0.0))
    f = {"f_avg": np.log(avg), "f_recent": np.log(recent / avg), "f_start": np.log(start / avg)}
    f["f_avg2"] = (f["f_avg"] - 1.6) ** 2
    f["f_avg_recent"] = (f["f_avg"] - 1.6) * f["f_recent"]
    f["f_avg_start"] = (f["f_avg"] - 1.6) * f["f_start"]
    if female is not None:
        f["female"] = np.asarray(female, dtype=float)
    if age is not None:
        a = (np.asarray(age, dtype=float) - 40.0) / 10.0
        f["age10"] = a
        f["age10sq"] = a ** 2
    return f, tc


def feature_names(variant):
    names = list(BASE_FEATURES)
    if "g" in variant:
        names += DEMO_FEATURES["g"]
    if "a" in variant:
        names += DEMO_FEATURES["a"]
    return names


def design(f, variant):
    cols = [f[k] for k in feature_names(variant)]
    return np.column_stack([np.ones_like(cols[0])] + cols)


def type_profile(T_at_marks, marks, upto, mean_pace):
    """Segment paces for the type segments fully inside [0, upto], divided by
    mean_pace (the predicted finish over 42.195 km). Scaling by the predicted
    finish, rather than by the mean of the segments seen, keeps the early
    segments comparable with the full-race centroids: a runner whose first
    half is 5% quicker than their projected average already looks like a
    fader. Returns (indices of segments used, profile)."""
    segs = [i for i, (a, b) in enumerate(TYPE_SEGS) if b <= upto + 1e-9]
    pts = sorted({0.0, *[x for s in segs for x in TYPE_SEGS[s]]})
    t = interp_at(np.asarray(marks, float), np.asarray(T_at_marks, float), np.array(pts))
    tm = dict(zip(pts, t))
    paces = np.array([(tm[TYPE_SEGS[s][1]] - tm[TYPE_SEGS[s][0]]) / (TYPE_SEGS[s][1] - TYPE_SEGS[s][0]) for s in segs])
    return segs, paces / mean_pace


def nearest_type(profile, segs, types):
    """Nearest centroid on the segments seen, distance weighted by segment length."""
    lens = np.array([TYPE_SEGS[s][1] - TYPE_SEGS[s][0] for s in segs])
    cen = np.asarray(types["centroids"])[:, segs]
    return int(np.argmin((((profile[None, :] - cen) ** 2) * lens).sum(1)))


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z > -50 else 0.0


def predict_finish(splits, gender=None, age=None):
    m = load_model()
    if not splits:
        raise ValueError("need at least one split")
    pts = sorted((checkpoint_key(k), float(v)) for k, v in splits.items())
    km = np.array([p[0] for p in pts])
    t = np.array([p[1] for p in pts])
    if np.any(t <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("split times must be positive and increase with distance")
    pace = t / km
    if np.any(pace < 2.5) or np.any(pace > 20):
        raise ValueError("average pace outside 2:30 to 20:00 per km; check the units (cumulative minutes)")
    c = float(km[-1])
    ck = m["checkpoints"][f"{c:g}"]
    g = None if gender is None else str(gender).strip().upper()[:1]
    female = None if g not in ("F", "W", "M") else (1.0 if g in ("F", "W") else 0.0)
    a = None
    if age is not None:
        try:
            a = float(age)
        except (TypeError, ValueError):
            a = None
        if a is not None and not (16 <= a <= 90):
            a = None
    variant = ("g" if female is not None else "") + ("a" if a is not None else "") or "none"
    f, tc = features_matrix(t[None, :], km, c, None if female is None else [female], None if a is None else [a])
    X = design(f, variant)[0]
    fm = ck["finish"][variant]
    y = float(X @ np.array(fm["coef"]))
    even = float(tc[0]) * FULL / c
    edges = ck["bins"]
    b = int(np.searchsorted(edges, f["f_avg"][0], side="right"))
    q = fm["resid_q"][b]  # residual quantiles in log space: 2.5, 10, 50, 90, 97.5
    out = {k: even * math.exp(y + q[i]) for i, k in enumerate(["lo95_min", "lo80_min", "median_min", "hi80_min", "hi95_min"])}
    # wall risk
    if c >= 40 - 1e-9:
        tt = interp_at(km, t, np.array([10.0, 20.0, 35.0, 40.0]))
        wall = 1.0 if (tt[3] - tt[2]) / 5 > 1.25 * (tt[1] - tt[0]) / 10 else 0.0
    else:
        wc = ck["wall"][variant]["coef"]
        wall = _sigmoid(float(X @ np.array(wc)))
    types = m["types"]
    segs, prof = type_profile(t, km, c, out["median_min"] / FULL)
    k = nearest_type(prof, segs, types)
    res = {k2: round(v, 2) for k2, v in out.items()}
    res = {"median_min": res["median_min"], "lo80_min": res["lo80_min"], "hi80_min": res["hi80_min"],
           "lo95_min": res["lo95_min"], "hi95_min": res["hi95_min"],
           "wall_risk": round(wall, 4), "pace_type": types["names"][k],
           "n_train": m["n_train"], "model_version": m["model_version"]}
    return res


if __name__ == "__main__":
    print(predict_finish({5: 22.0, 10: 44.5, 21: 95.0}, "F", 34))
