"""Load the two statistical models this coach calls as tools.

* ``pace_model.predict_finish`` from marathon-pacing-decay (finish time and wall
  risk from the splits so far).
* ``heat_model.heat_effect`` from marathon-heat-tax (minutes lost to weather).

Both are single-file modules (standard library plus NumPy) that load a JSON file
sitting next to them. The shipped copies live in ``src/tools/vendor/`` with a
``SOURCE`` file recording the commit they came from. If a vendored copy is
missing, the loader falls back to the sibling repo checkout, which is how the
coach was developed before the models were final. Set ``MSR_MODEL_SOURCE=sibling``
to force the sibling copies.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor"
_PROJECTS = Path(__file__).resolve().parents[3].parent  # .../6. Data Projects
SIBLINGS = {
    "pace_model": _PROJECTS / "Project 13 Hitting the Wall (Pacing Decay)" / "marathon-pacing-decay",
    "heat_model": _PROJECTS / "Project 14 The Heat Tax (Weather vs Marathon Times)" / "marathon-heat-tax",
}


# What the models' own holdout checks found, passed along with every result so the
# coach does not present either model as more certain than it is.
PACE_CAVEAT = (
    "Fitted on Boston splits; has no weather input. On the 2024 holdout its 80% "
    "interval at 35 km covered 74% of runners, so late-race intervals run narrow.")
HEAT_CAVEAT = (
    "Fitted on Boston results vs. race-day temperature (baseline 55F or cooler). Dew "
    "point is accepted but not used: it did not beat dry-bulb temperature in "
    "cross-validation. On its 2015-2019 holdout the model did not beat a flat "
    "no-weather guess (2.26% vs 2.15% error), so treat the slowdown as a rough guide.")


class ModelUnavailable(RuntimeError):
    pass


def _candidates(name: str) -> list[Path]:
    vendored = VENDOR / f"{name}.py"
    sibling = SIBLINGS[name] / f"{name}.py"
    if os.environ.get("MSR_MODEL_SOURCE") == "sibling":
        return [sibling]
    return [vendored, sibling]


@lru_cache(maxsize=None)
def load(name: str):
    for path in _candidates(name):
        if path.exists():
            spec = importlib.util.spec_from_file_location(f"msr_{name}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.__msr_path__ = str(path)
            return mod
    raise ModelUnavailable(
        f"{name}.py not found in src/tools/vendor/ or the sibling repo; "
        "copy it in with scripts/vendor_models.py")


def predict_finish(splits: dict, gender: str | None = None, age: int | None = None) -> dict:
    clean = {}
    for k, v in (splits or {}).items():
        km = int(round(float(k)))
        clean[21 if km == 21 else km] = float(v)
    if not clean:
        raise ValueError("splits must contain at least one km mark")
    out = dict(load("pace_model").predict_finish(clean, gender=gender, age=age))
    out["caveat"] = PACE_CAVEAT
    return out


def heat_effect(finish_min: float, temp_f: float, dewpoint_f: float | None = None,
                headwind_mph: float | None = None) -> dict:
    out = dict(load("heat_model").heat_effect(
        float(finish_min), float(temp_f),
        None if dewpoint_f is None else float(dewpoint_f),
        None if headwind_mph is None else float(headwind_mph)))
    # Convenience field so the coach never has to add numbers itself.
    out["adjusted_finish_min"] = round(float(finish_min) + float(out["expected_slowdown_min"]), 2)
    out["caveat"] = HEAT_CAVEAT
    return out


def source_info() -> dict:
    info = {}
    for name in SIBLINGS:
        try:
            mod = load(name)
            info[name] = mod.__msr_path__
        except ModelUnavailable as exc:
            info[name] = f"unavailable: {exc}"
    return info
