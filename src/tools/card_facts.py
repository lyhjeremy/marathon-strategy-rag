"""Numbers taken from the knowledge cards, each with the exact card text it comes from.

`pace_table` and the gold set's card-arithmetic questions read these values
instead of retyping them. `check()` confirms every quote still appears in its card,
so editing a card without updating this table fails loudly.

    python -m src.tools.card_facts      # verify every quote against knowledge/
"""

from __future__ import annotations

from ..config import KNOWLEDGE_DIR

# name: (value, card stem, quote that must appear verbatim in the card)
FACTS: dict[str, tuple[float, str, str]] = {
    # negative-split-reality: second half vs first half, Boston 2014
    "fade_pct_median": (14.0, "negative-split-reality", "**median slowdown was +14%**"),
    "fade_pct_sub3": (5.0, "negative-split-reality", "Sub-3:00 runners slowed only about **+5%**"),
    "fade_pct_over4": (22.0, "negative-split-reality", "Runners over 4:00 slowed roughly **+22%**"),
    # carb-loading
    "carb_load_g_per_kg_lo": (8.0, "carb-loading", "**8–12 g of carbohydrate per kg of body weight per day**"),
    "carb_load_g_per_kg_hi": (12.0, "carb-loading", "**8–12 g of carbohydrate per kg of body weight per day**"),
    # carb-fueling
    "fuel_g_per_h_lo": (30.0, "carb-fueling", "**~30–60 g of carbohydrate per hour**"),
    "fuel_g_per_h_hi": (60.0, "carb-fueling", "**~30–60 g of carbohydrate per hour**"),
    # super-shoes
    "shoe_gain_pct": (0.9, "super-shoes", "about **0.9% of a 2:05 marathon.**"),
    # heat-tax
    "heat_min_per_f": (1.0, "heat-tax", "approximately **1 minute slower per °F**"),
    # tapering
    "taper_cut_pct_lo": (40.0, "tapering", "**total volume by roughly 40–60%**"),
    "taper_cut_pct_hi": (60.0, "tapering", "**total volume by roughly 40–60%**"),
}


def value(name: str) -> float:
    return FACTS[name][0]


def cite(name: str) -> dict:
    v, card, quote = FACTS[name]
    return {"value": v, "card": card, "quote": quote.replace("**", "")}


def check() -> list[str]:
    problems = []
    for name, (_, card, quote) in FACTS.items():
        path = KNOWLEDGE_DIR / f"{card}.md"
        if not path.exists() or quote not in path.read_text(encoding="utf-8"):
            problems.append(f"{name}: quote not found in {card}.md")
    return problems


if __name__ == "__main__":
    bad = check()
    print("\n".join(bad) if bad else f"all {len(FACTS)} card facts match their cards")
    raise SystemExit(1 if bad else 0)
