"""Fast checks that need no model download and no LLM: python -m pytest -q"""

import pytest

from src.agent import parse_action
from src.eval import citation_validity, parse_final
from src.numcheck import check
from src.tools import call_tool
from src.tools.card_facts import check as facts_check
from src.tools.pace import pace_table, parse_time


def test_card_facts_match_cards():
    assert facts_check() == []


def test_even_pace_3_30():
    t = pace_table("3:30")
    assert t["first_half_min"] == 105.0
    assert t["pace_first_half_per_km"] == "4:59"
    assert t["splits"][-1]["elapsed"] == "3:30:00"


def test_fade_uses_card_bands():
    assert pace_table("2:50", "fade")["fade_pct"] == 5.0
    assert pace_table("3:30", "fade")["fade_pct"] == 14.0
    assert pace_table("4:30", "fade")["fade_pct"] == 22.0
    t = pace_table("4:00", "fade:10")
    assert abs(t["second_half_min"] / t["first_half_min"] - 1.10) < 1e-3


def test_parse_time():
    assert parse_time("3:30") == 210
    assert parse_time("2:59:59") == pytest.approx(179.983, abs=1e-3)
    assert parse_time(200) == 200


def test_calculator_rejects_names():
    assert call_tool("calculator", {"expression": "8*65"})["result"] == 520
    assert "error" in call_tool("calculator", {"expression": "__import__('os')"})


def test_unknown_tool_is_an_error_not_a_crash():
    assert "error" in call_tool("nope", {})


def test_parse_action_repairs():
    a = parse_action('```json\n{"action": "tool", "tool": "pace_table", "args": {"goal_time": "3:30",},}\n```')
    assert a["tool"] == "pace_table"
    f = parse_action('Sure. {"action": "final", "answer": "Run 4:59/km {even}."} thanks')
    assert f["answer"].startswith("Run")
    with pytest.raises(ValueError):
        parse_action("no json here")


def test_numcheck_flags_only_unsourced():
    ev = [{"first_half_min": 105.0, "pace_first_half_per_km": "4:59"}]
    r = check("Run 4:59 per km [T1], halfway in 1:45, about 5 min/km. Lose 37 min.\nFINAL: x=105",
              "Pace for 3:30?", ev)
    assert r["unsourced"] == ["37"]


def test_parse_final_and_citations():
    got = parse_final("blah\nFINAL: finish_min=213.5; wall_risk_pct=31%; pace_sec_per_km=4:59")
    assert got == {"finish_min": 213.5, "wall_risk_pct": 31.0, "pace_sec_per_km": 299.0}
    assert citation_validity("a [1] b [T2] c [7]", 5, 2) == pytest.approx(2 / 3)
    assert citation_validity("no refs", 5) is None


def test_bold_list_numbers_are_ignored():
    assert check("**3. Drink to thirst.** Do it.\n4) Sip.", "q", [])["unsourced"] == []
