"""The coach's tools: one registry shared by the agent loop and the MCP server.

Four domain tools (the MCP server exposes these):

    retrieve_cards(query, k)                        knowledge-card search
    predict_finish(splits, gender, age)             marathon-pacing-decay model
    heat_effect(finish_min, temp_f, dewpoint_f, headwind_mph)   marathon-heat-tax model
    pace_table(goal_time, strategy)                 even or fade split table

plus ``calculator(expression)`` for the agent only, so that any sum or product
in an answer comes out of a tool instead of the language model's head.
"""

from __future__ import annotations

import ast
import operator
from functools import lru_cache

from . import models
from .pace import pace_table

RETRIEVAL_MODE = "hybrid+rerank"

TOOL_SPECS = [
    {
        "name": "retrieve_cards",
        "description": (
            "Search the marathon knowledge cards (pacing, the wall, heat, fueling, "
            "tapering, hydration, shoes, course difficulty, BQ standards). Returns "
            "passages with the card title and source."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up."},
                "k": {"type": "integer", "description": "Passages to return (1-8).", "default": 4},
            },
            "required": ["query"],
        },
    },
    {
        "name": "predict_finish",
        "description": (
            "Predict a marathon finish time from the splits run so far, with 80% and "
            "95% intervals and the probability of hitting the wall (35-40 km pace more "
            "than 25% slower than 10-20 km pace). Fitted on Boston Marathon splits."),
        "input_schema": {
            "type": "object",
            "properties": {
                "splits": {
                    "type": "object",
                    "description": ("Cumulative elapsed MINUTES keyed by km mark passed: "
                                    "5, 10, 15, 20, 21 (halfway), 25, 30, 35, 40. "
                                    "Example: {\"10\": 47.5, \"21\": 102}."),
                    "additionalProperties": {"type": "number"},
                },
                "gender": {"type": "string", "enum": ["M", "F"], "description": "Optional."},
                "age": {"type": "integer", "description": "Optional."},
            },
            "required": ["splits"],
        },
    },
    {
        "name": "heat_effect",
        "description": (
            "Expected slowdown in minutes from race-day weather for a runner of a given "
            "cool-weather finish time, relative to the model's baseline conditions. "
            "Fitted on Boston Marathon results against reanalysis weather. Also returns "
            "adjusted_finish_min = finish_min + expected slowdown."),
        "input_schema": {
            "type": "object",
            "properties": {
                "finish_min": {"type": "number", "description": "Cool-weather finish time in minutes."},
                "temp_f": {"type": "number", "description": "Air temperature, deg F."},
                "dewpoint_f": {"type": "number", "description": "Dew point, deg F (optional)."},
                "headwind_mph": {"type": "number", "description": "Headwind, mph (optional)."},
            },
            "required": ["finish_min", "temp_f"],
        },
    },
    {
        "name": "pace_table",
        "description": (
            "Split table for a goal time. strategy 'even' (one pace), 'fade' (second half "
            "slower by the typical Boston fade for that level, from the negative-split "
            "card) or 'fade:<pct>'. Returns half splits, pace per km and per mile, and "
            "cumulative times at 5 km marks."),
        "input_schema": {
            "type": "object",
            "properties": {
                "goal_time": {"type": "string", "description": "h:mm or h:mm:ss, or minutes."},
                "strategy": {"type": "string", "default": "even"},
            },
            "required": ["goal_time"],
        },
    },
    {
        "name": "calculator",
        "description": ("Evaluate an arithmetic expression (+ - * / ** parentheses, "
                        "numbers only). Use it for any sum, difference or product you "
                        "want to state."),
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

DOMAIN_TOOLS = ["retrieve_cards", "predict_finish", "heat_effect", "pace_table"]


@lru_cache(maxsize=1)
def _retriever():
    from ..hybrid import HybridRetriever

    return HybridRetriever(mode=RETRIEVAL_MODE)


def retrieve_cards(query: str, k: int = 4) -> dict:
    k = max(1, min(int(k or 4), 8))
    hits = _retriever().search(query, k=k)
    return {"passages": [
        {"card": p.card, "title": p.title, "source": p.source, "text": p.text}
        for p in hits
    ]}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / ** ( ) are allowed")


def calculator(expression: str) -> dict:
    expr = str(expression).replace("^", "**")
    if len(expr) > 200:
        raise ValueError("expression too long")
    val = _eval(ast.parse(expr, mode="eval"))
    return {"expression": expression, "result": round(float(val), 4)}


FUNCS = {
    "retrieve_cards": retrieve_cards,
    "predict_finish": models.predict_finish,
    "heat_effect": models.heat_effect,
    "pace_table": pace_table,
    "calculator": calculator,
}


def call_tool(name: str, args: dict) -> dict:
    """Run a tool. Errors come back as {"error": ...} so the agent can recover."""
    if name not in FUNCS:
        return {"error": f"unknown tool {name!r}; choose one of {sorted(FUNCS)}"}
    try:
        return FUNCS[name](**(args or {}))
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:  # model unavailable, bad value, ...
        return {"error": f"{type(exc).__name__}: {exc}"}
