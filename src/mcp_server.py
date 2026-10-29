"""MCP server: the coach's four tools over stdio, for Claude Desktop or Claude Code.

Uses the official `mcp` Python SDK (version 2, where FastMCP became MCPServer).
The tools are the same functions the agent loop calls (`src/tools`), so an
MCP client gets the same numbers the coach does.

    python src/mcp_server.py          # speaks MCP on stdin/stdout

Setup snippets for Claude Desktop and `claude mcp add` are in the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Runnable as a plain script (`python src/mcp_server.py`) from any directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from src import tools  # noqa: E402

server = MCPServer(
    name="marathon-coach",
    log_level="WARNING",
    instructions=(
        "Marathon race-strategy tools. predict_finish and heat_effect are statistical "
        "models fitted on Boston Marathon data; pace_table is arithmetic on a goal time; "
        "retrieve_cards searches 14 knowledge cards. Quote the numbers these tools return "
        "rather than estimating your own."),
)

_DESC = {t["name"]: t["description"] for t in tools.TOOL_SPECS}


@server.tool(description=_DESC["retrieve_cards"])
def retrieve_cards(query: str, k: int = 4) -> dict:
    return tools.call_tool("retrieve_cards", {"query": query, "k": k})


@server.tool(description=_DESC["predict_finish"] + (
    " splits: cumulative elapsed minutes keyed by km mark (5, 10, 15, 20, 21 for "
    "halfway, 25, 30, 35, 40), e.g. {\"10\": 47.5, \"21\": 102}. gender: 'M' or 'F'."))
def predict_finish(splits: dict[str, float], gender: str | None = None,
                   age: int | None = None) -> dict:
    return tools.call_tool("predict_finish", {"splits": splits, "gender": gender, "age": age})


@server.tool(description=_DESC["heat_effect"])
def heat_effect(finish_min: float, temp_f: float, dewpoint_f: float | None = None,
                headwind_mph: float | None = None) -> dict:
    return tools.call_tool("heat_effect", {"finish_min": finish_min, "temp_f": temp_f,
                                           "dewpoint_f": dewpoint_f,
                                           "headwind_mph": headwind_mph})


@server.tool(description=_DESC["pace_table"] + " goal_time: 'h:mm', 'h:mm:ss' or minutes.")
def pace_table(goal_time: str, strategy: str = "even") -> dict:
    return tools.call_tool("pace_table", {"goal_time": goal_time, "strategy": strategy})


if __name__ == "__main__":
    server.run("stdio")
