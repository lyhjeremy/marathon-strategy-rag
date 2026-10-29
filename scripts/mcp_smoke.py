"""Start the MCP server over stdio, list its tools and call two of them.

    python -m scripts.mcp_smoke        # writes eval/mcp_smoke.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import date

from mcp import ClientSession, StdioServerParameters, stdio_client

from src.config import ROOT

CALLS = [
    ("pace_table", {"goal_time": "3:45", "strategy": "even"}),
    ("retrieve_cards", {"query": "how much does heat slow a marathon", "k": 2}),
]


async def main() -> dict:
    params = StdioServerParameters(command=sys.executable,
                                   args=[str(ROOT / "src" / "mcp_server.py")], cwd=str(ROOT))
    t0 = time.perf_counter()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = await session.list_tools()
            out = {"date": date.today().isoformat(),
                   "server": init.server_info.name,
                   "startup_s": round(time.perf_counter() - t0, 2),
                   "tools": [t.name for t in listed.tools], "calls": []}
            for name, args in CALLS:
                t1 = time.perf_counter()
                res = await session.call_tool(name, args)
                text = "".join(getattr(c, "text", "") for c in res.content)
                out["calls"].append({"tool": name, "args": args, "is_error": res.is_error,
                                     "seconds": round(time.perf_counter() - t1, 2),
                                     "result": json.loads(text) if text.startswith("{") else text})
    return out


if __name__ == "__main__":
    result = asyncio.run(main())
    (ROOT / "eval" / "mcp_smoke.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "calls"}, indent=1))
    for c in result["calls"]:
        print(c["tool"], "error" if c["is_error"] else "ok", c["seconds"], "s",
              json.dumps(c["result"])[:200])
