"""
Week 5 -- MCP server exposing WealthPilot's underwriting tools.

Written against mcp SDK 2.x's MCPServer, NOT the 1.x FastMCP API most
tutorials (and an earlier version of this project's own toolkit
reference doc) were written against. mcp 2.0.0 shipped 2026-07-28 --
after this assistant's training cutoff -- and renamed FastMCP to
MCPServer, moving mcp.server.fastmcp -> mcp.server.mcpserver. Confirmed
directly against the installed package (2.1.1) by trying the old import,
reading the ModuleNotFoundError it raises on purpose to point at this,
and checking call_tool()'s real signature -- not assumed from training
data. See docs/BUILD_LOG.md for the full story.

Design: build_server() is a factory, same DI shape as every prior week's
build_graph() / make_retrieve_policy_tool() -- it takes an ALREADY-BUILT
retrieve_policy tool (real index in production, fake-embedded index in
tests) rather than building one itself, so the MCP wiring is testable
without needing real embeddings.
"""

import json

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from .tools import calculate_ratios as _calculate_ratios_tool
from .tools import detect_red_flags as _detect_red_flags_tool


def build_server(retrieve_policy_tool) -> MCPServer:
    mcp = MCPServer("wealthpilot-tools")

    @mcp.tool()
    def calculate_ratios(statement_json: str) -> str:
        """Compute underwriting ratios from a financial statement.

        Input must be a JSON object with these fields: company_id, period,
        revenue, prior_period_revenue, cogs, operating_expenses, ebit,
        interest_expense, net_income, total_assets, current_assets,
        current_liabilities, total_liabilities, shareholder_equity,
        retained_earnings, total_debt_service.

        Returns DSCR, current_ratio, debt_to_equity, interest_coverage,
        revenue_growth_pct, altman_z, and altman_zone (safe/grey/distress).
        """
        # Delegates to the SAME LangChain tool object Week 2's agent and
        # Week 4's risk_analyst node call -- one implementation, three
        # different callers (LLM tool-calling, a plain Python node, and
        # now an MCP client), not three copies of the ratio logic.
        #
        # ValidationError is caught and re-raised as ToolError on purpose
        # -- confirmed directly that any OTHER exception type propagates
        # out of call_tool() as an opaque "Error executing tool" crash
        # with a full traceback, while ToolError comes back as a clean
        # is_error=True result with the actual message readable by
        # whatever model or client called this. A malformed statement
        # from a real client is an ANTICIPATED failure, not a crash.
        try:
            result = _calculate_ratios_tool.invoke({"statement_json": statement_json})
        except ValidationError as e:
            raise ToolError(f"Invalid financial statement: {e}") from e
        return json.dumps(result)

    @mcp.tool()
    def detect_red_flags(ratios_json: str) -> str:
        """Flag specific underwriting concerns from a ratios dict.

        Input is the JSON object returned by calculate_ratios (pass it
        through as-is). Returns a JSON array of concern strings, or an
        empty JSON array if none are breached.
        """
        try:
            return _detect_red_flags_tool.invoke({"ratios_json": ratios_json})
        except (ValueError, KeyError, TypeError) as e:
            raise ToolError(f"Invalid ratios JSON: {e}") from e

    @mcp.tool()
    def retrieve_policy(query: str, k: int = 4) -> str:
        """Retrieve relevant SME underwriting policy clauses for a risk
        question (e.g. "is high leverage a decline reason?"). Returns a
        JSON array of objects with 'text', 'source', and 'clause'.
        """
        return retrieve_policy_tool.invoke({"query": query, "k": k})

    return mcp


if __name__ == "__main__":
    # Live server -- needs real embeddings reachable over the network
    # (huggingface.co for the local model), unavailable in the sandbox
    # this was built in. Run this on your own machine, then point Claude
    # Desktop's MCP settings at it.
    from langchain_huggingface import HuggingFaceEmbeddings

    from .policy_docs import POLICY_DOCS
    from .rag import build_policy_index, make_retrieve_policy_tool

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    index = build_policy_index(POLICY_DOCS, embeddings)
    retrieve_tool = make_retrieve_policy_tool(index)

    server = build_server(retrieve_tool)
    server.run(transport="stdio")
