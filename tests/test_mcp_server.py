"""MCP 2.x tool registration, outputs and in-process error checks.

These tests do not exercise a client transport connection.
"""

import asyncio
import hashlib
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from wealthpilot.mcp_server import build_server
from wealthpilot.policy_docs import POLICY_DOCS
from wealthpilot.rag import build_policy_index, make_retrieve_policy_tool
from wealthpilot.ratios import compute_ratios
from wealthpilot.synthetic_data import generate_synthetic_case
from wealthpilot.tools import calculate_ratios as lc_calculate_ratios
from wealthpilot.tools import detect_red_flags as lc_detect_red_flags


class FakeEmbeddings:
    """Deterministic bag-of-words test embeddings for MCP tool checks."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture(scope="module")
def server():
    index = build_policy_index(POLICY_DOCS, FakeEmbeddings())
    retrieve_tool = make_retrieve_policy_tool(index)
    return build_server(retrieve_tool)


def run(coro):
    """pytest here has no async plugin installed -- wrapping each call in
    asyncio.run() avoids adding a dependency for three tests' worth of
    async calls."""
    return asyncio.run(coro)


class TestRegistration:
    def test_all_three_tools_are_registered(self, server):
        tools = run(server.list_tools())
        assert {t.name for t in tools} == {"calculate_ratios", "detect_red_flags", "retrieve_policy"}


class TestHappyPath:
    def test_calculate_ratios_via_mcp_matches_direct_tool_call(self, server):
        fs = generate_synthetic_case("overleveraged", 1)
        expected = compute_ratios(fs)

        result = run(server.call_tool("calculate_ratios", {"statement_json": fs.model_dump_json()}))
        assert result.is_error is False
        assert json.loads(result.content[0].text) == expected

    def test_detect_red_flags_via_mcp_matches_direct_tool_call(self, server):
        fs = generate_synthetic_case("declining", 1)
        ratios_json = json.dumps(compute_ratios(fs))
        expected = json.loads(lc_detect_red_flags.invoke({"ratios_json": ratios_json}))

        result = run(server.call_tool("detect_red_flags", {"ratios_json": ratios_json}))
        assert result.is_error is False
        assert json.loads(result.content[0].text) == expected

    def test_retrieve_policy_via_mcp_returns_citable_clauses(self, server):
        result = run(server.call_tool("retrieve_policy", {"query": "leverage debt equity", "k": 2}))
        assert result.is_error is False
        findings = json.loads(result.content[0].text)
        assert len(findings) == 2
        assert all({"text", "source", "clause"} <= set(f.keys()) for f in findings)
        with pytest.raises(ToolError) as exc_info:
            run(server.call_tool("retrieve_policy", {"query": "leverage", "k": 0}))
        assert not isinstance(exc_info.value, UnexpectedToolError)

    def test_healthy_case_end_to_end_through_mcp_produces_no_flags(self, server):
        """The same three-tool chain an MCP client would actually make:
        calculate_ratios -> detect_red_flags -> (skip retrieve_policy,
        nothing to look up) -- proving the tools compose correctly when
        called the way a real client calls them, not just individually."""
        fs = generate_synthetic_case("healthy", 1)
        ratios_result = run(server.call_tool("calculate_ratios", {"statement_json": fs.model_dump_json()}))
        flags_result = run(server.call_tool("detect_red_flags", {"ratios_json": ratios_result.content[0].text}))
        assert json.loads(flags_result.content[0].text) == []


class TestAnticipatedFailures:
    """Every case here is a ToolError -- an anticipated failure, not a
    crash -- confirmed by checking the exact exception type, not just
    that *something* was raised."""

    def test_malformed_statement_json_raises_tool_error(self, server):
        with pytest.raises(ToolError) as exc_info:
            run(server.call_tool("calculate_ratios", {"statement_json": "{not valid json"}))
        assert not isinstance(exc_info.value, UnexpectedToolError)  # anticipated, not a crash
        assert "invalid financial statement" in str(exc_info.value).lower()

    def test_negative_revenue_raises_tool_error(self, server):
        bad_statement = json.dumps({
            "company_id": "x", "period": "FY25", "revenue": -100,
            "prior_period_revenue": 100, "cogs": 1, "operating_expenses": 1,
            "ebit": 1, "interest_expense": 1, "net_income": 1, "total_assets": 100,
            "current_assets": 1, "current_liabilities": 1, "total_liabilities": 1,
            "shareholder_equity": 1, "retained_earnings": 1, "total_debt_service": 1,
        })
        with pytest.raises(ToolError) as exc_info:
            run(server.call_tool("calculate_ratios", {"statement_json": bad_statement}))
        assert not isinstance(exc_info.value, UnexpectedToolError)

    def test_malformed_ratios_json_raises_tool_error(self, server):
        with pytest.raises(ToolError) as exc_info:
            run(server.call_tool("detect_red_flags", {"ratios_json": "not json at all"}))
        assert not isinstance(exc_info.value, UnexpectedToolError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
