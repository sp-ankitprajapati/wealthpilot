"""Controlled tests of the single-agent tool-calling loop."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from wealthpilot.agent import build_graph
from wealthpilot.schemas import FinancialStatement
from wealthpilot.ratios import compute_ratios
from wealthpilot.synthetic_data import generate_synthetic_case


class FakeToolCallingModel:
    """Returns a tool call on the first invocation, then a plain-text
    final answer on the second — the minimum needed to exercise the real
    graph loop without a real API behind it."""

    def __init__(self, statement_json: str, tool_sequence=("calculate_ratios", "detect_red_flags")):
        self._statement_json = statement_json
        self._tool_sequence = tool_sequence
        self._call_count = 0

    def bind_tools(self, tools):
        return self  # no-op — this fake always follows its fixed script

    def invoke(self, messages):
        self._call_count += 1
        step = self._call_count - 1

        if step < len(self._tool_sequence):
            tool_name = self._tool_sequence[step]
            if tool_name == "calculate_ratios":
                args = {"statement_json": self._statement_json}
            else:  # detect_red_flags — needs the PRIOR tool's actual output
                last_tool_msg = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
                args = {"ratios_json": last_tool_msg.content}
            return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": f"call_{step}"}])

        # script exhausted — final answer, no more tool calls
        last_tool_msg = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
        return AIMessage(content=f"Risk summary based on red flags: {last_tool_msg.content}")


def _invoke_fake_agent(fs: FinancialStatement):
    fake_model = FakeToolCallingModel(fs.model_dump_json())
    app = build_graph(fake_model)
    return app.invoke({"messages": [HumanMessage(content="Assess this applicant.")]})


class TestGraphControlFlow:
    def test_loop_terminates_with_a_final_text_answer(self):
        fs = generate_synthetic_case("healthy", 1)
        result = _invoke_fake_agent(fs)
        final = result["messages"][-1]
        assert isinstance(final, AIMessage)
        assert not final.tool_calls  # must have stopped, not still asking for tools
        assert len(final.content) > 0

    def test_both_tools_actually_ran(self):
        """Not just that the graph terminated — that BOTH tools actually
        executed and their real outputs are in the message history, not
        placeholder text."""
        fs = generate_synthetic_case("overleveraged", 1)
        result = _invoke_fake_agent(fs)
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 2

        ratios_msg, flags_msg = tool_messages
        expected_ratios = compute_ratios(fs)
        # the tool's real output made it into the graph state — not a stub
        assert json.loads(ratios_msg.content)["dscr"] == expected_ratios["dscr"]
        assert "DSCR" in flags_msg.content  # overleveraged always trips this flag

    def test_healthy_case_produces_no_flags_in_the_transcript(self):
        fs = generate_synthetic_case("healthy", 1)
        result = _invoke_fake_agent(fs)
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        flags_msg = tool_messages[1]
        assert json.loads(flags_msg.content) == []

    def test_final_message_references_real_tool_output_not_a_hallucination(self):
        """A crude but useful check: the fake model's final answer is
        BUILT from the last ToolMessage's actual content, so this proves
        the tool output reached the final node — a real model failing to
        ground its answer in tool output would fail an equivalent check
        against a real transcript."""
        fs = generate_synthetic_case("declining", 1)
        result = _invoke_fake_agent(fs)
        final = result["messages"][-1]
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert tool_messages[-1].content in final.content

    @pytest.mark.parametrize("profile", ["healthy", "illiquid", "overleveraged", "declining"])
    def test_graph_runs_clean_for_every_profile(self, profile):
        """Not a judgment-quality test — just proves no profile's data
        shape breaks the tool-calling loop itself (e.g. an inf value from
        zero total_debt_service surviving JSON round-tripping)."""
        fs = generate_synthetic_case(profile, 1)
        result = _invoke_fake_agent(fs)
        assert result["messages"][-1].tool_calls == [] or result["messages"][-1].tool_calls is None
