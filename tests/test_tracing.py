"""
Week 7 — Observability.

Run with: pytest tests/test_tracing.py -v

The last test in this file (interrupt-safety) locks in something
confirmed manually first with a minimal reproduction (see
docs/BUILD_LOG.md, Week 7): wrapping an interrupt()-based node in
traced_node() does not break pause/resume. That finding is now a real,
running test against the actual case_graph, not just a one-off repro.
"""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from wealthpilot.case_graph import build_case_graph, run_case
from wealthpilot.synthetic_data import generate_synthetic_case
from wealthpilot.tracing import traced_node

from test_eval_harness import RuleBasedFakeModel, FakeEmbeddings
from wealthpilot.policy_docs import POLICY_DOCS
from wealthpilot.rag import build_policy_index, make_retrieve_policy_tool


@pytest.fixture(scope="module")
def retrieve_tool():
    index = build_policy_index(POLICY_DOCS, FakeEmbeddings())
    return make_retrieve_policy_tool(index)


class TestTracing:
    def test_traced_node_logs_node_name_case_id_and_latency(self, caplog):
        def dummy_node(state):
            return {"ratios": {"dscr": 1.5}}

        wrapped = traced_node(dummy_node, "dummy_node")
        with caplog.at_level("INFO", logger="wealthpilot"):
            wrapped({"case_id": "TRACE-1"})

        assert len(caplog.records) == 1
        entry = json.loads(caplog.records[0].message)
        assert entry["node"] == "dummy_node"
        assert entry["case_id"] == "TRACE-1"
        assert "latency_s" in entry

    def test_traced_node_handles_missing_case_id_without_crashing(self, caplog):
        wrapped = traced_node(lambda state: {}, "no_case_id_node")
        with caplog.at_level("INFO", logger="wealthpilot"):
            wrapped({})  # no case_id key at all
        entry = json.loads(caplog.records[0].message)
        assert entry["case_id"] is None

    def test_traced_node_extracts_token_usage_when_present(self, caplog):
        class FakeMessageWithUsage:
            usage_metadata = {"input_tokens": 120, "output_tokens": 45}

        def node_with_usage(state):
            return {"messages": [FakeMessageWithUsage()]}

        wrapped = traced_node(node_with_usage, "llm_node")
        with caplog.at_level("INFO", logger="wealthpilot"):
            wrapped({"case_id": "X"})
        entry = json.loads(caplog.records[0].message)
        assert entry["input_tokens"] == 120
        assert entry["output_tokens"] == 45

    def test_traced_node_does_not_break_a_real_interrupt_flow(self, retrieve_tool):
        """Confirmed manually earlier (see docs/BUILD_LOG.md, Week 7)
        that wrapping an interrupt()-based node is safe -- this locks
        that finding in as a real, running test against the actual
        case_graph, not just the minimal repro used to first check it."""

        model = RuleBasedFakeModel()
        app = build_case_graph(model, retrieve_tool, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "TRACE-INTERRUPT-1"}}
        fs = generate_synthetic_case("healthy", 1)

        paused = run_case(app, config, case_id="T1", raw_statement_json=fs.model_dump_json())
        assert "__interrupt__" in paused  # still pauses correctly, tracing wrapper didn't break it

        result = app.invoke(Command(resume={"approved": True}), config=config)
        assert result["final_status"] == "approve"  # still resumes and completes correctly





if __name__ == "__main__":
    pytest.main([__file__, "-v"])
