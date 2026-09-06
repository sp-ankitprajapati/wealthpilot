"""
Week 4 — Multi-agent orchestration + persistent case memory.

Run with: pytest test_week4.py -v

Same fake-first philosophy as every prior week: FakeCaseModel and
FakeEmbeddings prove the ORCHESTRATION and PERSISTENCE mechanics work,
deterministically, with no API key and no network call. Whether a real
model's policy-lookup reasoning and final verdict are actually GOOD needs
a live run with a real provider (see case_graph.py used from agent.py's
provider factory) -- not this test suite.

The persistence tests are the point of this week: they don't just check
that compile(checkpointer=...) doesn't crash, they PROVE a resumed case
doesn't recompute anything, using call counters as the evidence -- and
prove it survives a completely fresh graph object, not just a second
.invoke() on the same Python object, since a real restart rebuilds the
graph from scratch too.
"""

import hashlib
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from wealthpilot.case_graph import build_case_graph, run_case
from wealthpilot.policy_docs import POLICY_DOCS
from wealthpilot.rag import build_policy_index, make_retrieve_policy_tool
from wealthpilot.synthetic_data import generate_synthetic_case


class FakeEmbeddings:
    """Same bag-of-words fake as test_week3.py -- real embeddings still
    need huggingface.co, unavailable in this sandbox."""

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


class FakeCaseModel:
    """Branches on prompt CONTENT rather than call order, since the same
    model instance is bound differently at two different node call
    sites (credit_researcher binds a tool, final_review doesn't) --
    content is the only reliable signal for which node is calling."""

    def __init__(self, verdict_word: str = "review"):
        self.verdict_word = verdict_word
        self.retrieve_calls = 0
        self.final_calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        content = messages[-1].content
        if "retrieve_policy tool" in content:
            self.retrieve_calls += 1
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "retrieve_policy",
                    "args": {"query": "leverage debt equity coverage", "k": 3},
                    "id": f"call_{self.retrieve_calls}",
                }],
            )
        self.final_calls += 1
        return AIMessage(content=f"{self.verdict_word.capitalize()}: per the ratios and policy findings above.")


@pytest.fixture(scope="module")
def retrieve_tool():
    index = build_policy_index(POLICY_DOCS, FakeEmbeddings())
    return make_retrieve_policy_tool(index)


class TestOrchestration:
    def test_healthy_case_skips_policy_lookup_without_calling_the_model(self, retrieve_tool):
        """No red flags -> credit_researcher should return [] directly,
        never even calling the model -- there's nothing to ask it."""
        model = FakeCaseModel()
        app = build_case_graph(model, retrieve_tool)
        fs = generate_synthetic_case("healthy", 1)
        result = app.invoke({"case_id": "H-1", "raw_statement_json": fs.model_dump_json()})

        assert result["policy_findings"] == []
        assert model.retrieve_calls == 0  # the actual point of this test
        assert result["verdict"] in ("approve", "review", "decline")

    def test_flagged_case_triggers_policy_lookup_and_reaches_a_verdict(self, retrieve_tool):
        model = FakeCaseModel(verdict_word="decline")
        app = build_case_graph(model, retrieve_tool)
        fs = generate_synthetic_case("overleveraged", 1)
        result = app.invoke({"case_id": "O-1", "raw_statement_json": fs.model_dump_json()})

        assert model.retrieve_calls == 1
        assert len(result["policy_findings"]) > 0
        assert result["verdict"] == "decline"

    def test_invalid_input_short_circuits_before_risk_analyst_ever_runs(self, retrieve_tool):
        """A malformed statement should fail at document_parser and never
        reach risk_analyst at all -- proving the supervisor's verdict-set
        early exit actually works, not just that it compiles."""
        model = FakeCaseModel()
        app = build_case_graph(model, retrieve_tool)
        result = app.invoke({"case_id": "BAD-1", "raw_statement_json": json.dumps({"company_id": "x"})})

        assert result["verdict"] == "rejected_invalid_data"
        assert "risk_analyst_runs" not in result  # never ran
        assert model.retrieve_calls == 0
        assert model.final_calls == 0


class TestPersistence:
    def test_second_invoke_on_a_completed_case_does_not_recompute_anything(self, retrieve_tool, tmp_path):
        db_path = str(tmp_path / "cases.sqlite")
        fs = generate_synthetic_case("illiquid", 1)
        config = {"configurable": {"thread_id": "PERSIST-1"}}

        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            model = FakeCaseModel(verdict_word="review")
            app = build_case_graph(model, retrieve_tool, checkpointer=checkpointer)

            first = run_case(
                app, config, case_id="PERSIST-1", raw_statement_json=fs.model_dump_json()
            )
            assert first["document_parser_runs"] == 1
            assert first["risk_analyst_runs"] == 1

            # invoke AGAIN, same thread_id, no new financial data -- the
            # supervisor should see verdict already set and end immediately.
            # invoke() itself would return None here (see run_case's
            # docstring) -- this is exactly the case it exists to handle.
            second = run_case(app, config)
            assert second["document_parser_runs"] == 1  # NOT 2 -- didn't re-parse
            assert second["risk_analyst_runs"] == 1  # NOT 2 -- didn't recompute ratios
            assert second["verdict"] == first["verdict"]

    def test_persistence_survives_a_fresh_graph_object_not_just_a_fresh_invoke(self, retrieve_tool, tmp_path):
        """The real test: a completely NEW build_case_graph() call, as if
        the process had restarted, pointed at the SAME sqlite file, must
        still see the prior case's state. If this failed and the earlier
        test passed, persistence would only be an in-memory illusion."""
        db_path = str(tmp_path / "cases.sqlite")
        fs = generate_synthetic_case("declining", 1)
        config = {"configurable": {"thread_id": "RESTART-1"}}

        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            model_a = FakeCaseModel(verdict_word="decline")
            app_a = build_case_graph(model_a, retrieve_tool, checkpointer=checkpointer)
            first = run_case(
                app_a, config, case_id="RESTART-1", raw_statement_json=fs.model_dump_json()
            )
            assert first["verdict"] == "decline"

        # simulate a fresh process: new checkpointer connection to the
        # SAME file, new model instance, brand new graph object
        with SqliteSaver.from_conn_string(db_path) as checkpointer_2:
            model_b = FakeCaseModel(verdict_word="review")  # different fake -- proves it's NOT re-run
            app_b = build_case_graph(model_b, retrieve_tool, checkpointer=checkpointer_2)
            second = run_case(app_b, config)

            assert second["verdict"] == "decline"  # model_a's verdict, not model_b's
            assert second["document_parser_runs"] == 1
            assert model_b.final_calls == 0  # proves final_review did NOT re-run on the new model

    def test_two_thread_ids_are_fully_isolated(self, retrieve_tool, tmp_path):
        db_path = str(tmp_path / "cases.sqlite")
        fs_a = generate_synthetic_case("healthy", 1)
        fs_b = generate_synthetic_case("declining", 1)

        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            model = FakeCaseModel()
            app = build_case_graph(model, retrieve_tool, checkpointer=checkpointer)

            result_a = app.invoke(
                {"case_id": "CASE-A", "raw_statement_json": fs_a.model_dump_json()},
                config={"configurable": {"thread_id": "CASE-A"}},
            )
            result_b = app.invoke(
                {"case_id": "CASE-B", "raw_statement_json": fs_b.model_dump_json()},
                config={"configurable": {"thread_id": "CASE-B"}},
            )

            assert result_a["financial_statement"]["company_id"] != result_b["financial_statement"]["company_id"]
            assert result_a["red_flags"] == []  # healthy
            assert len(result_b["red_flags"]) > 0  # declining


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
