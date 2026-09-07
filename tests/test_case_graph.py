"""
Week 4 — Multi-agent orchestration + persistent case memory.

Run with: pytest test_week4.py -v

Updated for Week 6: a case no longer completes in one .invoke() call --
human_approval's interrupt() pauses it first. Every test here now does
invoke() -> confirm it paused with the expected recommendation -> resume
with Command(resume=...) -> confirm the final outcome. That two-step
shape isn't incidental to these tests; it's the real behavior Week 6
added, and pretending otherwise would make these tests pass against code
that doesn't match how the graph actually runs.

Same fake-first philosophy as every prior week: FakeCaseModel and
FakeEmbeddings prove the ORCHESTRATION and PERSISTENCE mechanics work,
deterministically, with no API key and no network call.
"""

import hashlib
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from wealthpilot.case_graph import build_case_graph, run_case, is_awaiting_approval
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
    def test_healthy_case_skips_policy_lookup_and_pauses_for_approval(self, retrieve_tool):
        """No red flags -> credit_researcher should return [] directly,
        never even calling the model -- there's nothing to ask it. The
        case should still PAUSE at human_approval, not skip straight to
        a final answer -- an AI recommendation is never self-approving."""
        model = FakeCaseModel()
        app = build_case_graph(model, retrieve_tool, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "H-1"}}
        fs = generate_synthetic_case("healthy", 1)

        paused = run_case(app, config, case_id="H-1", raw_statement_json=fs.model_dump_json())
        assert is_awaiting_approval(paused)
        assert paused["policy_findings"] == []
        assert model.retrieve_calls == 0  # the actual point of this sub-check
        assert paused["recommended_verdict"] in ("approve", "review", "decline")
        assert "final_status" not in paused  # not decided yet -- only recommended

        approved = app.invoke(Command(resume={"approved": True}), config=config)
        assert approved["final_status"] == paused["recommended_verdict"]

    def test_flagged_case_triggers_policy_lookup_then_pauses(self, retrieve_tool):
        model = FakeCaseModel(verdict_word="decline")
        app = build_case_graph(model, retrieve_tool, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "O-1"}}
        fs = generate_synthetic_case("overleveraged", 1)

        paused = run_case(app, config, case_id="O-1", raw_statement_json=fs.model_dump_json())
        assert model.retrieve_calls == 1
        assert len(paused["policy_findings"]) > 0
        assert paused["recommended_verdict"] == "decline"
        assert is_awaiting_approval(paused)

    def test_invalid_input_short_circuits_before_risk_analyst_or_approval(self, retrieve_tool):
        """A malformed statement should fail at document_parser and never
        reach risk_analyst OR human_approval -- there's nothing for a
        human to review when the input itself doesn't parse."""
        model = FakeCaseModel()
        app = build_case_graph(model, retrieve_tool, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "BAD-1"}}
        result = run_case(app, config, case_id="BAD-1", raw_statement_json=json.dumps({"company_id": "x"}))

        assert result["final_status"] == "rejected_invalid_data"
        assert not is_awaiting_approval(result)  # never paused -- never got that far
        assert "risk_analyst_runs" not in result
        assert model.retrieve_calls == 0
        assert model.final_calls == 0


class TestHumanApproval:
    def test_is_awaiting_approval_is_false_before_and_after_the_pause(self, retrieve_tool):
        model = FakeCaseModel()
        app = build_case_graph(model, retrieve_tool, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "FLAG-1"}}

        # before the case even exists, an empty dict is obviously not "awaiting"
        assert is_awaiting_approval({}) is False

        fs = generate_synthetic_case("healthy", 1)
        paused = run_case(app, config, case_id="FLAG-1", raw_statement_json=fs.model_dump_json())
        assert is_awaiting_approval(paused) is True

        done = app.invoke(Command(resume={"approved": True}), config=config)
        assert is_awaiting_approval(done) is False

    def test_send_back_records_reviewer_notes_and_does_not_set_a_verdict_as_final(self, retrieve_tool):
        model = FakeCaseModel(verdict_word="approve")
        app = build_case_graph(model, retrieve_tool, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "SENDBACK-1"}}

        fs = generate_synthetic_case("healthy", 1)
        run_case(app, config, case_id="SENDBACK-1", raw_statement_json=fs.model_dump_json())

        result = app.invoke(
            Command(resume={"approved": False, "notes": "want a second opinion"}), config=config
        )
        assert result["final_status"] == "sent_back"
        assert result["final_status"] != "approve"  # the AI's recommendation, explicitly NOT taken
        assert result["reviewer_notes"] == "want a second opinion"

    def test_override_verdict_replaces_the_ai_recommendation(self, retrieve_tool):
        model = FakeCaseModel(verdict_word="review")
        app = build_case_graph(model, retrieve_tool, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "OVERRIDE-1"}}

        fs = generate_synthetic_case("illiquid", 1)
        paused = run_case(app, config, case_id="OVERRIDE-1", raw_statement_json=fs.model_dump_json())
        assert paused["recommended_verdict"] == "review"

        result = app.invoke(
            Command(resume={"approved": True, "override_verdict": "approve", "notes": "seasonal, verified"}),
            config=config,
        )
        assert result["final_status"] == "approve"
        assert result["recommended_verdict"] == "review"  # original AI recommendation preserved, not overwritten





class TestPersistence:
    def test_second_invoke_on_a_completed_case_does_not_recompute_anything(self, retrieve_tool, tmp_path):
        db_path = str(tmp_path / "cases.sqlite")
        fs = generate_synthetic_case("illiquid", 1)
        config = {"configurable": {"thread_id": "PERSIST-1"}}

        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            model = FakeCaseModel(verdict_word="review")
            app = build_case_graph(model, retrieve_tool, checkpointer=checkpointer)

            paused = run_case(
                app, config, case_id="PERSIST-1", raw_statement_json=fs.model_dump_json()
            )
            assert paused["document_parser_runs"] == 1
            assert paused["risk_analyst_runs"] == 1

            first = app.invoke(Command(resume={"approved": True}), config=config)
            assert first["final_status"] == "review"

            # invoke AGAIN, same thread_id, nothing new -- supervisor sees
            # final_status already set and ends immediately. invoke()
            # itself would return None here (see run_case's docstring).
            second = run_case(app, config)
            assert second["document_parser_runs"] == 1  # NOT 2 -- didn't re-parse
            assert second["risk_analyst_runs"] == 1  # NOT 2 -- didn't recompute ratios
            assert second["final_status"] == first["final_status"]

    def test_persistence_survives_a_fresh_graph_object_not_just_a_fresh_invoke(self, retrieve_tool, tmp_path):
        """The real test: a completely NEW build_case_graph() call, as if
        the process had restarted, pointed at the SAME sqlite file, must
        still see the prior case's state -- including a still-paused
        interrupt, resumed on a graph object that never saw the original
        invoke() at all."""
        db_path = str(tmp_path / "cases.sqlite")
        fs = generate_synthetic_case("declining", 1)
        config = {"configurable": {"thread_id": "RESTART-1"}}

        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            model_a = FakeCaseModel(verdict_word="decline")
            app_a = build_case_graph(model_a, retrieve_tool, checkpointer=checkpointer)
            paused = run_case(
                app_a, config, case_id="RESTART-1", raw_statement_json=fs.model_dump_json()
            )
            assert paused["recommended_verdict"] == "decline"

        # simulate a fresh process: new checkpointer connection to the
        # SAME file, new model instance, brand new graph object -- and
        # resume the interrupt on THIS object, which never ran the
        # original invoke() at all
        with SqliteSaver.from_conn_string(db_path) as checkpointer_2:
            model_b = FakeCaseModel(verdict_word="approve")  # different fake -- proves it's NOT re-run
            app_b = build_case_graph(model_b, retrieve_tool, checkpointer=checkpointer_2)
            result = app_b.invoke(Command(resume={"approved": True}), config=config)

            assert result["final_status"] == "decline"  # model_a's recommendation, not model_b's
            assert result["document_parser_runs"] == 1
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
