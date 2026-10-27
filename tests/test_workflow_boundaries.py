import json
import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from wealthpilot.multi_agent import make_final_review
from wealthpilot.case_graph import build_case_graph, run_case
from wealthpilot.schemas import FinancialStatement
from wealthpilot.ratios import compute_ratios
from wealthpilot.synthetic_data import generate_synthetic_case
from wealthpilot.rag import build_policy_index
from test_case_graph import FakeCaseModel, FakeEmbeddings


@tool
def no_policy(query: str, k: int = 4) -> str:
    """Return no policy findings for a controlled graph test."""
    return "[]"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Decline: cannot approve this case", "decline"),
        ("Review: compare approval and decline options", "review"),
        ("Do not approve this application", "review"),
        ("No verdict supplied", "review"),
    ],
)
def test_text_verdict_is_never_selected_by_substring_priority(text, expected):
    class Model:
        def invoke(self, _):
            return AIMessage(content=text)

    result = make_final_review(Model())(
        {"ratios": {}, "red_flags": [], "policy_findings": []}
    )
    assert result["recommended_verdict"] == expected
    if not text.startswith(("Decline:", "Review:")):
        assert "recommendation_error" in result


@pytest.mark.parametrize(
    "verdict,expected", [("decline", "decline"), ("banana", "review")]
)
def test_structured_recommendation_schema_checks_enum(verdict, expected):
    class Model:
        def with_structured_output(self, schema):
            return self

        def invoke(self, _):
            return {"verdict": verdict, "rationale": "Based only on supplied evidence"}

    assert make_final_review(Model())({})["recommended_verdict"] == expected


def fresh_case():
    app = build_case_graph(FakeCaseModel(), no_policy, InMemorySaver())
    config = {"configurable": {"thread_id": "boundary"}}
    statement = generate_synthetic_case("healthy", 1)
    run_case(
        app, config, case_id="boundary", raw_statement_json=statement.model_dump_json()
    )
    return app, config, statement


@pytest.mark.parametrize(
    "decision",
    [
        {"approved": "false"},
        {"approved": 1},
        {"notes": "missing decision"},
        {"approved": True, "override_verdict": "banana"},
        {"approved": False, "override_verdict": "approve"},
    ],
)
def test_malformed_human_decision_cannot_set_final_status(decision):
    app, config, _ = fresh_case()
    with pytest.raises(ValueError):
        app.invoke(Command(resume=decision), config=config)
    assert "final_status" not in app.get_state(config).values


def test_revised_statement_requires_new_revision_and_recomputes():
    app, config, statement = fresh_case()
    app.invoke(Command(resume={"approved": True}), config=config)
    changed = generate_synthetic_case("overleveraged", 2)
    with pytest.raises(ValueError, match="new thread_id/revision"):
        run_case(app, config, raw_statement_json=changed.model_dump_json())
    assert (
        app.get_state(config).values["raw_statement_json"]
        == statement.model_dump_json()
    )
    new = run_case(
        app,
        {"configurable": {"thread_id": "boundary-r2"}},
        case_id="boundary-r2",
        raw_statement_json=changed.model_dump_json(),
    )
    assert new["ratios"] == compute_ratios(changed)
    assert "final_status" not in new


def test_wrapper_rejects_caller_injected_derived_state():
    app, config, _ = fresh_case()
    with pytest.raises(ValueError, match="derived state"):
        run_case(app, config, final_status="approve")


@pytest.mark.parametrize(
    "field",
    ["revenue", "ebit", "net_income", "shareholder_equity", "total_debt_service"],
)
def test_nonfinite_numbers_never_enter_financial_engine(field):
    data = generate_synthetic_case("healthy", 1).model_dump()
    data[field] = float("nan")
    with pytest.raises(ValidationError):
        FinancialStatement(**data)


def test_zero_denominators_are_explicit_and_strict_json_safe():
    data = generate_synthetic_case("healthy", 1).model_dump()
    data.update(
        total_debt_service=0,
        current_liabilities=0,
        total_liabilities=0,
        shareholder_equity=0,
        interest_expense=0,
        prior_period_revenue=0,
    )
    ratios = compute_ratios(FinancialStatement(**data))
    assert ratios["dscr"] is None and ratios["altman_zone"] == "undefined"
    assert "Infinity" not in json.dumps(ratios, allow_nan=False)
    assert ratios["equity_nonpositive"] is True


def test_negative_book_equity_gets_manual_review_flag():
    from wealthpilot.tools import detect_red_flags

    data = generate_synthetic_case("healthy", 1).model_dump()
    data["shareholder_equity"] = -1000
    flags = json.loads(
        detect_red_flags.invoke(
            {"ratios_json": json.dumps(compute_ratios(FinancialStatement(**data)))}
        )
    )
    assert any("Nonpositive book equity" in flag for flag in flags)


def test_boolean_financial_amount_is_rejected():
    data = generate_synthetic_case("healthy", 1).model_dump()
    data["total_debt_service"] = True
    with pytest.raises(ValidationError):
        FinancialStatement(**data)


def test_malformed_ratio_values_are_rejected_by_public_tool():
    from wealthpilot.tools import detect_red_flags

    ratios = compute_ratios(generate_synthetic_case("healthy", 1))
    ratios["dscr"] = float("nan")
    with pytest.raises(ValidationError):
        detect_red_flags.invoke({"ratios_json": json.dumps(ratios)})


def test_offline_demo_pauses_and_explicitly_sends_case_back():
    from wealthpilot.offline_demo import run_demo

    result = run_demo()
    assert result["measures_live_llm_quality"] is False
    assert result["paused_for_human"] is True
    assert result["final_status"] == "sent_back"


def test_two_policy_indexes_cannot_silently_share_default_collection():
    first = build_policy_index(
        [{"source": "first", "text": "## 1.1 First-only policy"}], FakeEmbeddings()
    )
    second = build_policy_index(
        [{"source": "second", "text": "## 2.1 Second-only policy"}], FakeEmbeddings()
    )
    assert {d.metadata["source"] for d in first.similarity_search("policy", k=5)} == {
        "first"
    }
    assert {d.metadata["source"] for d in second.similarity_search("policy", k=5)} == {
        "second"
    }
