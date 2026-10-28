"""Checkpointed case graph with explicit human approval.

build_case_graph connects the specialist nodes through a supervisor router.
A supplied checkpointer preserves state across calls and review interruptions.
"""

from langgraph.graph import StateGraph, END

from .multi_agent import (
    CaseState,
    document_parser,
    risk_analyst,
    make_credit_researcher,
    make_final_review,
    human_approval,
    supervisor,
)
from .tracing import traced_node


def build_case_graph(model, retrieve_policy_tool, checkpointer=None):
    graph = StateGraph(CaseState)
    graph.add_node("document_parser", traced_node(document_parser, "document_parser"))
    graph.add_node("risk_analyst", traced_node(risk_analyst, "risk_analyst"))
    graph.add_node(
        "credit_researcher",
        traced_node(
            make_credit_researcher(model, retrieve_policy_tool), "credit_researcher"
        ),
    )
    graph.add_node(
        "final_review", traced_node(make_final_review(model), "final_review")
    )
    graph.add_node("human_approval", traced_node(human_approval, "human_approval"))

    graph.set_conditional_entry_point(supervisor)
    for node in (
        "document_parser",
        "risk_analyst",
        "credit_researcher",
        "final_review",
        "human_approval",
    ):
        graph.add_conditional_edges(node, supervisor)

    return graph.compile(checkpointer=checkpointer)


def run_case(app, config: dict, **input_updates) -> dict:
    """Thin wrapper around app.invoke() -- exists because of a real,
    non-obvious LangGraph behavior found while testing this: if the
    supervisor's conditional entry point routes straight to END (a
    resumed case with nothing new to do -- final_status already set), NO
    node ever executes, and .invoke() returns None, even though the
    case's state is fully present and correct in the checkpoint.

    Confirmed directly: app.get_state(config).values shows every field
    intact right after an .invoke() call that itself returned None. A
    caller who doesn't know this will crash on `result["recommended_verdict"]`
    exactly the way an early version of this project's own test suite
    did. Any real caller of this graph should go through run_case(), not
    raw .invoke(), unless it's prepared to handle a None result itself.

    A PAUSED case (waiting on human_approval's interrupt()) is a
    different, valid situation -- .invoke() returns a real dict there,
    with a "__interrupt__" key, not None -- so this fallback doesn't
    fire, and is_awaiting_approval() below is how a caller should detect
    that case instead.
    """
    allowed = {"case_id", "raw_statement_json"}
    if set(input_updates) - allowed:
        raise ValueError(
            "Only case_id and raw_statement_json are accepted; derived state is internal"
        )
    saved = app.get_state(config).values if app.checkpointer is not None else {}
    if saved:
        for key, value in input_updates.items():
            previous = saved.get(key)
            same = value == previous
            if key == "raw_statement_json":
                import json

                try:
                    same = json.loads(value) == json.loads(previous)
                except (ValueError, TypeError):
                    pass
            if not same:
                raise ValueError("Changed input requires a new thread_id/revision")
    result = app.invoke(input_updates, config=config)
    if result is None:
        result = app.get_state(config).values
    return result


def is_awaiting_approval(result: dict) -> bool:
    """True if the case just paused at human_approval's interrupt() and
    is waiting for a real decision -- the payload passed to interrupt()
    is available at result["__interrupt__"][0].value."""
    return bool(result.get("__interrupt__"))
