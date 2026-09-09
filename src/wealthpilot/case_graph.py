"""
Week 4/6 — Multi-agent orchestration, persistent case memory, and
human-in-the-loop approval.

build_case_graph wires the five specialists from multi_agent.py behind a
supervisor router, with an optional checkpointer so a case's state
survives across separate invoke() calls -- across a completely fresh
graph object (a process restart), and now also across a real pause for
human review (Week 6's human_approval node).
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
        traced_node(make_credit_researcher(model, retrieve_policy_tool), "credit_researcher"),
    )
    graph.add_node("final_review", traced_node(make_final_review(model), "final_review"))
    graph.add_node("human_approval", traced_node(human_approval, "human_approval"))

    graph.set_conditional_entry_point(supervisor)
    for node in ("document_parser", "risk_analyst", "credit_researcher", "final_review", "human_approval"):
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
    result = app.invoke(input_updates, config=config)
    if result is None:
        result = app.get_state(config).values
    return result


def is_awaiting_approval(result: dict) -> bool:
    """True if the case just paused at human_approval's interrupt() and
    is waiting for a real decision -- the payload passed to interrupt()
    is available at result["__interrupt__"][0].value."""
    return bool(result.get("__interrupt__"))
