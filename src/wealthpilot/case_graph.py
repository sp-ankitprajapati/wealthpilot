"""
Week 4 — Multi-agent orchestration + persistent case memory.

build_case_graph wires the four specialists from multi_agent.py behind a
supervisor router, with an optional checkpointer so a case's state
survives across separate invoke() calls -- and, tested below, across a
completely fresh graph object (i.e. a process restart), as long as it's
pointed at the same underlying SQLite file.
"""

from langgraph.graph import StateGraph, END

from .multi_agent import (
    CaseState,
    document_parser,
    risk_analyst,
    make_credit_researcher,
    make_final_review,
    supervisor,
)


def build_case_graph(model, retrieve_policy_tool, checkpointer=None):
    graph = StateGraph(CaseState)
    graph.add_node("document_parser", document_parser)
    graph.add_node("risk_analyst", risk_analyst)
    graph.add_node("credit_researcher", make_credit_researcher(model, retrieve_policy_tool))
    graph.add_node("final_review", make_final_review(model))

    graph.set_conditional_entry_point(supervisor)
    for node in ("document_parser", "risk_analyst", "credit_researcher", "final_review"):
        graph.add_conditional_edges(node, supervisor)

    return graph.compile(checkpointer=checkpointer)


def run_case(app, config: dict, **input_updates) -> dict:
    """Thin wrapper around app.invoke() -- exists because of a real,
    non-obvious LangGraph behavior found while testing this: if the
    supervisor's conditional entry point routes straight to END (a
    resumed case with nothing new to do -- verdict already set), NO node
    ever executes, and .invoke() returns None, even though the case's
    state is fully present and correct in the checkpoint.

    Confirmed directly: app.get_state(config).values shows every field
    intact right after an .invoke() call that itself returned None. A
    caller who doesn't know this will crash on `result["verdict"]`
    exactly the way an early version of this project's own test suite
    did. Any real caller of this graph should go through run_case(), not
    raw .invoke(), unless it's prepared to handle a None result itself.
    """
    result = app.invoke(input_updates, config=config)
    if result is None:
        result = app.get_state(config).values
    return result
