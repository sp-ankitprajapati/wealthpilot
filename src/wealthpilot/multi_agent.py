"""
Week 4 — Multi-agent orchestration + persistent case memory.

Four specialists instead of one agent doing everything:
  document_parser   -- deterministic, validates raw input into a FinancialStatement
  risk_analyst       -- deterministic, calls Week 2's tools directly (no LLM)
  credit_researcher  -- LLM-driven, decides what to search for, calls Week 3's retrieve_policy
  final_review       -- LLM-driven, synthesizes everything into a cited verdict

A supervisor function routes between them based on what's already in state,
not a hardcoded linear chain -- each specialist returns to the supervisor
for re-routing, so a case can only ever be in exactly one of: needs
parsing, needs ratios, needs policy lookup, needs final synthesis, or done.
"""

import json
from typing import Optional, TypedDict, Annotated

from pydantic import ValidationError
from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.graph.message import add_messages

from .schemas import FinancialStatement
from .tools import calculate_ratios, detect_red_flags


class CaseState(TypedDict, total=False):
    """total=False because keys accumulate over the case's lifetime --
    'ratios' genuinely doesn't exist yet on a freshly-opened case, and
    that's different from existing-but-empty. Several bugs earlier in
    this project came from conflating "missing" with "falsy" (see the
    supervisor's routing logic below) -- total=False forces every node
    to use state.get() defensively rather than assume a key is there."""

    messages: Annotated[list, add_messages]
    case_id: str
    raw_statement_json: str
    financial_statement: dict
    ratios: dict
    red_flags: list
    policy_findings: list
    verdict: str
    parse_error: str
    # instrumentation, not production logic -- lets tests PROVE a node
    # didn't re-run after a case resumed from a checkpoint, rather than
    # just trusting that it didn't
    document_parser_runs: int
    risk_analyst_runs: int


def document_parser(state: CaseState) -> dict:
    """Validates the raw input FIRST, before any other specialist touches
    it. A malformed statement should fail loudly here, not surface as a
    confusing downstream error three nodes later."""
    runs = state.get("document_parser_runs", 0) + 1
    try:
        fs = FinancialStatement.model_validate_json(state["raw_statement_json"])
        return {"financial_statement": fs.model_dump(), "document_parser_runs": runs}
    except ValidationError as e:
        # A real, intentional early exit -- setting "verdict" here is what
        # tells the supervisor to route straight to END without ever
        # calling risk_analyst on data we already know is invalid.
        return {
            "parse_error": str(e),
            "verdict": "rejected_invalid_data",
            "document_parser_runs": runs,
        }


def risk_analyst(state: CaseState) -> dict:
    """No LLM here on purpose -- this is pure deterministic math, reusing
    the EXACT tool objects Week 2 built for an LLM to call. Same
    implementation, two different callers (an LLM via tool-calling in
    Week 2's agent, a plain Python call here) -- proof the tool boundary
    is genuinely reusable, not just conveniently reusable."""
    runs = state.get("risk_analyst_runs", 0) + 1
    statement_json = json.dumps(state["financial_statement"])
    ratios = calculate_ratios.invoke({"statement_json": statement_json})
    red_flags = json.loads(detect_red_flags.invoke({"ratios_json": json.dumps(ratios)}))
    return {"ratios": ratios, "red_flags": red_flags, "risk_analyst_runs": runs}


def make_credit_researcher(model, retrieve_policy_tool):
    """Factory, same DI shape as Week 2's build_graph(model) and Week 3's
    make_retrieve_policy_tool(vectorstore) -- this node needs BOTH a
    chat model and a specific tool instance, and closing over them here
    is what lets tests swap in fakes for both."""
    model_with_tool = model.bind_tools([retrieve_policy_tool])

    def credit_researcher(state: CaseState) -> dict:
        red_flags = state.get("red_flags", [])
        if not red_flags:
            # nothing concerning was found -- no policy question to ask,
            # and "no findings" is itself a valid, complete answer here,
            # not a "didn't run yet" placeholder (see supervisor below)
            return {"policy_findings": []}

        prompt = (
            "Using the retrieve_policy tool, find the underwriting policy "
            "clauses relevant to these concerns:\n"
            + "\n".join(f"- {flag}" for flag in red_flags)
        )
        response = model_with_tool.invoke([HumanMessage(content=prompt)])

        findings = []
        for call in getattr(response, "tool_calls", None) or []:
            if call["name"] == retrieve_policy_tool.name:
                result = retrieve_policy_tool.invoke(call["args"])
                findings.extend(json.loads(result))
        return {"policy_findings": findings}

    return credit_researcher


def make_final_review(model):
    """The only node that writes prose. Everything it needs (ratios,
    red_flags, policy_findings) was already gathered deterministically or
    semi-deterministically by earlier specialists -- this node's only job
    is synthesis and a verdict, not fact-gathering, which keeps its
    output groundable in what's already in state instead of the model
    inventing numbers."""

    def final_review(state: CaseState) -> dict:
        summary_input = {
            "ratios": state.get("ratios"),
            "red_flags": state.get("red_flags"),
            "policy_findings": state.get("policy_findings"),
        }
        prompt = (
            "Write a short SME loan underwriting risk summary from the data "
            "below. State a verdict — approve, review, or decline — as the "
            "first word of your response, then justify it by citing the "
            "specific ratios and policy clauses involved. Do not introduce "
            "any number not present in the data below.\n\n"
            + json.dumps(summary_input, indent=2)
        )
        response = model.invoke([HumanMessage(content=prompt)])
        text = response.content.lower()
        verdict = next((v for v in ("approve", "review", "decline") if v in text), "review")
        return {"verdict": verdict, "messages": [response]}

    return final_review


def supervisor(state: CaseState) -> str:
    """Routes on KEY PRESENCE ("not in state"), never on truthiness. An
    empty policy_findings list is a legitimate completed result (no red
    flags -> nothing to look up) and must not be mistaken for "hasn't run
    yet" -- using `not state.get(...)` here would silently re-run
    credit_researcher forever on every clean case, since [] is falsy.
    This is the same class of bug Week 2 found in LangGraph's tool-output
    serialization: don't let "empty" and "absent" collapse into each
    other."""
    if state.get("verdict"):
        return END
    if "financial_statement" not in state:
        return "document_parser"
    if "ratios" not in state:
        return "risk_analyst"
    if "policy_findings" not in state:
        return "credit_researcher"
    return "final_review"
