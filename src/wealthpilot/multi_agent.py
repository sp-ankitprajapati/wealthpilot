"""
Week 4/6 — Multi-agent orchestration, persistent case memory, and
human-in-the-loop approval.

Five specialists instead of one agent doing everything:
  document_parser   -- deterministic, validates raw input into a FinancialStatement
  risk_analyst       -- deterministic, calls Week 2's tools directly (no LLM)
  credit_researcher  -- LLM-driven, decides what to search for, calls Week 3's retrieve_policy
  final_review       -- LLM-driven, synthesizes everything into a RECOMMENDED verdict
  human_approval     -- pauses (interrupt()) for an explicit human decision (Week 6)

A supervisor function routes between them based on what's already in state,
not a hardcoded linear chain -- each specialist returns to the supervisor
for re-routing.

Week 6 splits two things that were previously conflated under one
"verdict" field:
  recommended_verdict -- the AI's suggestion (approve/review/decline) --
                          NEVER final on its own
  final_status         -- the actual outcome: "approved" (human agreed with
                          the recommendation), "sent_back" (human didn't),
                          or "rejected_invalid_data" (never reached a
                          human at all -- the input itself was invalid)
A case is only actually done once final_status exists -- an AI
recommendation by itself is not a decision.
"""

import json
from typing import Optional, TypedDict, Annotated

from pydantic import ValidationError
from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from .schemas import FinancialStatement
from .tools import calculate_ratios, detect_red_flags


class CaseState(TypedDict, total=False):
    """total=False because keys accumulate over the case's lifetime --
    'ratios' genuinely doesn't exist yet on a freshly-opened case, and
    that's different from existing-but-empty. Several bugs earlier in
    this project came from conflating "missing" with "falsy" (see the
    supervisor's routing logic below) -- total=False forces every node
    to use state.get() defensively rather than assume a key is there.

    Every field a node might return MUST be declared here -- confirmed
    directly (see docs/BUILD_LOG.md, Week 6) that LangGraph silently
    drops any key a node returns that isn't part of the compiled state
    schema. It doesn't error; it just never shows up in get_state() or
    the next node's input. An easy thing to not notice if you add a new
    field to a node's return value and forget to add it here too."""

    messages: Annotated[list, add_messages]
    case_id: str
    raw_statement_json: str
    financial_statement: dict
    ratios: dict
    red_flags: list
    policy_findings: list
    recommended_verdict: str
    rationale: str
    final_status: str
    reviewer_notes: str
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
        # A real, intentional early exit -- setting "final_status" here
        # (not "recommended_verdict") is what tells the supervisor to
        # route straight to END. This is a TECHNICAL rejection, not a
        # business decision, so it correctly never reaches a human
        # reviewer at all -- there's nothing for a human to approve.
        return {
            "parse_error": str(e),
            "final_status": "rejected_invalid_data",
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
    is synthesis and a RECOMMENDED verdict, not fact-gathering (keeps its
    output groundable in what's already in state, not invented numbers)
    and not a final decision (that's human_approval's job, Week 6)."""

    def final_review(state: CaseState) -> dict:
        # Deliberately excludes "financial_statement" -- the raw statement,
        # and anything on it like an applicant's region, never reaches
        # this prompt at all. Not "trusted to be ignored" -- structurally
        # absent from what the model can see. See fairness.py and
        # docs/BUILD_LOG.md, Week 6, for why that distinction matters.
        summary_input = {
            "ratios": state.get("ratios"),
            "red_flags": state.get("red_flags"),
            "policy_findings": state.get("policy_findings"),
        }
        prompt = (
            "Write a short SME loan underwriting risk summary from the data "
            "below. State a RECOMMENDED verdict — approve, review, or decline "
            "— as the first word of your response, then justify it by citing "
            "the specific ratios and policy clauses involved. This is a "
            "recommendation for human review, not a final decision. Do not "
            "introduce any number not present in the data below.\n\n"
            + json.dumps(summary_input, indent=2)
        )
        response = model.invoke([HumanMessage(content=prompt)])
        text = response.content.lower()
        verdict = next((v for v in ("approve", "review", "decline") if v in text), "review")
        return {"recommended_verdict": verdict, "rationale": response.content, "messages": [response]}

    return final_review


def human_approval(state: CaseState) -> dict:
    """interrupt() is the FIRST thing this node does, before any other
    logic -- confirmed directly (see docs/BUILD_LOG.md, Week 6) that
    LangGraph re-executes a node FROM THE TOP on resume, replaying
    anything before the interrupt() call. Putting side-effecting logic
    before it would re-run that logic every time a case resumes, not
    just once.

    The payload sent to interrupt() is everything a real reviewer needs
    to decide WITHOUT re-deriving it themselves -- the recommendation,
    the rationale, and the underlying ratios/flags/policy it's based on.
    """
    payload = {
        "case_id": state.get("case_id"),
        "recommended_verdict": state.get("recommended_verdict"),
        "rationale": state.get("rationale"),
        "ratios": state.get("ratios"),
        "red_flags": state.get("red_flags"),
        "policy_findings": state.get("policy_findings"),
    }
    decision = interrupt(payload)

    if decision.get("approved"):
        return {
            "final_status": decision.get("override_verdict") or state.get("recommended_verdict"),
            "reviewer_notes": decision.get("notes", ""),
        }
    return {"final_status": "sent_back", "reviewer_notes": decision.get("notes", "")}


def supervisor(state: CaseState) -> str:
    """Routes on KEY PRESENCE ("not in state"), never on truthiness. An
    empty policy_findings list is a legitimate completed result (no red
    flags -> nothing to look up) and must not be mistaken for "hasn't run
    yet" -- using `not state.get(...)` here would silently re-run
    credit_researcher forever on every clean case, since [] is falsy.
    This is the same class of bug Week 2 found in LangGraph's tool-output
    serialization: don't let "empty" and "absent" collapse into each
    other.

    "final_status" (not "recommended_verdict") is the ONLY thing that
    means a case is actually done -- an AI recommendation alone must
    always still route to human_approval, never straight to END."""
    if state.get("final_status"):
        return END
    if "financial_statement" not in state:
        return "document_parser"
    if "ratios" not in state:
        return "risk_analyst"
    if "policy_findings" not in state:
        return "credit_researcher"
    if "recommended_verdict" not in state:
        return "final_review"
    return "human_approval"
