"""Specialist nodes and routing for the human-reviewed case workflow.

Nodes validate input, compute deterministic ratios, retrieve policy evidence,
produce a recommendation and pause for a human decision. recommended_verdict
is separate from final_status; a model recommendation does not finish a case.
"""

import json
import re
from typing import Optional, TypedDict, Annotated

from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from .schemas import FinancialStatement
from .tools import calculate_ratios, detect_red_flags


class CaseState(TypedDict, total=False):
    """Fields accumulated by the case workflow.

    Keys may be absent until their node runs. Every node output belongs in
    this schema so it remains available in subsequent graph state.
    """

    messages: Annotated[list, add_messages]
    case_id: str
    raw_statement_json: str
    financial_statement: dict
    ratios: dict
    red_flags: list
    policy_findings: list
    recommended_verdict: str
    recommendation_error: str
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
    """Compute ratios and risk flags using the shared deterministic tools."""
    runs = state.get("risk_analyst_runs", 0) + 1
    statement_json = json.dumps(state["financial_statement"])
    ratios = calculate_ratios.invoke({"statement_json": statement_json})
    red_flags = json.loads(detect_red_flags.invoke({"ratios_json": json.dumps(ratios)}))
    return {"ratios": ratios, "red_flags": red_flags, "risk_analyst_runs": runs}


def make_credit_researcher(model, retrieve_policy_tool):
    """Create a research node with an injected chat model and retrieval tool."""
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


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["approve", "review", "decline"]
    rationale: str


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: StrictBool
    override_verdict: Literal["approve", "review", "decline"] | None = None
    notes: str = ""


def make_final_review(model):
    """Use a typed verdict when supported; legacy fakes must start with an exact verdict."""
    structured = (
        model.with_structured_output(Recommendation)
        if callable(getattr(model, "with_structured_output", None))
        else None
    )

    def final_review(state):
        summary = {
            key: state.get(key) for key in ("ratios", "red_flags", "policy_findings")
        }
        prompt = (
            "Write a short risk summary and RECOMMENDED verdict (approve/review/decline) for human review. "
            "Use only supplied ratios and retrieved policy citations. If using text, start with exactly Approve:, Review: or Decline:. "
            "No invented numbers. Evidence: " + json.dumps(summary, allow_nan=False)
        )
        try:
            if structured is not None:
                decision = Recommendation.model_validate(
                    structured.invoke([HumanMessage(content=prompt)])
                )
                return {
                    "recommended_verdict": decision.verdict,
                    "rationale": decision.rationale,
                }
            response = model.invoke([HumanMessage(content=prompt)])
            text = response.content if isinstance(response.content, str) else ""
            match = re.match(r"^\s*(approve|review|decline)\s*:", text, re.IGNORECASE)
            if not match:
                raise ValueError(
                    "Response does not begin with an explicit supported verdict"
                )
            return {
                "recommended_verdict": match.group(1).lower(),
                "rationale": text,
                "messages": [response],
            }
        except (ValueError, TypeError) as exc:
            return {
                "recommended_verdict": "review",
                "rationale": "Model output could not be validated; manual review required.",
                "recommendation_error": str(exc),
            }

    return final_review


def human_approval(state: CaseState) -> dict:
    """Pause before processing the reviewer's decision.

    LangGraph resumes by re-executing the node. Place side effects after
    the interruption to avoid repeating them when the case resumes.
    """
    payload = {
        "case_id": state.get("case_id"),
        "recommended_verdict": state.get("recommended_verdict"),
        "rationale": state.get("rationale"),
        "ratios": state.get("ratios"),
        "red_flags": state.get("red_flags"),
        "policy_findings": state.get("policy_findings"),
    }
    decision = ApprovalDecision.model_validate(interrupt(payload))
    if decision.override_verdict is not None and not decision.approved:
        raise ValueError("An override requires approved=true")
    if decision.approved:
        return {
            "final_status": decision.override_verdict or state["recommended_verdict"],
            "reviewer_notes": decision.notes,
        }
    return {"final_status": "sent_back", "reviewer_notes": decision.notes}


def supervisor(state: CaseState) -> str:
    """Route by field presence and final status.

    An empty policy_findings list is a completed result. A recommendation
    still requires human review; final_status controls task completion.
    """
    if "financial_statement" in state and "raw_statement_json" in state:
        parsed = FinancialStatement.model_validate_json(
            state["raw_statement_json"]
        ).model_dump()
        if parsed != state["financial_statement"]:
            raise ValueError(
                "Changed statement requires a new thread_id/revision; old derived state cannot be reused"
            )
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
