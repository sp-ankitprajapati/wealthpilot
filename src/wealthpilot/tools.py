"""
Week 2 — Single agent, single loop.

Two tools, both thin wrappers around Week 1's already-tested logic. That's
deliberate: the agent should never be doing arithmetic itself — it decides
WHEN to call a deterministic calculator and HOW to interpret the result,
the calculator does the actual math. Keeping that boundary sharp is what
makes the system's numbers trustworthy regardless of which LLM sits behind
the agent.
"""

import json

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict
from typing import Literal

from .schemas import FinancialStatement
from .ratios import compute_ratios


class RatioValues(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    dscr: float | None
    current_ratio: float | None
    debt_to_equity: float | None
    interest_coverage: float | None
    revenue_growth_pct: float | None
    altman_z: float | None
    altman_zone: Literal["safe", "grey", "distress", "undefined"]
    equity_nonpositive: bool = False
    undefined_ratios: list[str] = []


@tool
def calculate_ratios(statement_json: str) -> dict:
    """Compute underwriting ratios from a financial statement.

    Input must be a JSON object with these fields: company_id, period,
    revenue, prior_period_revenue, cogs, operating_expenses, ebit,
    interest_expense, net_income, total_assets, current_assets,
    current_liabilities, total_liabilities, shareholder_equity,
    retained_earnings, total_debt_service.

    Returns DSCR, current_ratio, debt_to_equity, interest_coverage,
    revenue_growth_pct, altman_z, and altman_zone (safe/grey/distress).
    Call this FIRST, before forming any opinion about the applicant —
    every downstream judgment should trace back to these numbers.
    """
    fs = FinancialStatement.model_validate_json(statement_json)
    return compute_ratios(fs)


@tool
def detect_red_flags(ratios_json: str) -> str:
    """Flag specific underwriting concerns from a ratios dict.

    Input is the JSON object returned by calculate_ratios (pass it through
    as-is). Checks five independent thresholds — DSCR below 1.0, Altman
    Z'-score in the distress zone, current ratio below 1.0, debt-to-equity
    above 2.5, and negative revenue growth — and returns a JSON array
    with one short string per breached threshold, or an empty JSON array
    ("[]") if none are breached.

    Call this AFTER calculate_ratios, never before — it has nothing to
    check without the ratios.
    """
    ratios = RatioValues.model_validate_json(ratios_json).model_dump()
    flags = []
    if ratios["dscr"] is not None and ratios["dscr"] < 1.0:
        flags.append(
            f"DSCR {ratios['dscr']} is below 1.0 — cannot cover debt service from operating income alone"
        )
    if ratios["altman_zone"] == "distress":
        flags.append(
            f"Altman Z'-score {ratios['altman_z']} is in the distress zone (below 1.23)"
        )
    if ratios["current_ratio"] is not None and ratios["current_ratio"] < 1.0:
        flags.append(
            f"Current ratio {ratios['current_ratio']} is below 1.0 — short-term liabilities exceed short-term assets"
        )
    if ratios["debt_to_equity"] is not None and ratios["debt_to_equity"] > 2.5:
        flags.append(
            f"Debt-to-equity {ratios['debt_to_equity']} exceeds 2.5 — high leverage relative to owner equity"
        )
    if ratios["revenue_growth_pct"] is not None and ratios["revenue_growth_pct"] < 0:
        flags.append(
            f"Revenue declined {abs(ratios['revenue_growth_pct'])}% year-over-year"
        )
    if ratios.get("equity_nonpositive"):
        flags.append(
            "Nonpositive book equity requires manual review; a negative debt-to-equity ratio does not indicate low risk"
        )
    if ratios.get("undefined_ratios"):
        flags.append(
            "Undefined ratios require context: " + ", ".join(ratios["undefined_ratios"])
        )
    # Explicit json.dumps here, not "return flags". LangGraph's ToolNode
    # auto-serializes a dict return to a JSON string, but an EMPTY list
    # return is left as a raw Python list (an empty list is technically
    # valid multimodal "content blocks", so the auto-conversion skips it) —
    # a real, confirmed inconsistency, not a hypothetical one. Returning
    # the string ourselves means ToolMessage.content is always a JSON
    # string, whether flags is empty or not — one format, no special case
    # for callers to remember.
    return json.dumps(flags)


TOOLS = [calculate_ratios, detect_red_flags]


if __name__ == "__main__":
    # Prove the tools work standalone, against Week 1's own labeled eval
    # set, with NO agent and NO LLM involved yet. If these don't produce
    # sensible flags per profile, wiring an LLM on top only hides the bug.
    from .paths import EVAL_CASES_PATH

    with open(EVAL_CASES_PATH) as f:
        eval_cases = json.load(f)

    for case in eval_cases[::5]:  # one example per profile (5 cases apart)
        statement_json = json.dumps(case["statement"])
        ratios = calculate_ratios.invoke({"statement_json": statement_json})
        flags = json.loads(detect_red_flags.invoke({"ratios_json": json.dumps(ratios)}))
        print(
            f"\n{case['statement']['company_id']}  (profile: {case['risk_profile']}, expected: {case['expected_verdict']})"
        )
        for flag in flags:
            print(f"  ⚠ {flag}")
        if not flags:
            print("  (no red flags)")
