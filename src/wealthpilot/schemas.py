"""
Week 1 — Foundation.

FinancialStatement is the single data contract every later component reads
from: the PDF parser will produce one, the risk-analyst agent will consume
one, the eval harness will grade against one. Getting this schema right now
means every later week is building on solid ground instead of a dict of
whatever-keys-happened-to-parse.
"""

from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


class FinancialStatement(BaseModel):
    company_id: str
    period: str  # e.g. "FY2025-Q4"

    # income statement
    revenue: float
    prior_period_revenue: float  # last year's revenue — lets us compute YoY trend
    cogs: float
    operating_expenses: float
    ebit: float
    interest_expense: float
    net_income: float

    # balance sheet
    total_assets: float
    current_assets: float
    current_liabilities: float
    total_liabilities: float
    shareholder_equity: float  # BOOK value — see note in ratios.py on why
    retained_earnings: float

    # debt service
    total_debt_service: float  # annual principal + interest due

    # Week 6 — fairness monitoring ONLY. Collected so a compliance review
    # can check for disparate impact across regions after the fact (see
    # fairness.py) — never read by compute_ratios, detect_red_flags, or
    # any prompt built in multi_agent.py. That's not "trusted to be
    # ignored" by convention; it's structurally absent from every
    # downstream input those functions actually receive. See
    # docs/BUILD_LOG.md, Week 6, for why that distinction is the point.
    applicant_region: Optional[str] = None

    @field_validator("revenue", "total_assets")
    @classmethod
    def must_be_positive(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def ebit_should_roughly_match_components(self) -> "FinancialStatement":
        """Not a hard constraint — real statements have adjustments the
        simple revenue - cogs - opex formula won't capture. But if EBIT is
        wildly inconsistent with its inputs, that's almost always a parsing
        bug, not a real business, so we flag it loudly rather than silently
        feeding a nonsense number to every downstream agent."""
        implied = self.revenue - self.cogs - self.operating_expenses
        if implied != 0 and abs(self.ebit - implied) / abs(implied) > 0.25:
            raise ValueError(
                f"EBIT ({self.ebit:.0f}) is more than 25% off from "
                f"revenue - cogs - opex ({implied:.0f}) — check the parse"
            )
        return self
