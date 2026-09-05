"""
Week 1 — Foundation.

Generates SME financial statements with a KNOWN risk profile, so we have
ground-truth labels before any agent exists to grade. Four profiles, each
stressing a different part of the ratio set on purpose — a real underwriter
has to tell these apart, and so will the agent, eventually.

    healthy        -> sound on every ratio                  -> approve
    illiquid        -> sound fundamentals, short-term crunch  -> review
    overleveraged  -> too much debt relative to earnings     -> decline
    declining      -> shrinking, margins compressing         -> decline

The ranges below were tuned by generating a batch and checking the actual
computed ratios land where each profile's story says they should (see
`if __name__ == "__main__"` at the bottom) — not chosen once and trusted.
"""

import random
from typing import Literal

from .schemas import FinancialStatement

RiskProfile = Literal["healthy", "illiquid", "overleveraged", "declining"]

EXPECTED_VERDICT = {
    "healthy": "approve",
    "illiquid": "review",
    "overleveraged": "decline",
    "declining": "decline",
}


def generate_synthetic_case(profile: RiskProfile, case_num: int) -> FinancialStatement:
    revenue = random.uniform(2_000_000, 8_000_000)

    if profile == "healthy":
        prior_revenue = revenue * random.uniform(0.90, 0.98)
        cogs_pct = random.uniform(0.52, 0.58)
        opex_pct = random.uniform(0.16, 0.21)
        debt_service_pct_of_ebit = random.uniform(0.42, 0.62)  # DSCR ~1.6-2.4
        cl_pct_of_assets = random.uniform(0.15, 0.25)
        ca_to_cl_ratio = random.uniform(1.6, 2.4)
        tl_pct_of_assets = random.uniform(0.35, 0.50)
        re_pct_of_equity = random.uniform(0.40, 0.70)
        assets_to_revenue = random.uniform(0.70, 0.90)  # asset-efficient — a real strength

    elif profile == "illiquid":
        prior_revenue = revenue * random.uniform(0.90, 0.98)
        cogs_pct = random.uniform(0.55, 0.62)
        opex_pct = random.uniform(0.18, 0.24)
        debt_service_pct_of_ebit = random.uniform(0.55, 0.75)  # DSCR ~1.3-1.8, still fine
        cl_pct_of_assets = random.uniform(0.28, 0.36)
        ca_to_cl_ratio = random.uniform(0.60, 0.90)  # <-- the actual problem
        tl_pct_of_assets = random.uniform(0.35, 0.50)  # leverage is FINE
        re_pct_of_equity = random.uniform(0.40, 0.70)
        assets_to_revenue = random.uniform(0.85, 1.05)  # ordinary, not a strength or weakness

    elif profile == "overleveraged":
        prior_revenue = revenue * random.uniform(0.97, 1.03)
        cogs_pct = random.uniform(0.60, 0.68)
        opex_pct = random.uniform(0.14, 0.19)
        debt_service_pct_of_ebit = random.uniform(1.05, 1.55)  # DSCR <1
        cl_pct_of_assets = random.uniform(0.15, 0.22)
        ca_to_cl_ratio = random.uniform(1.0, 1.3)
        tl_pct_of_assets = random.uniform(0.72, 0.85)  # <-- the actual problem
        re_pct_of_equity = random.uniform(0.10, 0.30)
        assets_to_revenue = random.uniform(0.95, 1.20)

    else:  # declining
        prior_revenue = revenue * random.uniform(1.08, 1.20)  # revenue SHRANK
        cogs_pct = random.uniform(0.68, 0.80)  # margins compressed
        opex_pct = random.uniform(0.20, 0.28)
        debt_service_pct_of_ebit = random.uniform(0.60, 0.90)
        cl_pct_of_assets = random.uniform(0.20, 0.30)
        ca_to_cl_ratio = random.uniform(0.70, 1.10)
        tl_pct_of_assets = random.uniform(0.55, 0.70)
        re_pct_of_equity = random.uniform(-0.20, 0.10)  # accumulated losses
        assets_to_revenue = random.uniform(1.05, 1.35)  # bloated, underutilized assets

    total_assets = revenue * assets_to_revenue
    cogs = revenue * cogs_pct
    operating_expenses = revenue * opex_pct
    ebit = revenue - cogs - operating_expenses

    total_debt_service = max(abs(ebit) * debt_service_pct_of_ebit, revenue * 0.02)
    interest_expense = total_debt_service * random.uniform(0.40, 0.60)
    pretax_income = ebit - interest_expense
    net_income = pretax_income * 0.75 if pretax_income > 0 else pretax_income

    current_liabilities = total_assets * cl_pct_of_assets
    current_assets = current_liabilities * ca_to_cl_ratio

    total_liabilities = total_assets * tl_pct_of_assets
    shareholder_equity = total_assets - total_liabilities
    retained_earnings = shareholder_equity * re_pct_of_equity

    return FinancialStatement(
        company_id=f"SME-{profile[:4].upper()}-{case_num:03d}",
        period="FY2025-Q4",
        revenue=round(revenue, 2),
        prior_period_revenue=round(prior_revenue, 2),
        cogs=round(cogs, 2),
        operating_expenses=round(operating_expenses, 2),
        ebit=round(ebit, 2),
        interest_expense=round(interest_expense, 2),
        net_income=round(net_income, 2),
        total_assets=round(total_assets, 2),
        current_assets=round(current_assets, 2),
        current_liabilities=round(current_liabilities, 2),
        total_liabilities=round(total_liabilities, 2),
        shareholder_equity=round(shareholder_equity, 2),
        retained_earnings=round(retained_earnings, 2),
        total_debt_service=round(total_debt_service, 2),
    )


def generate_eval_set(cases_per_profile: int = 5, seed: int = 7) -> list[dict]:
    """The held-out ground truth for Week 7's eval harness. Built once, with
    a fixed seed, and then left alone — regenerating it with a new seed
    every time you touch this file would mean your eval numbers aren't
    comparable week over week."""
    random.seed(seed)
    from .ratios import compute_ratios

    cases = []
    for profile in ("healthy", "illiquid", "overleveraged", "declining"):
        for i in range(1, cases_per_profile + 1):
            fs = generate_synthetic_case(profile, i)
            cases.append(
                {
                    "statement": fs.model_dump(),
                    "ratios": compute_ratios(fs),
                    "risk_profile": profile,
                    "expected_verdict": EXPECTED_VERDICT[profile],
                }
            )
    return cases


if __name__ == "__main__":
    # Sanity pass: generate a few of each profile and print the ratios a
    # human underwriter would actually look at, so we can eyeball whether
    # "healthy" really looks healthy before we trust this as ground truth.
    from .ratios import compute_ratios

    random.seed(42)
    for profile in ("healthy", "illiquid", "overleveraged", "declining"):
        print(f"\n--- {profile} (expected verdict: {EXPECTED_VERDICT[profile]}) ---")
        for i in range(1, 4):
            fs = generate_synthetic_case(profile, i)
            r = compute_ratios(fs)
            print(
                f"  {fs.company_id}  DSCR={r['dscr']:>5}  "
                f"current={r['current_ratio']:>5}  D/E={r['debt_to_equity']:>5}  "
                f"Z'={r['altman_z']:>5} ({r['altman_zone']:<8})  "
                f"rev_growth={r['revenue_growth_pct']:>6}%"
            )

    import json

    eval_set = generate_eval_set(cases_per_profile=5, seed=7)
    from .paths import EVAL_CASES_PATH
    EVAL_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_CASES_PATH, "w") as f:
        json.dump(eval_set, f, indent=2)
    print(f"\nWrote {len(eval_set)} labeled cases to {EVAL_CASES_PATH}")
