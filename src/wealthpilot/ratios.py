"""Deterministic financial ratios and simplified risk calculations."""

from .schemas import FinancialStatement

# Z'-score zone cutoffs for PRIVATE firms (Altman, 1983 revision).
# Not the same cutoffs as the original 1968 public-company Z-score —
# using those on a private SME is a common, avoidable mistake.
Z_SAFE_CUTOFF = 2.90
Z_DISTRESS_CUTOFF = 1.23


def compute_ratios(fs: FinancialStatement) -> dict:
    """Five numbers an underwriter actually looks at, plus the ones that
    feed the Altman Z'-score."""

    dscr = fs.ebit / fs.total_debt_service if fs.total_debt_service else None
    current_ratio = (
        fs.current_assets / fs.current_liabilities if fs.current_liabilities else None
    )
    debt_to_equity = (
        fs.total_liabilities / fs.shareholder_equity if fs.shareholder_equity else None
    )
    interest_coverage = fs.ebit / fs.interest_expense if fs.interest_expense else None
    revenue_growth_pct = (
        (fs.revenue - fs.prior_period_revenue) / fs.prior_period_revenue * 100
        if fs.prior_period_revenue
        else None
    )

    # Altman Z'-score for PRIVATE firms:
    #   Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
    # X1 = working capital / total assets      (liquidity)
    # X2 = retained earnings / total assets     (accumulated profitability)
    # X3 = EBIT / total assets                  (operating profitability)
    # X4 = BOOK equity / total liabilities       (leverage — book, not market:
    #      a private SME has no market cap, and forcing the original
    #      public-company formula onto one by substituting book value in is
    #      a known-bad shortcut, not a harmless approximation)
    # X5 = revenue / total assets                (asset efficiency)
    working_capital = fs.current_assets - fs.current_liabilities
    x1 = working_capital / fs.total_assets
    x2 = fs.retained_earnings / fs.total_assets
    x3 = fs.ebit / fs.total_assets
    x4 = fs.shareholder_equity / fs.total_liabilities if fs.total_liabilities else None
    x5 = fs.revenue / fs.total_assets

    altman_z = (
        (0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5)
        if x4 is not None
        else None
    )

    if altman_z is None:
        z_zone = "undefined"
    elif altman_z > Z_SAFE_CUTOFF:
        z_zone = "safe"
    elif altman_z > Z_DISTRESS_CUTOFF:
        z_zone = "grey"
    else:
        z_zone = "distress"

    return {
        "dscr": round(dscr, 2) if dscr is not None else None,
        "current_ratio": round(current_ratio, 2) if current_ratio is not None else None,
        "debt_to_equity": round(debt_to_equity, 2)
        if debt_to_equity is not None
        else None,
        "interest_coverage": round(interest_coverage, 2)
        if interest_coverage is not None
        else None,
        "revenue_growth_pct": round(revenue_growth_pct, 1)
        if revenue_growth_pct is not None
        else None,
        "altman_z": round(altman_z, 2) if altman_z is not None else None,
        "altman_zone": z_zone,
        "equity_nonpositive": fs.shareholder_equity <= 0,
        "undefined_ratios": [
            name
            for name, value in {
                "dscr": dscr,
                "current_ratio": current_ratio,
                "debt_to_equity": debt_to_equity,
                "interest_coverage": interest_coverage,
                "revenue_growth_pct": revenue_growth_pct,
                "altman_z": altman_z,
            }.items()
            if value is None
        ],
    }
