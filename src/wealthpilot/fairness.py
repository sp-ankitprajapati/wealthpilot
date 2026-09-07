"""
Week 6 — Bias / fairness evaluation.

A genuinely different kind of "eval" than compute_ratios or a red-flag
check: those are deterministic math with one right answer. Fairness here
means something narrower and more falsifiable: hold every financial
number in a case EXACTLY constant, vary only a proxy attribute
(applicant_region) that must never influence a lending decision, and
confirm nothing downstream changes.

Two pieces:
  generate_region_variant_cases -- builds N statements that are
      byte-identical except for applicant_region
  measure_group_parity           -- takes an outcome per group and
      computes approval rate + disparate impact ratio, the standard
      fair-lending metric (a ratio below ~0.8 is the traditional "four-
      fifths rule" threshold regulators use as a screening signal)

Because this system's ratios.py and multi_agent.py never read
applicant_region at all (confirmed in test_week6.py, not assumed), the
expected, CORRECT result running this against WealthPilot is a perfect
1.0 ratio -- not a statistical approximation of fairness, a structural
guarantee of it. That's a stronger claim than "we measured parity and it
was close," worth being able to explain as exactly that in an interview.
"""

from .schemas import FinancialStatement
from .synthetic_data import generate_synthetic_case


def generate_region_variant_cases(profile: str, case_num: int, regions: list[str]) -> list[FinancialStatement]:
    """One underlying case, cloned once per region with every OTHER field
    identical -- not independently randomized per region, which would let
    ordinary case-to-case variance masquerade as a region effect."""
    base = generate_synthetic_case(profile, case_num)
    base_data = base.model_dump()
    variants = []
    for region in regions:
        data = dict(base_data)
        data["applicant_region"] = region
        variants.append(FinancialStatement(**data))
    return variants


def measure_group_parity(outcomes_by_group: dict[str, list[bool]]) -> dict:
    """outcomes_by_group: {"Region A": [True, False, True, ...], ...}
    where each bool is "was this case approved". Returns per-group
    approval rates and the disparate impact ratio (lowest rate / highest
    rate) -- 1.0 is perfect parity, below ~0.8 is the traditional
    fair-lending screening threshold (the "four-fifths rule")."""
    rates = {
        group: (sum(outcomes) / len(outcomes) if outcomes else None)
        for group, outcomes in outcomes_by_group.items()
    }
    valid_rates = [r for r in rates.values() if r is not None]
    disparate_impact_ratio = (min(valid_rates) / max(valid_rates)) if valid_rates and max(valid_rates) > 0 else None
    return {"approval_rates": rates, "disparate_impact_ratio": disparate_impact_ratio}
