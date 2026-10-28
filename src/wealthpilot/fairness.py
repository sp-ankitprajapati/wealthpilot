"""Synthetic region-variant checks and group approval-rate summaries.

Cases hold financial values constant while varying applicant_region.
The review pathway excludes that field from its model context. These checks
cover the tested input variation; group parity on synthetic cases does not
establish population fairness, proxy exclusion or legal compliance.
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
