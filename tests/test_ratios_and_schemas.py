"""
Week 1 — Foundation.

Run with: pytest test_week1.py -v

These aren't exhaustive, but they cover the two failure modes that would
otherwise silently corrupt every later week: (1) the ratio math itself
being wrong, and (2) division-by-zero on the edge cases real SME data
actually produces (a debt-free business has total_debt_service == 0).
"""

import random

import pytest
from pydantic import ValidationError

from wealthpilot.schemas import FinancialStatement
from wealthpilot.ratios import compute_ratios
from wealthpilot.synthetic_data import generate_synthetic_case, generate_eval_set, EXPECTED_VERDICT


def _make_statement(**overrides) -> FinancialStatement:
    """A clean baseline statement — tests override only the fields they
    care about, so each test reads as 'what's different here', not a
    20-field wall of setup."""
    defaults = dict(
        company_id="TEST-001",
        period="FY2025-Q4",
        revenue=1_000_000,
        prior_period_revenue=900_000,
        cogs=550_000,
        operating_expenses=200_000,
        ebit=250_000,
        interest_expense=50_000,
        net_income=150_000,
        total_assets=900_000,
        current_assets=300_000,
        current_liabilities=150_000,
        total_liabilities=400_000,
        shareholder_equity=500_000,
        retained_earnings=250_000,
        total_debt_service=125_000,
    )
    defaults.update(overrides)
    return FinancialStatement(**defaults)


class TestRatioMath:
    def test_dscr_is_ebit_over_debt_service(self):
        fs = _make_statement(ebit=250_000, total_debt_service=125_000)
        assert compute_ratios(fs)["dscr"] == 2.0

    def test_dscr_below_one_is_a_real_warning_signal(self):
        # cogs raised so revenue - cogs - opex = 100,000, matching the ebit below
        fs = _make_statement(cogs=700_000, ebit=100_000, total_debt_service=150_000)
        assert compute_ratios(fs)["dscr"] < 1.0

    def test_zero_debt_service_gives_infinite_dscr_not_a_crash(self):
        """A debt-free SME is a real, if rare, case — total_debt_service
        can legitimately be 0. Dividing by it must not crash the graph."""
        fs = _make_statement(total_debt_service=0)
        assert compute_ratios(fs)["dscr"] == float("inf")

    def test_current_ratio(self):
        fs = _make_statement(current_assets=300_000, current_liabilities=150_000)
        assert compute_ratios(fs)["current_ratio"] == 2.0

    def test_revenue_growth_sign_matches_direction(self):
        # ebit adjusted to match each revenue so the consistency validator
        # doesn't reject the fixture — cogs/opex (750,000 combined) unchanged
        growing = _make_statement(revenue=1_100_000, ebit=350_000, prior_period_revenue=1_000_000)
        shrinking = _make_statement(revenue=900_000, ebit=150_000, prior_period_revenue=1_000_000)
        assert compute_ratios(growing)["revenue_growth_pct"] > 0
        assert compute_ratios(shrinking)["revenue_growth_pct"] < 0

    def test_altman_z_zone_boundaries(self):
        # a statement engineered to land comfortably in each zone, to prove
        # the zone cutoffs (2.90 / 1.23) are wired correctly, not just that
        # *some* number comes out. cogs is adjusted alongside ebit in each
        # case so revenue - cogs - opex still matches (the consistency
        # validator rejects them otherwise, as it should).
        safe = _make_statement(
            cogs=400_000, ebit=400_000, retained_earnings=400_000, current_assets=500_000,
            current_liabilities=100_000, total_liabilities=200_000, shareholder_equity=700_000,
        )
        distressed = _make_statement(
            cogs=850_000, ebit=-50_000, retained_earnings=-100_000, current_assets=150_000,
            current_liabilities=200_000, total_liabilities=700_000, shareholder_equity=200_000,
        )
        assert compute_ratios(safe)["altman_zone"] == "safe"
        assert compute_ratios(distressed)["altman_zone"] == "distress"


class TestSchemaValidation:
    def test_negative_revenue_is_rejected(self):
        with pytest.raises(ValidationError):
            _make_statement(revenue=-100)

    def test_ebit_wildly_inconsistent_with_components_is_rejected(self):
        """revenue - cogs - opex = 250,000 here; claiming ebit = 900,000
        is almost certainly a parsing bug, not a real business."""
        with pytest.raises(ValidationError):
            _make_statement(ebit=900_000)

    def test_ebit_within_tolerance_is_accepted(self):
        # real statements have adjustments the simple formula won't
        # capture — a SMALL mismatch must not be rejected
        fs = _make_statement(ebit=270_000)  # ~8% off 250,000 — should pass
        assert fs.ebit == 270_000


class TestSyntheticDataMatchesItsOwnStory:
    """The point of labeled synthetic data is that the label is TRUE. If
    'healthy' cases don't actually land in the safe zone, the eval set is
    lying to every week that depends on it — so we check the story, not
    just that the generator runs without crashing.

    Every test here seeds the RNG explicitly before generating. Without
    that, 'case 8 failed' is meaningless — it depends on whatever random
    calls happened to run earlier in the test session, so a failure can't
    be reproduced or debugged. Seeded, a failure points at an exact,
    reproducible statement every time.
    """

    @pytest.mark.parametrize("profile", ["healthy", "illiquid", "overleveraged", "declining"])
    def test_generator_produces_valid_statements(self, profile):
        random.seed(1)
        fs = generate_synthetic_case(profile, case_num=1)
        assert fs.company_id.startswith("SME-")

    def test_healthy_cases_land_in_safe_zone(self):
        random.seed(42)
        for i in range(1, 11):
            fs = generate_synthetic_case("healthy", i)
            assert compute_ratios(fs)["altman_zone"] == "safe", f"case {i} broke the healthy story"

    def test_illiquid_cases_have_weak_current_ratio_but_ok_dscr(self):
        random.seed(42)
        for i in range(1, 11):
            fs = generate_synthetic_case("illiquid", i)
            r = compute_ratios(fs)
            assert r["current_ratio"] < 1.0, f"case {i}: illiquid should mean current ratio < 1"
            assert r["dscr"] > 1.2, f"case {i}: illiquid should NOT mean weak DSCR too"

    def test_overleveraged_cases_have_low_dscr_and_high_de(self):
        random.seed(42)
        for i in range(1, 11):
            fs = generate_synthetic_case("overleveraged", i)
            r = compute_ratios(fs)
            assert r["dscr"] < 1.0, f"case {i}: overleveraged should mean DSCR < 1"
            assert r["debt_to_equity"] > 2.5, f"case {i}: overleveraged should mean D/E > 2.5"

    def test_declining_cases_show_negative_revenue_growth(self):
        random.seed(42)
        for i in range(1, 11):
            fs = generate_synthetic_case("declining", i)
            assert compute_ratios(fs)["revenue_growth_pct"] < 0, f"case {i} isn't actually declining"

    def test_eval_set_label_distribution(self):
        eval_set = generate_eval_set(cases_per_profile=5, seed=7)
        assert len(eval_set) == 20
        verdicts = [c["expected_verdict"] for c in eval_set]
        assert verdicts.count("approve") == 5
        assert verdicts.count("review") == 5
        assert verdicts.count("decline") == 10  # two profiles map to decline — expected, not a bug
