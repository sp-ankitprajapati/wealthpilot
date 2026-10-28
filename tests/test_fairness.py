"""Regression checks for region variants in the synthetic review pathway."""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from wealthpilot.case_graph import build_case_graph, run_case
from wealthpilot.fairness import generate_region_variant_cases, measure_group_parity
from wealthpilot.policy_docs import POLICY_DOCS
from wealthpilot.rag import build_policy_index, make_retrieve_policy_tool
from wealthpilot.ratios import compute_ratios
from wealthpilot.tools import detect_red_flags

from test_case_graph import FakeCaseModel, FakeEmbeddings


@pytest.fixture(scope="module")
def retrieve_tool():
    index = build_policy_index(POLICY_DOCS, FakeEmbeddings())
    return make_retrieve_policy_tool(index)


class TestFairness:
    """Every test here is a falsifiable structural claim, not a
    statistical approximation. If applicant_region ever started
    influencing an outcome, every single one of these would fail --
    they're written to catch a REGRESSION as much as to prove a current
    guarantee."""

    def test_ratios_are_identical_across_regions(self):
        variants = generate_region_variant_cases("overleveraged", 1, ["Region A", "Region B", "Region C"])
        results = [compute_ratios(fs) for fs in variants]
        assert all(r == results[0] for r in results)

    def test_red_flags_are_identical_across_regions(self):
        variants = generate_region_variant_cases("declining", 1, ["Region A", "Region B"])
        flag_sets = [
            json.loads(detect_red_flags.invoke({"ratios_json": json.dumps(compute_ratios(fs))}))
            for fs in variants
        ]
        assert flag_sets[0] == flag_sets[1]

    def test_final_review_prompt_never_contains_region_or_the_raw_statement(self, retrieve_tool):
        """The strong claim: not "the model ignores region" but "region
        never reaches the model's context window at all". Proven by
        capturing the EXACT prompt text final_review sends and checking
        for the region string and the word "region" itself -- and
        confirming the prompt is byte-identical across two cases whose
        financials are identical but whose region differs."""

        captured_prompts = []

        class RecordingModel(FakeCaseModel):
            def invoke(self, messages):
                captured_prompts.append(messages[-1].content)
                return super().invoke(messages)

        variants = generate_region_variant_cases("healthy", 1, ["Alpha District", "Beta District"])
        for fs in variants:
            model = RecordingModel()
            app = build_case_graph(model, retrieve_tool, checkpointer=InMemorySaver())
            run_case(
                app,
                {"configurable": {"thread_id": f"REGION-{fs.applicant_region}"}},
                case_id=f"REGION-{fs.applicant_region}",
                raw_statement_json=fs.model_dump_json(),
            )

        final_review_prompts = [p for p in captured_prompts if "RECOMMENDED verdict" in p]
        assert len(final_review_prompts) == 2
        assert "Alpha District" not in final_review_prompts[0]
        assert "Beta District" not in final_review_prompts[1]
        assert "region" not in final_review_prompts[0].lower()
        assert final_review_prompts[0] == final_review_prompts[1]  # byte-identical

    def test_end_to_end_outcome_is_identical_across_regions(self, retrieve_tool):
        variants = generate_region_variant_cases("overleveraged", 1, ["Region A", "Region B", "Region C"])
        recommendations = []
        for fs in variants:
            model = FakeCaseModel(verdict_word="decline")
            app = build_case_graph(model, retrieve_tool, checkpointer=InMemorySaver())
            paused = run_case(
                app,
                {"configurable": {"thread_id": f"E2E-{fs.applicant_region}"}},
                case_id=f"E2E-{fs.applicant_region}",
                raw_statement_json=fs.model_dump_json(),
            )
            recommendations.append((paused["ratios"], paused["red_flags"], paused["recommended_verdict"]))

        assert all(r == recommendations[0] for r in recommendations)

    def test_disparate_impact_ratio_is_perfectly_1_across_regions(self, retrieve_tool):
        """The compliance-facing metric a real fair-lending review would
        actually compute. Runs 3 profiles x 3 regions = 9 cases through
        the FULL graph (including human approval, auto-approved here)
        and checks the standard disparate impact ratio -- expected to be
        exactly 1.0 given the structural guarantee proven above, not
        approximately close to it."""
        regions = ["Region A", "Region B", "Region C"]
        outcomes_by_region = {r: [] for r in regions}

        for profile in ("healthy", "illiquid", "overleveraged"):
            variants = generate_region_variant_cases(profile, 1, regions)
            for fs in variants:
                model = FakeCaseModel(verdict_word="approve" if profile == "healthy" else "decline")
                app = build_case_graph(model, retrieve_tool, checkpointer=InMemorySaver())
                thread_id = f"PARITY-{profile}-{fs.applicant_region}"
                run_case(
                    app, {"configurable": {"thread_id": thread_id}},
                    case_id=thread_id, raw_statement_json=fs.model_dump_json(),
                )
                result = app.invoke(Command(resume={"approved": True}), config={"configurable": {"thread_id": thread_id}})
                outcomes_by_region[fs.applicant_region].append(result["final_status"] == "approve")

        parity = measure_group_parity(outcomes_by_region)
        assert parity["disparate_impact_ratio"] == 1.0
        rates = list(parity["approval_rates"].values())
        assert all(r == rates[0] for r in rates)





if __name__ == "__main__":
    pytest.main([__file__, "-v"])
