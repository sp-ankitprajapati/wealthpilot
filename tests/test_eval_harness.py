"""
Week 7 — Eval harness.

Run with: pytest tests/test_eval_harness.py -v

Uses TWO fakes on purpose: a rule-based one that reads the same red-flag
strings tools.py's detect_red_flags produces (non-trivial, should score
well against Week 1's labeled set) and a "dumb" one that always says the
same thing (should score badly on whichever classes it doesn't match).
If the harness's accuracy computation had a bug that made everything
score 100% regardless of input, only testing the smart fake would never
catch it -- the dumb fake is what proves the number means something.

RuleBasedFakeModel and FakeEmbeddings are defined here (not duplicated)
and imported by test_tracing.py, which needs them for one test that
exercises a real graph run.
"""

import hashlib
import json

import pytest
from langchain_core.messages import AIMessage

from wealthpilot.eval_harness import run_eval_harness, build_report
from wealthpilot.paths import EVAL_CASES_PATH
from wealthpilot.policy_docs import POLICY_DOCS
from wealthpilot.rag import build_policy_index, make_retrieve_policy_tool

from langgraph.checkpoint.memory import InMemorySaver


class FakeEmbeddings:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class RuleBasedFakeModel:
    """Reads the SAME human-readable red-flag strings tools.py's
    detect_red_flags actually produces, embedded verbatim in the
    final_review prompt via json.dumps. Not a real LLM -- but non-trivial
    enough that getting it wrong on a case is a real, meaningful miss,
    which is what makes it useful for proving the harness measures
    something real."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        content = messages[-1].content
        if "retrieve_policy tool" in content:
            # Query built from the ACTUAL red-flag text in the prompt, not
            # a generic placeholder. A generic single-word query has near-
            # zero word overlap with any clause, so every clause ties at
            # ~zero similarity -- confirmed this caused real, reproducible
            # non-determinism ACROSS separate process runs, most likely
            # from how Chroma breaks ties among equally-scored results.
            # Same root cause in spirit as Week 1's unseeded-RNG test bug:
            # a fixture with a hidden source of randomness masquerading
            # as deterministic.
            query_text = content.split("concerns:")[-1].strip()
            return AIMessage(content="", tool_calls=[{
                "name": "retrieve_policy", "args": {"query": query_text, "k": 2}, "id": "c1",
            }])
        if "cannot cover debt service" in content or "distress zone" in content:
            verdict = "decline"
        elif "Current ratio" in content and "below 1.0" in content:
            verdict = "review"
        elif '"red_flags": []' in content:
            verdict = "approve"
        else:
            verdict = "review"
        return AIMessage(content=f"{verdict.capitalize()}: rule-based assessment.")


class AlwaysApproveFakeModel:
    """Deliberately bad -- ignores the input entirely. Exists ONLY to
    prove the harness's accuracy number actually moves when the model's
    quality changes, rather than being wired to report a fixed number
    regardless of what's fed into it."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        content = messages[-1].content
        if "retrieve_policy tool" in content:
            return AIMessage(content="", tool_calls=[{
                "name": "retrieve_policy", "args": {"query": "policy", "k": 2}, "id": "c1",
            }])
        return AIMessage(content="Approve: looks fine.")


@pytest.fixture(scope="module")
def retrieve_tool():
    index = build_policy_index(POLICY_DOCS, FakeEmbeddings())
    return make_retrieve_policy_tool(index)


@pytest.fixture(scope="module")
def eval_cases():
    with open(EVAL_CASES_PATH) as f:
        return json.load(f)


class TestReportArithmetic:
    """Unit-level checks on build_report()'s math, with synthetic rows --
    no graph run needed to verify accuracy/confusion-matrix arithmetic."""

    def test_accuracy_with_all_correct(self):
        rows = [{"expected": "approve", "predicted": "approve", "correct": True, "latency_s": 0.1}] * 5
        report = build_report(rows)
        assert report["accuracy"] == 1.0

    def test_accuracy_with_half_correct(self):
        rows = (
            [{"expected": "approve", "predicted": "approve", "correct": True, "latency_s": 0.1}] * 5
            + [{"expected": "decline", "predicted": "approve", "correct": False, "latency_s": 0.1}] * 5
        )
        report = build_report(rows)
        assert report["accuracy"] == 0.5
        assert report["per_class_accuracy"]["approve"]["accuracy"] == 1.0
        assert report["per_class_accuracy"]["decline"]["accuracy"] == 0.0

    def test_confusion_matrix_counts_transitions(self):
        rows = [
            {"expected": "decline", "predicted": "approve", "correct": False, "latency_s": 0.1},
            {"expected": "decline", "predicted": "approve", "correct": False, "latency_s": 0.1},
            {"expected": "decline", "predicted": "decline", "correct": True, "latency_s": 0.1},
        ]
        report = build_report(rows)
        assert report["confusion_matrix"]["decline->approve"] == 2
        assert report["confusion_matrix"]["decline->decline"] == 1

    def test_empty_results_does_not_crash(self):
        report = build_report([])
        assert report["accuracy"] == 0.0
        assert report["n_cases"] == 0


class TestHarnessDifferentiatesModelQuality:
    """The actual point of this file: prove the accuracy number MEANS
    something, by running two models of clearly different quality
    against the same 20-case labeled set and confirming the harness
    reports clearly different accuracy for them."""

    def test_rule_based_model_scores_well(self, retrieve_tool, eval_cases):
        report = run_eval_harness(
            RuleBasedFakeModel(), retrieve_tool, eval_cases, checkpointer=InMemorySaver()
        )
        assert report["accuracy"] >= 0.8  # non-trivial rule reasoning, should do well

    def test_always_approve_model_scores_badly(self, retrieve_tool, eval_cases):
        report = run_eval_harness(
            AlwaysApproveFakeModel(), retrieve_tool, eval_cases, checkpointer=InMemorySaver()
        )
        # 5 of 20 cases are labeled "approve" -- a model that always says
        # approve should score close to exactly that fraction, not high
        assert report["accuracy"] <= 0.35
        assert report["per_class_accuracy"]["approve"]["accuracy"] == 1.0  # right by accident
        assert report["per_class_accuracy"]["decline"]["accuracy"] == 0.0  # wrong every time

    def test_the_two_models_produce_meaningfully_different_scores(self, retrieve_tool, eval_cases):
        """The actual assertion that matters most: same eval set, same
        harness code, different model quality -> different reported
        accuracy. If this failed, the harness would be measuring
        something other than the model's actual behavior."""
        good_report = run_eval_harness(
            RuleBasedFakeModel(), retrieve_tool, eval_cases, checkpointer=InMemorySaver()
        )
        bad_report = run_eval_harness(
            AlwaysApproveFakeModel(), retrieve_tool, eval_cases, checkpointer=InMemorySaver()
        )
        assert good_report["accuracy"] > bad_report["accuracy"] + 0.3  # not just noise





if __name__ == "__main__":
    pytest.main([__file__, "-v"])
