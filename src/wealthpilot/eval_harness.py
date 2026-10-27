"""
Week 7 — Eval harness.

Runs the FULL graph (document_parser -> risk_analyst -> credit_researcher
-> final_review -> human_approval) against every case in Week 1's labeled
eval set, auto-approving each recommendation (Command(resume={"approved":
True})) so the harness measures the AI's recommendation quality, not a
human reviewer's judgment -- the auto-approval is the harness's choice,
not a claim that real cases should skip human review.

Deliberately separate from the tests: a test asserts a specific expected
result and fails the build if it's wrong. This harness measures whatever
the given model actually does and reports it -- accuracy 62% is a valid,
useful harness result; it would be a failing test.
"""

import json
import time
from uuid import uuid4
from collections import Counter

from langgraph.types import Command

from .case_graph import build_case_graph, run_case


def run_eval_harness(
    model, retrieve_policy_tool, eval_cases: list[dict], checkpointer
) -> dict:
    app = build_case_graph(model, retrieve_policy_tool, checkpointer=checkpointer)
    results = []

    run_id = uuid4().hex
    for index, case in enumerate(eval_cases):
        case_id = case["statement"]["company_id"]
        config = {"configurable": {"thread_id": f"EVAL-{run_id}-{index}-{case_id}"}}
        start = time.perf_counter()

        paused = run_case(
            app,
            config,
            case_id=case_id,
            raw_statement_json=json.dumps(case["statement"]),
        )
        final = app.invoke(Command(resume={"approved": True}), config=config)

        results.append(
            {
                "case_id": case_id,
                "profile": case["risk_profile"],
                "expected": case["expected_verdict"],
                "predicted": final.get("final_status"),
                "correct": final.get("final_status") == case["expected_verdict"],
                "latency_s": round(time.perf_counter() - start, 3),
            }
        )

    return build_report(results)


def build_report(results: list[dict]) -> dict:
    """Split out from run_eval_harness so tests can feed it synthetic
    result rows directly, without needing a real graph run just to check
    the accuracy/confusion-matrix arithmetic is correct."""
    n = len(results)
    accuracy = sum(r["correct"] for r in results) / n if n else 0.0

    per_class = {}
    for expected_class in sorted(set(r["expected"] for r in results)):
        class_rows = [r for r in results if r["expected"] == expected_class]
        per_class[expected_class] = {
            "n": len(class_rows),
            "accuracy": sum(r["correct"] for r in class_rows) / len(class_rows),
        }

    confusion = Counter((r["expected"], r["predicted"]) for r in results)

    latencies = [r["latency_s"] for r in results]
    latency_stats = {
        "mean_s": round(sum(latencies) / n, 3) if n else 0.0,
        "max_s": round(max(latencies), 3) if latencies else 0.0,
    }

    return {
        "n_cases": n,
        "accuracy": round(accuracy, 3),
        "per_class_accuracy": per_class,
        "confusion_matrix": {
            f"{exp}->{pred}": count for (exp, pred), count in confusion.items()
        },
        "latency": latency_stats,
        "results": results,
    }


def print_report(report: dict) -> None:
    """Human-readable summary -- the thing you'd actually paste into a
    demo or a PR description, not the raw dict."""
    print(f"Cases: {report['n_cases']}   Accuracy: {report['accuracy']:.1%}")
    print(
        f"Latency: mean {report['latency']['mean_s']}s, max {report['latency']['max_s']}s"
    )
    print("\nPer-class accuracy:")
    for cls, stats in report["per_class_accuracy"].items():
        print(f"  {cls:10s}  {stats['accuracy']:.1%}  (n={stats['n']})")
    print("\nConfusion matrix (expected -> predicted):")
    for transition, count in sorted(report["confusion_matrix"].items()):
        print(f"  {transition}: {count}")
    misses = [r for r in report["results"] if not r["correct"]]
    if misses:
        print(f"\n{len(misses)} miss(es):")
        for r in misses:
            print(
                f"  {r['case_id']} ({r['profile']}): expected {r['expected']}, got {r['predicted']}"
            )


if __name__ == "__main__":
    # Live run -- needs a real API key. See the Toolkit's Setup chapter,
    # or agent.py's get_model() for the provider factory.
    import os
    from langgraph.checkpoint.memory import InMemorySaver

    from .agent import get_model
    from .paths import EVAL_CASES_PATH
    from .policy_docs import POLICY_DOCS
    from .rag import build_policy_index, make_retrieve_policy_tool

    with open(EVAL_CASES_PATH) as f:
        eval_cases = json.load(f)

    provider = os.getenv("WEALTHPILOT_PROVIDER", "gemini")
    key_var = {
        "gemini": "GOOGLE_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "grok": "XAI_API_KEY",
    }[provider]
    if not os.getenv(key_var):
        print(f"No {key_var} found in the environment -- set it to run this live.")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        model = get_model(provider)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        index = build_policy_index(POLICY_DOCS, embeddings)
        retrieve_tool = make_retrieve_policy_tool(index)

        report = run_eval_harness(
            model, retrieve_tool, eval_cases, checkpointer=InMemorySaver()
        )
        print_report(report)
