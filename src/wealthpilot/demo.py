"""
Week 7 — Demo script.

Run with: python3 demo.py

Walks ONE case through the entire pipeline with readable output at each
stage -- meant to be the thing you actually run on screen in an
interview, not a test and not a full eval sweep (that's eval_harness.py).
Shows the recommendation, the rationale, the interrupt payload a real
reviewer would see, and the final decision after approving it.

Needs a real API key (see the Toolkit's Setup chapter) and real internet
access for the embeddings model -- this will not run inside the
network-sandboxed environment this project was built in. That's not a
bug to fix; every prior week's __main__ block has the same honest
limitation, documented each time it came up.
"""

import json
import os
import sys

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .agent import get_model
from .case_graph import build_case_graph, run_case, is_awaiting_approval
from .paths import EVAL_CASES_PATH
from .policy_docs import POLICY_DOCS
from .rag import build_policy_index, make_retrieve_policy_tool


def _line(char: str = "-", width: int = 60) -> None:
    print(char * width)


def run_demo(model, embeddings, case_index: int = 10) -> None:
    with open(EVAL_CASES_PATH) as f:
        eval_cases = json.load(f)
    case = eval_cases[case_index]

    print(f"Applicant: {case['statement']['company_id']}  (profile: {case['risk_profile']})")
    print(f"Revenue: ${case['statement']['revenue']:,.0f}   "
          f"Prior year: ${case['statement']['prior_period_revenue']:,.0f}")
    _line()

    index = build_policy_index(POLICY_DOCS, embeddings)
    retrieve_tool = make_retrieve_policy_tool(index)

    checkpointer = InMemorySaver()
    app = build_case_graph(model, retrieve_tool, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": case["statement"]["company_id"]}}

    print("Running document_parser -> risk_analyst -> credit_researcher -> final_review ...")
    paused = run_case(
        app, config,
        case_id=case["statement"]["company_id"],
        raw_statement_json=json.dumps(case["statement"]),
    )

    if not is_awaiting_approval(paused):
        # document_parser rejected the input outright -- nothing to approve
        print(f"\nCase auto-rejected: {paused.get('final_status')}")
        print(f"Reason: {paused.get('parse_error')}")
        return

    _line()
    print("RATIOS:")
    for k, v in paused["ratios"].items():
        print(f"  {k}: {v}")
    print("\nRED FLAGS:")
    for flag in paused["red_flags"] or ["(none)"]:
        print(f"  - {flag}")
    print("\nPOLICY FINDINGS:")
    for finding in paused["policy_findings"] or [{"source": "(none needed)", "clause": "", "text": ""}]:
        print(f"  [{finding['source']} / {finding.get('clause', '')}] {finding.get('text', '')[:100]}")
    _line()
    print(f"\nRECOMMENDED VERDICT: {paused['recommended_verdict'].upper()}")
    print(f"Rationale: {paused['rationale']}")
    _line("=")

    print(f"\n(This is where a real reviewer would approve, send back, or override --")
    print(f" auto-approving for this demo.)\n")
    final = app.invoke(Command(resume={"approved": True}), config=config)

    print(f"FINAL STATUS: {final['final_status'].upper()}")
    print(f"(expected, from Week 1's labeled eval set: {case['expected_verdict']})")


if __name__ == "__main__":
    provider = os.getenv("WEALTHPILOT_PROVIDER", "gemini")
    key_var = {"gemini": "GOOGLE_API_KEY", "claude": "ANTHROPIC_API_KEY", "grok": "XAI_API_KEY"}[provider]
    if not os.getenv(key_var):
        print(f"No {key_var} found in the environment.")
        print(f"Set it, or set WEALTHPILOT_PROVIDER to a different provider, then re-run:")
        print(f"  export {key_var}=...")
        print(f"  python3 demo.py")
        sys.exit(1)

    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"WealthPilot demo -- provider: {provider}")
    _line("=")
    run_demo(model=get_model(provider), embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
