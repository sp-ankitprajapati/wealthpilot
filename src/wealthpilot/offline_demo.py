"""A clearly labeled workflow demonstration; no claim of real LLM credit judgment."""

import json
import re
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from .case_graph import build_case_graph, run_case, is_awaiting_approval
from .policy_docs import POLICY_DOCS
from .rag import chunk_policy_docs
from .synthetic_data import generate_synthetic_case


class OfflineDemoModel:
    """Controlled tool request + fixed review recommendation, for tracing code only."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        text = messages[-1].content
        if "retrieve_policy tool" in text:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_policy",
                        "args": {"query": text, "k": 3},
                        "id": "offline-policy",
                    }
                ],
            )
        return AIMessage(
            content="Review: controlled offline demonstration. A qualified reviewer must examine the supplied evidence."
        )


@tool
def retrieve_policy(query: str, k: int = 3) -> str:
    """Offline lexical overlap over real bundled policy clauses; no embedding quality claim."""
    texts, metadata = chunk_policy_docs(POLICY_DOCS)
    tokens = set(re.findall(r"\w+", query.lower()))
    scored = sorted(
        enumerate(texts),
        key=lambda pair: len(tokens & set(re.findall(r"\w+", pair[1].lower()))),
        reverse=True,
    )
    return json.dumps(
        [
            {
                "text": text,
                "source": metadata[i]["source"],
                "clause": metadata[i]["clause_id"],
            }
            for i, text in scored[:k]
        ]
    )


def run_demo():
    graph = build_case_graph(OfflineDemoModel(), retrieve_policy, InMemorySaver())
    config = {"configurable": {"thread_id": "offline-demo"}}
    statement = generate_synthetic_case("overleveraged", 1)
    paused = run_case(
        graph,
        config,
        case_id="offline-demo",
        raw_statement_json=statement.model_dump_json(),
    )
    final = graph.invoke(
        Command(
            resume={"approved": False, "notes": "Offline example sent back for review"}
        ),
        config=config,
    )
    return {
        "mode": "controlled_offline_demo",
        "measures_live_llm_quality": False,
        "ratios": paused["ratios"],
        "red_flags": paused["red_flags"],
        "policy_findings": paused["policy_findings"],
        "recommended_verdict": paused["recommended_verdict"],
        "paused_for_human": is_awaiting_approval(paused),
        "final_status": final["final_status"],
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, allow_nan=False))
