"""
Week 7 — Observability.

traced_node wraps any node function with structured logging: node name,
case_id, latency, and token usage IF the node's result carries a message
with usage_metadata (real LLM responses do; the deterministic nodes and
test fakes don't, and that's handled as an absence, not an error).

Confirmed safe around interrupt()-based nodes before relying on it: when
a node pauses via interrupt(), LangGraph's internal signal propagates
THROUGH this wrapper without the wrapper doing anything wrong -- the
logger.info() call after node_fn(state) simply doesn't run yet, because
node_fn(state) hasn't returned yet. It fires exactly once, on the resume
call that actually completes the node. Verified directly with a minimal
repro before wrapping the real human_approval node with it -- a decorator
that silently broke pause/resume would be a bad trade for a log line.
"""

import json
import logging
import time

logger = logging.getLogger("wealthpilot")


def traced_node(node_fn, name: str):
    def wrapper(state):
        start = time.time()
        result = node_fn(state)
        latency_s = round(time.time() - start, 4)

        log_entry = {
            "node": name,
            "case_id": state.get("case_id"),
            "latency_s": latency_s,
        }

        # Real LLM responses (ChatAnthropic/ChatGoogleGenerativeAI/ChatXAI)
        # attach usage_metadata to AIMessage; our test fakes and the
        # deterministic nodes don't. Absence isn't an error -- just
        # nothing to report for this node.
        messages = result.get("messages") if isinstance(result, dict) else None
        if messages:
            for message in messages:
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    log_entry["input_tokens"] = usage.get("input_tokens")
                    log_entry["output_tokens"] = usage.get("output_tokens")

        logger.info(json.dumps(log_entry))
        return result

    return wrapper
