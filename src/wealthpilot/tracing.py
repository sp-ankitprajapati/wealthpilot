"""Structured case-node logging with latency and optional token usage.

Interrupted nodes propagate the pause signal. Completed node calls emit
their trace record after the wrapped function returns.
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
