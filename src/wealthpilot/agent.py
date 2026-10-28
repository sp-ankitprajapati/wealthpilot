"""Single-agent tool-calling loop for financial ratios and risk flags.

This implementation has no policy retrieval, persistent case workflow or
human-approval gate. Use case_graph for the checkpointed review workflow.
"""

import os

from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from .tools import TOOLS


def build_graph(model):
    """Takes anything that behaves like a LangChain chat model — real
    (ChatGoogleGenerativeAI / ChatAnthropic / ChatXAI) or a test double —
    and wires it into a two-node graph: agent decides, tools execute,
    repeat until the agent stops asking for tools.

    Provider-agnostic on purpose: this function never imports a specific
    provider's SDK. That happens only in get_model() below, and only for
    whichever provider you actually ask for.
    """
    model_with_tools = model.bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict:
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def get_model(provider: str):
    """One function, one place to look when you swap providers — matches
    the WealthPilot Toolkit's Ch. 01. Each import is lazy (inside the
    branch, not at module top) so you only need the ONE provider package
    actually installed, not all three."""
    if not os.getenv("WEALTHPILOT_MODEL"):
        raise ValueError(
            "Set WEALTHPILOT_MODEL to a currently available model for your chosen provider"
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=os.environ["WEALTHPILOT_MODEL"])
    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=os.environ["WEALTHPILOT_MODEL"])
    elif provider == "grok":
        from langchain_xai import ChatXAI

        return ChatXAI(model=os.environ["WEALTHPILOT_MODEL"])
    else:
        raise ValueError(f"unknown provider: {provider!r} (use gemini/claude/grok)")


SYSTEM_PROMPT = (
    "You are an SME loan underwriting assistant. When given a financial "
    "statement, ALWAYS call calculate_ratios first, then detect_red_flags "
    "on the result. Only after seeing both outputs, write a short risk "
    "summary: state the verdict you'd recommend (approve / review / "
    "decline), and justify it by naming the specific ratios and flags "
    "that drove the call. Never state a verdict without having called "
    "both tools first."
)


if __name__ == "__main__":
    # Live run against a real provider — needs an API key in the
    # environment (see the Toolkit's Setup chapter for which env var).
    provider = os.getenv("WEALTHPILOT_PROVIDER", "gemini")
    key_var = {
        "gemini": "GOOGLE_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "grok": "XAI_API_KEY",
    }[provider]

    if not os.getenv(key_var):
        print(f"No {key_var} found in the environment.")
        print(
            f"Set it, or set WEALTHPILOT_PROVIDER to a different provider, then re-run:"
        )
        print(f"  export {key_var}=...")
        print(f"  python3 -m wealthpilot.agent")
    else:
        import json
        from .paths import EVAL_CASES_PATH

        with open(EVAL_CASES_PATH) as f:
            eval_cases = json.load(f)
        sample_case = eval_cases[10]  # an overleveraged case

        model = get_model(provider)
        app = build_graph(model)
        result = app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            f"{SYSTEM_PROMPT}\n\nHere is the financial statement:\n"
                            f"{json.dumps(sample_case['statement'])}"
                        )
                    )
                ]
            }
        )
        print(result["messages"][-1].content)
