# WealthPilot

An AI agent for SME loan underwriting and credit research — built as a
mentor-guided project, incrementally, one week at a time.

WealthPilot takes an SME's financial statements and produces a cited,
policy-grounded underwriting recommendation (approve / review / decline),
using a multi-agent LangGraph pipeline: deterministic financial-ratio
tools, a RAG layer over lending policy documents, and a human-approval
checkpoint before any final decision.

## Why this project

Built to sit at the intersection of two things: agentic AI engineering
(LangGraph orchestration, RAG, MCP, tool use, eval-driven development) and
domain fluency in credit/financial analysis. Full design rationale for
each week — including three real bugs found by actually running the code,
not just writing it — is in [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md).

## Status

| Week | Scope | Status |
|---|---|---|
| 1 | Financial statement schema, ratio engine, labeled synthetic eval set | ✅ done |
| 2 | Single-agent LangGraph loop (ratio + red-flag tools) | ✅ done |
| 3 | Policy RAG (structure-aware chunking, Chroma retrieval, citations) | ✅ done |
| 4 | Multi-agent orchestration + persistent case memory | ✅ done |
| 5 | MCP server exposing the underwriting tools | ✅ done |
| 6 | Human-in-the-loop approval + bias/fairness eval | ✅ done |
| 7 | Observability, eval harness, final polish | ✅ done |

All 7 weeks of the original build plan are complete. 65 tests pass across the full test suite (`pytest tests/ -v`). See [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) for the complete week-by-week story, including every real bug found along the way — that log is the actual interview-prep material this project produced, arguably more than the code itself.

## Architecture

```
Financial statement (JSON)
        │
        ▼
┌───────────────────┐     ┌──────────────────────┐
│ calculate_ratios   │────▶│ detect_red_flags      │
│ (DSCR, D/E, Z'...) │     │ (5 threshold checks)  │
└───────────────────┘     └──────────────────────┘
        │                            │
        └──────────────┬─────────────┘
                        ▼
              ┌───────────────────┐        ┌────────────────────┐
              │   LangGraph agent  │◀──────▶│  retrieve_policy     │
              │ (provider-agnostic)│        │  (Chroma + citations)│
              └───────────────────┘        └────────────────────┘
                        │
                        ▼
              Cited risk summary + RECOMMENDED verdict
                        │
                        ▼
              ┌────────────────────┐
              │   human_approval    │   interrupt() -- pauses here for
              │  (Week 6, HITL)     │   a real reviewer's approve /
              └────────────────────┘   send-back / override decision
                        │
                        ▼
              final_status (the actual decision)
```

Every tool returns a **deterministic, tested** result — the LLM decides
*when* to call a tool and *how to interpret* the result, never does the
arithmetic itself. See `src/wealthpilot/ratios.py` for why the Altman
Z**'**-score (private-firm variant), not the original public-company
formula, is used here.

## Choosing an LLM provider

The agent is provider-agnostic — Claude, Gemini, or Grok all plug in via
LangChain's chat-model wrapper with no changes to the graph itself. For a
free/no-budget build, **Gemini's free tier is the most sustainable**
(Grok's is promotional credits, not permanent; Claude's API has no
ongoing free tier at all). See `.env.example` and `get_model()` in
`src/wealthpilot/agent.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,local-embeddings,gemini]"   # swap gemini -> claude / grok
cp .env.example .env   # fill in your provider's key
```

Generate the labeled eval set (already included in `data/eval_cases.json`,
regenerate only if you change the synthetic-data ranges):

```bash
python -m wealthpilot.synthetic_data
```

Run the tests:

```bash
pytest tests/ -v
```

Run the agent live (needs a real API key — see Setup):

```bash
python -m wealthpilot.agent
```

Run the MCP server (needs `mcp[cli]` or a real MCP client to actually connect — Claude Desktop is the fastest way to try it: point its MCP settings at `python -m wealthpilot.mcp_server`):

```bash
python -m wealthpilot.mcp_server
```

Run the eval harness against all 20 labeled cases (needs a real API key):

```bash
python -m wealthpilot.eval_harness
```

Run the demo (a single case, formatted for actually presenting the system — needs a real API key):

```bash
python -m wealthpilot.demo
```

## A note on how the tests are split

Every week's tests split into two concerns on purpose:

- **Control-flow / plumbing** — does the graph call the right tool in the
  right order, does chunking keep one clause per chunk, does retrieval
  respect `k`. Fully testable right now, deterministically, with **no
  API key and no network call** — see the `Fake*` test doubles in
  `tests/test_agent.py` and `tests/test_rag.py`.
- **Judgment / semantic quality** — is the model's reasoning actually
  good, is real embedding-based retrieval actually relevant. Needs a real
  LLM and (for Week 3) real embeddings reachable over the network — run
  the `__main__` block in `agent.py` / `rag.py` on your own machine for
  this, not the test suite.

Conflating these two is a common mistake — it either makes your tests
flaky and slow, or means you never test orchestration logic at all.

## Known sandbox limitation (Weeks 3, 5 & 7)

This repo was scaffolded inside a network-sandboxed environment that
couldn't reach `huggingface.co` to download real embedding model weights,
or any LLM provider's API. `tests/test_rag.py`, `tests/test_mcp_server.py`,
and `tests/test_eval_harness.py` all use a deterministic bag-of-words fake
embedding (and, for the eval harness, a rule-based fake model) to prove
the *pipeline* works — chunking, indexing, retrieval, citation format,
orchestration, and accuracy/confusion-matrix arithmetic. None of this
proves real semantic retrieval quality or real LLM judgment quality. Run
`python -m wealthpilot.eval_harness` or `python -m wealthpilot.demo` on a
machine with normal internet access and a real API key to see both for
real.

## Project layout

```
src/wealthpilot/
  schemas.py        # FinancialStatement — the one data contract everything reads from
  ratios.py          # DSCR, current ratio, D/E, interest coverage, Altman Z'-score
  synthetic_data.py   # Labeled synthetic SME cases, 4 risk profiles
  tools.py            # calculate_ratios, detect_red_flags (LangChain @tool)
  agent.py             # Single-agent loop + provider-agnostic model factory (Week 2)
  policy_docs.py       # Illustrative SME lending policy text (Week 3)
  rag.py               # Chunking, Chroma indexing, retrieve_policy tool (Week 3)
  multi_agent.py       # CaseState, 4 specialist nodes, supervisor router (Week 4)
  case_graph.py        # build_case_graph + run_case (checkpointed multi-agent graph)
  mcp_server.py         # MCP server exposing the same tools via mcp 2.x's MCPServer (Week 5)
  fairness.py           # Region-variant case generation + disparate impact ratio (Week 6)
  tracing.py            # traced_node -- structured logging wrapper for any graph node (Week 7)
  eval_harness.py        # Runs the full graph against data/eval_cases.json, reports accuracy (Week 7)
  demo.py                # Single-case walkthrough with formatted output -- the interview demo script
  paths.py             # Single source of truth for data file locations
tests/                # pytest, organized by SOURCE MODULE, not by week --
                        # test_case_graph.py covers case_graph.py AND
                        # multi_agent.py (orchestration, persistence, and
                        # human-in-the-loop approval together, since all
                        # three exercise the same graph); test_fairness.py
                        # covers fairness.py; test_tracing.py and
                        # test_eval_harness.py cover Week 7's observability
                        # and eval harness respectively. See "how the
                        # tests are split" above for the broader philosophy.
data/eval_cases.json  # 20 labeled synthetic cases (5 per risk profile)
docs/BUILD_LOG.md      # Design rationale + real bugs found each week
```

## License

MIT — see [LICENSE](LICENSE).
