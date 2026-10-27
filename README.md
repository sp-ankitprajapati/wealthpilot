# WealthPilot

A portfolio SME-underwriting workflow using deterministic ratio tools, policy RAG, LangGraph case checkpoints and explicit human approval. It exposes three tools through an MCP 2.x server. The original single-agent loop remains a separate learning example; the case graph is the interface with the approval gate.

## Verified status

**89 local tests pass**, including the original 65 cases and24 additional boundary cases. New checks cover verdict ambiguity, typed recommendations, malformed human decisions, changed statements, injected derived state, nonfinite financial values, JSON-safe zero denominators, negative book equity and policy-index isolation. CI runs the suite without provider credentials or model downloads.

Tests use controlled models/embeddings to verify orchestration. They do not measure live LLM underwriting quality, real population fairness or deployment suitability. The reported historical7.6 s to 1.06 s import improvement remains build-log history. A new reproducible three-process measurement is saved in [startup benchmark](results/startup_benchmark.json), with median 2.14 s on the checked Windows environment. Fresh-process import timing is different from OS cold-cache or request latency.

## Setup

Use Python 3.12 in a fresh environment:

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m wealthpilot.offline_demo
```

The offline demo uses real deterministic ratios and bundled policy text, controlled lexical retrieval, and a fixed review recommendation. It pauses for human approval, then explicitly sends the demonstration case back. Its output labels that no live LLM quality was measured. No keys or downloaded embedding weights are needed.

## Case workflow

1. Validate statement JSON and reject nonfinite values or negative expense/debt-service amounts.
2. Compute ratios and risk flags with Python tools.
3. Retrieve policy clauses for flagged cases, retaining source and clause identifiers.
4. Produce a recommended verdict using a typed approve/review/decline schema when the model supports structured output. Text-only test doubles must begin with an exact verdict prefix. Invalid output routes to manual review.
5. Pause with a checkpoint. A strictly Boolean human approved decision and validated optional override control the final outcome.

The final-review context excludes applicant_region and the raw statement. Tests show region exclusion for otherwise identical synthetic financials in this multi-agent pathway. This limited safeguard does not establish real-world fairness or protect against all proxies or reviewer effects. The older single-agent prompt receives the supplied statement and is outside this claim.

## Revisions and human approval

Use build_case_graph(..., checkpointer=...) and run_case(app, config, case_id=..., raw_statement_json=...). Supply a stable configurable.thread_id for pause/resume. Resume with LangGraph Command(resume={"approved":true,"notes":"..."}); optional override_verdict must be approve/review/decline and requires approved=true.

The wrapper accepts only source input keys and refuses changed statement/case inputs on an existing thread. Use a **new thread ID for each revised statement**. A supervisor check also detects changed raw statements behind cached derived state. This prevents an old completed recommendation from silently representing new financial numbers. Direct app.invoke/update_state calls are low-level APIs; callers must preserve the same lifecycle contract.

final_status values are approve/review/decline when the human accepts or overrides the recommendation, sent_back when the human declines to accept it, and rejected_invalid_data for failed parsing. The model recommendation alone does not complete a case.

## Numeric contracts

Ratios use the project's simplified EBIT-based DSCR and private-firm Altman Z-prime formula. Undefined ratios are JSON null with an explicit undefined_ratios list, rather than nonstandard Infinity. A zero prior revenue does not become fabricated 0% growth. Nonpositive book equity receives an explicit manual-review flag rather than appearing reassuring through a negative debt-to-equity ratio. The public flagging tool validates its ratio schema.

These are educational formulas and synthetic policy thresholds, not a validated credit-risk model or lending advice. Altman-model applicability varies by business type and requires domain assessment. The application does not implement real document OCR, default prediction, portfolio validation or regulated underwriting controls.

## Live providers and embeddings

Install only the extra you intend to use: `.[gemini]`, `.[claude]` or `.[grok]`. Set WEALTHPILOT_PROVIDER, WEALTHPILOT_MODEL and the matching GOOGLE_API_KEY / ANTHROPIC_API_KEY / XAI_API_KEY in the process environment. Choose a model available to your account. The project does not automatically load .env and makes no free-tier availability claim.

For the live Chroma path, install `.[local-embeddings]` to use Hugging Face embeddings; this may download model weights. build_policy_index creates a distinct collection per index so unrelated in-process indexes cannot silently share a default collection. The current helper creates ephemeral policy collections; case persistence comes from the supplied LangGraph checkpointer, including SQLite.

```bash
python -m wealthpilot.demo
python -m wealthpilot.eval_harness
python -m wealthpilot.mcp_server
```

These live entry points need the relevant provider/embedding configuration. The eval harness uses fresh run-specific thread IDs to avoid old checkpoint decisions contaminating a new evaluation. Its automatic approval exists only for scoring controlled evaluations and is not the production human-review policy.

## MCP and source map

MCP 2.x exposes calculate_ratios, detect_red_flags and retrieve_policy through build_server(retrieve_tool). In-process tests check registration, outputs and anticipated ToolError exceptions; transport behavior and external client interoperability require separate client verification. This server exposes tools, not the full approval workflow.

| File | Responsibility |
| --- | --- |
| schemas.py / ratios.py / tools.py | Input contracts, arithmetic and flags |
| rag.py / policy_docs.py | Clause boundaries, isolated Chroma indexes and citable results |
| multi_agent.py | Specialist nodes, typed recommendation and strict human decision |
| case_graph.py | Checkpoints and revision guards |
| fairness.py | Controlled region-variant checks |
| eval_harness.py / tracing.py | Evaluation and structured node logs |
| mcp_server.py | Three reusable tools |
| offline_demo.py | Honest no-key workflow demonstration |

Read [updated Hinglish learning guide](docs/LEARNING_HINGLISH.md), [verification evidence](results/verification.json) and [historical build log](docs/BUILD_LOG.md). Model judgment, policy retrieval relevance, proxy fairness and legal applicability still need independent live evaluation. No production-readiness claim is made.

## Accurate CV wording

- Built a LangGraph underwriting prototype with deterministic ratio tools, clause-grounded policy retrieval, persistent case checkpoints and strict human approval.
- Added typed verdict validation, revision isolation, finite JSON contracts and89 automated tests, including controlled region-exclusion checks.
- Exposed calculation and research tools through MCP 2.x and published reproducible startup measurements with clearly scoped evaluation evidence.
