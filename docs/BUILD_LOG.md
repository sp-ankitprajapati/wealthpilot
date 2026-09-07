# Build Log

Design decisions and real bugs, week by week — written to actually be
useful in an interview, not as a changelog. Every bug below was caught by
running the code and inspecting real output, not discovered by inspection
or assumed to exist.

## Week 1 — Foundation

**Decision: Altman Z'-score (private-firm variant), not the original
1968 public-company Z-score.** The original formula needs a market value
of equity, which a private SME simply doesn't have. Substituting book
equity into the public formula is a known-bad shortcut. The Z'-score
(Altman, 1983) uses different coefficients and different zone cutoffs
(safe >2.90, distress <1.23, vs. the original's 2.99/1.81) specifically
for firms without traded equity.

**Bug: my own test fixtures failed against my own validator.**
`FinancialStatement` has a `model_validator` that rejects a statement if
`ebit` is more than 25% off from `revenue - cogs - operating_expenses`.
Three early test fixtures overrode `ebit` without adjusting `cogs`/`opex`
to match, and the validator correctly rejected them. Fixed the fixtures,
not the validator — proof it catches real inconsistency, including from
whoever wrote the test.

**Bug: a nondeterministic test.** `test_healthy_cases_land_in_safe_zone`
failed once, citing "case 8" — but the test never seeded the RNG, so
which statement "case 8" referred to depended on whatever random calls
had already happened earlier in the pytest session. Unreproducible
failures are nearly useless. Fixed by seeding explicitly in every test
that generates synthetic data.

## Week 2 — Single-agent LangGraph loop

**Decision: separate control-flow tests from judgment-quality tests.**
Whether the graph calls the right tool, in order, and terminates
correctly is a different question from whether the LLM's reasoning is
actually good. The first is testable deterministically, with no API key,
using `FakeToolCallingModel` — a test double implementing only
`.bind_tools()` and `.invoke()`. The second needs a real model and lives
in `agent.py`'s `__main__`, not the test suite.

**Bug: LangGraph's `ToolNode` serializes tool output inconsistently by
return-value truthiness.** A tool returning a non-empty `list` gets its
`ToolMessage.content` JSON-serialized automatically. The *same* tool
returning an **empty** list gets left as a raw Python list — because an
empty list happens to be technically valid as multimodal "content
blocks," so the auto-conversion skips it. Confirmed directly (see the
`[] <class 'list'>` vs. JSON-string output in the commit history / build
notes). Fix: `detect_red_flags` now does `return json.dumps(flags)`
explicitly — never trust a framework's implicit serialization for
structured tool output.

## Week 3 — Policy RAG

**Bug #1: chunk_size too small split a real clause in half.** First
version used `chunk_size=500`. Clause 3.4 (Distress Zone Handling) is 543
characters — the splitter fell through to sentence-level separators and
produced an orphan `n/a`-tagged fragment. Measuring the actual longest
clause (543 chars) and raising `chunk_size` above it (to 700) was the
obvious next move —

**Bug #2: which caused a worse bug.** With `chunk_size=700`,
`RecursiveCharacterTextSplitter` didn't just stop cutting clause 3.4 in
half — it started *merging* adjacent short clauses (3.1 + 3.2) into a
single chunk, tagged only with the first clause's ID. A citation pointing
at "clause 3.1" could now actually be about clause 3.2 (leverage limits).
The splitter optimizes for filling each chunk close to `chunk_size`; it
has no concept of "these are two distinct citable units." **The real
fix** was to stop using the recursive splitter for the top-level split
entirely — explicit `str.split("\n## ")` on the clause boundary
guarantees exactly one clause per chunk, with the recursive splitter
demoted to a fallback used only when a single clause alone exceeds the
size limit (in which case every sub-chunk correctly keeps that one
clause's ID — no ambiguity, because it's genuinely one clause).

**Sandbox limitation, not a bug:** real embeddings need
`huggingface.co` (for the local `sentence-transformers` model) or a
provider's hosted embeddings API, both unreachable from the environment
this repo was scaffolded in. `tests/test_rag.py` uses a deterministic
bag-of-words `FakeEmbeddings` to prove the *pipeline* — chunking,
indexing, retrieval, metadata, citation format — works correctly. It does
not, and cannot, prove real semantic retrieval quality. That needs a live
run of `rag.py`'s `__main__` on a machine with normal internet access.

**Housekeeping:** switched from `langchain_community.vectorstores.Chroma`
to the standalone `langchain-chroma` package after a deprecation warning
— `langchain-community`'s vectorstore integrations are being sunset in
favor of standalone packages.

## Week 4 — Multi-agent orchestration + persistent case memory

**Design decision, applied proactively from a Week 2 lesson, not
discovered by a failure this time:** the supervisor routes on key
*presence* (`"policy_findings" not in state`), never on truthiness
(`not state.get("policy_findings")`). An empty `policy_findings` list is
a legitimate completed result — no red flags means nothing to look up —
and would be indistinguishable from "hasn't run yet" under a truthiness
check, since `[]` is falsy. This is the same class of issue Week 2's
empty-list serialization bug belonged to (empty vs. absent), and this
time it was designed around correctly from the start rather than found
by a failing test.

**Bug, actually caught by a failing test this time:** `.invoke()` on an
already-completed case (same `thread_id`, nothing new to do) returns
`None` — even though `app.get_state(config).values` shows the case's
full state intact and correct. Confirmed directly: printing `.invoke()`'s
raw return value showed `None` while `get_state()` right after showed
every field present. Root cause: when the supervisor's conditional entry
point routes straight to `END`, zero graph nodes execute, and
`.invoke()`, which returns whatever the last executed step emitted, has
nothing to return. **Real consequence:** any caller who does
`result["verdict"]` on that return value crashes on a case that's
actually fine — exactly what this project's own test suite did on the
first run. Fixed with a small wrapper, `run_case()`, that falls back to
`get_state(config).values` whenever `.invoke()` returns `None`. Every
real caller of this graph should go through `run_case()`, not raw
`.invoke()`.

**Test design point:** the persistence tests don't just check that
`compile(checkpointer=...)` doesn't crash — they build a completely
*second* `build_case_graph()` object, with a fresh `SqliteSaver`
connection to the same file, and a *different* fake model, then confirm
the resumed case still returns the first model's verdict and never calls
the second model at all. A test that only calls `.invoke()` twice on the
*same* Python object would still pass even if persistence were an
in-memory illusion with nothing actually reaching disk — a real restart
rebuilds the graph object too, so the test has to as well.

## Week 5 — MCP server

**Discovery: the `mcp` SDK had a breaking major version release after
this project's tooling knowledge was current.** `mcp` 2.0.0 shipped
2026-07-28 and renamed `FastMCP` to `MCPServer`, moving
`mcp.server.fastmcp` → `mcp.server.mcpserver`. The old import doesn't
just fail silently — it raises a `ModuleNotFoundError` whose message
names the migration guide, specifically so v1 code gives a useful signal
that the installed SDK is a different major version rather than a plain
"module not found." Confirmed by trying the old import first, reading
that message, then checking the new API's real signatures with
`inspect.signature()` against the installed 2.1.1 rather than trusting a
search result. `pyproject.toml` now pins `mcp>=2,<3` — the same lesson
`langchain-chroma` already taught in Week 3: a range spanning two majors
can resolve to a version that can't even import.

**Discovery: anticipated vs. unanticipated tool failures are genuinely
different exception types, but not converted to `is_error=True` at the
layer this project's tests actually call.** Read directly from the
installed SDK's source (`mcp/server/mcpserver/tools/base.py`): a
deliberately-raised `ToolError` gets logged at INFO and re-wrapped as
`ToolError`; any other exception gets wrapped as `UnexpectedToolError`
and logged at ERROR with a full traceback, the client seeing only a
generic message. **Both still propagate as raised Python exceptions**
out of `MCPServer.call_tool()` when called in-process, the way a test
does — they are not returned as a `CallToolResult(is_error=True)` at
this level. That conversion happens only at the real transport/protocol
layer, which a direct in-process call doesn't exercise. First version of
`mcp_server.py` let a `pydantic.ValidationError` from a malformed
financial statement propagate unguarded — confirmed as an
`UnexpectedToolError`-shaped crash by actually running it and reading
the traceback, not assumed. Fixed by catching `ValidationError` in
`calculate_ratios` and re-raising as `ToolError` with a clear message,
and by testing with `pytest.raises(ToolError)` rather than checking
`result.is_error`, which would have silently tested nothing.

**Design decision, not a bug:** every MCP tool delegates to the *same*
underlying LangChain tool objects Week 2's agent and Week 4's
`risk_analyst` node already call (`_calculate_ratios_tool.invoke(...)`
inside the MCP wrapper, not a re-implementation). One set of business
logic, three different callers now — LLM tool-calling, a plain Python
graph node, and an MCP client.

## Week 6 — Human-in-the-loop approval + bias/fairness eval

No new dependency needed this week — a real contrast to Weeks 3 and 5,
both of which needed a package swap. `interrupt()`, `Command`, and
`InMemorySaver` are all already part of `langgraph`, already a
dependency since Week 4.

**Design change: split one "verdict" field into two.**
`recommended_verdict` (the AI's suggestion) and `final_status` (the
actual outcome: `"approved"`/an overridden verdict, `"sent_back"`, or
`"rejected_invalid_data"`) used to be conflated under one `"verdict"`
key. The supervisor's "is this case done" check now looks at
`final_status` only — an AI recommendation alone must never be treated
as final, which is the entire point of human-in-the-loop. This also
meant updating every Week 4 test that checked `result["verdict"]`; they
now invoke, confirm the case paused with a recommendation, resume with
`Command(resume=...)`, and check `final_status` — a real behavior
change, not a cosmetic rename, so the old tests couldn't just be patched
in place, they had to be re-thought around the new two-phase flow.

**Finding, confirmed by direct test:** `interrupt()` re-executes the
*entire* node from the top on resume, replaying anything before the
`interrupt()` call. `human_approval` calls `interrupt()` as the very
first line for exactly this reason — any side-effecting logic placed
before it would silently re-run on every resume, not just once.

**Finding, confirmed by direct test:** a node's return value that isn't
declared in the compiled `CaseState` schema is *silently dropped* — not
an error, just absent from `get_state()` and the next node's input. Ran
a minimal reproduction with `MessagesState` (which only declares
`messages`) returning an undeclared `approved` key from a node — it
never appeared anywhere afterward. `CaseState` already declares every
field nodes use, so this wasn't a bug in this project's own code, but
it's a real, easy-to-hit footgun for anyone who adds a new field to a
node's return value and forgets to add it to the state schema too.

**Fairness: a structural claim, not a statistical one.** Added
`applicant_region` to `FinancialStatement` as an explicitly-unused field
— collected for compliance monitoring, never read by `compute_ratios`,
`detect_red_flags`, or any prompt `multi_agent.py` builds. Rather than
computing an approval-rate disparity on invented protected-attribute
data (this system has none), `fairness.py` and `tests/test_fairness.py`
prove something stronger and directly falsifiable: the *exact* prompt
text sent to `final_review`'s model is captured and shown to be
byte-identical across two otherwise-identical cases differing only in
region, and the standard fair-lending disparate impact ratio, computed
end-to-end across 3 profiles × 3 regions through the real graph
(including human approval), comes back at *exactly* 1.0 — not
approximately close to it. That's the difference between "we measured
fairness and it looked fine" and "the architecture makes the unfair
outcome structurally impossible" — a stronger, more specific claim to
be able to defend in an interview.

## Week 7 onward

Not yet built — see the README status table.
