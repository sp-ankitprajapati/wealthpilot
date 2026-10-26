> Historical source walkthrough. Current behavior, numbers and fixes: README.md and LEARNING_HINGLISH.md.

# WealthPilot: loan analysis se graph code tak

CarbonLedger ke baad padho. Dono projects ka shared lesson hai: AI interpretation aur deterministic arithmetic ki responsibilities alag rakho. WealthPilot mein state, policy retrieval, memory aur reviewer pause extra hain. Exact source `07_WEALTHPILOT_CODE_REFERENCE.md` mein milega.

## 1. Problem aur actual input

SME ka matlab small/medium enterprise. Loan underwriting mein lender applicant ki financial position aur repayment ability assess karta hai. WealthPilot is process ka preliminary analysis prototype hai.

Actual inspected input **financial JSON** hai. `document_parser` naam se yeh assume mat karo ki yeh PDF padhkar figures extract karta hai. Current node provided JSON ko FinancialStatement schema se validate karta hai.

Output pehle **recommended_verdict** aur rationale hai. Phir human approval interrupt. Reviewer ke response ke baad **final_status** set hota hai. Invalid input pe validation rejection ho sakti hai, jo loan reject decision se different hai.

## 2. Financial terms bilkul basics se

| Field/concept | Meaning | Simple illustration |
|---|---|---|
| Revenue | Sales/business se generated income, costs subtract karne se pehle | Sales 100 |
| Prior revenue | Previous comparable period ki sales | Last year 90 |
| COGS | Sold goods ki direct cost | Materials/production cost 50 |
| Operating expenses | Business operations ke other expenses | Admin/rent etc. 20 |
| EBIT | Earnings before interest and taxes | Simplified 100 - 50 - 20 = 30 |
| Interest expense | Borrowed funds ka interest cost | Interest 5 |
| Net income | Relevant expenses/taxes ke baad bottom-line profit/loss | EBIT se same nahi |
| Total assets | Business ke economic resources ka total | Cash, inventory, machinery |
| Current assets | Short-term/current operating cycle assets | Cash, receivables, inventory |
| Current liabilities | Short-term/current operating cycle obligations | Near-term payables/debt |
| Total liabilities | Business ke total obligations | Current + longer-term obligations |
| Equity | Owners ka accounting interest | Simplified assets minus liabilities |
| Retained earnings | Past profits ka accumulated retained portion, losses se negative bhi | Cash balance ke equal zaroori nahi |
| Debt service | Specified period mein due principal + interest | Annual 20 |
| Liquidity | Near-term obligations meet karne ki position | Current ratio ek indicator |
| Leverage | Owners' funds ke comparison mein obligations/debt exposure | D/E ek indicator |
| Risk flag | Specific rule breach ka signal | DSCR below 1 |

Numbers ka period aur units consistent hone chahiye. Monthly EBIT ko annual debt service se bina adjustment compare karna meaningful result nahi dega. Currency, reporting period, accounting definitions aur data quality real implementation mein clarify karni hongi.

Yeh descriptions project samajhne ke liye hain. Implemented ratio conventions ko kisi lender ki universal policy mat samjho.

## 3. High-level workflow

```text
Financial JSON
  document_parser: schema validation
  risk_analyst: ratios + red flags
  credit_researcher: relevant policy evidence
  final_review: model recommendation + explanation
  human_approval: pause, reviewer input, final status
```

Supervisor Python function decide karta hai next kaunsa step missing hai. Current graph specialists ko sequence mein route karta hai. Yeh multiple LLM workers ka parallel execution nahi hai.

## 4. File map

| Module | Purpose |
|---|---|
| schemas.py | FinancialStatement contract aur validators |
| ratios.py | Fixed arithmetic |
| synthetic_data.py | Known profiles ke sample financial cases |
| tools.py | Ratio/flag functions ko LangChain tools banana |
| agent.py | Earlier single-agent tool loop aur provider factory |
| policy_docs.py | Illustrative lending policy clauses |
| rag.py | Chunking, index aur retrieval tool |
| multi_agent.py | State, specialists aur supervisor |
| case_graph.py | Graph build, run wrapper aur pause detection |
| mcp_server.py | Three tools ka MCP interface |
| fairness.py | Region variants aur group outcome metric |
| tracing.py | Node timing/log wrapper |
| eval_harness.py | Labeled cases pe results/accuracy/latency report |
| demo.py | Single-case presentation flow |
| paths.py | Data file location constants |
| __init__.py | Package initialization/version metadata |
| tests/ | Modules ke automated checks with fakes |
| docs/BUILD_LOG.md | Design choices, bugs aur reported profiling story |

## 5. Pydantic schema: data boundary

Actual class ke field examples:

```python
class FinancialStatement(BaseModel):
    company_id: str
    period: str
    revenue: float
    prior_period_revenue: float
    # Original class mein aur financial fields bhi hain.
```

Yeh excerpt full class nahi hai. Actual class mein income, assets/liabilities, equity, retained earnings, debt service aur optional applicant_region fields hain.

Important actual validation:

```python
@field_validator("revenue", "total_assets")
@classmethod
def must_be_positive(cls, v: float, info) -> float:
    if v <= 0:
        raise ValueError(f"{info.field_name} must be positive, got {v}")
    return v
```

Decorator fields choose karta hai. `v` provided parsed value hai. Non-positive value reject hoti hai. Yeh sirf selected fields ka check hai. Har liability/expense/finite value automatically validate ho gayi, aisa nahi.

Model validator ka relevant calculation:

```python
implied = self.revenue - self.cogs - self.operating_expenses
if implied != 0 and abs(self.ebit - implied) / abs(implied) > 0.25:
    raise ValueError(...)
```

Last line yahan shortened hai; original detailed message reference mein hai. Revenue-costs-opex se expected EBIT nikala. Given EBIT ka relative difference 25% se zyada ho toh reject. `abs` magnitude leta hai. `implied != 0` division-by-zero avoid karta hai, lekin zero implied EBIT case ki mismatch validation skip bhi karta hai.

`model_validate_json` JSON text parse + model validation karta hai. `model_dump` model se Python dict deta hai. `model_dump_json` JSON text deta hai. Yeh representations distinguish kar pao.

## 6. Ratios: formulas aur intuition

### DSCR, project definition

```python
dscr = fs.ebit / fs.total_debt_service if fs.total_debt_service else float("inf")
```

EBIT 30, debt service 20: DSCR 1.5. Project ke numerator se income debt-service amount ki 1.5 times hai. EBIT 15 ho toh 0.75. Denominator zero pe code infinity deta hai. Yeh explicit implementation convention hai. Real lender different cash-flow definition use kar sakta hai.

### Current ratio

Current assets / current liabilities. 60 / 40 = 1.5. Short-term resources vs obligations ka indicator. Inventory actual cash jitni immediately usable ho, zaroori nahi. Single ratio complete conclusion nahi.

### Debt-to-equity

Project total_liabilities / shareholder_equity use karta hai. 80 / 20 = 4. Project ke formula mein equity ke har 1 unit ke against 4 units liabilities hain. Negative/zero equity ke treatment ko explicit policy chahiye. Zero pe infinity code path hai; negative value high-leverage rule ko confuse kar sakti hai.

### Interest coverage

EBIT / interest_expense. 30 / 5 = 6. Project 6 report karega. Interest zero pe infinity. Principal repayment is formula mein denominator ka part nahi.

### Revenue growth

```text
(current revenue - previous revenue) / previous revenue × 100
(110 - 100) / 100 × 100 = 10%
```

Previous revenue zero pe project 0.0 return karta hai. "No prior denominator" ko zero growth kehna business-wise questionable ho sakta hai. Yeh current code ka convention hai, universally valid financial interpretation nahi.

### Altman Z-prime

Project multiple balance-sheet/income ratios ko weighted score mein combine karta hai:

```text
X1 = (current assets - current liabilities) / total assets
X2 = retained earnings / total assets
X3 = EBIT / total assets
X4 = book equity / total liabilities
X5 = revenue / total assets

Z' = 0.717 X1 + 0.847 X2 + 3.107 X3 + 0.420 X4 + 0.998 X5
```

Weighted formula implementation ko samjho. Score default probability nahi. Model ki applicability har sector/company pe automatically established nahi hai.

Actual branch:

```python
if altman_z > Z_SAFE_CUTOFF:
    z_zone = "safe"
elif altman_z > Z_DISTRESS_CUTOFF:
    z_zone = "grey"
else:
    z_zone = "distress"
```

Constants 2.90 aur 1.23 hain. Exactly 2.90 grey hai, exactly 1.23 distress. Greater-than aur greater-than-or-equal ka boundary difference interview mein pooch sakte hain.

## 7. Tools: same business logic ko reusable banana

`calculate_ratios(statement_json)` Pydantic se validate karke `compute_ratios` call karta hai. `@tool` decorator LangChain interface expose karta hai. `risk_analyst` ise `.invoke({"statement_json": ...})` se call karta hai.

`detect_red_flags` ratios JSON parse karta hai aur five rules check karta hai:

1. DSCR < 1.0.
2. altman_zone == distress.
3. Current ratio < 1.0.
4. Debt-to-equity > 2.5.
5. Revenue growth < 0.

Return `json.dumps(flags)` hai. Empty list ka string `"[]"` caller ko consistent text format deta hai. Repo build log/tool comments mein tool-message empty-list handling ka version-specific bug explain hai.

Subtlety: ratio function values round karke return karta hai; flags returned values check karte hain. True DSCR 0.996 rounded 1.00 ho toh `< 1` rule miss ho sakta hai. Calculation precision aur display rounding separate karna useful improvement hai.

## 8. Sample case trace

Dataset case `SME-OVER-001` mein inspected calculation se:

```text
DSCR = 0.72                  flag
Current ratio = 1.27         no current-ratio flag
Debt-to-equity = 4.60        flag
Interest coverage = 1.51     displayed, no separate flag rule here
Revenue growth = -1.1%       flag
Altman Z-prime = 1.54        grey, no distress flag
```

Dataset expected verdict decline hai. Deterministic ratio results previous review mein reproduce hue. Live LLM ka actual response verify nahi hua. Ratios/flags ke baad model policy context se recommendation synthesize karta hai; dataset label aur real model decision ko same evidence mat samjho.

## 9. Earlier single-agent loop: agent.py

`build_graph(model)` model ko TOOLS bind karta hai. `call_model` messages se response leta hai. `should_continue` last message mein tool_calls dekhkar tool node ya END choose karta hai. Tool node execute hone ke baad flow model ko wapas jata hai.

```text
Model response
  tool call present? ToolNode execute, result model ko
  tool call absent? END
```

Prompt bolta hai calculate_ratios first, flags next. Earlier graph sequencing ko strictly enforce nahi karta; model behavior pe depend karta hai. Later case graph parser/analyst order more structurally define karta hai.

`get_model(provider)` selected provider ka wrapper import/create karta hai. Imports branch ke andar hain, isliye har provider ka package import/startup pe load karna necessary nahi. Hardcoded model IDs future mein provider availability ke saath verify karne padenge.

## 10. State: case ki working notebook

CaseState ko ek shared notebook samjho. Initially raw_statement_json aur case_id hain. Parser financial_statement add karta hai. Analyst ratios/red_flags add karta hai. Research policy_findings add karta hai. Final review recommendation/rationale add karta hai. Human step final_status/reviewer_notes add karta hai.

TypedDict ke `total=False` ka matlab all declared keys ka initially present hona required nahi. `messages: Annotated[list, add_messages]` message list updates ke combine behavior ko specify karta hai. `document_parser_runs` aur `risk_analyst_runs` counters tests ko duplicate work detect karne mein help karte hain.

Actual supervisor:

```python
if state.get("final_status"):
    return END
if "financial_statement" not in state:
    return "document_parser"
if "ratios" not in state:
    return "risk_analyst"
if "policy_findings" not in state:
    return "credit_researcher"
if "recommended_verdict" not in state:
    return "final_review"
return "human_approval"
```

Yeh LLM prompt nahi, regular Python branch logic hai. Har step ke baad router state dobara inspect karta hai.

**Critical example:** `policy_findings=[]` means research complete but no findings. Missing key means research not run. `if not state.get("policy_findings")` dono situations same treat karega aur clean case mein repeated research/loop cause kar sakta hai.

## 11. Specialist nodes

**document_parser:** FinancialStatement.model_validate_json. Valid pe dict store. ValidationError pe parse_error aur final_status rejected_invalid_data. Supervisor immediately END route kar sakta hai.

**risk_analyst:** Financial dict JSON banata hai, ratio tool call karta hai, flag tool call karta hai, results state mein return. Is node mein LLM nahi.

**make_credit_researcher:** Factory injected model aur retrieve tool use karti hai. No red_flags pe `policy_findings=[]`. Otherwise model ko policy-search prompt; returned tool calls mein matching tool execute; JSON findings combine. Model tool request na de toh bhi empty findings ho sakti hain. Current code mandatory evidence retrieval success enforce nahi karta.

**make_final_review:** Ratios, flags aur findings ka summary_input banata hai. Raw financial_statement/region prompt mein directly include nahi. Model se recommendation/rationale mangta hai. Generated output validate karne ka current approach weak hai, next section dekho.

**human_approval:** Evidence payload ke saath interrupt, then reviewer decision process. Yeh actual case graph completion boundary hai.

## 12. Verdict parsing ka interview-worthy bug

Actual source line:

```python
verdict = next((v for v in ("approve", "review", "decline") if v in text), "review")
```

`next` first match leta hai. Word order approve, review, decline hai. Text `"decline; do not approve"` ho toh approve substring present hai, isliye approve choose ho sakta hai. First word maangne wali prompt actual parser ko safe nahi banati.

Better practice: model se separate recommended_verdict enum aur rationale fields lo. Pydantic/schema se allowed values validate karo. Citation IDs aur numeric consistency ko independently check karo. Reviewer override bhi allowed enum hona chahiye. Yeh suggested change hai.

## 13. Policy RAG

`policy_docs.py` illustrative policies rakhta hai. `chunk_policy_docs` headings/clause boundaries pe split karta hai aur source/clause_id metadata attach karta hai. Oversized clause ke liye `_simple_recursive_split` fallback hai. Generic chunk size fill karne ke liye two policy clauses merge ho jaayein toh citation wrong ho sakti hai.

Actual index function:

```python
def build_policy_index(docs: list[dict], embeddings) -> Chroma:
    texts, metadatas = chunk_policy_docs(docs)
    return Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
```

Texts numeric embeddings mein map hote hain aur Chroma searchable index banata hai. `make_retrieve_policy_tool` us index ko closure mein hold karta hai. Similarity search query ke top-k clauses laati hai. Tool output objects mein text, source aur clause milte hain.

Live example local HuggingFace embedding model use karta hai. Tests fake embeddings use karte hain. In fake tests ki retrieval wiring pass hona actual semantic retrieval quality prove nahi karta. Policy corpus se missing rule ko LLM memory se invent karna avoid karne ke liye insufficient-evidence path useful hai.

## 14. Graph building aur dependency injection

`build_case_graph(model, retrieve_policy_tool, checkpointer=None)` nodes register karta hai. Conditional entry supervisor se; har specialist ke baad supervisor routing; `compile(checkpointer=...)` runnable app banata hai.

Model/tool/checkpointer arguments se pass hue. Yeh dependency injection hai. Test fake model pass kar sakta hai, live demo provider model. Same graph business sequence reuse hota hai.

`run_case` invoke call karta hai; documented repo behavior mein completed state pe no node run ho aur result None ho toh checkpoint values fallback leta hai. `is_awaiting_approval` result mein __interrupt__ check karta hai. Exact framework return behavior installed version ke saath verify karo.

## 15. Pause aur resume

Actual approval code ka important part:

```python
decision = interrupt(payload)

if decision.get("approved"):
    return {
        "final_status": decision.get("override_verdict") or state.get("recommended_verdict"),
        "reviewer_notes": decision.get("notes", ""),
    }
return {"final_status": "sent_back", "reviewer_notes": decision.get("notes", "")}
```

Checkpointer + thread_id se graph pause state identify hota hai. Resume input example:

```python
app.invoke(Command(resume={"approved": True}), config=config)
```

Same config ke thread_id se correct case resume hota hai. `approved=True` ka matlab reviewer recommendation/override accept kar raha hai. Recommendation decline thi toh final_status decline bhi ho sakta hai. Isko unconditional loan approval mat samjho.

Interrupted node resume pe beginning se run hota hai. Payload preparation repeat ho sakti hai. Email/payment/database insert jaise side effects before interrupt hon toh duplicate effects ka risk hai. Is project ka logging wrapper completion ke baad log karta hai.

## 16. Memory aur case lifecycle

InMemorySaver process memory mein state rakhta hai. Process restart pe durable memory claim nahi kar sakte. SQLite-backed checkpoint tests fresh graph object ke saath saved state access verify karte hain. Checkpointer optional parameter hone se har execution durable nahi ho jata.

New thread_id separate case state. Same completed thread_id ke saath changed financial numbers bheje toh old final_status/ratios remain kar sakte hain aur supervisor immediate END kar sakta hai. Resume aur new revision ka distinction design karna hai.

Practice: finish case A; changed revenue same ID pe submit; state inspect. Phir new revision ID ya derived-state invalidation design karo. Proposed fix ko actual test/reproducer ke saath explain karo.

## 17. MCP server

`build_server(retrieve_policy_tool)` MCPServer banakar ratio, flags aur retrieve_policy register karta hai. Ratio wrapper underlying LangChain tool invoke karta hai. Ek formula, multiple callers: model tool loop, deterministic specialist aur MCP client.

ValidationError ko ToolError mein convert karna caller ko understandable tool-level failure dene ki koshish hai. SDK method ke in-process versus protocol behavior/version ka difference repo log mein documented hai.

Server full underwriting approval graph ko single tool mein expose nahi karta. Real external-client interoperability prove karne ke liye client discover tools, call tool aur invalid input scenario demonstrate karo. In-process tests narrower evidence hain.

## 18. Fairness: exactly kya safeguard hai?

`applicant_region` collect ho sakta hai, lekin ratio calculation aur final-review prompt directly use nahi karte. `generate_region_variant_cases` same financial data ko regions change karke clone karta hai. Tests identical ratios, flags, captured prompts aur fake-model outcomes check karte hain.

`measure_group_parity` har group approval rate aur min rate / max rate nikalta hai. Agar rates 0.6 aur 0.8 hon toh ratio 0.75. Yeh mathematical illustration hai, legal threshold/advice nahi. No approvals/insufficient data pe undefined case handle karna hai.

Narrow conclusion: direct region exclusion ke controlled regression checks hain. Broad conclusion "system proven fair for lending" supported nahi. Proxies, different financial populations, model behavior, policies aur human decisions further evaluation demand karte hain.

## 19. Tracing, evaluation aur synthetic data

`traced_node` function wrapper original node ka start/end measure karke node name, case_id, latency log karta hai. Returned messages mein usage_metadata ho toh token information attach karta hai. Har model call necessarily captured nahi: researcher apna model response messages mein return nahi karta.

`generate_synthetic_case` healthy, illiquid, overleveraged aur declining profiles ke random financial data banata hai. `generate_eval_set` cases aur expected labels produce karta hai. Dataset 20 cases ka hai. Synthetic labels useful controlled practice hain, real underwriting expert-ground-truth ka substitute nahi.

`run_eval_harness` graph run karta hai, paused recommendation auto-accept karta hai, predicted/expected compare karta hai. `build_report` accuracy, per-class results, confusion counts, latency aur rows collect karta hai. `print_report` display. Invalid-case/pause failures ke robust handling ko production exercise mein evaluate karna chahiye.

`run_demo` one case ke stages dikhata hai, InMemorySaver use karta hai aur demonstration ke liye auto-accept karta hai. Console demo ka auto-accept real human interface nahi.

## 20. Tests aur performance claims

Static count 59 test functions, visible parametrizations ke baad 65 cases. Tests ratio/schema, single-agent, RAG, checkpointed graph/HITL, MCP, fairness, tracing aur eval arithmetic cover karte hain. Full suite is review mein run nahi hui; README passing result independently reproduce karna baaki hai.

Build log reported import timing 7.6s se 1.06s. Heavy text splitter ki transitive dependencies remove karke local fallback add kiya gaya. 7.6/1.06 about 7.17x import speedup hai. End-to-end LLM inference/request seven times faster hua, yeh claim mat karo. Raw timings, environment aur exact command reproduce karo.

## 21. Interview explanation

"WealthPilot structured SME financial data se preliminary underwriting recommendation banata hai. Pydantic validation aur Python ratio tools numeric layer handle karte hain. LangGraph specialist steps ko state ke basis pe route karta hai. Policy RAG recommendation ko retrieved evidence de sakta hai. Case graph human review interrupt pe pause hota hai. MCP same tools external clients ke liye expose karta hai. Current limitations mein synthetic evaluation, generated verdict validation aur fairness evidence ka limited scope hain."

Ab code band karke state keys ka flow draw karo. Pehle financial ratios samjho, phir graph. Saare libraries ek saath install/run karke flow samajhne ka pressure mat lo.

Sources: [pinned WealthPilot snapshot](https://github.com/sp-ankitprajapati/wealthpilot/tree/851c96c6d6fc66f95e326696b951cdf474698fd5), [LangGraph interrupt behavior](https://docs.langchain.com/oss/python/langgraph/interrupts), [MCP architecture](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture).
