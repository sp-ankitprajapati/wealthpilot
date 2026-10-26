# WealthPilot update: Hinglish code aur interview guide

Current source mein 89 tests pass hain. Historical original walkthrough se current behavior ke differences yahan explain hain.

## 1. Schema aur undefined numbers

FinancialStatement Pydantic model input contract hai. NaN/Infinity aur Boolean amounts reject hote hain. Negative profit/equity possible hain, lekin negative debt service / expense input invalid maana hai. Zero denominator par infinity JSON mein bhejna standard JSON nahi. Ab value null aur undefined_ratios list mein naam aata hai. Zero prior revenue par growth 0% fabricate nahi karte. Negative equity ko “negative debt/equity means low risk” nahi maana; manual review flag aata hai.

## 2. Verdict parsing ka bug

Old code response mein approve word pehle search karta tha. “Decline: cannot approve” mein approve milkar wrong recommendation aa sakti thi. Ab model.with_structured_output supported ho toh Recommendation schema verdict enum validate karta hai. Text-only test fake ke liye exact first word prefix required hai. Invalid output manual review banata hai. Provider failure successful recommendation nahi banata.

## 3. Human approval strict kyun?

Python mein nonempty string “false” truthy hoti hai. Old if decision.get('approved') usko approval samajh sakta tha. StrictBool sirf true/false Boolean accepts karta hai. Override sirf approve/review/decline ho sakta hai aur approved=true par hi valid hai. Wrong resume payload final status set nahi kar sakta. Tests real graph pause/resume se verify karte hain.

## 4. Checkpoint aur revision

thread_id case ka stored state select karta hai. Agar same ID par changed financial input daal do aur old final status bacha ho, supervisor old decision return kar sakta tha. run_case ab changed input reject karta hai; new thread ID/revision chahiye. Direct graph callers ke liye supervisor bhi raw statement aur cached validated statement compare karta hai. Caller derived ratios/final status inject nahi kar sakta through run_case.

Practice: healthy case complete karo. Same ID par overleveraged statement bhejo, expected error dekho. New ID par bhejo, updated ratios aur new human pause check karo. Memory persist hona aur correct business lifecycle same cheez nahi.

## 5. RAG collection isolation

Chroma.from_texts default collection share ho sakti hai jab same process mein multiple indexes bante hain. Ab per index unique collection name hai. Ek policy index ka text dusre mein leak nahi hona chahiye. Tests different sources ke indexes bana kar isolation prove karte hain. Current policy index ephemeral hai; persistent case memory LangGraph checkpointer se aati hai. Clause chunking source+clause ID preserve karti hai. Citation present hona relevance correct hone ka proof nahi.

## 6. Offline demo aur live model

python -m wealthpilot.offline_demo deterministic ratio engine, bundled policy text aur controlled model se graph trace karta hai. Recommendation fixed review hai. Human pause demonstrate karke explicitly sent_back karta hai. Isko real LLM accuracy mat bolo. Live provider / embedding paths separate dependencies aur credentials maangte hain. WEALTHPILOT_MODEL explicitly choose karo; environment example automatically loaded nahi hota.

## 7. Fairness aur performance ka scope

Region variation tests identical financial data par applicant_region ko multi-case final-review prompt se exclude karna prove karte hain. Proxies, population distribution, other pathways aur reviewers ka real fairness established nahi. Older single agent raw statement prompt mein leta hai; us path par same exclusion claim applicable nahi.

New benchmark mein 3 fresh process imports ka median about 2.14 s hai on checked Windows runtime. Yeh OS cold cache ya full request latency nahi. Historical 7.6→1.06 story docs/BUILD_LOG.md mein old observation hai; new run ne woh historical speedup ratio reproduce nahi kiya.

## 8. Interview aur coding practice

Explain TypedDict versus Pydantic versus dataclass. Missing key versus empty list routing. Tool binding versus invoke. Interrupt first kyun? StrictBool kyun? Revisions ke liye new thread kyun? Null versus Infinity? RAG chunk versus document? Policy collection versus checkpoint database? Fake embedding tests kya prove karte? MCP server kaunse 3 tools expose karta? Typed verdict mein extra enum value kaise fail hoti? Same financials / different region test overall fairness kyun nahi?

Coding drills: exact verdict parser, StrictBool approval schema, finite financial JSON, zero-denominator record, two isolated indexes aur changed revision guard khud implement karo. Tests/test_workflow_boundaries.py ka expected behavior pehle words mein bolo. Ek bug reproduce karke before/after demonstrate karo. All possible interview questions ka guarantee nahi; assumptions aur failure cases explain karna ownership ka practical proof hai.

Technology coverage: Python/typing/closures, JSON/Pydantic, deterministic ratios, LangChain tools / messages, LangGraph state / routing, RAG/Chroma/embeddings, SQLite checkpoints, human approval, fairness checks, MCP 2.x, pytest/fakes/fixtures, tracing, evaluation, dependency extras, packaging aur GitHub CI. Original detailed foundations companion files mein hain.


Detailed reading: [Foundations](FOUNDATIONS_HINGLISH.md), [original walkthrough](ORIGINAL_WALKTHROUGH_HINGLISH.md), [practice code](practice_basics.py).
