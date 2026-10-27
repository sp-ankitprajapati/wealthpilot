# Skills zero se: naam, meaning, example aur project use

Is file ko glossary aur small lessons ki tarah use karo. Har term pehle read karke expert banna zaroori nahi. Relevant project lesson ke saath wapas aao.

## 1. Project, repository, module aur pipeline

**Project:** Ek problem solve karne wala complete kaam. Isme code, data, requirements, tests aur documentation ho sakte hain.

**Repository/repo:** Git se track hone wala project folder. GitHub us repository ko online host kar sakta hai. Git aur GitHub same cheez nahi: Git history/version tracking ka tool hai, GitHub hosting/collaboration service hai.

**Module:** Ek Python file. `calculator.py` module hai. **Package:** Related modules ka importable group. `app` aur `wealthpilot` packages ke examples hain. WealthPilot mein `src/` folder ke andar actual package hai.

**Dependency/library:** Kisi aur ka reusable code jo project import karta hai. `scikit-learn` ML library hai. Package ka installation name aur Python import name different ho sakte hain: `scikit-learn` install hota hai, `sklearn` import hota hai.

**Pipeline:** Steps ka sequence. Jaise invoice text lena, items nikalna, category choose karna aur amount calculate karna. Har step ka input/output samajhna architecture samajhne ki shuruaat hai.

## 2. Python mein data kaise represent hota hai?

```python
company = "ABC Ltd"                 # str: text
quantity = 100                      # int: whole number
factor = 2.68                       # float: decimal approximation
needs_review = True                 # bool: True ya False
items = ["diesel", "electricity"]    # list: ordered collection
item = {"quantity": 100, "unit": "litre"}  # dict: key-value data
unknown = None                      # abhi value available nahi
```

`item["quantity"]` se 100 milta hai. Missing key par square bracket access `KeyError` de sakta hai. `item.get("source_label", "")` missing hone par empty string dega. Default dene ka matlab yeh nahi ki missing business data hamesha accept karna chahiye.

List aur dictionary ko confuse mat karna: list mein multiple items; dict ek record ke named fields rakh sakta hai. Project mein line items ki **list** hai aur har item ek **dict** ho sakta hai.

### Function

```python
def emissions_kg(quantity, factor):
    return quantity * factor
```

Function ek named operation hai. Parameters input ke naam hain. `return` output caller ko deta hai. `print` screen pe dikhata hai. Dono same nahi hain. Function ko dusra function call kar sakta hai.

### Loop aur condition

```python
for item in items:
    print(item)

if confidence < 0.55:
    needs_review = True
else:
    needs_review = False
```

Loop har item pe kaam repeat karta hai. Condition decide karti hai kaunsa branch chalega. CarbonLedger ka classifier gate isi idea ka use karta hai.

### Class, object, method aur self

Class related data/behavior ka structure hai. Object us class ka actual instance hai. Class ke andar function ko method kehte hain. `self` current object ko refer karta hai.

`CarbonLedgerAgent()` se object banta hai. `agent.process_document(text)` us object ka method call hai. `__init__` object banne pe initial setup karta hai, jaise classifier aur retriever load karna.

### Dataclass

```python
from dataclasses import dataclass

@dataclass
class ActivityRecord:
    activity_key: str
    quantity: float
    source_label: str = ""
```

Dataclass fields wale objects banana convenient karti hai aur initializer jaise routine methods generate karti hai. `quantity: float` type hint hai. Dataclass normally is hint ko enforce karke bad string reject nahi karti. Pydantic ka role alag hai.

`field(default_factory=list)` ka matlab har instance ko fresh list do. Mutable default ko galat jagah reuse karna shared-data bugs create kar sakta hai. CarbonLedger ki EmissionsReport records list ke liye default_factory use karti hai.

### Type hints aur TypedDict

`def f(text: str) -> dict` reader/type checker ko expected types batata hai. Python is hint se automatically har call validate nahi karta. `TypedDict` dictionary ke expected keys/types document karta hai. WealthPilot ka CaseState typed graph schema bhi define karta hai, lekin woh Pydantic ki financial validation ka replacement nahi hai.

### List comprehension aur generator

```python
quantities = [item["quantity"] for item in records]
total = sum(record.quantity for record in activities)
```

Pehli line list banati hai. Dusri generator se values `sum` ko deti hai. Inhe compact loop samjho. Pehle normal loop likhkar samjho, phir concise syntax read karo.

### Empty aur missing

`[]` empty list hai, `None` no value ho sakta hai, aur absent dictionary key alag situation hai. `if not value` kai falsy values ko ek saath treat karta hai: `0`, `False`, `None`, `""`, `[]`. WealthPilot routing mein missing aur empty ko alag rakhna bahut important hai.

### Exceptions

Unexpected/invalid operation par error raise hota hai. `try/except` specific expected error handle kar sakta hai. Example: invalid JSON pe `JSONDecodeError`, invalid schema pe Pydantic `ValidationError`.

Sab errors ko catch karke empty result return kar dena failure hide kar sakta hai. Interview mein explain karo: kis error ko recover karoge, kis error ko user review tak bhejoge aur kya log karoge.

### Decorators, factories aur closures

Decorator `@tool` function ko tool system mein usable banata hai. Pydantic ka `@field_validator` custom field-check register karta hai. Decorator magic word nahi: function/class ke behavior ya registration ko modify karne ka Python pattern hai.

Factory function koi configured object ya function return karti hai. `make_retrieve_policy_tool(vectorstore)` ek function banati hai jo us specific vectorstore ko yaad rakhta hai. Inner function ka outer variable yaad rakhna closure hai.

### Imports, main guard aur paths

`from app.calculator import calculate_emissions` module ka function import karta hai. `if __name__ == "__main__":` wala block file ko directly/module command se run karne par execute hota hai, normal import par nahi.

Top-level code import ke waqt run ho sakta hai. Isi wajah se import ke waqt heavy model load karna startup ko slow kar sakta hai. Path current working directory se dependent ho sakta hai; project mein `__file__`/paths module ki help se stable file locations banayi gayi hain.

## 3. JSON, schema aur Pydantic

JSON data exchange ka text format hai. Python dict memory mein Python object hai. Similar dikhne se dono same nahi hote.

```python
import json
text = '{"quantity": 100, "unit": "litre"}'
data = json.loads(text)       # JSON text se Python dict
again = json.dumps(data)      # Python dict se JSON text
```

JSON syntactically valid ho sakta hai lekin business-wise wrong: `{"quantity": -100}`. Schema expected fields/types define karta hai. Pydantic input ko parse/validate karta hai aur custom rules bhi run kar sakta hai. Default configuration mein kuch types convert bhi ho sakte hain, isliye strict validation alag choice hai.

WealthPilot ka FinancialStatement Pydantic BaseModel hai. Revenue positive hona aur EBIT ka components se roughly match karna checks hain. "Pydantic use hua" ka matlab financial data ke har possible issue ka check implement ho gaya, yeh assume mat karna.

## 4. Frontend, backend, API aur HTTP

**Frontend/UI:** User ka input lena aur result dikhana. CarbonLedger mein Streamlit. **Backend:** Data processing/business logic. FastAPI backend entry point provide karta hai.

**API:** Caller aur service ke beech input/output ka defined interface. Restaurant analogy mein menu aur order format samjho. Caller ko andar ka calculation code copy nahi karna, defined request deni hai.

**HTTP request:** Client server ko method, path aur optional data bhejta hai. **GET** usually information read karne ke liye; **POST** data submit/process karne ke liye. **Endpoint/route:** Jaise `/process`. **Request body:** Jaise `{"document_text": "..."}`. **Response:** Summary/audit data wala JSON.

**FastAPI:** Python web API framework. `@app.post("/process")` function ko route se connect karta hai. **Uvicorn:** Application serve karne ka ASGI server. Port 8000 local address ka ek listening endpoint hai.

Important actual architecture: CarbonLedger ka Streamlit frontend shared Python agent ko directly call karta hai. Current code mein Streamlit se FastAPI ko HTTP request nahi jaati. UI aur API dono same business logic ke separate entry points hain.

**CORS:** Browser ko cross-origin requests ke rules batata hai. Authentication nahi hai. API key aur app login/auth alag cheezein hain.

## 5. Streamlit, pandas aur ReportLab

Streamlit Python statements se text fields, buttons, tables aur metrics wali interface banata hai. Interaction par script rerun ho sakti hai. `st.session_state` ek user session ke reruns ke beech values rakhne mein help karta hai. CarbonLedger yahan agent object store karta hai. Yeh durable audit database nahi hai.

pandas ka DataFrame rows/columns wala in-memory tabular data hai. Audit entries ko DataFrame mein convert karke display kiya gaya hai. DataFrame hona database persistence ka proof nahi hai.

ReportLab Python se PDF banane ki library hai. Paragraphs, tables aur document layout create karta hai. Narrative words Gemini de sakta hai, lekin PDF file ka layout/build ReportLab karta hai. CarbonLedger mein full audit list PDF ko nahi jaati; audit decision count jaata hai.

## 6. AI, ML, LLM aur Gemini

**AI:** Broad area jisme systems tasks perform karte hain jo intelligence se related hote hain. **Machine learning:** Data se patterns/parameters learn karke predictions banana. **LLM:** Large language model jo text/context ko process karke language output ya tool requests generate kar sakta hai.

**Gemini API:** Project remote provider model ko request bhejta hai. Project ne Gemini ko scratch se train nahi kiya. **API key:** Access credential. **Prompt:** Model ko diya input/instruction. **System instruction:** Behavior/task guidance. **Temperature:** Generation randomness ko affect karne wali setting; low temperature correctness ki guarantee nahi.

**Token:** Model ke text processing ka unit, often word ka part bhi ho sakta hai. **Context window:** Ek request mein model jitna context handle kar sakta hai uski limit. **Latency:** Response aane mein laga time. **Hallucination:** Fluent output mein unsupported/incorrect information aana.

**Structured output:** Expected schema mein result mangna, jaise JSON with a verdict field. Valid JSON milna aur correct business decision milna alag checks hain. Output ko validate karna padega.

## 7. Supervised learning: labels, training aur inference

Example text `"generator diesel fuel"`, correct label `"diesel_litre"`. Aise labeled examples se category predict karna supervised classification hai.

**Training:** `fit(texts, labels)` se model parameters learn hote hain. **Inference:** Trained model naya text classify karta hai. **Feature:** Input ki numeric representation. **Class:** Possible label. **Prediction:** Chosen output label. **Confidence score:** Model ki reported probability-like score, correctness ka proof nahi.

**Train/validation/test split:** Training pe learn karo, validation pe choices tune karo, final held-out test pe estimate karo. Near-duplicate invoice lines splits mein aa gayi toh score misleadingly high ho sakta hai. Isko leakage/contamination ke angle se inspect karo.

**Overfitting:** Training examples pe achha aur genuinely new inputs pe weak performance. CarbonLedger ke 48 examples/15 labels small demo dataset hain. Real performance ke liye representative evaluation chahiye.

## 8. TF-IDF, n-grams aur sparse vectors

Computer ko text ko numbers mein represent karna hota hai. TF-IDF words/terms ko weights deta hai: document mein term kitna aata hai aur corpus mein kitna uncommon hai.

Tiny illustration: vocabulary `[diesel, electricity, generator]` ho toh different descriptions ke vectors mein alag positions pe weights honge. Yeh illustration hai; sklearn actual tokenization, frequency normalization aur IDF formula use karta hai.

**Unigram:** Ek token, jaise diesel. **Bigram:** Do adjacent tokens, jaise diesel fuel. **Stop words:** Common words ko ignore karne ka configured set. **Vocabulary:** Training se learned known terms. **Sparse vector:** Bahut positions zero hoti hain, isliye nonzero entries efficiently store ki jaati hain.

`fit_transform` vocabulary/weights learn karke transform karta hai. `transform` already learned vocabulary use karta hai. New prediction ke waqt nayi vocabulary fit karna feature alignment kharab kar sakta hai.

## 9. MLP aur scikit-learn Pipeline

**MLP:** Multi-Layer Perceptron, ek feedforward neural network. Weighted numeric inputs hidden layers se pass hote hain aur category scores bante hain. CarbonLedger mein hidden layer sizes `(64, 32)` hain. Yeh tumhara local trained classifier hai, Gemini model nahi.

**Weights:** Learned numerical parameters. **Loss:** Prediction error ko summarize karne wala training objective. **Backpropagation:** Loss ke basis par gradients calculate karne ka method. **Adam:** Parameters update karne wala optimizer. **ReLU:** Activation function, simple form `max(0, x)`. **Epoch/iterations:** Training passes/updates ke related terms; exact library behavior docs se verify karo.

**random_state=42:** Certain random choices reproducible banane mein help karta hai. Generalization ya universal numerical reproducibility ki guarantee nahi.

**Pipeline:** sklearn ka object jo vectorizer aur classifier ko ek sequence mein fit/use karta hai. Isse preprocessing aur model ko saath save/reuse karna convenient hai. **joblib:** Trained pipeline serialize/load karta hai. Serialization model ko train karne se alag operation hai.

## 10. Confidence gate aur calibration

Project maximum class probability ko 0.55 se compare karta hai. Kam score par expensive model path chalta hai. Isko escalation samjho: easy/familiar work local, uncertain work further review.

0.8 ka score automatically yeh prove nahi karta ki 80% predictions correct honge. **Calibration** mein score aur observed correctness ka relation check karte hain. **Threshold selection** mein accepted predictions ki accuracy aur escalation frequency ke beech tradeoff dekhte hain.

Accuracy = correct predictions / total predictions. **Precision:** Kisi label ki predicted positives mein kitni sahi. **Recall:** Actual positives mein kitne detect hue. **F1:** Precision aur recall ka harmonic mean. Multiclass setup mein per-class/macro/micro reporting choices matter karti hain. In sabko repo ne measure kiya hua mat assume karo.

## 11. Retrieval, cosine similarity, embeddings aur RAG

**Retrieval:** Existing evidence search karna. CarbonLedger factor descriptions search karta hai. WealthPilot policy clauses search karta hai.

**Cosine similarity:** Do vectors ki direction similarity compare karta hai. **Top-k:** Highest-ranked k candidates. Highest-ranked ka matlab relevant hone ki guarantee nahi; sab candidates weak bhi ho sakte hain.

**Embedding:** Text ko numeric vector mein map karna. Common usage mein pretrained semantic embedding model related meanings ko nearby represent karne ki koshish karta hai. TF-IDF bhi numeric text representation hai, lekin project mein lexical features aur learned semantic embeddings ko distinguish karna useful hai.

**Vector store:** Vectors aur metadata store/search karne wali system. WealthPilot Chroma use karta hai. **Metadata:** Extra identifying data, jaise source aur clause number.

**RAG:** Retrieval-Augmented Generation. Pehle evidence retrieve karo, phir model ko context ke saath answer karne do. CarbonLedger mein TF-IDF + Gemini candidates; WealthPilot mein Chroma + policy clauses. RAG ke liye hamesha neural embedding ya vector DB mandatory nahi.

**Chunk:** Document ka smaller piece. **Chunk overlap:** Adjacent pieces mein kuch shared content. **Clause-aware chunking:** Policy ki actual clause boundaries preserve karna. **Citation:** Generated answer ka supporting source/clause reference. Citation likh dena source ke claim ko support karne ki guarantee nahi.

## 12. LangChain, tool calling aur LangGraph

**LangChain:** Model integrations, messages aur tools jaise abstractions. `@tool` function ka usable schema/description expose karta hai. **bind_tools:** Model wrapper ko available tool definitions deta hai. **tool_calls:** Model ka request ki kaunsa tool kis arguments ke saath execute ho.

Model Python function ko telepathy se execute nahi karta. App/model wrapper tool request receive karta hai, tool execute karta hai aur result workflow ko deta hai.

**LangGraph:** Stateful workflows define/run karne ka framework. **Node:** Kaam karne wala step. **Edge:** Next possible step ka connection. **State:** Current case ki shared information. **Supervisor/router:** State dekhkar next node select karta hai. WealthPilot supervisor Python function hai.

**Compile:** Defined graph ko runnable app banana. **invoke:** Input dekar run karna. **END:** Current workflow completion. **Reducer:** Multiple updates ko combine karne ka rule. WealthPilot messages ke liye `add_messages` use karta hai.

**Multi-agent:** Multiple specialist roles. Har role necessarily different LLM ya parallel process nahi hota. WealthPilot ke kuch nodes deterministic hain aur graph sequentially route karta hai.

## 13. Persistence, thread_id, SQLite aur approval

**Persistence:** Data process ke baad bhi save rahe. **Checkpoint:** Graph ka saved state. **Checkpointer:** Save/load mechanism. InMemorySaver process memory use karta hai. SQLite file-based database ke through durable checkpoint possible hai, as exercised by repo tests.

**thread_id:** Particular graph conversation/case state ka identifier. OS thread nahi. Same ID old saved state select kar sakta hai. Completed case mein changed inputs bhejne ke liye revision strategy chahiye.

**Human-in-the-loop:** Human ke input/review ko workflow ka real step banana. **interrupt:** Pause aur caller ko payload. **Command(resume=...):** Resume value supply karta hai. Interrupted node beginning se rerun hota hai. Isliye pehle side effect hua toh repeat ho sakta hai.

**Idempotency:** Operation repeat hone se unwanted extra effect na ho. Example: same decision ID se duplicate database rows create na hon. Sirf function ka same return value idempotent business side effects ki guarantee nahi.

## 14. MCP

Model Context Protocol tool access ke liye common client/server interface deta hai. **Host:** AI application. **Client:** Host ka server se connect karne wala component. **Server:** Tools/resources jaise capabilities expose karta hai.

WealthPilot server teen tools expose karta hai. MCP loan ratio ka formula decide nahi karta. Formula project code mein hai. MCP boundary us function ko compatible clients tak accessible banati hai.

**stdio transport:** Client/server standard input/output streams se messages exchange karte hain. Transport protocol ka communication path hai. SDK version aur external client compatibility independently verify karni hoti hai.

## 15. Testing aur evaluation

**pytest:** Automated checks run karne ka framework. **assert:** Expected condition. **Fixture:** Test setup data/dependency. **Parametrize:** Same test ko multiple inputs pe run karna. **Fake model:** Controlled responses dene wala replacement. **Unit test:** Chhote component ka check. **Integration test:** Components ka saath behavior. **End-to-end:** Chosen complete path ko exercise karna; mocked dependencies hone par claim ka scope phir bhi limited hai.

**Evaluation:** Model/system performance measure karna. **Ground truth/label:** Comparison ka expected answer. **Confusion matrix:** Actual aur predicted classes ki counts. **Regression:** Pehle working behavior change se break ho jana.

Tests pass hone ka matlab output har real-world case mein correct hai, aisa nahi. Fake LLM graph routing prove kar sakta hai. Live reasoning quality ke liye separate evaluation chahiye.

## 16. Observability, performance aur cost

**Logging:** Events/details record karna. **Audit trail:** Decision kis input/evidence se bana, uski traceability. **Observability:** Logs/metrics se system ke behavior ko investigate karna.

**Import time:** Module load ka time. **Startup time:** Service usable hone tak time. **Request latency:** Input se response tak time. **Cold start:** Fresh process/environment startup, definition clarify karo. **Warm run:** Already initialized resources ke saath run. In metrics ko interchangeable mat use karo.

**Profiling:** Kaunsa operation/import kitna time/resources le raha hai measure karna. **Benchmark:** Defined workload par repeatable measurement. **Caching:** Reusable result/resource save karke repeated work bachana. **p50/p95:** Latency distribution ke percentiles; mean se slow-tail behavior hide ho sakta hai.

Cost ka measure calls ke saath tokens, input size aur provider pricing pe depend karega. 90% classification calls kam hona exactly 90% money saving prove nahi karta.

## 17. Environment aur security basics

Virtual environment project dependencies ko separate rakhne mein help karta hai. `requirements.txt`/`pyproject.toml` packages declare karte hain. Broad minimum versions exact reproducibility ensure nahi karte. Lock/pin karna useful production improvement ho sakta hai, current feature ka claim nahi.

Environment variable process ko configuration deta hai. `.env` file automatically har Python app read nahi karti. Loader/configuration chahiye. CarbonLedger inspected path mein direct environment read karta hai. Credentials share/commit mat karo.

Untrusted input ko validate karo. Prompt injection mein document/tool text model ko irrelevant instructions follow karne ko keh sakta hai. Tool permissions aur output validation app ki responsibility hain. API authentication, authorization, input size limits aur safe output filenames production concerns hain; repo mein complete implementation assume mat karo.

## 18. Domain basics kahan padhenge?

CarbonLedger guide mein emissions, activity, factor, scopes aur units. WealthPilot guide mein revenue, costs, assets, liabilities, equity, liquidity, leverage, ratios aur underwriting. Domain meaning samjhe bina library names ratna useful nahi.

## Official references, jab deeper padhna ho

- [Python tutorial](https://docs.python.org/3/tutorial/)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [GHG Protocol scope definitions](https://ghgprotocol.org/calculation-tools-faq)
- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [MCP architecture](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture)

Yeh file concepts ka introduction hai. Project guides exact repository code se inhe connect karengi. Advanced topics ko implementation mein present tabhi bolo jab actual code/evidence dikha sako.
