# InsightForge — AI-Powered Business Intelligence Assistant

Capstone project for **Advanced Generative AI**.

InsightForge turns a raw sales dataset into an analyst you can talk to. It
computes the business metrics with pandas, indexes them (alongside reference BI
literature) into a vector store, and puts a LangChain RAG pipeline with
conversational memory on top — then wraps the whole thing in a Streamlit UI with
dashboards, a chained insight report, QAEvalChain evaluation and LLMOps
monitoring.

---

## Quick start

```bash
# 1. environment (already created in this folder)
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 2. key — OpenRouter is the configured provider
copy .env.example .env           # then paste your OPENROUTER_API_KEY
#   (or just export it: set OPENROUTER_API_KEY=sk-or-...)

# 3. run the whole pipeline once (analysis + charts + report + evaluation)
python run_pipeline.py

# 4. launch the app
streamlit run app.py
```

The app opens on <http://localhost:8501>.

> **No API key?** Everything still runs. The assistant drops into an *offline
> extractive mode* — retrieval, memory, charts, evaluation and the UI all work,
> answers are pulled verbatim from the computed statistics instead of being
> written by an LLM.

---

## How the problem statement maps onto the code

| # | Requirement | Where |
|---|---|---|
| 1 | Data preparation | [`insightforge/data_loader.py`](insightforge/data_loader.py) — typing, calendar parts, age bands, satisfaction bands |
| 2 | Knowledge base creation | [`insightforge/knowledge_base.py`](insightforge/knowledge_base.py) — metrics → narrative documents + PDF chunks → FAISS |
| 3 | Advanced data summary | [`insightforge/analysis.py`](insightforge/analysis.py) — time periods, product/region, demographics, statistics |
| 3 | Custom retriever | [`insightforge/retriever.py`](insightforge/retriever.py) — keyword routing + live pandas slicing + vector back-fill |
| 3 | Prompt engineering | [`insightforge/prompts.py`](insightforge/prompts.py) |
| 4 | Chain prompts | [`insightforge/rag_chain.py`](insightforge/rag_chain.py) — `full_report()`: insights → recommendations → executive summary |
| 5 | RAG system setup | [`insightforge/rag_chain.py`](insightforge/rag_chain.py) — LCEL chain over the custom retriever |
| 6 | Memory integration | `WindowedMemory` + question condensation in [`rag_chain.py`](insightforge/rag_chain.py) |
| 7 | Model evaluation (QAEvalChain) | [`insightforge/evaluation.py`](insightforge/evaluation.py) |
| 7 | Data visualisation | [`insightforge/visualization.py`](insightforge/visualization.py) — nine charts across the four required families |
| 7 | Streamlit UI | [`app.py`](app.py) |
| 7 | Monitoring | [`insightforge/monitoring.py`](insightforge/monitoring.py) — JSONL log of latency, tokens, provider, errors |

---

## Architecture

```
sales_data.csv ──► data_loader ──► DataAnalyzer ──► metrics (dict)
                                       │
 PDF Folder ──► PyPDFLoader ──┐        ├──► narrative documents ──► FAISS index
                              └────────┘                                 │
                                                                         ▼
        question ──► condense (memory) ──► BusinessStatsRetriever ──► context
                                              │  keyword routing            │
                                              │  live pandas slice          ▼
                                              └─ vector search       QA_PROMPT ──► LLM ──► answer
                                                                         │
                                                              monitoring log · sources
```

### Why a custom retriever?

Pure vector search over metric documents answers *"tell me about regions"* well
but fails on *"how did Widget B do in the West in 2024?"* — no pre-computed
document holds that exact cell. `BusinessStatsRetriever` therefore:

1. **routes** the question to the metric topics that can answer it
   (time / product / region / demographics / statistics),
2. **computes** the exact slice with pandas at query time when the question
   names a product, region, gender, age band, year, quarter or month, and
3. **back-fills** the remaining slots from the FAISS index, which is also where
   the reference PDFs enter the answer.

That combination is what makes the numbers in the answers exact rather than
plausible.

### Memory

`WindowedMemory` keeps the last six exchanges. Before retrieval, a follow-up is
rewritten into a standalone question against that history, so this works:

```
> Which product generates the most revenue?
  Widget A, with 375,235.00 (27.13% of total revenue).
> And how does it do in the West?
  (condensed to: How does Widget A perform in the West region?)
```

### Embeddings

OpenRouter serves chat completions only — it has no `/embeddings` route — so the
index is built locally with a TF-IDF + LSA embedding model
([`embeddings.py`](insightforge/embeddings.py)) that implements the LangChain
`Embeddings` interface. Indexing is therefore free and offline while the
reasoning runs on a hosted model. Set `OPENAI_API_KEY` and the code switches to
`OpenAIEmbeddings` automatically.

---

## Evaluation

Ground truth is **computed from the dataset**, not hand-written, so it can never
drift from the CSV. Every prediction is graded twice:

* `QAEvalChain` — LangChain's LLM-as-judge with a custom grading prompt,
* **numeric grounding** — a deterministic check that the reference figures
  actually appear in the answer (this is what scores the offline mode).

```bash
python run_pipeline.py --skip-figures --skip-report   # evaluation only
```

Results land in `outputs/reports/evaluation.csv` and are also shown in the
app's **Evaluation** tab.

### Unit tests

26 fast tests cover the deterministic half of the system — enrichment, every
KPI against a pandas recomputation, retriever routing, live-slice arithmetic,
the memory window, grade parsing and the monitoring aggregations. They need no
API key and no network:

```bash
python -m pytest tests -q
```

---

## Layout

```
Capstone Project/
├── app.py                     Streamlit UI (6 tabs)
├── run_pipeline.py            end-to-end CLI runner
├── requirements.txt
├── .env.example
├── insightforge/
│   ├── config.py              paths, provider resolution, tunables
│   ├── data_loader.py         load + enrich
│   ├── analysis.py            DataAnalyzer: every metric
│   ├── knowledge_base.py      metrics → documents → vector store
│   ├── embeddings.py          OpenAI or offline TF-IDF/LSA embeddings
│   ├── retriever.py           BusinessStatsRetriever (hybrid)
│   ├── prompts.py             prompt library
│   ├── llm.py                 provider factory + offline model
│   ├── rag_chain.py           assistant: RAG + chains + memory
│   ├── evaluation.py          QAEvalChain harness
│   ├── visualization.py       nine matplotlib/seaborn charts
│   └── monitoring.py          interaction telemetry
├── notebooks/
│   ├── InsightForge_Capstone.ipynb    guided walkthrough of all 7 steps
│   └── build_notebook.py              regenerates the notebook
├── tests/
│   └── test_insightforge.py   26 offline tests
├── Datasets/                  sales_data.csv + reference PDFs (provided)
└── outputs/                   figures, reports, vector store, logs
```

---

## Configuration

All settings resolve through `insightforge/config.py` and can be overridden
with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | enables the hosted LLM |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `CHAT_MODEL` | `openai/gpt-4o-mini` | any OpenRouter model slug |
| `LLM_PROVIDER` | `auto` | `openrouter` \| `openai` \| `google` \| `offline` |
| `TEMPERATURE` | `0.1` | low, because BI answers must be stable |
| `MAX_TOKENS` | `900` | answer length cap |

---

## Note on dependency versions

The course-supplied `Datasets/requirements_genai.txt` pins releases from 2023
(`langchain==0.0.335`, `openai==0.28.1`) that do not build on modern Python.
`requirements.txt` here is the same stack at current versions, running on
Python 3.13. `QAEvalChain` moved from `langchain` to `langchain_classic` in
LangChain 1.0, so `evaluation.py` imports it through a shim that works with
either release. `langchain-community` (the home of the FAISS vector store and
`PyPDFLoader`) prints a sunset warning on import; it is still the canonical
import path for both, so the warning is expected and harmless.
