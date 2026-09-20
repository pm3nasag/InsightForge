"""Generates `InsightForge_Capstone.ipynb`.

Keeping the notebook in a generator script means the walkthrough stays in sync
with the package and can be regenerated after any refactor:

    python notebooks/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


CELLS = [
    md(
        """
# InsightForge — AI-Powered Business Intelligence Assistant

**Advanced Generative AI · Capstone**

This notebook walks through all seven steps of the problem statement against the
`insightforge` package that lives next to it.

| Step | Topic |
|---|---|
| 1 | Data preparation |
| 2 | Knowledge base creation |
| 3 | LLM application development (advanced summary + RAG integration) |
| 4 | Chain prompts |
| 5 | RAG system setup |
| 6 | Memory integration |
| 7 | LLMOps — QAEvalChain evaluation, visualisation, monitoring, Streamlit |

> Run `pip install -r ../requirements.txt` and set `OPENROUTER_API_KEY` first.
> Without a key everything still runs in offline extractive mode.
"""
    ),
    code(
        """
import json
import sys
from pathlib import Path

# make the package importable when running from notebooks/
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import pandas as pd

from insightforge.config import settings

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

print("provider resolved to:", settings.resolve_provider())
print("chat model:", settings.chat_model)
print("dataset:", settings.data_file)
"""
    ),
    md(
        """
---
## Step 1 · Data preparation

The dataset is already clean, so this step is about *enrichment*: calendar
parts for time-series analysis, age bands and satisfaction bands for
segmentation.
"""
    ),
    code(
        """
from insightforge.data_loader import dataset_profile, load_sales_data

df = load_sales_data()
print(df.shape)
df.head()
"""
    ),
    code(
        """
print(json.dumps(dataset_profile(df), indent=2))
"""
    ),
    md(
        """
---
## Step 3a · Advanced data summary

`DataAnalyzer` produces the four families of metric the brief asks for:
sales by time period, product & regional analysis, customer segmentation, and
statistical measures.
"""
    ),
    code(
        """
from insightforge.analysis import DataAnalyzer

analyzer = DataAnalyzer(df)
pd.Series(analyzer.kpis())
"""
    ),
    code(
        """
# 1 · Sales performance by time period
analyzer.sales_by_period("Quarter").head(12)
"""
    ),
    code(
        """
print(json.dumps(analyzer.growth_summary("Month"), indent=2))
"""
    ),
    code(
        """
# 2 · Product and regional analysis
display(analyzer.product_performance())
display(analyzer.regional_performance())
analyzer.product_region_matrix()
"""
    ),
    code(
        """
# 3 · Customer segmentation by demographics
segments = analyzer.customer_segments()
for name, table in segments.items():
    print(f"--- {name} ---")
    display(table)
analyzer.segment_matrix()
"""
    ),
    code(
        """
# 4 · Statistical measures
print(json.dumps(analyzer.statistics(), indent=2))
"""
    ),
    md(
        """
---
## Step 2 · Knowledge base creation

Two knowledge sources are combined:

* the **metrics above**, rendered as short factual documents, and
* the **reference PDFs** in `Datasets/PDF Folder`, chunked with
  `RecursiveCharacterTextSplitter`.

Both go into a FAISS index. Because OpenRouter serves chat completions only,
embeddings are produced locally by a TF-IDF + LSA model implementing the
LangChain `Embeddings` interface.
"""
    ),
    code(
        """
from insightforge.knowledge_base import KnowledgeBase

kb = KnowledgeBase(df, include_pdfs=True)
kb.build_vector_store()
kb.describe()
"""
    ),
    code(
        """
# what a metric document actually looks like
print(kb.stat_documents[0].page_content)
print("\\n---\\n")
print(kb.stats_by_topic(["regional_performance"])[0].page_content)
"""
    ),
    md(
        """
---
## Step 3b · The custom retriever

Vector search alone cannot answer *"how did Widget B do in the West in 2024?"* —
no pre-computed document holds that cell. `BusinessStatsRetriever` routes the
question by keyword, slices the dataframe with pandas at query time, and only
then back-fills from the vector index.
"""
    ),
    code(
        """
from insightforge.retriever import BusinessStatsRetriever, format_documents, route_topics

retriever = BusinessStatsRetriever(knowledge_base=kb, k=5)

question = "How did Widget B do in the West in 2024?"
print("routed topics:", route_topics(question))
docs = retriever.invoke(question)
print(f"{len(docs)} documents retrieved\\n")
print(docs[0].page_content)
"""
    ),
    md(
        """
---
## Steps 4, 5 & 6 · RAG chain, chained prompts and memory

`InsightForgeAssistant` wires the retriever, the prompt library, the chat model,
a sliding-window memory and the monitoring log into one object.
"""
    ),
    code(
        """
from insightforge.rag_chain import InsightForgeAssistant

assistant = InsightForgeAssistant(kb)
print("provider:", assistant.provider)

result = assistant.ask("Which product generates the most revenue and what share of the total is that?")
print(result["answer"])
print("\\n[", result["provider"], "|", result["latency_s"], "s |", len(result["sources"]), "sources ]")
"""
    ),
    code(
        """
# Memory in action: the follow-up has no subject of its own
follow_up = assistant.ask("And how does it do in the West region?")
print("condensed to:", follow_up["standalone_question"])
print()
print(follow_up["answer"])
"""
    ),
    code(
        """
# what the model was actually given
for source in result["sources"][:3]:
    print(f"[{source['source']} · {source['topic']}]")
    print(source["excerpt"][:300], "\\n")
"""
    ),
    md(
        """
### Step 4 · Chained prompts

`full_report()` runs three prompts in sequence, each consuming the previous
output: **metrics → insights → recommendations → executive summary**.
"""
    ),
    code(
        """
report = assistant.full_report(n_insights=5)
print(report["executive_summary"])
"""
    ),
    code(
        """
print(report["insights"])
"""
    ),
    code(
        """
print(report["recommendations"])
"""
    ),
    md(
        """
---
## Step 7a · Model evaluation with QAEvalChain

The ground truth is computed from the dataset itself, so it can never drift.
Each answer is graded twice: by LangChain's `QAEvalChain` (LLM-as-judge) and by
a deterministic numeric-grounding check.
"""
    ),
    code(
        """
from insightforge.evaluation import build_eval_set, evaluate, save_report

examples = build_eval_set(kb)
pd.DataFrame(examples).head()
"""
    ),
    code(
        """
evaluation = evaluate(assistant, examples)
print(json.dumps(evaluation["metrics"], indent=2))
evaluation["table"][["question", "qaeval_grade", "numerically_grounded", "latency_s"]]
"""
    ),
    code(
        """
save_report(evaluation)
"""
    ),
    md(
        """
---
## Step 7b · Data visualisation

Four families of chart: sales trends over time, product performance
comparisons, regional analysis, and customer demographics/segmentation.
"""
    ),
    code(
        """
%matplotlib inline
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")

from insightforge.visualization import (
    plot_correlation_matrix,
    plot_customer_demographics,
    plot_product_performance,
    plot_product_trend,
    plot_regional_analysis,
    plot_sales_trend,
    plot_seasonality,
    plot_segment_heatmap,
)

plot_sales_trend(df, "Month");
"""
    ),
    code(
        """
plot_seasonality(df);
"""
    ),
    code(
        """
plot_product_performance(df);
plot_product_trend(df);
"""
    ),
    code(
        """
plot_regional_analysis(df);
"""
    ),
    code(
        """
plot_customer_demographics(df);
plot_segment_heatmap(df);
plot_correlation_matrix(df);
"""
    ),
    md(
        """
---
## Step 7c · Monitoring

Every call is appended to `outputs/interactions.jsonl` with latency, tokens,
provider and errors — the telemetry behind the app's Monitoring tab.
"""
    ),
    code(
        """
from insightforge.monitoring import load_interactions, usage_report

print(json.dumps(usage_report(), indent=2))
pd.DataFrame(load_interactions()).tail(8)[["timestamp", "question", "provider", "latency_s", "total_tokens"]]
"""
    ),
    md(
        """
---
## Step 7d · Streamlit UI

The interface lives in `app.py` at the project root. From a terminal in the
project folder:

```bash
streamlit run app.py
```

It exposes seven tabs — Chat, Dashboard, Insights, Evaluation, Monitoring and
Knowledge base — over exactly the objects built in this notebook.

---

### Summary

| Step | Delivered by |
|---|---|
| 1 Data preparation | `insightforge/data_loader.py` |
| 2 Knowledge base | `insightforge/knowledge_base.py` |
| 3 Advanced summary + custom retriever | `analysis.py`, `retriever.py` |
| 4 Chain prompts | `prompts.py`, `rag_chain.full_report()` |
| 5 RAG system | `rag_chain.py` (LCEL over the custom retriever) |
| 6 Memory | `rag_chain.WindowedMemory` + question condensation |
| 7 LLMOps | `evaluation.py`, `visualization.py`, `monitoring.py`, `app.py` |
"""
    ),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


if __name__ == "__main__":
    path = HERE / "InsightForge_Capstone.ipynb"
    path.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(CELLS)} cells)")
