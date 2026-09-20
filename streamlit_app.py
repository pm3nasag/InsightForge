"""InsightForge - Streamlit user interface (Part 2, step 7).

Run with::

    streamlit run streamlit_app.py

Tabs:
  Chat        - conversational RAG assistant with memory
  Dashboard   - KPIs and the four required visualisation families
  Insights    - the chained prompts (insights -> recommendations -> summary)
  Evaluation  - QAEvalChain scoring of the assistant
  Monitoring  - latency / token / usage telemetry from the interaction log
  Knowledge   - what is actually in the retrieval index

The tabs are dynamic (``on_change="rerun"``), so only the visible tab's
content is computed - the dashboard renders nine matplotlib figures and should
not run while the user is chatting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from insightforge.analysis import DataAnalyzer
from insightforge.config import settings
from insightforge.data_loader import load_sales_data
from insightforge.evaluation import build_eval_set, evaluate
from insightforge.knowledge_base import KnowledgeBase
from insightforge.monitoring import load_interactions, usage_report
from insightforge.rag_chain import InsightForgeAssistant
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

st.set_page_config(
    page_title="InsightForge - BI assistant",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading the sales dataset...", ttl=3600)
def _load_data(path: str) -> pd.DataFrame:
    return load_sales_data(path)


@st.cache_resource(show_spinner="Building the knowledge base and vector index...")
def _build_kb(path: str, include_pdfs: bool) -> KnowledgeBase:
    kb = KnowledgeBase(load_sales_data(path), include_pdfs=include_pdfs)
    kb.build_vector_store()
    return kb


def _get_assistant(
    kb: KnowledgeBase, provider: str, k: int, api_key: str | None
) -> InsightForgeAssistant:
    """One assistant per (provider, k, key, knowledge base), so memory survives
    reruns but a newly pasted API key really does build a new client.

    The signature stores a fingerprint rather than the key itself.
    """
    fingerprint = hashlib.sha256((api_key or "").encode()).hexdigest()[:16]
    signature = (provider, k, fingerprint, id(kb))
    if st.session_state.get("assistant_signature") != signature:
        st.session_state.assistant = InsightForgeAssistant(
            kb, provider=provider, k=k, api_key=api_key
        )
        st.session_state.assistant_signature = signature
    return st.session_state.assistant


def _render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources used ({len(sources)})"):
        for source in sources:
            st.markdown(f"**{source['source']}** · `{source['topic']}` · {source['kind']}")
            st.code(source["excerpt"], language="text")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("chat", [])
st.session_state.setdefault("report", None)
st.session_state.setdefault("evaluation", None)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("InsightForge")
st.sidebar.caption("AI-powered business intelligence assistant")

# --- API key ----------------------------------------------------------------
# A visitor can paste their own key instead of relying on the deployment's
# secrets. The value stays in this session's widget state and is passed down
# to the LLM client per session - it is never written onto the shared
# `settings` singleton, because one process serves every visitor and a stored
# key would leak from one to the next.
api_key = st.sidebar.text_input(
    "OpenRouter API key",
    type="password",
    placeholder="sk-or-v1-...",
    help=(
        "Optional. Overrides the deployment's own key for this session only - "
        "it is never logged or written to disk. Get one at openrouter.ai/keys. "
        "Leave empty to use the configured key, or none at all to run in "
        "offline mode."
    ),
)
session_key = api_key.strip() or None
if api_key and not api_key.strip().startswith("sk-or-"):
    st.sidebar.warning("OpenRouter keys normally start with `sk-or-`.")

data_path = st.sidebar.text_input("Dataset", value=str(settings.data_file))
include_pdfs = st.sidebar.checkbox(
    "Index the reference PDFs",
    value=True,
    help="Adds the BI research PDFs from Datasets/PDF Folder to the RAG corpus.",
)
provider = st.sidebar.selectbox(
    "LLM provider",
    options=["auto", "openrouter", "openai", "google", "offline"],
    help="'auto' uses OPENROUTER_API_KEY when available, otherwise the offline model.",
)
top_k = st.sidebar.slider("Documents retrieved (k)", 3, 12, settings.top_k)

resolved = (
    settings.resolve_provider(openrouter_key=session_key)
    if provider == "auto"
    else provider
)
if resolved == "offline":
    st.sidebar.warning(
        "No API key. Running in **offline extractive mode**: retrieval, charts, "
        "memory and evaluation all work, but answers are extracted from the "
        "statistics rather than written. Paste a key above for full answers."
    )
else:
    source = "your key" if session_key else "the configured key"
    st.sidebar.success(f"Provider: **{resolved}** · `{settings.chat_model}` · using {source}")

if not Path(data_path).exists():
    st.error(f"Dataset not found at `{data_path}`.")
    st.stop()

df = _load_data(data_path)
kb = _build_kb(data_path, include_pdfs)
analyzer = DataAnalyzer(df)
assistant = _get_assistant(kb, provider, top_k, session_key)

st.sidebar.divider()
st.sidebar.metric("Transactions", f"{len(df):,}")
st.sidebar.metric("Indexed documents", f"{len(kb.documents):,}")
st.sidebar.caption(f"Embeddings: `{kb.embedding_backend}`")
if st.sidebar.button("Clear conversation memory", width="stretch"):
    assistant.reset_memory()
    st.session_state.chat = []
    st.sidebar.success("Memory cleared.")


# ---------------------------------------------------------------------------
# Tabs - dynamic so only the visible one computes
# ---------------------------------------------------------------------------
tab_chat, tab_dash, tab_insights, tab_eval, tab_monitor, tab_kb = st.tabs(
    [
        ":material/chat: Chat",
        ":material/monitoring: Dashboard",
        ":material/lightbulb: Insights",
        ":material/fact_check: Evaluation",
        ":material/speed: Monitoring",
        ":material/library_books: Knowledge base",
    ],
    on_change="rerun",
)


# --- Chat -------------------------------------------------------------------
if tab_chat.open is not False:
    with tab_chat:
        st.subheader("Ask InsightForge about the business")
        st.caption(
            "Answers are grounded in statistics computed from the dataset. "
            'Follow-ups use conversation memory, so "and in the West?" works '
            "after a question about a product."
        )

        starters = [
            "Overview of performance",
            "Best product and region",
            "Revenue trend over time",
            "Which segment to target",
            "Median order value and spread",
        ]
        starter_prompts = {
            "Overview of performance": "Give me an overview of business performance.",
            "Best product and region": "Which product and region combination performs best?",
            "Revenue trend over time": "How has revenue trended over time?",
            "Which segment to target": "Which customer segment should we target and why?",
            "Median order value and spread": (
                "What is the median order value and how variable are sales?"
            ),
        }
        picked = st.pills("Starter questions", starters, key="starter")

        for turn in st.session_state.chat:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn.get("meta"):
                    st.caption(turn["meta"])
                if turn.get("sources"):
                    _render_sources(turn["sources"])

        typed = st.chat_input(
            "e.g. Which region grew fastest last year?", submit_mode="disable"
        )
        # `st.chat_input` only returns a value on the run that submits it, but a
        # pill keeps its selection across reruns - so the pill path (and only the
        # pill path) needs de-duplication.
        from_pill = starter_prompts.get(picked) if picked else None
        question = typed or from_pill
        if question is not None and typed is None and question == st.session_state.get(
            "last_starter"
        ):
            question = None

        if question:
            st.session_state.last_starter = from_pill
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving statistics and reasoning..."):
                    result = assistant.ask(question)
                st.markdown(result["answer"])
                meta = " · ".join(
                    part
                    for part in (
                        result["provider"],
                        f"{result['latency_s']}s",
                        f"{result['total_tokens']} tokens" if result.get("total_tokens") else "",
                        f"rewritten as: {result['standalone_question']}"
                        if result["standalone_question"] != question
                        else "",
                    )
                    if part
                )
                st.caption(meta)
                _render_sources(result["sources"])
            st.session_state.chat.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "meta": meta,
                    "sources": result["sources"],
                }
            )


# --- Dashboard --------------------------------------------------------------
if tab_dash.open:
    with tab_dash:
        st.subheader("Business dashboard")
        kpis = analyzer.kpis()

        with st.container(border=True):
            row = st.columns(4)
            row[0].metric("Total revenue", f"{kpis['total_sales']:,.0f}")
            row[1].metric("Orders", f"{kpis['total_orders']:,}")
            row[2].metric("Avg order value", f"{kpis['average_order_value']:,.2f}")
            row[3].metric("Median order value", f"{kpis['median_order_value']:,.2f}")

            row = st.columns(4)
            row[0].metric("Top product", kpis["top_product"], f"{kpis['top_product_sales']:,.0f}")
            row[1].metric("Top region", kpis["top_region"], f"{kpis['top_region_sales']:,.0f}")
            row[2].metric("Avg satisfaction", f"{kpis['average_satisfaction']:.2f} / 5")
            row[3].metric("Sales std dev", f"{kpis['sales_std_dev']:,.2f}")
            st.caption(f"Period covered: {kpis['date_range']}")

        st.markdown("#### 1 · Sales trends over time")
        period = st.segmented_control(
            "Granularity", ["Month", "Quarter", "Year"], default="Month"
        )
        st.pyplot(plot_sales_trend(df, period or "Month"))
        st.pyplot(plot_seasonality(df))

        st.markdown("#### 2 · Product performance comparisons")
        st.pyplot(plot_product_performance(df))
        st.pyplot(plot_product_trend(df))
        st.dataframe(analyzer.product_performance())

        st.markdown("#### 3 · Regional analysis")
        st.pyplot(plot_regional_analysis(df))
        st.dataframe(analyzer.regional_performance())

        st.markdown("#### 4 · Customer demographics and segmentation")
        st.pyplot(plot_customer_demographics(df))
        left, right = st.columns(2)
        with left:
            st.pyplot(plot_segment_heatmap(df))
        with right:
            st.pyplot(plot_correlation_matrix(df))

        with st.expander("Descriptive statistics"):
            st.json(analyzer.statistics())
        with st.expander("Raw data"):
            st.dataframe(df, height=320)


# --- Insights ---------------------------------------------------------------
if tab_insights.open is not False:
    with tab_insights:
        st.subheader("Chained analysis: insights, recommendations, executive summary")
        st.caption(
            "Three prompts run in sequence, each consuming the output of the "
            "previous one - the capstone's prompt-chaining requirement."
        )
        n_insights = st.slider("Number of insights", 3, 8, 5)
        if st.button("Generate full report", type="primary"):
            with st.spinner("Running the insight chain..."):
                st.session_state.report = assistant.full_report(n_insights)

        report = st.session_state.report
        if report:
            st.success(f"Generated with {report['provider']} in {report['latency_s']}s")
            st.markdown("### Executive summary")
            st.markdown(report["executive_summary"])
            st.markdown("### Key insights")
            st.markdown(report["insights"])
            st.markdown("### Recommendations")
            st.markdown(report["recommendations"])
            st.download_button(
                "Download report (Markdown)",
                data=(
                    "# InsightForge report\n\n## Executive summary\n\n"
                    f"{report['executive_summary']}\n\n## Key insights\n\n"
                    f"{report['insights']}\n\n## Recommendations\n\n"
                    f"{report['recommendations']}\n"
                ),
                file_name="insightforge_report.md",
                mime="text/markdown",
            )
        else:
            st.info("Press **Generate full report** to run the chain.")


# --- Evaluation -------------------------------------------------------------
if tab_eval.open:
    with tab_eval:
        st.subheader("Model evaluation with QAEvalChain")
        st.caption(
            "Ground-truth answers are computed from the dataset with pandas, so the "
            "reference can never drift. Each answer is graded by LangChain's "
            "QAEvalChain and by a deterministic numeric-grounding check."
        )
        examples = build_eval_set(kb)
        with st.expander(f"Evaluation set ({len(examples)} questions)"):
            st.dataframe(pd.DataFrame(examples))

        if st.button("Run evaluation", type="primary"):
            with st.spinner("Answering and grading..."):
                st.session_state.evaluation = evaluate(assistant, examples)

        evaluation = st.session_state.evaluation
        if evaluation:
            metrics = evaluation["metrics"]
            with st.container(border=True):
                cols = st.columns(4)
                cols[0].metric("Questions", metrics["questions"])
                cols[1].metric(
                    "QAEvalChain accuracy", f"{(metrics['qaeval_accuracy'] or 0) * 100:.0f}%"
                )
                cols[2].metric(
                    "Numeric grounding",
                    f"{(metrics['numeric_grounding_rate'] or 0) * 100:.0f}%",
                )
                cols[3].metric("Avg latency", f"{metrics['avg_latency_s']}s")
            st.dataframe(evaluation["table"])
            st.download_button(
                "Download results (CSV)",
                data=evaluation["table"].to_csv(index=False),
                file_name="insightforge_evaluation.csv",
                mime="text/csv",
            )
        else:
            st.info("Press **Run evaluation** to score the assistant.")


# --- Monitoring -------------------------------------------------------------
if tab_monitor.open:
    with tab_monitor:
        st.subheader("LLMOps monitoring")
        records = load_interactions()
        report = usage_report(records)

        with st.container(border=True):
            cols = st.columns(5)
            cols[0].metric("Interactions", report["interactions"])
            cols[1].metric("Avg latency", f"{report['avg_latency_s'] or 0}s")
            cols[2].metric("P95 latency", f"{report['p95_latency_s'] or 0}s")
            cols[3].metric("Total tokens", f"{report['total_tokens']:,}")
            cols[4].metric("Errors", report["errors"])

        if records:
            log = pd.DataFrame(records)
            log["timestamp"] = pd.to_datetime(log["timestamp"], errors="coerce")

            st.markdown("#### Latency per interaction")
            st.line_chart(log, x="timestamp", y="latency_s", y_label="seconds")

            st.markdown("#### Requests by provider")
            st.bar_chart(
                pd.DataFrame(
                    {"provider": list(report["providers"]), "requests": list(report["providers"].values())}
                ),
                x="provider",
                y="requests",
            )

            st.markdown("#### Interaction log")
            columns = [
                c
                for c in ("timestamp", "question", "provider", "latency_s", "total_tokens", "error")
                if c in log.columns
            ]
            st.dataframe(log[columns].sort_values("timestamp", ascending=False), height=320)
        else:
            st.info("No interactions logged yet - ask a question in the Chat tab.")


# --- Knowledge base ---------------------------------------------------------
if tab_kb.open:
    with tab_kb:
        st.subheader("What the assistant can retrieve")
        with st.container(border=True):
            cols = st.columns(4)
            described = kb.describe()
            cols[0].metric("Rows analysed", f"{described['rows']:,}")
            cols[1].metric("Metric documents", described["stat_documents"])
            cols[2].metric("PDF chunks", described["pdf_documents"])
            cols[3].metric("Total indexed", described["total_documents"])
            st.caption(f"Embedding backend: `{described['embedding_backend']}`")

        st.markdown("#### Try the retriever directly")
        probe = st.text_input("Retrieval probe", value="Which region sells the most?")
        if probe:
            docs = assistant.retrieve(probe)
            st.caption(f"{len(docs)} documents retrieved")
            for doc in docs:
                with st.expander(
                    f"{doc.metadata.get('source', '?')} · {doc.metadata.get('topic', '?')}"
                ):
                    st.code(doc.page_content, language="text")

        with st.expander("All computed metrics"):
            st.json(json.loads(json.dumps(kb.summary, default=str)))
