"""Fast, offline tests for the deterministic half of InsightForge.

No network and no API key required::

    python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insightforge.analysis import DataAnalyzer
from insightforge.data_loader import enrich
from insightforge.evaluation import build_eval_set, numeric_grounding, parse_grade
from insightforge.knowledge_base import KnowledgeBase, build_stat_documents
from insightforge.llm import OfflineChatModel, get_llm
from insightforge.monitoring import usage_report
from insightforge.rag_chain import InsightForgeAssistant, WindowedMemory
from insightforge.retriever import BusinessStatsRetriever, route_topics


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """A small synthetic dataset with known answers."""
    rng = np.random.default_rng(0)
    n = 400
    return enrich(
        pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
                "Product": rng.choice(["Widget A", "Widget B"], n),
                "Region": rng.choice(["North", "South"], n),
                "Sales": rng.integers(100, 1000, n),
                "Customer_Age": rng.integers(18, 70, n),
                "Customer_Gender": rng.choice(["Male", "Female"], n),
                "Customer_Satisfaction": rng.uniform(1, 5, n),
            }
        )
    )


@pytest.fixture(scope="module")
def kb(df: pd.DataFrame) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(df, include_pdfs=False)
    knowledge_base.build_vector_store()
    return knowledge_base


# --- data preparation -------------------------------------------------------
def test_enrich_adds_derived_columns(df: pd.DataFrame) -> None:
    for column in ("Year", "Quarter", "Month", "Day_Of_Week", "Age_Group", "Satisfaction_Band"):
        assert column in df.columns
    assert df["Age_Group"].isin(
        ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    ).all()
    assert df["Date"].is_monotonic_increasing


# --- analysis ---------------------------------------------------------------
def test_kpis_agree_with_pandas(df: pd.DataFrame) -> None:
    kpis = DataAnalyzer(df).kpis()
    assert kpis["total_sales"] == pytest.approx(df["Sales"].sum())
    assert kpis["total_orders"] == len(df)
    assert kpis["median_order_value"] == pytest.approx(df["Sales"].median())
    assert kpis["top_product"] == df.groupby("Product")["Sales"].sum().idxmax()


def test_shares_sum_to_one_hundred(df: pd.DataFrame) -> None:
    for table in (
        DataAnalyzer(df).product_performance(),
        DataAnalyzer(df).regional_performance(),
    ):
        assert table["share_of_sales_pct"].sum() == pytest.approx(100.0, abs=0.05)


def test_summary_is_json_serialisable(df: pd.DataFrame) -> None:
    import json

    json.dumps(DataAnalyzer(df).summary())  # must not raise


# --- knowledge base ---------------------------------------------------------
def test_stat_documents_cover_every_topic_family(df: pd.DataFrame) -> None:
    docs = build_stat_documents(DataAnalyzer(df).summary())
    topics = {d.metadata["topic"] for d in docs}
    for expected in (
        "overview",
        "kpis",
        "product_performance",
        "regional_performance",
        "segment_by_age_group",
        "statistics_sales",
        "correlations",
    ):
        assert expected in topics


def test_vector_store_returns_documents(kb: KnowledgeBase) -> None:
    assert kb.vector_store.similarity_search("regional revenue", k=3)


# --- retriever --------------------------------------------------------------
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the monthly trend?", "sales_by_month"),
        ("Which product sells best?", "product_performance"),
        ("Compare the North and South regions", "regional_performance"),
        ("Which age group buys most?", "segment_by_age_group"),
        ("What is the standard deviation?", "statistics_sales"),
    ],
)
def test_routing(question: str, expected: str) -> None:
    assert expected in route_topics(question)


def test_unroutable_question_falls_back_to_overview() -> None:
    assert route_topics("zzzz") == ["overview", "kpis"]


def test_computed_facts_match_pandas(kb: KnowledgeBase, df: pd.DataFrame) -> None:
    retriever = BusinessStatsRetriever(knowledge_base=kb, k=5)
    docs = retriever.computed_facts("How did Widget A do in the North?")
    assert docs, "a named product and region must produce a computed fact"

    expected = df[(df["Product"] == "Widget A") & (df["Region"] == "North")]["Sales"].sum()
    assert f"{expected:,.2f}" in docs[0].page_content
    assert docs[0].metadata["kind"] == "computed"


def test_retriever_returns_relevant_documents(kb: KnowledgeBase) -> None:
    retriever = BusinessStatsRetriever(knowledge_base=kb, k=5)
    docs = retriever.invoke("Which region has the highest sales?")
    assert docs
    assert any("region" in d.metadata.get("topic", "").lower() for d in docs)


# --- memory -----------------------------------------------------------------
def test_memory_respects_its_window() -> None:
    memory = WindowedMemory(window=2)
    for i in range(5):
        memory.add_exchange(f"q{i}", f"a{i}")
    assert len(memory) == 2
    assert "q4" in memory.as_text()
    assert "q0" not in memory.as_text()
    memory.clear()
    assert memory.as_text() == "(no previous turns)"


# --- offline model + assistant ---------------------------------------------
def test_offline_provider_is_selectable() -> None:
    llm, description = get_llm("offline")
    assert isinstance(llm, OfflineChatModel)
    assert description == "offline:extractive"


def test_assistant_answers_offline(kb: KnowledgeBase) -> None:
    assistant = InsightForgeAssistant(kb, provider="offline", enable_logging=False)
    result = assistant.ask("Which region has the highest sales?")
    assert result["error"] is None
    assert result["sources"]
    assert result["answer"].strip()
    assert len(assistant.memory) == 1


# --- evaluation -------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("GRADE: CORRECT", "CORRECT"),
        ("reasoning here\n\nGRADE: INCORRECT", "INCORRECT"),
        ("CORRECT", "CORRECT"),
        ("the answer is incorrect", "INCORRECT"),
        ("", "UNKNOWN"),
    ],
)
def test_parse_grade(raw: str, expected: str) -> None:
    assert parse_grade(raw) == expected


def test_numeric_grounding() -> None:
    assert numeric_grounding("Total revenue is 1,383,220.00.", "We made 1383220 in revenue.")
    assert not numeric_grounding("Total revenue is 1,383,220.00.", "We made 12 in revenue.")
    assert numeric_grounding("No figures here.", "Neither here.")


def test_eval_set_is_grounded_in_the_data(kb: KnowledgeBase) -> None:
    examples = build_eval_set(kb)
    assert len(examples) >= 10
    total = kb.summary["kpis"]["total_sales"]
    assert f"{total:,.2f}" in examples[0]["answer"]


# --- monitoring -------------------------------------------------------------
def test_usage_report_on_empty_log() -> None:
    report = usage_report([])
    assert report["interactions"] == 0
    assert report["avg_latency_s"] is None


def test_usage_report_aggregates() -> None:
    report = usage_report(
        [
            {"latency_s": 1.0, "total_tokens": 10, "provider": "offline:extractive"},
            {"latency_s": 3.0, "total_tokens": 20, "provider": "offline:extractive", "error": "x"},
        ]
    )
    assert report["interactions"] == 2
    assert report["avg_latency_s"] == 2.0
    assert report["total_tokens"] == 30
    assert report["errors"] == 1
