"""Step 3b - A custom retriever that extracts *relevant statistics*.

A plain vector search over the metric documents works, but it is blind to the
structure of a business question: "how did Widget B do in the West?" needs the
Widget-B-in-West number, not the three chunks whose wording happens to be
closest.

:class:`BusinessStatsRetriever` therefore does three things per query:

1. **Routing** - keyword rules map the question to the metric topics that can
   answer it (time, product, region, demographics, statistics, ...).
2. **Live computation** - when the question names a concrete product, region,
   age band, gender, year, quarter or month, the retriever slices the
   dataframe with pandas *at query time* and emits a freshly computed fact
   document.  This is what keeps answers numerically exact.
3. **Semantic back-fill** - the remaining slots are filled from the FAISS
   vector store, which is also where the reference PDFs come in.

The class implements ``langchain_core.retrievers.BaseRetriever``, so it drops
straight into any LangChain chain.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from insightforge.config import settings

# --- keyword -> metric topic routing table ---------------------------------
TOPIC_ROUTES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (
        ("trend", "over time", "monthly", "month", "seasonal", "growth", "decline", "forecast"),
        ("sales_by_month", "growth_month", "growth_quarter"),
    ),
    (("year", "yearly", "annual", "yoy"), ("sales_by_year", "growth_year")),
    (("quarter", "quarterly", "q1", "q2", "q3", "q4"), ("sales_by_quarter", "growth_quarter")),
    (("weekday", "day of week", "weekend", "daily"), ("sales_by_day_of_week",)),
    (
        ("product", "widget", "sku", "item", "best seller", "bestseller", "worst"),
        ("product_performance", "product_region_matrix"),
    ),
    (
        ("region", "north", "south", "east", "west", "geograph", "territory", "market"),
        ("regional_performance", "product_region_matrix"),
    ),
    (
        ("customer", "demographic", "age", "gender", "male", "female", "segment", "young", "old"),
        ("segment_by_age_group", "segment_by_gender", "segment_age_gender"),
    ),
    (
        ("satisfaction", "csat", "happy", "promoter", "detractor", "nps"),
        ("segment_by_satisfaction_band", "statistics_customer_satisfaction", "correlations"),
    ),
    (
        ("median", "standard deviation", "std", "variance", "distribution", "outlier",
         "skew", "spread", "statistic", "average", "mean", "correlat"),
        ("statistics_sales", "statistics_customer_age", "statistics_customer_satisfaction",
         "correlations"),
    ),
    (
        ("overview", "summary", "kpi", "overall", "total", "how many", "dataset", "describe"),
        ("overview", "kpis"),
    ),
    (
        ("recommend", "improve", "strategy", "action", "should we", "opportunity", "risk"),
        ("kpis", "product_performance", "regional_performance", "growth_month"),
    ),
]

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def route_topics(question: str) -> list[str]:
    """Return the metric topics a question should be answered from."""
    lowered = question.lower()
    topics: list[str] = []
    for keywords, mapped in TOPIC_ROUTES:
        if any(keyword in lowered for keyword in keywords):
            topics.extend(t for t in mapped if t not in topics)
    if not topics:  # a question we cannot route still gets the headline numbers
        topics = ["overview", "kpis"]
    return topics


class BusinessStatsRetriever(BaseRetriever):
    """Hybrid retriever: rule-routed metrics + live pandas + vector search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    knowledge_base: Any
    k: int = Field(default_factory=lambda: settings.top_k)
    use_vector_search: bool = True

    # -- LangChain entry point ---------------------------------------------
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        docs: list[Document] = []
        seen: set[str] = set()

        def push(candidates: list[Document]) -> None:
            for doc in candidates:
                key = doc.page_content[:160]
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)

        push(self.computed_facts(query))
        push(self.knowledge_base.stats_by_topic(route_topics(query)))

        if self.use_vector_search and len(docs) < self.k:
            try:
                push(
                    self.knowledge_base.vector_store.similarity_search(
                        query, k=self.k
                    )
                )
            except Exception:  # a broken index must not break the answer
                pass

        return docs[: max(self.k, 4)]

    # -- live pandas slicing ------------------------------------------------
    def computed_facts(self, query: str) -> list[Document]:
        """Compute exact figures for the entities named in the question."""
        df: pd.DataFrame = self.knowledge_base.df
        lowered = query.lower()
        facts: list[Document] = []

        filters: dict[str, str] = {}
        for product in df["Product"].unique():
            if str(product).lower() in lowered:
                filters["Product"] = product
                break
        for region in df["Region"].unique():
            if re.search(rf"\b{re.escape(str(region).lower())}\b", lowered):
                filters["Region"] = region
                break
        for gender in df["Customer_Gender"].unique():
            if re.search(rf"\b{re.escape(str(gender).lower())}\b", lowered):
                filters["Customer_Gender"] = gender
                break
        for age_group in df["Age_Group"].unique():
            if str(age_group).lower() in lowered:
                filters["Age_Group"] = age_group
                break

        mask = pd.Series(True, index=df.index)
        for column, value in filters.items():
            mask &= df[column] == value

        # time filters
        time_label = ""
        year_match = re.search(r"\b(20\d{2})\b", lowered)
        if year_match:
            year = int(year_match.group(1))
            mask &= df["Year"] == year
            time_label = f" in {year}"
        quarter_match = re.search(r"\bq([1-4])\b", lowered)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            mask &= df["Date"].dt.quarter == quarter
            time_label += f" in Q{quarter}"
        for name, number in MONTHS.items():
            if re.search(rf"\b{name}\b", lowered):
                mask &= df["Date"].dt.month == number
                time_label += f" in {name.title()}"
                break

        if not filters and not time_label:
            return facts

        subset = df[mask]
        scope = ", ".join(f"{k.replace('_', ' ')} = {v}" for k, v in filters.items())
        scope = (scope or "all transactions") + time_label

        if subset.empty:
            facts.append(
                Document(
                    page_content=f"Computed fact: there are no transactions for {scope}.",
                    metadata={"source": "pandas", "kind": "computed", "topic": "slice"},
                )
            )
            return facts

        lines = [
            f"Computed fact (calculated live with pandas) for {scope}:",
            f"- transactions: {len(subset):,}",
            f"- total sales: {subset['Sales'].sum():,.2f}",
            f"- average sale: {subset['Sales'].mean():,.2f}",
            f"- median sale: {subset['Sales'].median():,.2f}",
            f"- standard deviation: {subset['Sales'].std():,.2f}",
            f"- share of overall revenue: {subset['Sales'].sum() / df['Sales'].sum() * 100:,.2f}%",
            f"- average customer age: {subset['Customer_Age'].mean():,.1f}",
            f"- average satisfaction: {subset['Customer_Satisfaction'].mean():,.2f} / 5",
        ]

        # a small breakdown makes comparative questions answerable in one hop
        for dimension in ("Product", "Region"):
            if dimension not in filters and subset[dimension].nunique() > 1:
                breakdown = (
                    subset.groupby(dimension, observed=True)["Sales"].sum().sort_values(ascending=False)
                )
                lines.append(
                    f"- by {dimension.lower()}: "
                    + "; ".join(f"{idx} {val:,.2f}" for idx, val in breakdown.items())
                )

        facts.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    "source": "pandas",
                    "kind": "computed",
                    "topic": "slice",
                    "filters": scope,
                },
            )
        )
        return facts


def format_documents(docs: list[Document]) -> str:
    """Render retrieved documents as the ``context`` block of a prompt."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        kind = doc.metadata.get("kind", "context")
        blocks.append(f"[{i}] (source: {source} | type: {kind})\n{doc.page_content}")
    return "\n\n".join(blocks) if blocks else "No context retrieved."
