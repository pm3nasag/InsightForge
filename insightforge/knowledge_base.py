"""Step 2 & 5 - Knowledge base creation and the RAG vector store.

Two very different kinds of knowledge feed InsightForge:

* **Structured knowledge** - the metrics produced by
  :class:`~insightforge.analysis.DataAnalyzer`, rendered as short, factual,
  self-contained narrative documents.  These are what let the LLM answer
  "which region sells most?" with a real number instead of a hallucination.
* **Unstructured knowledge** - the reference PDFs shipped in
  ``Datasets/PDF Folder`` (BI approaches, Walmart sales analysis, ...), which
  give the assistant business context and best-practice vocabulary.

Both are chunked into ``langchain_core.documents.Document`` objects and pushed
into a FAISS vector store (with an in-memory store as fallback).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from insightforge.analysis import DataAnalyzer
from insightforge.config import settings
from insightforge.embeddings import get_embeddings


# ---------------------------------------------------------------------------
# Structured knowledge: metrics -> narrative documents
# ---------------------------------------------------------------------------
def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _table_text(title: str, rows: list[dict], index_key: str) -> str:
    lines = [title]
    for row in rows:
        label = row.get(index_key, "?")
        parts = [f"{k.replace('_', ' ')}: {_fmt(v)}" for k, v in row.items() if k != index_key]
        lines.append(f"- {label} -> " + "; ".join(parts))
    return "\n".join(lines)


def build_stat_documents(summary: dict) -> list[Document]:
    """Turn the analysis summary into retrievable, human-readable documents."""
    docs: list[Document] = []
    kpis = summary["kpis"]
    profile = summary["profile"]

    def add(topic: str, text: str, **extra) -> None:
        docs.append(
            Document(
                page_content=text.strip(),
                metadata={"source": "sales_data.csv", "kind": "statistics", "topic": topic, **extra},
            )
        )

    # -- overview -----------------------------------------------------------
    add(
        "overview",
        f"""
Dataset overview for the InsightForge sales dataset.
The dataset holds {_fmt(profile['rows'])} transactions recorded between {profile['date_min']} and {profile['date_max']}.
Products sold: {', '.join(profile['products'])}. Regions covered: {', '.join(profile['regions'])}.
Total revenue is {_fmt(kpis['total_sales'])} across {_fmt(kpis['total_orders'])} orders.
Average order value is {_fmt(kpis['average_order_value'])} and the median order value is {_fmt(kpis['median_order_value'])}.
Average customer satisfaction is {_fmt(kpis['average_satisfaction'])} out of 5 and the average customer age is {_fmt(kpis['average_customer_age'])} years.
The dataset contains {profile['missing_values']} missing values and {profile['duplicate_rows']} duplicate rows.
""",
    )

    # -- headline KPIs ------------------------------------------------------
    add(
        "kpis",
        "Headline key performance indicators (KPIs):\n"
        + "\n".join(f"- {k.replace('_', ' ')}: {_fmt(v)}" for k, v in kpis.items()),
    )

    # -- time periods -------------------------------------------------------
    for period, rows in summary["time"].items():
        add(
            f"sales_by_{period.lower()}",
            _table_text(
                f"Sales performance by {period.replace('_', ' ').lower()}:", rows, period
            ),
            period=period,
        )

    for period, growth in summary["growth"].items():
        add(
            f"growth_{period.lower()}",
            f"Growth analysis at {period.lower()} granularity:\n"
            + "\n".join(f"- {k.replace('_', ' ')}: {_fmt(v)}" for k, v in growth.items()),
            period=period,
        )

    # -- products and regions ----------------------------------------------
    add("product_performance", _table_text("Product performance analysis:", summary["product"], "Product"))
    add("regional_performance", _table_text("Regional performance analysis:", summary["region"], "Region"))
    add(
        "product_region_matrix",
        "Total sales for every product and region combination:\n"
        + "\n".join(
            f"- {product}: " + "; ".join(f"{region} {_fmt(v)}" for region, v in cells.items())
            for product, cells in summary["product_region_matrix"].items()
        ),
    )

    # -- customer segmentation ---------------------------------------------
    label_for = {
        "by_age_group": ("Customer segmentation by age group:", "Age_Group"),
        "by_gender": ("Customer segmentation by gender:", "Customer_Gender"),
        "by_satisfaction_band": (
            "Customer segmentation by satisfaction band "
            "(Detractor <=2.0, Neutral 2.0-3.5, Promoter >3.5):",
            "Satisfaction_Band",
        ),
    }
    for key, rows in summary["segments"].items():
        title, index_key = label_for[key]
        add(f"segment_{key}", _table_text(title, rows, index_key))

    add(
        "segment_age_gender",
        "Average sale value by age group and gender:\n"
        + "\n".join(
            f"- {age}: " + "; ".join(f"{gender} {_fmt(v)}" for gender, v in cells.items())
            for age, cells in summary["age_gender_avg_sale"].items()
        ),
    )

    # -- statistics ---------------------------------------------------------
    for column, stats in summary["statistics"].items():
        if column == "correlations":
            add(
                "correlations",
                "Pearson correlations between numeric fields:\n"
                + "\n".join(f"- {k}: {_fmt(v)}" for k, v in stats.items()),
            )
        else:
            add(
                f"statistics_{column.lower()}",
                f"Descriptive statistics for {column}:\n"
                + "\n".join(f"- {k.replace('_', ' ')}: {_fmt(v)}" for k, v in stats.items()),
            )

    return docs


# ---------------------------------------------------------------------------
# Unstructured knowledge: reference PDFs
# ---------------------------------------------------------------------------
def load_pdf_documents(pdf_dir: Path | None = None) -> list[Document]:
    """Load and chunk every PDF in the reference folder."""
    pdf_dir = Path(pdf_dir) if pdf_dir is not None else settings.pdf_dir
    if not pdf_dir.exists():
        return []

    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:  # pragma: no cover
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
        except Exception:  # a corrupt PDF must not kill the pipeline
            continue
        for chunk in splitter.split_documents(pages):
            text = chunk.page_content.strip()
            if len(text) < 80:  # skip headers / page numbers
                continue
            chunk.page_content = text
            chunk.metadata.update(
                {"source": pdf_path.name, "kind": "reference", "topic": pdf_path.stem}
            )
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# The knowledge base itself
# ---------------------------------------------------------------------------
class KnowledgeBase:
    """Holds the dataframe, its metrics, the documents and the vector store."""

    def __init__(self, df: pd.DataFrame, include_pdfs: bool = True) -> None:
        self.df = df
        self.analyzer = DataAnalyzer(df)
        self.summary = self.analyzer.summary()
        self.stat_documents = build_stat_documents(self.summary)
        self.pdf_documents = load_pdf_documents() if include_pdfs else []
        self.embedding_backend: str = "not-built"
        self._vector_store = None

    # -- documents ----------------------------------------------------------
    @property
    def documents(self) -> list[Document]:
        return self.stat_documents + self.pdf_documents

    # -- vector store -------------------------------------------------------
    def build_vector_store(self):
        """Embed every document and build the searchable index."""
        embeddings, backend = get_embeddings()
        self.embedding_backend = backend
        docs = self.documents

        try:
            from langchain_community.vectorstores import FAISS

            store = FAISS.from_documents(docs, embeddings)
        except Exception:  # FAISS unavailable -> pure-python fallback
            from langchain_core.vectorstores import InMemoryVectorStore

            store = InMemoryVectorStore.from_documents(docs, embeddings)

        self._vector_store = store
        return store

    @property
    def vector_store(self):
        if self._vector_store is None:
            self.build_vector_store()
        return self._vector_store

    def as_retriever(self, k: int | None = None):
        return self.vector_store.as_retriever(
            search_kwargs={"k": k or settings.top_k}
        )

    # -- persistence --------------------------------------------------------
    def save(self, directory: Path | None = None) -> Path:
        """Persist the metrics and the document corpus for inspection."""
        directory = Path(directory) if directory is not None else settings.vector_store_dir
        directory.mkdir(parents=True, exist_ok=True)

        (directory / "summary.json").write_text(
            json.dumps(self.summary, indent=2, default=str), encoding="utf-8"
        )
        (directory / "documents.jsonl").write_text(
            "\n".join(
                json.dumps({"text": d.page_content, "metadata": d.metadata})
                for d in self.documents
            ),
            encoding="utf-8",
        )

        # The FAISS index can only be reloaded when the embeddings are
        # reproducible across processes (i.e. the API-backed model).
        if self.embedding_backend.startswith("openai") and hasattr(
            self._vector_store, "save_local"
        ):
            self._vector_store.save_local(str(directory / "faiss_index"))
        return directory

    # -- convenience --------------------------------------------------------
    def stats_by_topic(self, topics: Iterable[str]) -> list[Document]:
        wanted = set(topics)
        return [d for d in self.stat_documents if d.metadata.get("topic") in wanted]

    def describe(self) -> dict:
        return {
            "rows": len(self.df),
            "stat_documents": len(self.stat_documents),
            "pdf_documents": len(self.pdf_documents),
            "total_documents": len(self.documents),
            "embedding_backend": self.embedding_backend,
        }
