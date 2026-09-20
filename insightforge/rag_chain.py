"""Steps 4, 5 and 6 - the RAG chain, the prompt chain, and memory.

:class:`InsightForgeAssistant` is the object the notebook, the CLI and the
Streamlit app all talk to.  It wires together:

* the custom retriever (:mod:`insightforge.retriever`),
* the prompt library (:mod:`insightforge.prompts`),
* the chat model (:mod:`insightforge.llm`),
* conversational memory, and
* the monitoring log (:mod:`insightforge.monitoring`).

**Memory** is a sliding window of the last ``settings.memory_window``
exchanges held in a ``langchain_core`` chat history.  Before retrieval, a
follow-up question is condensed into a standalone question against that
history - which is what lets "and in the West?" work after "how did Widget B
do?".

**Chained prompts** are exposed through :meth:`generate_insights`,
:meth:`recommend` and :meth:`executive_summary`; :meth:`full_report` runs all
three in sequence, feeding each step's output into the next.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from insightforge.config import settings
from insightforge.llm import get_llm
from insightforge.monitoring import extract_token_usage, log_interaction, timed
from insightforge.prompts import (
    CONDENSE_PROMPT,
    EXEC_SUMMARY_PROMPT,
    INSIGHT_PROMPT,
    QA_PROMPT,
    RECOMMENDATION_PROMPT,
)
from insightforge.retriever import BusinessStatsRetriever, format_documents


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
class WindowedMemory:
    """A bounded conversational memory over ``InMemoryChatMessageHistory``."""

    def __init__(self, window: int | None = None) -> None:
        self.window = window or settings.memory_window
        self.history = InMemoryChatMessageHistory()

    def add_exchange(self, question: str, answer: str) -> None:
        self.history.add_user_message(question)
        self.history.add_ai_message(answer)
        # keep only the last `window` question/answer pairs
        max_messages = self.window * 2
        if len(self.history.messages) > max_messages:
            self.history.messages = self.history.messages[-max_messages:]

    def as_text(self) -> str:
        if not self.history.messages:
            return "(no previous turns)"
        lines = []
        for message in self.history.messages:
            role = "User" if message.type == "human" else "InsightForge"
            content = str(message.content)
            if len(content) > 600:
                content = content[:600] + " ..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.history.clear()

    def __len__(self) -> int:
        return len(self.history.messages) // 2


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------
class InsightForgeAssistant:
    """The Business Intelligence Assistant: retrieval + prompts + memory."""

    def __init__(
        self,
        knowledge_base: Any,
        llm: Any | None = None,
        provider: str | None = None,
        k: int | None = None,
        enable_logging: bool = True,
        api_key: str | None = None,
    ) -> None:
        self.kb = knowledge_base
        if llm is None:
            self.llm, self.provider = get_llm(provider, api_key=api_key)
        else:
            self.llm, self.provider = llm, provider or "custom"

        self.retriever = BusinessStatsRetriever(
            knowledge_base=knowledge_base, k=k or settings.top_k
        )
        self.memory = WindowedMemory()
        self.enable_logging = enable_logging

        # --- the RAG chain, expressed with LCEL ---------------------------
        # the QA chain stops at the model so token usage stays available
        self.qa_chain = QA_PROMPT | self.llm
        self.condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser()

        # --- the sequential prompt chain ----------------------------------
        self.insight_chain = INSIGHT_PROMPT | self.llm | StrOutputParser()
        self.recommendation_chain = RECOMMENDATION_PROMPT | self.llm | StrOutputParser()
        self.summary_chain = EXEC_SUMMARY_PROMPT | self.llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------
    def condense(self, question: str) -> str:
        """Rewrite a follow-up into a standalone question using memory."""
        if len(self.memory) == 0 or self.provider.startswith("offline"):
            return question
        try:
            rewritten = self.condense_chain.invoke(
                {"chat_history": self.memory.as_text(), "question": question}
            ).strip()
            return rewritten or question
        except Exception:
            return question

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)

    def ask(self, question: str, use_memory: bool = True) -> dict:
        """Answer a business question. Returns answer, sources and telemetry."""
        standalone = self.condense(question) if use_memory else question

        with timed() as clock:
            error = None
            usage: dict[str, Any] = {}
            try:
                docs = self.retrieve(standalone)
                context = format_documents(docs)
                response = self.qa_chain.invoke(
                    {
                        "context": context,
                        "chat_history": self.memory.as_text() if use_memory else "(memory disabled)",
                        "question": standalone,
                    }
                )
                answer = str(getattr(response, "content", response)).strip()
                usage = extract_token_usage(response)
            except Exception as exc:  # surfaced to the UI instead of crashing it
                docs, answer, error = [], f"The assistant hit an error: {exc}", str(exc)

        if use_memory and not error:
            self.memory.add_exchange(question, answer)

        result = {
            "question": question,
            "standalone_question": standalone,
            "answer": answer,
            "sources": [
                {
                    "source": d.metadata.get("source", "unknown"),
                    "topic": d.metadata.get("topic", ""),
                    "kind": d.metadata.get("kind", ""),
                    "excerpt": d.page_content[:400],
                }
                for d in docs
            ],
            "documents": docs,
            "provider": self.provider,
            "latency_s": clock["elapsed"],
            "error": error,
            **usage,
        }

        if self.enable_logging:
            log_interaction(
                {
                    key: result[key]
                    for key in (
                        "question",
                        "standalone_question",
                        "answer",
                        "provider",
                        "latency_s",
                        "error",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                    if key in result
                }
                | {"retrieved": [s["topic"] for s in result["sources"]]}
            )
        return result

    def reset_memory(self) -> None:
        self.memory.clear()

    # ------------------------------------------------------------------
    # Chained prompts: metrics -> insights -> recommendations -> summary
    # ------------------------------------------------------------------
    def _metrics_block(self) -> str:
        summary = self.kb.summary
        compact = {
            "kpis": summary["kpis"],
            "growth": summary["growth"],
            "product": summary["product"],
            "region": summary["region"],
            "segments": summary["segments"],
            "statistics": summary["statistics"],
        }
        return json.dumps(compact, indent=2, default=str)

    def generate_insights(self, n_insights: int = 5) -> str:
        return self.insight_chain.invoke(
            {"metrics": self._metrics_block(), "n_insights": n_insights}
        ).strip()

    def recommend(self, insights: str | None = None) -> str:
        insights = insights or self.generate_insights()
        return self.recommendation_chain.invoke({"insights": insights}).strip()

    def executive_summary(self, insights: str, recommendations: str) -> str:
        return self.summary_chain.invoke(
            {"insights": insights, "recommendations": recommendations}
        ).strip()

    def full_report(self, n_insights: int = 5) -> dict:
        """Run the three prompts in sequence - the capstone's 'chain prompts'."""
        with timed() as clock:
            insights = self.generate_insights(n_insights)
            recommendations = self.recommend(insights)
            summary = self.executive_summary(insights, recommendations)

        report = {
            "insights": insights,
            "recommendations": recommendations,
            "executive_summary": summary,
            "provider": self.provider,
            "latency_s": clock["elapsed"],
        }
        if self.enable_logging:
            log_interaction(
                {
                    "question": "[chained report]",
                    "answer": summary[:500],
                    "provider": self.provider,
                    "latency_s": clock["elapsed"],
                }
            )
        return report


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------
def build_assistant(
    include_pdfs: bool = True,
    provider: str | None = None,
    api_key: str | None = None,
) -> InsightForgeAssistant:
    """Load data -> build knowledge base -> index -> return a ready assistant."""
    from insightforge.data_loader import load_sales_data
    from insightforge.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(load_sales_data(), include_pdfs=include_pdfs)
    kb.build_vector_store()
    return InsightForgeAssistant(kb, provider=provider, api_key=api_key)
