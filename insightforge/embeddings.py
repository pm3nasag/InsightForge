"""Embedding back-ends for the RAG store.

Preference order (resolved by :func:`get_embeddings`):

1. ``OpenAIEmbeddings`` - used whenever ``OPENAI_API_KEY`` is present.
2. :class:`TfidfEmbeddings` - a dependency-light, fully offline fallback.

The second one is the normal path for this project: OpenRouter (the chat
provider configured here) exposes chat completions only and has no
``/embeddings`` route, so the retrieval index is built locally.  That keeps
indexing free and offline while the *reasoning* still runs on a hosted LLM.

The fallback is a genuine embedding model in the LangChain sense: it exposes
``embed_documents`` / ``embed_query`` and returns fixed-length dense vectors.
It is built from TF-IDF followed by truncated SVD (LSA), which gives sensible
semantic-ish similarity on a small corpus without downloading any weights.
"""

from __future__ import annotations

import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from insightforge.config import settings


class TfidfEmbeddings(Embeddings):
    """Offline TF-IDF + LSA embeddings, fitted on the corpus being indexed."""

    def __init__(self, n_components: int = 256) -> None:
        self.n_components = n_components
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        self._svd: TruncatedSVD | None = None
        self._fitted = False

    # -- LangChain interface ------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.fit_transform(texts)
        n_components = int(min(self.n_components, max(1, min(matrix.shape) - 1)))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = self._svd.fit_transform(matrix)
        self._fitted = True
        return self._normalise(dense).tolist()

    def embed_query(self, text: str) -> list[float]:
        if not self._fitted or self._svd is None:
            raise RuntimeError(
                "TfidfEmbeddings must index documents before embedding a query."
            )
        dense = self._svd.transform(self._vectorizer.transform([text]))
        return self._normalise(dense)[0].tolist()

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TfidfEmbeddings(n_components={self.n_components}, fitted={self._fitted})"


def get_embeddings() -> tuple[Embeddings, str]:
    """Return ``(embeddings, backend_name)`` for the current environment."""
    if settings.supports_api_embeddings:
        try:
            from langchain_openai import OpenAIEmbeddings

            return (
                OpenAIEmbeddings(
                    model=settings.embedding_model,
                    api_key=settings.openai_api_key,
                ),
                f"openai:{settings.embedding_model}",
            )
        except Exception:  # pragma: no cover - network/import problems
            pass
    return TfidfEmbeddings(), "offline:tfidf-lsa"
