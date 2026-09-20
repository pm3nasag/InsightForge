"""Central configuration for InsightForge.

Everything that could differ between machines (paths, model names, API keys)
is resolved here exactly once, so the rest of the package never has to touch
``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional - only needed when a .env file is used
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime settings, overridable through environment variables."""

    # --- paths -------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_file: Path = PROJECT_ROOT / "Datasets" / "sales_data.csv"
    pdf_dir: Path = PROJECT_ROOT / "Datasets" / "PDF Folder"
    vector_store_dir: Path = PROJECT_ROOT / "outputs" / "vector_store"
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    reports_dir: Path = PROJECT_ROOT / "outputs" / "reports"
    log_file: Path = PROJECT_ROOT / "outputs" / "interactions.jsonl"
    eval_file: Path = PROJECT_ROOT / "eval" / "qa_pairs.json"

    # --- LLM ---------------------------------------------------------------
    # provider: "openrouter" | "openai" | "google" | "offline" | "auto"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    openrouter_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
    )
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    google_api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    chat_model: str = field(
        default_factory=lambda: os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "900")))

    # --- RAG ---------------------------------------------------------------
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 5
    memory_window: int = 6  # number of past exchanges kept in memory

    def __post_init__(self) -> None:
        for directory in (
            self.vector_store_dir,
            self.figures_dir,
            self.reports_dir,
            self.log_file.parent,
            self.eval_file.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # --- helpers -----------------------------------------------------------
    def resolve_provider(self) -> str:
        """Return the provider actually usable on this machine.

        ``auto`` prefers OpenRouter (the provider configured for this
        project), then a direct OpenAI key, then Google, and finally falls
        back to the offline deterministic model so that the whole pipeline
        (and the Streamlit demo) still runs without any API key.
        """
        provider = (self.llm_provider or "auto").lower()
        if provider != "auto":
            return provider
        if self.openrouter_api_key:
            return "openrouter"
        if self.openai_api_key:
            return "openai"
        if self.google_api_key:
            return "google"
        return "offline"

    @property
    def has_api_key(self) -> bool:
        return bool(self.openrouter_api_key or self.openai_api_key or self.google_api_key)

    @property
    def supports_api_embeddings(self) -> bool:
        """OpenRouter exposes chat completions only - no /embeddings route."""
        return bool(self.openai_api_key)


settings = Settings()
