"""LLM factory.

This project talks to LLMs through **OpenRouter**, which speaks the OpenAI
chat-completions protocol, so ``ChatOpenAI`` is used with OpenRouter's base
URL.  The pipeline also has to be runnable (and gradable) on a machine with
no key at all, so the factory resolves one of four back-ends:

``openrouter`` ``ChatOpenAI`` pointed at ``https://openrouter.ai/api/v1`` -
             the default whenever ``OPENROUTER_API_KEY`` is set.
``openai``   ``ChatOpenAI`` against the OpenAI API directly.
``google``   ``ChatGoogleGenerativeAI`` - used when ``GOOGLE_API_KEY`` is set
             and ``langchain-google-genai`` is installed.
``offline``  :class:`OfflineChatModel` - a deterministic, extractive model
             that answers straight from the retrieved context.  It is not a
             generative model; it exists so that retrieval, chaining, memory,
             evaluation and the Streamlit UI can all be demonstrated end to
             end without network access.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from insightforge.config import settings


# ---------------------------------------------------------------------------
# Offline fallback model
# ---------------------------------------------------------------------------
class OfflineChatModel(BaseChatModel):
    """Deterministic extractive 'LLM' used when no API key is available.

    It reads the ``Context`` block that the prompt template builds, keeps the
    lines that overlap most with the question, and renders them as a short
    analyst-style answer.  Answers are therefore always grounded and never
    hallucinated - at the cost of not being fluent prose.
    """

    max_lines: int = 12

    @property
    def _llm_type(self) -> str:
        return "insightforge-offline"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        answer = self._compose(prompt)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=answer))]
        )

    # -- extraction ---------------------------------------------------------
    @staticmethod
    def _split_prompt(prompt: str) -> tuple[str, str]:
        context = ""
        question = ""
        ctx = re.search(r"Context:\s*(.*?)(?:\n\s*(?:Chat history|Question):|\Z)", prompt, re.S)
        if ctx:
            context = ctx.group(1)
        qst = re.search(r"Question:\s*(.*?)\s*(?:\n[A-Z][a-z]+:|\Z)", prompt, re.S)
        if qst:
            question = qst.group(1)
        return context.strip(), question.strip()

    def _compose(self, prompt: str) -> str:
        context, question = self._split_prompt(prompt)
        if not context:
            return (
                "Offline mode: no context was retrieved for this question, so no "
                "grounded answer can be given. Set OPENROUTER_API_KEY to enable "
                "the generative model."
            )

        tokens = {t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) > 3}
        scored: list[tuple[int, str]] = []
        for line in context.splitlines():
            clean = line.strip()
            if len(clean) < 12 or clean.startswith("["):
                continue
            words = set(re.findall(r"[a-z0-9]+", clean.lower()))
            score = len(tokens & words)
            has_number = bool(re.search(r"\d", clean))
            scored.append((score * 2 + int(has_number), clean))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        chosen = [line for score, line in scored[: self.max_lines] if score > 0]
        if not chosen:
            chosen = [line for _, line in scored[: self.max_lines]]

        body = "\n".join(f"- {line.lstrip('- ')}" for line in chosen)
        return (
            "**Findings (offline extractive mode - figures taken directly from the "
            "retrieved statistics):**\n"
            f"{body}\n\n"
            "**Note:** running with an `OPENROUTER_API_KEY` replaces this "
            "extraction with a full narrative answer and recommendations."
        )

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGeneration]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_llm(
    provider: str | None = None,
    temperature: float | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[BaseChatModel, str]:
    """Return ``(llm, description)`` for the requested / resolved provider.

    ``api_key`` overrides the configured OpenRouter key for this client only.
    The Streamlit sidebar uses it so a visitor can supply their own key without
    touching the shared ``settings`` singleton, which every session reads.
    """
    api_key = (api_key or "").strip() or None
    provider = (provider or "auto").lower()
    if provider == "auto":
        # "auto" is a request to resolve, not a provider name - without this the
        # call would fall through every branch and land in offline mode.
        provider = settings.resolve_provider(openrouter_key=api_key)
    temperature = settings.temperature if temperature is None else temperature

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        model = model or settings.chat_model
        return (
            ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=settings.max_tokens,
                api_key=api_key or settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                default_headers={
                    # optional OpenRouter attribution headers
                    "HTTP-Referer": "https://github.com/insightforge-capstone",
                    "X-Title": "InsightForge BI Assistant",
                },
            ),
            f"openrouter:{model}",
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = model or settings.chat_model
        return (
            ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=settings.max_tokens,
                api_key=settings.openai_api_key,
            ),
            f"openai:{model}",
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = model or "gemini-1.5-flash"
        return (
            ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=settings.google_api_key,
            ),
            f"google:{model}",
        )

    return OfflineChatModel(), "offline:extractive"
