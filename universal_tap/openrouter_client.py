"""OpenRouter chat client for Universal TAP.

A thin, UI-free client built on the OpenAI 1.x SDK, pointed at OpenRouter. Credentials
come from the user's existing ``.env`` (never written to code, logs or configs):

* ``OPENROUTER_API_KEY`` is preferred;
* ``OPENAI_COMPATIBLE_API_KEY`` is accepted *only* when ``OPENAI_COMPATIBLE_BASE_URL``
  is itself an OpenRouter URL (as in the reference project's ``.env``).

This mirrors ``structural_attack/ui/chateditor.py::_client`` but without importing the
UI server. Attacker / target / judge each build their own client and model config so
their requests, histories and roles stay independent.

Structured ``tool_calls``, call IDs, tool results and ``finish_reason`` are preserved:
``chat()`` returns the raw OpenAI ``ChatCompletion`` so callers can read tool calls and
usage. ``CountingClient`` tallies request and token counts per client for cost logging.

Errors are classified so the search can treat them differently (see ``plan.md`` §3):
rate limits and transient 5xx are retried with backoff; authentication failures and
unknown-model errors are raised immediately (``FatalAPIError``) rather than silently
switching models.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Reference project's tentative Qwen 235B id on OpenRouter (plan.md §3). Verify
# availability and tool-calling support at run time; do not silently fall back.
DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"

# Transient conditions worth retrying (rate limits, timeouts, provider 5xx).
_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class FatalAPIError(RuntimeError):
    """Non-retryable API failure: bad auth, unknown model, malformed request."""


def load_api_key(env_path: str | os.PathLike[str] | None = None) -> str:
    """Resolve the OpenRouter API key from the environment / ``.env``.

    Args:
        env_path: explicit ``.env`` file to load; otherwise dotenv's default search.

    Returns:
        The API key string.

    Raises:
        FatalAPIError: if no suitable key is configured.
    """
    if env_path is not None:
        load_dotenv(Path(env_path))
    else:
        load_dotenv()

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
        if "openrouter.ai" in base_url:
            key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    if not key:
        raise FatalAPIError(
            "No OpenRouter API key found. Set OPENROUTER_API_KEY in your .env "
            "(or OPENAI_COMPATIBLE_API_KEY when OPENAI_COMPATIBLE_BASE_URL is an "
            "OpenRouter URL)."
        )
    return key


@dataclass
class UsageCounters:
    """Running totals for one client, for cost reporting (plan.md §10, §11)."""

    n_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, completion: Any) -> None:
        self.n_requests += 1
        usage = getattr(completion, "usage", None)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.total_tokens += getattr(usage, "total_tokens", 0) or 0

    def as_dict(self) -> dict[str, int]:
        return {
            "n_requests": self.n_requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def snapshot(self) -> "UsageCounters":
        return UsageCounters(**self.as_dict())

    def delta(self, earlier: "UsageCounters") -> dict[str, int]:
        return {
            "n_requests": self.n_requests - earlier.n_requests,
            "prompt_tokens": self.prompt_tokens - earlier.prompt_tokens,
            "completion_tokens": self.completion_tokens - earlier.completion_tokens,
            "total_tokens": self.total_tokens - earlier.total_tokens,
        }


class _Completions:
    def __init__(self, parent: "CountingClient") -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> Any:
        completion = self._parent._inner.chat.completions.create(**kwargs)
        self._parent.counters.add(completion)
        return completion


class _Chat:
    def __init__(self, parent: "CountingClient") -> None:
        self.completions = _Completions(parent)


class CountingClient:
    """Proxy around ``openai.OpenAI`` exposing ``.chat.completions.create`` and tallying
    requests/tokens. Accepted anywhere agentdojo expects an ``openai.OpenAI`` client,
    since the pipeline only calls ``client.chat.completions.create(...)``."""

    def __init__(self, inner: openai.OpenAI) -> None:
        self._inner = inner
        self.counters = UsageCounters()
        self.chat = _Chat(self)

    @property
    def inner(self) -> openai.OpenAI:
        return self._inner


@dataclass
class ModelConfig:
    """Per-role model/sampling configuration (attacker, target, or judge)."""

    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    request_timeout: float = 120.0
    max_retries: int = 4

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
        }


class OpenRouterClient:
    """One role's OpenRouter client plus its model config.

    Use ``openai_client`` as the ``openai.OpenAI``-compatible object for agentdojo's
    pipeline (it counts usage), and ``chat()`` for attacker/judge calls made directly
    here.
    """

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        self.config = config
        inner = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=config.request_timeout,
            max_retries=0,  # we do our own classified retries in chat()
        )
        self.openai_client = CountingClient(inner)

    @property
    def counters(self) -> UsageCounters:
        return self.openai_client.counters

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        """One chat-completion request with classified retries.

        Returns the raw ``ChatCompletion`` (callers read ``.choices[0].message`` for
        content and ``tool_calls``, and ``.usage`` is already tallied). OpenRouter
        occasionally reports provider errors inside a 200 response with no choices;
        that is surfaced as a retryable error.
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        effective_max = max_tokens if max_tokens is not None else self.config.max_tokens
        if effective_max is not None:
            kwargs["max_tokens"] = effective_max
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                completion = self.openai_client.chat.completions.create(**kwargs)
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                raise FatalAPIError(f"authentication/permission error: {exc}") from exc
            except openai.NotFoundError as exc:
                raise FatalAPIError(
                    f"model {self.config.model!r} not found on OpenRouter: {exc}"
                ) from exc
            except openai.BadRequestError as exc:
                raise FatalAPIError(f"bad request: {exc}") from exc
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(min(2.0 * (2**attempt), 30.0))
                continue

            if not getattr(completion, "choices", None):
                # Provider error reported in a 200 response.
                last_exc = RuntimeError(
                    getattr(completion, "error", None) or "empty choices in response"
                )
                if attempt >= self.config.max_retries:
                    break
                time.sleep(min(2.0 * (2**attempt), 30.0))
                continue

            return completion

        raise RuntimeError(
            f"chat request failed after {self.config.max_retries + 1} attempts: {last_exc}"
        )


def build_clients(
    *,
    attacker: ModelConfig,
    target: ModelConfig,
    judge: ModelConfig,
    env_path: str | os.PathLike[str] | None = None,
) -> dict[str, OpenRouterClient]:
    """Build independent attacker/target/judge clients sharing one API key."""
    api_key = load_api_key(env_path)
    return {
        "attacker": OpenRouterClient(attacker, api_key),
        "target": OpenRouterClient(target, api_key),
        "judge": OpenRouterClient(judge, api_key),
    }
