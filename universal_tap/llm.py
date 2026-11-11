"""OpenAI-compatible chat clients for the attacker, target and judge.

Each role has its own client and model config, so requests and histories stay
independent. A role's ``provider`` is either:

* ``"vllm"`` — a local server; base URL and key come from ``OPENAI_COMPATIBLE_BASE_URL``
  / ``OPENAI_COMPATIBLE_API_KEY``, normally the endpoint file that
  ``structural_attack/vllm/serve_multinode.sh`` writes (``env_file`` in the config);
* ``"openrouter"`` — key from ``OPENROUTER_API_KEY``.

Values in the endpoint file override the process environment, which may still hold a
previous job's node. Keys are never written to logs or configs.

Rate limits, timeouts and 5xx are retried with backoff; auth failures, unknown models and
bad requests raise ``FatalAPIError`` at once rather than silently switching models.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class FatalAPIError(RuntimeError):
    """Non-retryable API failure: bad auth, unknown model, malformed request, no endpoint."""


def resolve_endpoint(provider: str, env_file: str | None) -> tuple[str, str]:
    """Return ``(base_url, api_key)`` for ``provider``."""
    if env_file is not None:
        if not Path(env_file).exists():
            raise FatalAPIError(
                f"endpoint file {env_file} not found (the vLLM serve script removes it when "
                "its job ends; is the server still running?)"
            )
        load_dotenv(env_file, override=True)
    else:
        load_dotenv()

    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
    if provider == "vllm":
        if not base_url:
            raise FatalAPIError("OPENAI_COMPATIBLE_BASE_URL is not set; point env_file at the vLLM endpoint file")
        if "openrouter.ai" in base_url:
            raise FatalAPIError(f"provider 'vllm' but OPENAI_COMPATIBLE_BASE_URL is OpenRouter ({base_url})")
        return base_url, os.getenv("OPENAI_COMPATIBLE_API_KEY") or "EMPTY"
    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key and "openrouter.ai" in base_url:
            key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        if not key:
            raise FatalAPIError("OPENROUTER_API_KEY is not set")
        return OPENROUTER_BASE_URL, key
    raise FatalAPIError(f"unknown provider {provider!r}; expected 'vllm' or 'openrouter'")


@dataclass
class ModelConfig:
    """One role's model and sampling settings."""

    model: str
    provider: str = "vllm"
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    request_timeout: float = 120.0
    max_retries: int = 4

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Usage:
    """Running request/token totals for one client."""

    n_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, completion: Any) -> None:
        self.n_requests += 1
        usage = getattr(completion, "usage", None)
        if usage is not None:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0
            self.total_tokens += usage.total_tokens or 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


_usage_lock = threading.Lock()
_tracked = threading.local()  # per thread: {id(client usage): Usage of the current track() block}


class _CountingOpenAI:
    """Stands in for ``openai.OpenAI`` where agentdojo calls
    ``client.chat.completions.create(...)``, tallying every request. Thread-safe."""

    def __init__(self, inner: openai.OpenAI, usage: Usage) -> None:
        self._inner = inner
        self._usage = usage
        self.chat = self.completions = self  # client.chat.completions.create -> self.create

    def create(self, **kwargs: Any) -> Any:
        completion = self._inner.chat.completions.create(**kwargs)
        with _usage_lock:
            self._usage.add(completion)
        scoped = getattr(_tracked, "usages", {}).get(id(self._usage))
        if scoped is not None:
            scoped.add(completion)
        return completion


class LLMClient:
    """One role's client. ``openai_client`` goes to agentdojo's pipeline; ``chat()`` is for
    calls made here (attacker, judge, smoke test)."""

    def __init__(self, config: ModelConfig, base_url: str, api_key: str) -> None:
        self.config = config
        self.base_url = base_url
        self.usage = Usage()
        inner = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=config.request_timeout,
            max_retries=0,  # chat() does its own classified retries
        )
        self.openai_client = _CountingOpenAI(inner, self.usage)

    @contextmanager
    def track(self) -> Iterator[Usage]:
        """Usage of this client's requests made by the current thread inside the block
        (other threads' concurrent requests are not counted)."""
        usages = _tracked.__dict__.setdefault("usages", {})
        scoped = usages[id(self.usage)] = Usage()
        try:
            yield scoped
        finally:
            usages.pop(id(self.usage), None)

    def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> Any:
        """One chat completion with classified retries; returns the raw ``ChatCompletion``."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            if attempt:
                time.sleep(min(2.0 * 2 ** (attempt - 1), 30.0))
            try:
                completion = self.openai_client.create(**kwargs)
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                raise FatalAPIError(f"authentication/permission error: {exc}") from exc
            except openai.NotFoundError as exc:
                raise FatalAPIError(f"model {self.config.model!r} not found at {self.base_url}: {exc}") from exc
            except openai.BadRequestError as exc:
                raise FatalAPIError(f"bad request: {exc}") from exc
            except _RETRYABLE as exc:
                last_exc = exc
                continue
            if not getattr(completion, "choices", None):  # provider error inside a 200 response
                last_exc = RuntimeError(getattr(completion, "error", None) or "empty choices in response")
                continue
            return completion
        raise RuntimeError(f"chat request failed after {self.config.max_retries + 1} attempts: {last_exc}")


def build_clients(configs: dict[str, ModelConfig], env_file: str | None) -> dict[str, LLMClient]:
    """One independent client per role (e.g. attacker/target/judge)."""
    endpoints: dict[str, tuple[str, str]] = {}
    clients = {}
    for role, config in configs.items():
        if config.provider not in endpoints:
            endpoints[config.provider] = resolve_endpoint(config.provider, env_file)
        clients[role] = LLMClient(config, *endpoints[config.provider])
    return clients
