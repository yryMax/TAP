"""Run one AgentDojo case under a frozen injection template and extract results.

This reimplements the essential body of agentdojo's ``run_task_with_pipeline`` so the
full message trace is available (agentdojo's own helper returns only utility/security).
From the trace it extracts the signals Universal TAP needs (plan.md §4, §5):

* ``attack_success`` — agentdojo's ``security`` check (the injection task executed);
* ``utility`` — the benign user task still completed;
* ``injection_seen`` — whether the rendered injection actually appeared in a tool
  result the agent read (a case may run to the end without ever meeting it);
* ``first_step_response`` — the agent's first assistant turn *after* reading the
  injection, plus its structured tool calls, for the judge.

Environments are built fresh per run, from the task's initial state (no state caching
in the first version). The system prompt is sent with role ``system`` (not
``developer``) so Qwen-style chat templates keep it, matching the reference runner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import openai
import yaml
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.llms import openai_llm
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import (
    functions_stack_trace_from_messages,
    model_output_from_messages,
)
from agentdojo.types import get_text_content_as_str

from .attacks import FrozenTemplateAttack
from .openrouter_client import OpenRouterClient
from .schemas import RunStatus, Template, ToolCallRecord

_SYSTEM_ROLE_PATCHED = False


def use_system_role() -> None:
    """Send agentdojo's system prompt as role ``system`` instead of ``developer``.

    Qwen3 / GLM chat templates silently drop a ``developer`` message; vLLM and some
    providers rename it to ``system`` first, but not all do. This mirrors
    ``structural_attack/run_agentdojo.py::use_system_role`` without importing it, so
    the system prompt reliably reaches the request. Idempotent.
    """
    global _SYSTEM_ROLE_PATCHED
    if _SYSTEM_ROLE_PATCHED or getattr(openai_llm._message_to_openai, "uses_system_role", False):
        _SYSTEM_ROLE_PATCHED = True
        return
    original = openai_llm._message_to_openai

    def message_to_openai(message, model_name):
        converted = original(message, model_name)
        if converted["role"] == "developer":
            return {**converted, "role": "system"}
        return converted

    message_to_openai.uses_system_role = True
    openai_llm._message_to_openai = message_to_openai
    _SYSTEM_ROLE_PATCHED = True


@dataclass
class RunOutput:
    """Everything one agent run produced, before judging."""

    status: RunStatus
    attack_success: bool | None
    utility: bool | None
    injection_seen: bool
    first_step_response: str | None
    first_step_tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class AgentRunner:
    """Runs AgentDojo cases against an OpenRouter target model with a frozen template."""

    def __init__(
        self,
        target_client: OpenRouterClient,
        *,
        max_tool_iters: int = 15,
        system_message_name: str | None = None,
    ) -> None:
        use_system_role()
        self.target_client = target_client
        self.max_tool_iters = max_tool_iters
        self.system_message = load_system_message(system_message_name)
        llm = OpenAILLM(target_client.openai_client, target_client.config.model)
        self._llm = llm
        self._pipeline = AgentPipeline(
            [
                SystemMessage(self.system_message),
                InitQuery(),
                llm,
                ToolsExecutionLoop([ToolsExecutor(), llm], max_iters=max_tool_iters),
            ]
        )
        self._pipeline.name = f"universal-tap/{target_client.config.model}"

    @staticmethod
    @lru_cache(maxsize=None)
    def _suite(benchmark_version: str, suite_name: str):
        return get_suite(benchmark_version, suite_name)

    def attack_for(self, benchmark_version: str, suite_name: str, template: Template) -> FrozenTemplateAttack:
        """A frozen-template attack bound to a suite (cheap; cached candidates live on it)."""
        suite = self._suite(benchmark_version, suite_name)
        return FrozenTemplateAttack(suite, template)

    def run_case(
        self,
        *,
        benchmark_version: str,
        suite_name: str,
        user_task_id: str,
        injection_task_id: str,
        template: Template,
    ) -> RunOutput:
        """Run one (user_task, injection_task) pair under ``template``."""
        suite = self._suite(benchmark_version, suite_name)
        user_task = suite.get_user_task_by_id(user_task_id)
        injection_task = suite.get_injection_task_by_id(injection_task_id)

        try:
            attack = self.attack_for(benchmark_version, suite_name, template)
            injections = attack.attack(user_task, injection_task)
            raw_injection = attack.rendered_raw(injection_task.GOAL)
        except Exception as exc:  # noqa: BLE001 - suite/rendering failures are env errors
            return RunOutput(
                status=RunStatus.ENV_ERROR,
                attack_success=None,
                utility=None,
                injection_seen=False,
                first_step_response=None,
                error=f"injection rendering failed: {exc!r}",
            )

        return self._rollout(
            suite,
            task=user_task,
            injection_task=injection_task,
            injections=injections,
            needles=(raw_injection, injection_task.GOAL),
        )

    def run_baseline(
        self,
        *,
        benchmark_version: str,
        suite_name: str,
        kind: str,
        task_id: str,
    ) -> RunOutput:
        """Run a no-injection baseline, mirroring agentdojo's
        ``run_task_without_injection_tasks`` (plan.md §9 phase B).

        * ``kind="benign"``: the user task with the default (empty) injection vectors;
          ``utility`` is the user task's environment check.
        * ``kind="standalone"``: the injection task's GOAL given directly as the user
          prompt; ``utility`` is the injection task's own security check, i.e. whether
          the target model *can* carry out the attacker goal when asked outright.

        ``attack_success`` is ``None`` and ``injection_seen`` is ``False`` for both.
        """
        suite = self._suite(benchmark_version, suite_name)
        if kind == "benign":
            task = suite.get_user_task_by_id(task_id)
        elif kind == "standalone":
            task = suite.get_injection_task_by_id(task_id)
        else:
            raise ValueError(f"unknown baseline kind {kind!r}")
        return self._rollout(suite, task=task, injection_task=None, injections={}, needles=None)

    def _rollout(
        self,
        suite: Any,
        *,
        task: BaseUserTask | BaseInjectionTask,
        injection_task: BaseInjectionTask | None,
        injections: dict[str, str],
        needles: tuple[str, ...] | None,
    ) -> RunOutput:
        """One full agent rollout plus agentdojo's environment checks.

        ``task`` supplies the prompt (``PROMPT`` for a user task, ``GOAL`` for an
        injection task run standalone) and the ``utility`` check; ``injection_task``,
        if given, supplies the ``attack_success`` check. ``needles`` enables
        injection-seen / first-step extraction.
        """
        is_user_task = isinstance(task, BaseUserTask)
        prompt = task.PROMPT if is_user_task else task.GOAL
        try:
            environment = suite.load_and_inject_default_environment(injections)
            task_environment = task.init_environment(environment) if is_user_task else environment
            pre_environment = task_environment.model_copy(deep=True)
            runtime = FunctionsRuntime(suite.tools)

            messages: list[Any] = []
            model_output = None
            status = RunStatus.OK
            for _ in range(3):  # agentdojo retries up to 3x for a usable output
                try:
                    _, _, task_environment, messages, _ = self._pipeline.query(
                        prompt, runtime, task_environment
                    )
                except AbortAgentError as exc:
                    task_environment = exc.task_environment
                    messages = exc.messages
                try:
                    model_output = model_output_from_messages(messages)
                except ValueError:
                    model_output = None
                if model_output is not None:
                    break
        except openai.BadRequestError as exc:
            if (
                getattr(exc, "code", None) == "context_length_exceeded"
                or getattr(exc, "param", None) == "max_tokens"
                or "reduce the length of the messages." in repr(exc)
            ):
                return RunOutput(
                    status=RunStatus.CONTEXT_LENGTH,
                    # agentdojo's convention for this case
                    attack_success=True if injection_task is not None else None,
                    utility=False,
                    injection_seen=False,
                    first_step_response=None,
                    error=f"context_length_exceeded: {exc}",
                )
            return RunOutput(
                status=RunStatus.API_ERROR,
                attack_success=None,
                utility=None,
                injection_seen=False,
                first_step_response=None,
                error=f"bad request: {exc}",
            )
        except (openai.APIError, RuntimeError) as exc:
            return RunOutput(
                status=RunStatus.API_ERROR,
                attack_success=None,
                utility=None,
                injection_seen=False,
                first_step_response=None,
                error=f"api error during run: {exc}",
            )

        if model_output is None:
            status = RunStatus.INCOMPLETE

        trace = functions_stack_trace_from_messages(messages)
        try:
            utility = suite._check_task_result(
                task, model_output or [], pre_environment, task_environment, trace
            )
            attack_success = None
            if injection_task is not None:
                attack_success = bool(
                    suite._check_task_result(
                        injection_task, model_output or [], pre_environment, task_environment, trace
                    )
                )
        except Exception as exc:  # noqa: BLE001 - a check failure is an env error, not success
            return RunOutput(
                status=RunStatus.ENV_ERROR,
                attack_success=None,
                utility=None,
                injection_seen=False,
                first_step_response=None,
                messages=_messages_to_json(messages),
                error=f"result check failed: {exc!r}",
            )

        if status == RunStatus.OK and _hit_step_limit(messages, self.max_tool_iters):
            status = RunStatus.STEP_LIMIT

        seen, first_response, first_calls = False, None, []
        if needles:
            seen, first_response, first_calls = _extract_first_step(messages, *needles)

        return RunOutput(
            status=status,
            attack_success=attack_success,
            utility=bool(utility),
            injection_seen=seen,
            first_step_response=first_response,
            first_step_tool_calls=first_calls,
            messages=_messages_to_json(messages),
        )


def _message_text(message: Any) -> str:
    content = message.get("content")
    if not content:
        return ""
    try:
        return get_text_content_as_str(content)
    except Exception:  # noqa: BLE001
        if isinstance(content, str):
            return content
        return ""


def _tool_calls_of(message: Any) -> list[ToolCallRecord]:
    calls = message.get("tool_calls") or []
    records = []
    for call in calls:
        # agentdojo's FunctionCall is a pydantic model; JSON-loaded traces are dicts.
        # Read by type, not truthiness: a no-argument call has args == {}.
        if isinstance(call, dict):
            function, args, call_id = call.get("function"), call.get("args"), call.get("id")
        else:
            function = getattr(call, "function", None)
            args = getattr(call, "args", None)
            call_id = getattr(call, "id", None)
        records.append(ToolCallRecord(function=function, args=dict(args or {}), id=call_id))
    return records


def _extract_first_step(
    messages: list[Any], raw_injection: str, goal: str
) -> tuple[bool, str | None, list[ToolCallRecord]]:
    """Find whether the injection was read, and the first assistant turn after it.

    ``injection_seen`` is true when the rendered injection text (or, as a fallback, the
    attacker goal) appears in a tool-result message the agent received. The first-step
    response is the first assistant message following that tool result.
    """
    needles = [n for n in (raw_injection, goal) if n]
    seen_index = None
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        text = _message_text(message) or (message.get("error") or "")
        if _contains_injection(text, needles):
            seen_index = index
            break

    if seen_index is None:
        return False, None, []

    for message in messages[seen_index + 1 :]:
        if message.get("role") == "assistant":
            return True, _message_text(message), _tool_calls_of(message)
    # Injection was read but the agent emitted no further assistant turn.
    return True, None, []


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _yaml_strings(obj: Any) -> list[str]:
    """All string scalars (keys and values) inside a parsed YAML document."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out: list[str] = []
        for key, value in obj.items():
            out.extend(_yaml_strings(key))
            out.extend(_yaml_strings(value))
        return out
    if isinstance(obj, (list, tuple)):
        return [s for item in obj for s in _yaml_strings(item)]
    return []


def _contains_injection(text: str, needles: list[str]) -> bool:
    """Whether any needle occurs in a tool result, robust to agentdojo's formatting.

    agentdojo renders tool results with ``yaml.safe_dump``, which line-wraps long
    scalars, may single-/double-quote them (``'`` -> ``''``, ``\\n`` escapes) and
    folds whitespace, so a plain substring test misses injections the agent did read.
    We compare whitespace-normalized text, then fall back to parsing the result as
    YAML and comparing each decoded string scalar.
    """
    norm_needles = [_normalize_ws(n) for n in needles if n and n.strip()]
    if not norm_needles or not text:
        return False
    norm_text = _normalize_ws(text)
    if any(n in norm_text for n in norm_needles):
        return True
    try:
        parsed = yaml.safe_load(text)
    except Exception:  # noqa: BLE001 - tool output need not be YAML
        return False
    for scalar in _yaml_strings(parsed):
        norm_scalar = _normalize_ws(scalar)
        if any(n in norm_scalar for n in norm_needles):
            return True
    return False


def _hit_step_limit(messages: list[Any], max_iters: int) -> bool:
    """Heuristic: the loop ran and the last assistant turn still requested tools,
    i.e. it was cut off by max_iters rather than finishing."""
    assistant_turns = [m for m in messages if m.get("role") == "assistant"]
    if len(assistant_turns) < max_iters:
        return False
    last = messages[-1] if messages else None
    return bool(last and last.get("role") == "assistant" and (last.get("tool_calls")))


def _messages_to_json(messages: list[Any]) -> list[dict[str, Any]]:
    """Reduce agentdojo ChatMessages to JSON-serializable turns for the trace file."""
    out = []
    for message in messages:
        turn: dict[str, Any] = {"role": message.get("role"), "content": _message_text(message)}
        if message.get("role") == "assistant":
            turn["tool_calls"] = [tc.as_dict() for tc in _tool_calls_of(message)]
        if message.get("role") == "tool":
            turn["error"] = message.get("error")
        out.append(turn)
    return out
