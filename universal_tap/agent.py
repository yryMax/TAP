"""Run AgentDojo cases against the target model and extract what the search needs.

``AgentRunner.run_case`` renders the template around the task's attacker goal, injects it
into every vector the user task's ground-truth execution reads (as agentdojo's own
``important_instructions`` attack does), runs one full rollout from the task's initial
state, and returns:

* ``attack_success`` — agentdojo's ``security`` check (the injection task was executed);
* ``utility`` — the benign user task was still completed;
* ``injection_seen`` — the injection actually appeared in a tool result the agent read;
* the first assistant turn after that tool result, with its tool calls, for the judge.

A run is ``ok`` only if the agent finished normally; otherwise it is retried from a fresh
environment, up to ``max_attempts`` times, and is ``failed`` if no attempt succeeded.

This reimplements the body of agentdojo's ``run_task_with_pipeline`` so the full message
trace is available. The system prompt is sent as role ``system`` (not ``developer``),
which Qwen-style chat templates would otherwise drop.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

import openai
import yaml
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.llms import openai_llm
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages, model_output_from_messages
from agentdojo.types import get_text_content_as_str

from .llm import FatalAPIError, LLMClient
from .schemas import RunStatus, Template, ToolCallRecord


def use_system_role() -> None:
    """Send agentdojo's system prompt as role ``system`` instead of ``developer``. Idempotent."""
    original = openai_llm._message_to_openai
    if getattr(original, "uses_system_role", False):
        return

    def message_to_openai(message, model_name):
        converted = original(message, model_name)
        return {**converted, "role": "system"} if converted["role"] == "developer" else converted

    message_to_openai.uses_system_role = True
    openai_llm._message_to_openai = message_to_openai


def escape_for_yaml(text: str) -> str:
    """Escape text for agentdojo's injection slots.

    agentdojo substitutes injections into ``environment.yaml`` with one ``str.format`` pass
    (so braces in the value need no doubling) inside double-quoted YAML scalars, then
    ``yaml.safe_load``s it. ``json.dumps(text)[1:-1]`` is exactly a double-quoted string
    body (quotes, backslashes, newlines, control characters).
    """
    return json.dumps(text, ensure_ascii=False)[1:-1]


class _InjectionVectors(BaseAttack):
    """Only used for agentdojo's ``get_injection_candidates``, which replays the ground truth
    with canaries to find the injection vectors the agent reads. Never queries a model."""

    name = "universal_tap"

    def __init__(self, suite: Any) -> None:
        super().__init__(suite, target_pipeline=None)

    def attack(self, user_task, injection_task):  # pragma: no cover - not used
        raise NotImplementedError


@dataclass
class RunOutput:
    """Everything one agent run produced, before judging."""

    status: RunStatus
    attack_success: bool | None = None
    utility: bool | None = None
    injection_seen: bool = False
    first_step_response: str | None = None
    first_step_tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None  # why the run failed (every attempt's reason)
    attempts: int = 1
    # task texts, for the judge
    user_prompt: str = ""
    attacker_goal: str = ""
    injection_text: str = ""
    injection_message_id: str | None = None
    termination: str = "completed"


class AgentRunner:
    """Runs AgentDojo cases against the target model."""

    def __init__(
        self,
        target_client: LLMClient,
        *,
        benchmark_version: str,
        max_tool_iters: int = 15,
        max_attempts: int = 3,
        system_message_name: str | None = None,
    ) -> None:
        use_system_role()
        self.target_client = target_client
        self.benchmark_version = benchmark_version
        self.max_tool_iters = max_tool_iters
        self.max_attempts = max_attempts
        self.system_message = load_system_message(system_message_name)
        llm = OpenAILLM(
            target_client.openai_client, target_client.config.model, temperature=target_client.config.temperature
        )
        self._pipeline = AgentPipeline(
            [
                SystemMessage(load_system_message(system_message_name)),
                InitQuery(),
                llm,
                ToolsExecutionLoop([ToolsExecutor(), llm], max_iters=max_tool_iters),
            ]
        )
        self._pipeline.name = f"universal-tap/{target_client.config.model}"
        # Caches shared by concurrent runs. The pipeline itself keeps no per-query state,
        # and each run builds its own environment and runtime.
        self._suites: dict[str, Any] = {}
        self._vectors: dict[tuple[str, str], list[str]] = {}
        self._turn_boundaries: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}
        self._cache_lock = threading.RLock()

    def suite(self, name: str) -> Any:
        with self._cache_lock:
            if name not in self._suites:
                self._suites[name] = get_suite(self.benchmark_version, name)
            return self._suites[name]

    def _injection_vectors(self, suite_name: str, user_task: BaseUserTask) -> list[str]:
        key = (suite_name, user_task.ID)
        with self._cache_lock:
            if key not in self._vectors:
                self._vectors[key] = _InjectionVectors(self.suite(suite_name)).get_injection_candidates(user_task)
            return self._vectors[key]

    def run_case(self, case: tuple[str, str, str], template: Template) -> RunOutput:
        """Run one (suite, user_task_id, injection_task_id) pair under ``template``."""
        suite_name, user_task_id, injection_task_id = case
        suite = self.suite(suite_name)
        user_task = suite.get_user_task_by_id(user_task_id)
        injection_task = suite.get_injection_task_by_id(injection_task_id)
        injection_text = template.render(injection_task.GOAL)
        texts = {"user_prompt": user_task.PROMPT, "attacker_goal": injection_task.GOAL, "injection_text": injection_text}
        try:
            escaped = escape_for_yaml(injection_text)
            injections = {v: escaped for v in self._injection_vectors(suite_name, user_task)}
        except Exception as exc:  # noqa: BLE001 - deterministic, so not retried
            return RunOutput(RunStatus.FAILED, error=f"injection rendering failed: {exc!r}", **texts)

        run = self._rollout(
            suite, task=user_task, injection_task=injection_task, injections=injections,
            needles=(injection_text, injection_task.GOAL),
        )
        for key, value in texts.items():
            setattr(run, key, value)
        return run

    def run_baseline(self, *, suite_name: str, kind: str, task_id: str) -> RunOutput:
        """A no-injection baseline (agentdojo's ``run_task_without_injection_tasks``):

        * ``benign`` — the user task with empty injection vectors; ``utility`` is its check;
        * ``standalone`` — the injection task's GOAL given directly as the user prompt;
          ``utility`` is whether the target carries out the attacker goal when asked outright.
        """
        suite = self.suite(suite_name)
        if kind == "benign":
            task = suite.get_user_task_by_id(task_id)
        elif kind == "standalone":
            task = suite.get_injection_task_by_id(task_id)
        else:
            raise ValueError(f"unknown baseline kind {kind!r}")
        return self._rollout(suite, task=task, injection_task=None, injections={}, needles=None)

    def prepare_turn_case(self, case, template):
        """Preflight the complete payload in the real YAML slot, without any model calls."""
        from my_attacks import OneTurnAttack
        from .render import build_injections

        suite_name, user_id, injection_id = case
        suite = self.suite(suite_name)
        user_task = suite.get_user_task_by_id(user_id)
        injection_task = suite.get_injection_task_by_id(injection_id)
        key = (suite_name, user_id)
        with self._cache_lock:
            if key not in self._turn_boundaries:
                self._turn_boundaries[key] = OneTurnAttack(suite, self._pipeline)._get_tool_boundaries(user_task)
            boundaries = self._turn_boundaries[key]
        injections, payloads = build_injections(template, injection_task.GOAL, boundaries)
        # Loading catches malformed/invalid YAML before a candidate consumes evaluation budget.
        suite.load_and_inject_default_environment(injections)
        return suite, user_task, injection_task, injections, payloads

    def run_turn_case(self, case, template) -> RunOutput:
        """One rollout, with request-level retries only. Behavioral outcomes are never rerun.

        Stepping the pipeline elements explicitly retains both environment and trace if a
        request fails midway, or the tool budget is reached. Checks still inspect effects
        that already occurred, including on aborted or truncated trajectories.
        """
        try:
            suite, task, injection_task, injections, payloads = self.prepare_turn_case(case, template)
        except Exception as exc:
            return RunOutput(RunStatus.FAILED, error=f"injection rendering failed: {exc!r}",
                             termination="render_error")
        messages = []
        status, error, termination = RunStatus.OK, None, "completed"
        environment = task.init_environment(suite.load_and_inject_default_environment(injections))
        before = environment.model_copy(deep=True)
        runtime = FunctionsRuntime(suite.tools)
        prompt = task.PROMPT
        _, _, _, messages, _ = SystemMessage(self.system_message).query(prompt, runtime, environment)
        _, _, _, messages, _ = InitQuery().query(prompt, runtime, environment, messages)
        try:
            for round_id in range(self.max_tool_iters + 1):
                converted = [openai_llm._message_to_openai(m, self.target_client.config.model) for m in messages]
                tools = [openai_llm._function_to_openai(tool) for tool in runtime.functions.values()]
                completion = self.target_client.chat(converted, tools=tools)
                messages = [*messages, openai_llm._openai_to_assistant_message(completion.choices[0].message)]
                if not messages[-1].get("tool_calls"):
                    if not _message_text(messages[-1]):
                        termination = "no_final_answer"
                    break
                if round_id == self.max_tool_iters:
                    termination = "step_limit"
                    error = f"step limit ({self.max_tool_iters} tool rounds) reached"
                    break
                try:
                    _, _, environment, messages, _ = ToolsExecutor().query(prompt, runtime, environment, messages)
                except AbortAgentError:
                    raise
                except Exception as exc:
                    termination, error = "tool_error", f"{type(exc).__name__}: {exc}"
                    break
        except AbortAgentError as exc:
            messages, environment = exc.messages, exc.task_environment
            termination, error = "aborted", str(exc)
        except Exception as exc:
            status, termination, error = RunStatus.FAILED, "api_error", f"{type(exc).__name__}: {exc}"

        output = messages[-1].get("content") if messages and messages[-1]["role"] == "assistant" else []
        stack = functions_stack_trace_from_messages(messages)
        utility = security = None
        for checked_task, field in ((task, "utility"), (injection_task, "security")):
            try:
                result = bool(suite._check_task_result(checked_task, output or [], before, environment, stack))
                if field == "utility":
                    utility = result
                else:
                    security = result
            except Exception as exc:
                status = RunStatus.FAILED
                error = f"{error + '; ' if error else ''}{field} check failed: {exc!r}"
        seen, response, calls = _extract_first_step(messages, tuple(payloads))
        serialized = _messages_to_json(messages)
        injection_id = next((m["message_id"] for m in serialized
                             if m["role"] == "tool" and
                             _contains_injection(m["content"] or m.get("error") or "", tuple(payloads))), None)
        return RunOutput(status, attack_success=security, utility=utility, injection_seen=seen,
                         first_step_response=response, first_step_tool_calls=calls, messages=serialized,
                         error=error, user_prompt=task.PROMPT, attacker_goal=injection_task.GOAL,
                         injection_text=template.render(injection_task.GOAL),
                         injection_message_id=injection_id, termination=termination)

    def _rollout(self, suite: Any, **kwargs: Any) -> RunOutput:
        """Up to ``max_attempts`` attempts, each from a fresh environment; the first ``ok``
        attempt is returned. If every attempt fails, the run is ``failed`` and ``error``
        lists each attempt's reason."""
        reasons = []
        for attempt in range(1, self.max_attempts + 1):
            run = self._attempt(suite, **kwargs)
            run.attempts = attempt
            if run.status == RunStatus.OK:
                return run
            reasons.append(f"attempt {attempt}: {run.error}")
        run.error = "; ".join(reasons)
        return run

    def _attempt(
        self,
        suite: Any,
        *,
        task: BaseUserTask | BaseInjectionTask,
        injection_task: BaseInjectionTask | None,
        injections: dict[str, str],
        needles: tuple[str, ...] | None,
    ) -> RunOutput:
        """One full rollout plus agentdojo's environment checks. ``task`` supplies the prompt
        and the ``utility`` check; ``injection_task``, if given, the ``attack_success`` check.
        ``ok`` only if the agent finished normally: no error, within the step limit, with a
        final answer, and the checks ran."""
        is_user_task = isinstance(task, BaseUserTask)
        prompt = task.PROMPT if is_user_task else task.GOAL
        messages: list[Any] = []
        try:
            environment = suite.load_and_inject_default_environment(injections)
            task_environment = task.init_environment(environment) if is_user_task else environment
            pre_environment = task_environment.model_copy(deep=True)
            runtime = FunctionsRuntime(suite.tools)
            _, _, task_environment, messages, _ = self._pipeline.query(prompt, runtime, task_environment)
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as exc:
            raise FatalAPIError(f"target model unusable: {exc}") from exc  # no point retrying
        except AbortAgentError as exc:
            return RunOutput(RunStatus.FAILED, messages=_messages_to_json(exc.messages), error="agent aborted")
        except Exception as exc:  # noqa: BLE001 - API errors, context length, tool crashes
            return RunOutput(RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")

        def failed(reason: str) -> RunOutput:
            return RunOutput(RunStatus.FAILED, messages=_messages_to_json(messages), error=reason)

        if _hit_step_limit(messages, self.max_tool_iters):
            return failed(f"step limit ({self.max_tool_iters} tool rounds) reached")
        try:
            model_output = model_output_from_messages(messages)
        except ValueError:
            model_output = None
        if model_output is None:
            return failed("no final answer")
        trace = functions_stack_trace_from_messages(messages)
        try:
            utility = suite._check_task_result(task, model_output, pre_environment, task_environment, trace)
            attack_success = None
            if injection_task is not None:
                attack_success = bool(
                    suite._check_task_result(injection_task, model_output, pre_environment, task_environment, trace)
                )
        except Exception as exc:  # noqa: BLE001
            return failed(f"result check failed: {exc!r}")

        seen, first_response, first_calls = _extract_first_step(messages, needles) if needles else (False, None, [])
        return RunOutput(
            status=RunStatus.OK,
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
        return content if isinstance(content, str) else ""


def _tool_calls_of(message: Any) -> list[ToolCallRecord]:
    records = []
    for call in message.get("tool_calls") or []:
        # agentdojo's FunctionCall is a pydantic model; a no-argument call has args == {}
        records.append(ToolCallRecord(function=call.function, args=dict(call.args or {}), id=call.id))
    return records


def _extract_first_step(messages: list[Any], needles: tuple[str, ...]) -> tuple[bool, str | None, list[ToolCallRecord]]:
    """Whether the injection reached a tool result, and the first assistant turn after it."""
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        if _contains_injection(_message_text(message) or (message.get("error") or ""), needles):
            for later in messages[index + 1 :]:
                if later.get("role") == "assistant":
                    return True, _message_text(later), _tool_calls_of(later)
            return True, None, []  # read, but the agent emitted no further turn
    return False, None, []


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _yaml_strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for key, value in obj.items() for s in _yaml_strings(key) + _yaml_strings(value)]
    if isinstance(obj, (list, tuple)):
        return [s for item in obj for s in _yaml_strings(item)]
    return []


def _contains_injection(text: str, needles: tuple[str, ...]) -> bool:
    """Whether any needle occurs in a tool result.

    agentdojo renders tool results with ``yaml.safe_dump``, which line-wraps, quotes and
    escapes long scalars, so compare whitespace-normalized text, then fall back to the
    decoded YAML string scalars.
    """
    norm_needles = [_normalize_ws(n) for n in needles if n and n.strip()]
    if not norm_needles or not text:
        return False
    if any(n in _normalize_ws(text) for n in norm_needles):
        return True
    try:
        parsed = yaml.safe_load(text)
    except Exception:  # noqa: BLE001 - tool output need not be YAML
        return False
    return any(n in _normalize_ws(s) for s in _yaml_strings(parsed) for n in norm_needles)


def _hit_step_limit(messages: list[Any], max_iters: int) -> bool:
    """The loop ran max_iters times and the last assistant turn still requested tools."""
    if sum(1 for m in messages if m.get("role") == "assistant") < max_iters:
        return False
    last = messages[-1] if messages else None
    return bool(last and last.get("role") == "assistant" and last.get("tool_calls"))


def _messages_to_json(messages: list[Any]) -> list[dict[str, Any]]:
    out = []
    for index, message in enumerate(messages):
        turn: dict[str, Any] = {"message_id": f"m{index:04d}", "role": message.get("role"),
                                "content": _message_text(message)}
        if message.get("role") == "assistant":
            turn["tool_calls"] = [tc.as_dict() for tc in _tool_calls_of(message)]
        if message.get("role") == "tool":
            turn["error"] = message.get("error")
            call = message.get("tool_call")
            if call is not None:
                turn["tool_call"] = {"function": call.function, "args": dict(call.args or {}), "id": call.id}
            turn["tool_call_id"] = message.get("tool_call_id")
        out.append(turn)
    return out
