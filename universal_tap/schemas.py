"""Data objects for Universal TAP: tasks, templates, candidates and results.

These are plain dataclasses with explicit JSON (de)serialization so the local
JSON/JSONL logs are stable and re-readable (plan.md §4, §8). AgentDojo task objects
are loaded lazily from IDs and never serialized.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    """Outcome of a single agent run (distinct from attack success)."""

    OK = "ok"  # agent ran to a natural stop with a model output
    STEP_LIMIT = "step_limit"  # hit the tool-execution loop's max_iters: executed but unfinished
    CONTEXT_LENGTH = "context_length"  # provider refused for context length
    API_ERROR = "api_error"  # network/auth/provider failure during the run
    ENV_ERROR = "env_error"  # suite/environment error (e.g. injection rendering)
    INCOMPLETE = "incomplete"  # produced no usable model output


class JudgeStatus(str, Enum):
    OK = "ok"
    PARSE_ERROR = "parse_error"  # judge output could not be parsed into 1-10
    SKIPPED = "skipped"  # not scored (e.g. run failed, or injection not seen)
    API_ERROR = "api_error"


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Template:
    """A frozen injection template shared across all tasks.

    The rendered injection for a task with attacker goal ``g`` is
    ``prefix + separator + g + separator + suffix``. The separator is a fixed config,
    not searched. ``prefix``/``suffix`` are the search variables; they are identical
    for every training task.
    """

    prefix: str
    suffix: str
    separator: str = ""

    @property
    def template_hash(self) -> str:
        return _hash(self.prefix, self.separator, self.suffix)

    def render(self, goal: str) -> str:
        return f"{self.prefix}{self.separator}{goal}{self.separator}{self.suffix}"

    def as_dict(self) -> dict[str, str]:
        return {"prefix": self.prefix, "suffix": self.suffix, "separator": self.separator}


@dataclass(frozen=True)
class TaskCase:
    """A single AgentDojo (user_task, injection_task) pair in a suite.

    ``injection_strategy`` names the fixed, reproducible rule for where the injection
    goes; the first version uses ``"read_vectors"`` — render into every injection
    vector that the ground-truth execution actually reads.
    """

    suite: str
    user_task_id: str
    injection_task_id: str
    benchmark_version: str = "v1.2.2"
    injection_strategy: str = "read_vectors"

    @property
    def case_id(self) -> str:
        return f"{self.suite}/{self.user_task_id}/{self.injection_task_id}"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskCase":
        return cls(
            suite=data["suite"],
            user_task_id=data["user_task_id"],
            injection_task_id=data["injection_task_id"],
            benchmark_version=data.get("benchmark_version", "v1.2.2"),
            injection_strategy=data.get("injection_strategy", "read_vectors"),
        )


@dataclass
class Candidate:
    """A template plus the search bookkeeping that produced it."""

    candidate_id: str
    prefix: str
    suffix: str
    separator: str
    improvement: str
    parent_id: str | None
    root_id: str
    depth: int

    @property
    def template(self) -> Template:
        return Template(self.prefix, self.suffix, self.separator)

    @property
    def template_hash(self) -> str:
        return self.template.template_hash

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["template_hash"] = self.template_hash
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            candidate_id=data["candidate_id"],
            prefix=data["prefix"],
            suffix=data["suffix"],
            separator=data.get("separator", ""),
            improvement=data.get("improvement", ""),
            parent_id=data.get("parent_id"),
            root_id=data["root_id"],
            depth=data["depth"],
        )


@dataclass
class ToolCallRecord:
    """A structured tool call as the agent emitted it (name + arguments)."""

    function: str
    args: dict[str, Any]
    id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"function": self.function, "args": self.args, "id": self.id}


@dataclass
class TrialResult:
    """One evaluation of one candidate on one task pair, for one repeat/seed."""

    candidate_id: str
    template_hash: str
    suite: str
    user_task_id: str
    injection_task_id: str
    repeat_id: int
    seed: int
    status: RunStatus
    injection_seen: bool
    judge_status: JudgeStatus
    judge_score: int | None
    attack_success: bool | None  # agentdojo `security`: injection task executed
    utility: bool | None  # benign user task still completed
    first_step_response: str | None
    first_step_tool_calls: list[ToolCallRecord] = field(default_factory=list)
    n_requests: int = 0
    total_tokens: int = 0
    duration: float = 0.0
    trace_path: str | None = None
    error: str | None = None

    @property
    def is_valid_for_scoring(self) -> bool:
        """A trial contributes to a candidate's judge mean only if the run itself did
        not hit an infrastructure error. A clean run that never surfaced the injection
        is valid and scores the configured minimum (handled by the evaluator)."""
        return self.status in (RunStatus.OK, RunStatus.STEP_LIMIT, RunStatus.CONTEXT_LENGTH)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["judge_status"] = self.judge_status.value
        d["first_step_tool_calls"] = [tc.as_dict() for tc in self.first_step_tool_calls]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialResult":
        return cls(
            candidate_id=data["candidate_id"],
            template_hash=data["template_hash"],
            suite=data["suite"],
            user_task_id=data["user_task_id"],
            injection_task_id=data["injection_task_id"],
            repeat_id=data["repeat_id"],
            seed=data["seed"],
            status=RunStatus(data["status"]),
            injection_seen=data["injection_seen"],
            judge_status=JudgeStatus(data["judge_status"]),
            judge_score=data.get("judge_score"),
            attack_success=data.get("attack_success"),
            utility=data.get("utility"),
            first_step_response=data.get("first_step_response"),
            first_step_tool_calls=[
                ToolCallRecord(**tc) for tc in data.get("first_step_tool_calls", [])
            ],
            n_requests=data.get("n_requests", 0),
            total_tokens=data.get("total_tokens", 0),
            duration=data.get("duration", 0.0),
            trace_path=data.get("trace_path"),
            error=data.get("error"),
        )


@dataclass
class CandidateSummary:
    """Aggregate of a candidate across the training set (plan.md §4, §5)."""

    candidate_id: str
    template_hash: str
    n_tasks_total: int
    n_tasks_covered: int  # tasks with >=1 valid trial
    mean_judge_score: float | None  # paper's ranking signal; None if nothing scorable
    proxy_pass_rate: float | None  # fraction of tasks whose mean judge >= proxy threshold
    real_asr: float | None  # fraction of task pairs with attack_success in full rollout
    utility_rate: float | None
    n_errors: int
    n_requests: int
    total_tokens: int
    fully_evaluated: bool  # every task had >=1 valid trial

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TemplateArtifact:
    """The frozen deliverable: a selected template plus provenance (plan.md §4, §8)."""

    prefix: str
    suffix: str
    separator: str
    render_rule: str
    train_manifest_hash: str
    model_config_fingerprint: dict[str, Any]
    run_config: dict[str, Any]
    selection_reason: str
    candidate_id: str
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "TemplateArtifact":
        with open(path, encoding="utf-8") as fh:
            return cls(**json.load(fh))
