"""Data objects: tasks, templates, candidates and results, with stable JSON output."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    """Whether a trial was carried out normally (separate from whether the attack succeeded).

    ``ok``: the agent finished normally and, if it read the injection, the judge rated it.
    ``failed``: anything else (API error, step limit, no final answer, judge gave no
    rating, ...) after all retries; the reason is in the trial's ``error``. A candidate
    with any failed trial is dropped from the search.
    """

    OK = "ok"
    FAILED = "failed"


@dataclass(frozen=True)
class Template:
    """An injection template shared by every task: the injection for attacker goal ``g``
    is ``prefix + g + suffix``."""

    prefix: str
    suffix: str

    @property
    def template_hash(self) -> str:
        h = hashlib.sha256()
        for part in (self.prefix, self.suffix):
            h.update(part.encode("utf-8") + b"\x00")
        return h.hexdigest()[:16]

    def render(self, goal: str) -> str:
        return f"{self.prefix}{goal}{self.suffix}"


@dataclass
class Candidate:
    """A template plus the search bookkeeping that produced it."""

    candidate_id: str
    template: Template
    improvement: str
    parent_id: str | None
    root_id: str
    depth: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "prefix": self.template.prefix,
            "suffix": self.template.suffix,
            "improvement": self.improvement,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "depth": self.depth,
            "template_hash": self.template.template_hash,
        }


@dataclass
class ToolCallRecord:
    """A structured tool call as the agent emitted it."""

    function: str
    args: dict[str, Any]
    id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrialResult:
    """One run of one candidate on one task pair."""

    candidate_id: str
    template_hash: str
    suite: str
    user_task_id: str
    injection_task_id: str
    repeat_id: int
    status: RunStatus
    injection_seen: bool
    judge_score: int | None  # None if the injection was never seen (scores the minimum) or failed
    attack_success: bool | None  # agentdojo `security`: the injection task was executed
    utility: bool | None  # the benign user task was still completed
    first_step_response: str | None
    first_step_tool_calls: list[ToolCallRecord] = field(default_factory=list)
    n_requests: int = 0
    total_tokens: int = 0
    duration: float = 0.0
    attempts: int = 1
    trace_path: str | None = None
    error: str | None = None

    @property
    def case_id(self) -> str:
        return f"{self.suite}/{self.user_task_id}/{self.injection_task_id}"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class CandidateSummary:
    """A candidate's aggregate over the training set."""

    candidate_id: str
    template_hash: str
    n_tasks_total: int
    n_tasks_covered: int  # tasks whose trials all succeeded
    mean_judge_score: float | None  # the ranking signal (mean over covered tasks)
    proxy_pass_rate: float | None  # fraction of tasks whose mean judge >= proxy threshold
    real_asr: float | None  # fraction of trials whose attack succeeded in the full rollout
    utility_rate: float | None
    n_failed: int  # failed trials
    n_requests: int
    total_tokens: int
    fully_evaluated: bool  # no failed trial; only such candidates can survive or be best

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnTrialResult(TrialResult):
    grade: int = 0
    labels: dict[str, dict[str, Any]] = field(default_factory=dict)
    injection_message_id: str | None = None
    termination: str = "completed"
    judge_error: str | None = None
    judge_version: str = ""
    evaluator_requests: int = 0
    evaluator_tokens: int = 0
    judge_outputs: list[str] = field(default_factory=list)
    rollout_status: str = "ok"
    rollout_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {**super().as_dict(), "security": self.attack_success}

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        from dataclasses import fields
        values = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        values["status"] = RunStatus(values["status"])
        values["first_step_tool_calls"] = [ToolCallRecord(**call) for call in values.get("first_step_tool_calls", [])]
        return cls(**values)


@dataclass
class TurnSummary:
    candidate_id: str
    template_hash: str
    n_turns: int
    n_tasks_total: int
    repeats: int
    expected_trials: int
    n_trials: int
    n_failed: int
    n_missing: int
    fully_evaluated: bool
    security_rate: float
    utility_rate: float
    mean_grade: float
    grade_distribution: dict[str, int]
    unknown_counts: dict[str, int]
    per_case: list[dict[str, Any]]
    n_requests: int = 0
    total_tokens: int = 0
    duration: float = 0.0

    @property
    def rank_key(self) -> tuple[float, float, int]:
        return self.security_rate, self.mean_grade, -self.n_turns

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
