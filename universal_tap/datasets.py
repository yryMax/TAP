"""Task-manifest loading and split bookkeeping (plan.md §4, §10).

A manifest is a JSON file listing explicit ``(suite, user_task, injection_task)``
pairs for a split (train / validation / test). Manifests are explicit rather than
"any 18 tasks": the paper's training set is 3 user tasks x 2 injection tasks in each of
Workspace, Banking and Slack (18 pairs), with Travel held out. The concrete task IDs
must be verified against the pinned framework and written here; this module does not
invent them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .schemas import TaskCase


@dataclass
class TaskManifest:
    name: str
    benchmark_version: str
    cases: list[TaskCase]

    @property
    def suites(self) -> list[str]:
        seen: list[str] = []
        for case in self.cases:
            if case.suite not in seen:
                seen.append(case.suite)
        return seen

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(
            [c.case_id for c in sorted(self.cases, key=lambda c: c.case_id)],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def corpus_overview(self) -> dict[str, object]:
        """A compact description of the corpus for the attacker (plan.md §6): the
        diversity of goals, suites and tool sets — but never per-task test feedback."""
        user_tasks = sorted({f"{c.suite}/{c.user_task_id}" for c in self.cases})
        injection_tasks = sorted({f"{c.suite}/{c.injection_task_id}" for c in self.cases})
        return {
            "n_task_pairs": len(self.cases),
            "suites": self.suites,
            "n_user_tasks": len(user_tasks),
            "n_injection_tasks": len(injection_tasks),
        }

    def describe_goals(self, benchmark_version: str | None = None) -> list[str]:
        """Resolve each distinct injection task's GOAL text, for the attacker's corpus
        overview. Loaded lazily so importing this module needs no suites."""
        from agentdojo.task_suite.load_suites import get_suite

        version = benchmark_version or self.benchmark_version
        goals: dict[str, str] = {}
        for case in self.cases:
            key = f"{case.suite}/{case.injection_task_id}"
            if key in goals:
                continue
            suite = get_suite(version, case.suite)
            goals[key] = suite.get_injection_task_by_id(case.injection_task_id).GOAL
        return [f"[{k}] {v}" for k, v in sorted(goals.items())]


def load_manifest(path: str | Path) -> TaskManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    benchmark_version = data.get("benchmark_version", "v1.2.2")
    cases = []
    for raw in data["cases"]:
        raw.setdefault("benchmark_version", benchmark_version)
        cases.append(TaskCase.from_dict(raw))
    return TaskManifest(
        name=data.get("name", Path(path).stem),
        benchmark_version=benchmark_version,
        cases=cases,
    )


def assert_disjoint(train: TaskManifest, held_out: TaskManifest) -> None:
    """Guard that a held-out split shares no task pair with training (plan.md §10).

    Note: non-overlapping *pairs* alone does not prove unseen user tasks, goals or
    domains; the caller should additionally track which components overlap.
    """
    train_ids = {c.case_id for c in train.cases}
    clash = sorted(train_ids & {c.case_id for c in held_out.cases})
    if clash:
        raise ValueError(f"held-out split {held_out.name!r} overlaps training pairs: {clash}")


def split_overlap_report(train: TaskManifest, held_out: TaskManifest) -> dict[str, object]:
    """Describe how a held-out split relates to training: which user tasks, injection
    goals and suites are genuinely unseen vs. merely recombined (plan.md §10)."""
    def user_keys(m: TaskManifest) -> set[str]:
        return {f"{c.suite}/{c.user_task_id}" for c in m.cases}

    def inj_keys(m: TaskManifest) -> set[str]:
        return {f"{c.suite}/{c.injection_task_id}" for c in m.cases}

    return {
        "held_out": held_out.name,
        "shares_pairs": sorted({c.case_id for c in train.cases} & {c.case_id for c in held_out.cases}),
        "unseen_user_tasks": sorted(user_keys(held_out) - user_keys(train)),
        "unseen_injection_goals": sorted(inj_keys(held_out) - inj_keys(train)),
        "unseen_suites": sorted(set(held_out.suites) - set(train.suites)),
    }
