"""Task manifests: JSON files listing explicit (suite, user task, injection task) pairs.

The paper trains on 3 user tasks x 2 injection tasks in each of Workspace, Banking and
Slack (18 pairs) and holds out Travel; see ``data/universal/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import BENCHMARK_VERSION

Case = tuple[str, str, str]  # (suite, user_task_id, injection_task_id), as in structural_attack


@dataclass
class TaskManifest:
    name: str
    benchmark_version: str
    cases: list[Case]

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(sorted("/".join(c) for c in self.cases), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def corpus_overview(self) -> dict[str, object]:
        """What the attacker is told about the training corpus (never per-task test feedback)."""
        return {
            "n_task_pairs": len(self.cases),
            "suites": list(dict.fromkeys(suite for suite, _, _ in self.cases)),
            "n_user_tasks": len({(suite, user) for suite, user, _ in self.cases}),
            "n_injection_tasks": len({(suite, inj) for suite, _, inj in self.cases}),
        }

    def describe_goals(self) -> list[str]:
        """Each distinct injection task's GOAL, for the attacker's system prompt."""
        from agentdojo.task_suite.load_suites import get_suite

        goals: dict[str, str] = {}
        for suite_name, _, injection_task_id in self.cases:
            key = f"{suite_name}/{injection_task_id}"
            if key not in goals:
                suite = get_suite(self.benchmark_version, suite_name)
                goals[key] = suite.get_injection_task_by_id(injection_task_id).GOAL
        return [f"[{key}] {goal}" for key, goal in sorted(goals.items())]


def load_manifest(path: str | Path) -> TaskManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TaskManifest(
        name=data.get("name", Path(path).stem),
        benchmark_version=data.get("benchmark_version", BENCHMARK_VERSION),
        cases=[(c["suite"], c["user_task_id"], c["injection_task_id"]) for c in data["cases"]],
    )
