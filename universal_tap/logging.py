"""Local result, trace, checkpoint and cost logging (plan.md §8).

The first version uses local JSON/JSONL as the primary artifact (WandB optional and not
required). Every candidate, trial and summary is recorded — not just the final winner —
so a candidate's aggregate score can be recomputed from raw records.

Result identity (plan.md §8): each trace is keyed by run_id, template hash, task ID,
repeat_id and a config fingerprint (model, provider, system message, benchmark
revision, injection strategy, sampling). This search reruns the same task under many
templates, so the original runner's suite/user_task/attack skip condition must not be
reused as a completion check; a record counts as done only when its fields are complete
and its status is terminal.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import Candidate, CandidateSummary, TrialResult


def config_fingerprint(run_config: dict[str, Any]) -> str:
    payload = json.dumps(run_config, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class RunLogger:
    """Single-writer logger for one search run."""

    def __init__(self, run_dir: str | Path, run_config: dict[str, Any]) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir = self.run_dir / "traces"
        self.traces_dir.mkdir(exist_ok=True)
        self.run_config = run_config
        self.fingerprint = config_fingerprint(run_config)
        self.run_id = run_config.get("run_id") or time.strftime("%Y%m%d-%H%M%S")
        self._candidates = self.run_dir / "candidates.jsonl"
        self._trials = self.run_dir / "trials.jsonl"
        self._summaries = self.run_dir / "summaries.jsonl"
        self._cost = self.run_dir / "cost.jsonl"
        _atomic_write(
            self.run_dir / "run_config.json",
            json.dumps(
                {"run_id": self.run_id, "fingerprint": self.fingerprint, **run_config},
                ensure_ascii=False,
                indent=2,
            ),
        )

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_candidate(self, candidate: Candidate) -> None:
        self._append(self._candidates, {"run_id": self.run_id, **candidate.as_dict()})

    def log_trial(self, trial: TrialResult, trace: list[dict[str, Any]] | None = None) -> None:
        """Record a trial *before* any pruning, and its full trace (plan.md §8)."""
        trace_path = None
        if trace is not None:
            case = f"{trial.suite}__{trial.user_task_id}__{trial.injection_task_id}"
            cand_dir = self.traces_dir / trial.template_hash
            cand_dir.mkdir(parents=True, exist_ok=True)
            trace_file = cand_dir / f"{case}__r{trial.repeat_id}.json"
            _atomic_write(
                trace_file,
                json.dumps(
                    {
                        "run_id": self.run_id,
                        "fingerprint": self.fingerprint,
                        "candidate_id": trial.candidate_id,
                        "template_hash": trial.template_hash,
                        "case_id": f"{trial.suite}/{trial.user_task_id}/{trial.injection_task_id}",
                        "repeat_id": trial.repeat_id,
                        "seed": trial.seed,
                        "status": trial.status.value,
                        "messages": trace,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            trace_path = str(trace_file.relative_to(self.run_dir))
            trial.trace_path = trace_path
        record = {"run_id": self.run_id, "fingerprint": self.fingerprint, **trial.as_dict()}
        self._append(self._trials, record)

    def log_baseline(self, record: dict[str, Any], trace: list[dict[str, Any]] | None = None) -> None:
        """Record one no-injection baseline run (benign user task or standalone
        injection task) and its full trace (plan.md §9 phase B)."""
        if trace is not None:
            base_dir = self.traces_dir / "baselines"
            base_dir.mkdir(parents=True, exist_ok=True)
            name = f"{record['kind']}__{record['suite']}__{record['task_id']}__r{record['repeat_id']}.json"
            trace_file = base_dir / name
            _atomic_write(
                trace_file,
                json.dumps(
                    {"run_id": self.run_id, "fingerprint": self.fingerprint, **record, "messages": trace},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            record = {**record, "trace_path": str(trace_file.relative_to(self.run_dir))}
        self._append(self.run_dir / "baselines.jsonl", {"run_id": self.run_id, "fingerprint": self.fingerprint, **record})

    def log_summary(self, summary: CandidateSummary, depth: int) -> None:
        self._append(
            self._summaries,
            {"run_id": self.run_id, "depth": depth, **summary.as_dict()},
        )

    def log_cost(self, label: str, usage: dict[str, int]) -> None:
        self._append(self._cost, {"run_id": self.run_id, "ts": time.time(), "label": label, **usage})

    def save_checkpoint(self, state: dict[str, Any]) -> None:
        _atomic_write(
            self.run_dir / "checkpoint.json",
            json.dumps({"run_id": self.run_id, "fingerprint": self.fingerprint, **state}, ensure_ascii=False, indent=2),
        )

    def load_checkpoint(self) -> dict[str, Any] | None:
        path = self.run_dir / "checkpoint.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("fingerprint") != self.fingerprint:
            raise ValueError(
                f"checkpoint fingerprint {data.get('fingerprint')!r} != current "
                f"{self.fingerprint!r}; refusing to resume a run with a different config "
                "(model/provider/benchmark/injection strategy)."
            )
        return data

    def save_artifact(self, artifact_dict: dict[str, Any], name: str = "best_template.json") -> Path:
        path = self.run_dir / name
        _atomic_write(path, json.dumps(artifact_dict, ensure_ascii=False, indent=2))
        return path

    def load_done_trial_keys(self) -> set[tuple[str, str, int]]:
        """Keys (template_hash, case_id, repeat_id) of trials already completed with a
        terminal status, for resume. A trial is done only if its JSON parses and carries
        a terminal status field."""
        done: set[tuple[str, str, int]] = set()
        if not self._trials.exists():
            return done
        for line in self._trials.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "status" not in rec:
                continue
            case_id = f"{rec['suite']}/{rec['user_task_id']}/{rec['injection_task_id']}"
            done.add((rec["template_hash"], case_id, rec["repeat_id"]))
        return done
