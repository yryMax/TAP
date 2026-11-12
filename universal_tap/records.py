"""Everything a run writes into its run directory.

* ``run_config.json`` — models, manifest, parameters, plus a fingerprint of them;
* ``log.txt`` — everything printed to the terminal (and any traceback);
* ``candidates.jsonl`` / ``trials.jsonl`` / ``summaries.jsonl`` — every candidate, every
  trial (recorded before pruning) and every candidate summary, so any score can be
  recomputed from raw records;
* ``baselines.jsonl`` — no-injection baseline runs;
* ``traces/`` — the full message trace of each trial;
* ``best_template.json`` — the frozen template a search selected.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .schemas import Candidate, CandidateSummary, TrialResult


def _atomic_write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class _Tee:
    """Write a stream to the terminal and to a log file at the same time."""

    def __init__(self, stream: Any, log_file: Any) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._log_file.write(text)  # log first: a broken terminal never costs a log line
        self._log_file.flush()
        self._stream.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class RunLogger:
    """Single writer for one run directory. Creating it also mirrors stdout/stderr into
    ``log.txt``."""

    def __init__(self, run_dir: str | Path, run_config: dict[str, Any], *, resume: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.traces_dir = self.run_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        payload = json.dumps(run_config, ensure_ascii=False, sort_keys=True)
        self.fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        self.run_config = run_config
        if resume:
            saved = json.loads((self.run_dir / "run_config.json").read_text(encoding="utf-8"))
            if saved["fingerprint"] != self.fingerprint:
                raise ValueError("resume config differs from the saved run (models, data, renderer or search parameters)")
            self.run_id = saved["run_id"]
        else:
            _atomic_write(
                self.run_dir / "run_config.json",
                {"run_id": self.run_id, "fingerprint": self.fingerprint, **run_config},
            )

        log_file = (self.run_dir / "log.txt").open("a", encoding="utf-8")
        self._log_file = log_file
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(sys.stdout, log_file)
        sys.stderr = _Tee(sys.stderr, log_file)
        print(f"run dir: {self.run_dir}  (output also saved to {self.run_dir / 'log.txt'})")

    def close(self) -> None:
        if isinstance(sys.stdout, _Tee) and sys.stdout._log_file is self._log_file:
            sys.stdout = self._stdout
        if isinstance(sys.stderr, _Tee) and sys.stderr._log_file is self._log_file:
            sys.stderr = self._stderr
        self._log_file.close()

    def _append(self, name: str, record: dict[str, Any]) -> None:
        with (self.run_dir / name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run_id": self.run_id, **record}, ensure_ascii=False) + "\n")

    def _write_trace(self, path: Path, record: dict[str, Any], messages: list[dict[str, Any]]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, {"run_id": self.run_id, "fingerprint": self.fingerprint, **record, "messages": messages})
        return str(path.relative_to(self.run_dir))

    def log_candidate(self, candidate: Candidate) -> None:
        self._append("candidates.jsonl", candidate.as_dict())

    def log_trial(self, trial: TrialResult, trace: list[dict[str, Any]]) -> None:
        name = f"{trial.suite}__{trial.user_task_id}__{trial.injection_task_id}__r{trial.repeat_id}.json"
        trial.trace_path = self._write_trace(
            self.traces_dir / trial.template_hash / name,
            {"candidate_id": trial.candidate_id, "template_hash": trial.template_hash,
             "case_id": trial.case_id, "repeat_id": trial.repeat_id, "status": trial.status.value},
            trace,
        )
        extra = {"messages": trace} if hasattr(trial, "grade") else {}
        self._append("trials.jsonl", {"fingerprint": self.fingerprint, **trial.as_dict(), **extra})

    def log_invalid_candidate(self, record: dict[str, Any]) -> None:
        self._append("candidates.jsonl", record)

    def log_mutation(self, record: dict[str, Any]) -> None:
        self._append("mutations.jsonl", record)

    def log_baseline(self, record: dict[str, Any], trace: list[dict[str, Any]]) -> None:
        name = f"{record['kind']}__{record['suite']}__{record['task_id']}__r{record['repeat_id']}.json"
        trace_path = self._write_trace(self.traces_dir / "baselines" / name, record, trace)
        self._append("baselines.jsonl", {"fingerprint": self.fingerprint, **record, "trace_path": trace_path})

    def log_summary(self, summary: CandidateSummary, depth: int) -> None:
        self._append("summaries.jsonl", {"depth": depth, **summary.as_dict()})

    def save_artifact(self, artifact: dict[str, Any], name: str = "best_template.json") -> Path:
        path = self.run_dir / name
        _atomic_write(path, artifact)
        return path
