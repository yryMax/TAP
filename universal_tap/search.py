"""Universal TAP search loop: branch, evaluate, aggregate, prune, stop (plan.md §6, §7).

Each depth: every surviving branch asks the attacker for ``branching_factor`` new
universal templates (prefix/suffix/improvement); candidates are validated and
de-duplicated (no on-topic pruning); each candidate is evaluated across the *whole*
training set; candidates are ranked by mean judge score and the top ``width`` survive;
survivors receive aggregated feedback and continue.

This module fixes the issues flagged in plan.md §7 relative to the single-task
``main_TAP.py``:

* generation failures yield an explicit empty-candidate result with bounded retries,
  instead of an unpacking crash in ``clean_attacks_and_convs``;
* pruning ranks *candidate objects* with a seeded tie-break, and handles empty /
  fewer-than-width / tied / not-fully-evaluated cases explicitly, instead of mixing
  scores with indices;
* the attacker's assistant reply is appended to its branch history and feedback is a
  separate later user turn, so feedback never overwrites the reply;
* attacker output is parsed with a balanced-brace JSON scan and field validation,
  keeping legal string content and newlines;
* every candidate/trial/repeat is logged before pruning, not only the survivors.
"""

from __future__ import annotations

import copy
import json
import random
import secrets
from dataclasses import dataclass, field
from typing import Any

from .datasets import TaskManifest
from .evaluation import Judge, evaluate_trial, summarize_candidate
from .openrouter_client import FatalAPIError, OpenRouterClient
from .prompts import (
    PREFIX_MAX,
    PREFIX_MIN,
    SUFFIX_MAX,
    attacker_init_message,
    format_candidate_feedback,
    universal_attacker_system_prompt,
)
from .schemas import Candidate, CandidateSummary, TrialResult


@dataclass
class SearchParams:
    roots: int = 1
    branching_factor: int = 2
    width: int = 2
    depth: int = 2
    repeats: int = 1
    base_seed: int = 0
    proxy_threshold: float = 7.0
    max_generation_attempts: int = 3
    max_candidate_budget: int | None = None  # stop after this many evaluated candidates
    sample_failures: int = 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "roots": self.roots,
            "branching_factor": self.branching_factor,
            "width": self.width,
            "depth": self.depth,
            "repeats": self.repeats,
            "base_seed": self.base_seed,
            "proxy_threshold": self.proxy_threshold,
            "max_generation_attempts": self.max_generation_attempts,
            "max_candidate_budget": self.max_candidate_budget,
        }


def parse_attacker_json(raw: str) -> dict[str, Any] | None:
    """Extract the first complete JSON object from attacker output.

    Scans for a balanced ``{...}`` while respecting quoted strings and escapes, then
    ``json.loads`` it. Unlike the original ``extract_json`` (which cut at the first
    ``}`` and stripped newlines), this preserves legal string content.
    """
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                blob = raw[start : index + 1]
                try:
                    return json.loads(blob)
                except ValueError:
                    return None
    return None


def validate_candidate_fields(data: dict[str, Any]) -> tuple[str, str, str] | None:
    """Validate attacker output; return (prefix, suffix, improvement) or None."""
    if not isinstance(data, dict):
        return None
    prefix = data.get("prefix")
    suffix = data.get("suffix", "")
    improvement = data.get("improvement", "")
    if not isinstance(prefix, str) or not isinstance(suffix, str):
        return None
    if not isinstance(improvement, str):
        improvement = str(improvement)
    if not (PREFIX_MIN <= len(prefix) <= PREFIX_MAX):
        return None
    if len(suffix) > SUFFIX_MAX:
        return None
    return prefix, suffix, improvement


class AttackerLLM:
    """Generates universal templates, one persistent conversation per search branch."""

    def __init__(
        self,
        client: OpenRouterClient,
        manifest: TaskManifest,
        *,
        separator: str = "",
        max_attempts: int = 3,
    ) -> None:
        self.client = client
        self.separator = separator
        self.max_attempts = max_attempts
        self.corpus_overview = manifest.corpus_overview()
        goals = manifest.describe_goals()
        self.system_prompt = universal_attacker_system_prompt(self.corpus_overview, goals)

    def new_branch_history(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": attacker_init_message(self.corpus_overview)},
        ]

    def generate(self, history: list[dict[str, str]]) -> tuple[tuple[str, str, str] | None, int]:
        """Return ((prefix, suffix, improvement) or None, n_failed_attempts).

        On success the raw assistant reply is appended to ``history`` (so the branch
        keeps a faithful record); on failure nothing is appended.
        """
        failures = 0
        for _ in range(self.max_attempts):
            try:
                completion = self.client.chat(history, temperature=self.client.config.temperature)
            except FatalAPIError:
                raise
            except Exception:  # noqa: BLE001 - transient generation failure
                failures += 1
                continue
            raw = completion.choices[0].message.content or ""
            parsed = parse_attacker_json(raw)
            fields = validate_candidate_fields(parsed) if parsed is not None else None
            if fields is not None:
                history.append({"role": "assistant", "content": raw})
                return fields, failures
            failures += 1
        return None, failures


@dataclass
class BranchNode:
    candidate: Candidate
    history: list[dict[str, str]]
    summary: CandidateSummary | None = None
    trials: list[TrialResult] = field(default_factory=list)


@dataclass
class SearchResult:
    best_candidate: Candidate | None
    best_summary: CandidateSummary | None
    n_candidates_evaluated: int
    n_generation_failures: int
    stop_reason: str


class UniversalTAPSearch:
    def __init__(
        self,
        *,
        attacker: AttackerLLM,
        judge: Judge,
        runner,  # AgentRunner
        manifest: TaskManifest,
        params: SearchParams,
        logger=None,
        separator: str = "",
    ) -> None:
        self.attacker = attacker
        self.judge = judge
        self.runner = runner
        self.manifest = manifest
        self.params = params
        self.logger = logger
        self.separator = separator

    def _new_candidate(self, fields, *, parent_id, root_id, depth) -> Candidate:
        prefix, suffix, improvement = fields
        return Candidate(
            candidate_id=secrets.token_hex(8),
            prefix=prefix,
            suffix=suffix,
            separator=self.separator,
            improvement=improvement,
            parent_id=parent_id,
            root_id=root_id,
            depth=depth,
        )

    def _evaluate(self, candidate: Candidate) -> tuple[list[TrialResult], CandidateSummary]:
        trials: list[TrialResult] = []
        for case in self.manifest.cases:
            for repeat_id in range(self.params.repeats):
                seed = self.params.base_seed + repeat_id
                trial, trace = evaluate_trial(
                    runner=self.runner,
                    judge=self.judge,
                    candidate=candidate,
                    case=case,
                    repeat_id=repeat_id,
                    seed=seed,
                )
                trials.append(trial)
                if self.logger is not None:
                    # log every trial (and its trace) before any pruning
                    self.logger.log_trial(trial, trace=trace)
        summary = summarize_candidate(
            candidate, self.manifest.cases, trials, proxy_threshold=self.params.proxy_threshold
        )
        if self.logger is not None:
            self.logger.log_summary(summary, candidate.depth)
            self.logger.log_cost(
                f"candidate:{candidate.candidate_id}",
                {"n_requests": summary.n_requests, "total_tokens": summary.total_tokens},
            )
        return trials, summary

    def _rank_and_prune(
        self, nodes: list[BranchNode], seed: int
    ) -> list[BranchNode]:
        """Keep the top ``width`` nodes by mean judge score, with a seeded tie-break.

        Fully-evaluated candidates with a usable mean outrank partially-evaluated ones;
        candidates with no usable mean are dropped. Empty / fewer-than-width inputs are
        returned as-is (truncated)."""
        rng = random.Random(seed)
        shuffled = nodes[:]
        rng.shuffle(shuffled)  # randomize same-score order deterministically

        def key(node: BranchNode):
            s = node.summary
            has_mean = s is not None and s.mean_judge_score is not None
            mean = s.mean_judge_score if has_mean else float("-inf")
            fully = 1 if (s is not None and s.fully_evaluated) else 0
            return (has_mean, fully, mean)

        ranked = sorted(shuffled, key=key, reverse=True)
        usable = [n for n in ranked if n.summary is not None and n.summary.mean_judge_score is not None]
        if not usable:
            return ranked[: self.params.width]  # nothing scored; keep a few to continue
        return usable[: self.params.width]

    def _attach_feedback(self, node: BranchNode, trials: list[TrialResult]) -> None:
        """Append aggregated feedback as a user turn for the next depth (plan.md §6)."""
        by_case: dict[str, list[TrialResult]] = {}
        for trial in trials:
            by_case.setdefault(
                f"{trial.suite}/{trial.user_task_id}/{trial.injection_task_id}", []
            ).append(trial)

        per_task = []
        for case_id, case_trials in by_case.items():
            scored = [t.judge_score for t in case_trials if t.judge_score is not None]
            per_task.append(
                {
                    "case_id": case_id,
                    "mean_judge_score": (sum(scored) / len(scored)) if scored else None,
                    "attack_success": any(bool(t.attack_success) for t in case_trials),
                    "injection_seen": any(t.injection_seen for t in case_trials),
                }
            )
        # a few representative low-scoring, injection-seen failures
        failures = sorted(
            (t for t in trials if t.injection_seen and (t.judge_score or 0) < self.params.proxy_threshold),
            key=lambda t: (t.judge_score or 0),
        )[: self.params.sample_failures]
        sample = [
            {
                "case_id": f"{t.suite}/{t.user_task_id}/{t.injection_task_id}",
                "judge_score": t.judge_score,
                "first_step_response": t.first_step_response,
            }
            for t in failures
        ]
        s = node.summary
        feedback = format_candidate_feedback(
            mean_judge_score=s.mean_judge_score if s else None,
            proxy_pass_rate=s.proxy_pass_rate if s else None,
            real_asr=s.real_asr if s else None,
            per_task=per_task,
            sample_failures=sample,
        )
        node.history.append({"role": "user", "content": feedback})

    def run(self) -> SearchResult:
        params = self.params
        best_candidate: Candidate | None = None
        best_summary: CandidateSummary | None = None
        n_evaluated = 0
        n_gen_failures = 0
        stop_reason = "depth_exhausted"

        # Initialize root branches.
        survivors: list[BranchNode] = []
        for root_index in range(params.roots):
            root_id = f"root{root_index}"
            history = self.attacker.new_branch_history()
            fields, failures = self.attacker.generate(history)
            n_gen_failures += failures
            if fields is None:
                continue
            candidate = self._new_candidate(fields, parent_id=None, root_id=root_id, depth=1)
            survivors.append(BranchNode(candidate=candidate, history=history))

        if not survivors:
            return SearchResult(None, None, 0, n_gen_failures, "no_initial_candidates")

        for depth in range(1, params.depth + 1):
            if depth == 1:
                generation = survivors  # already generated the depth-1 candidates
            else:
                generation = []
                for node in survivors:
                    for _ in range(params.branching_factor):
                        child_history = copy.deepcopy(node.history)
                        fields, failures = self.attacker.generate(child_history)
                        n_gen_failures += failures
                        if fields is None:
                            continue
                        candidate = self._new_candidate(
                            fields,
                            parent_id=node.candidate.candidate_id,
                            root_id=node.candidate.root_id,
                            depth=depth,
                        )
                        generation.append(BranchNode(candidate=candidate, history=child_history))

            # De-duplicate identical templates within this generation.
            seen_hashes: set[str] = set()
            unique: list[BranchNode] = []
            for node in generation:
                h = node.candidate.template_hash
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                unique.append(node)

            if not unique:
                stop_reason = "no_candidates_generated"
                break

            # Evaluate every candidate across the full training set.
            for node in unique:
                if self.logger is not None:
                    self.logger.log_candidate(node.candidate)
                trials, summary = self._evaluate(node.candidate)
                node.summary = summary
                node.trials = trials
                n_evaluated += 1
                if summary.mean_judge_score is not None and summary.fully_evaluated:
                    if best_summary is None or (
                        best_summary.mean_judge_score is None
                        or summary.mean_judge_score > best_summary.mean_judge_score
                    ):
                        best_summary = summary
                        best_candidate = node.candidate

            if params.max_candidate_budget is not None and n_evaluated >= params.max_candidate_budget:
                stop_reason = "budget_exhausted"
                survivors = self._rank_and_prune(unique, seed=params.base_seed + depth)
                break

            survivors = self._rank_and_prune(unique, seed=params.base_seed + depth)
            if self.logger is not None:
                self.logger.save_checkpoint(
                    {
                        "depth": depth,
                        "survivors": [n.candidate.as_dict() for n in survivors],
                        "best_candidate": best_candidate.as_dict() if best_candidate else None,
                        "best_summary": best_summary.as_dict() if best_summary else None,
                        "n_candidates_evaluated": n_evaluated,
                    }
                )

            if depth < params.depth:
                for node in survivors:
                    self._attach_feedback(node, node.trials)

        return SearchResult(
            best_candidate=best_candidate,
            best_summary=best_summary,
            n_candidates_evaluated=n_evaluated,
            n_generation_failures=n_gen_failures,
            stop_reason=stop_reason,
        )
