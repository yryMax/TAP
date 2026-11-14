#!/usr/bin/env python3
"""Universal TAP for indirect prompt injection in AgentDojo.

Subcommands:

* ``smoke``      — one tool-call round trip against the target model (connectivity check);
* ``baselines``  — no-injection baselines: each user task on its own, and each attacker
                   goal given directly as the user prompt;
* ``eval``       — evaluate ONE fixed template on a task manifest (no search). Give it
                   ``--template best_template.json``, or ``--prefix/--suffix``; with
                   neither, the bare attacker goal is injected;
* ``search``     — the universal TAP search; saves the best template.

Usage, from this directory::

    source ../structural_attack/.venv/bin/activate
    python main_universal_tap.py search                                  # 3-task smoke search
    python main_universal_tap.py search --config configs/train18.json    # paper's 18-task training
    python main_universal_tap.py eval --manifest data/universal/test.json \\
        --template runs/universal/search-<time>/best_template.json

Output goes to the terminal and to ``<run dir>/log.txt``; the run dir defaults to a new
``runs/universal/<command>-<time>`` for prefix/suffix mode, or
``runs/turn_search/<command>-<time>`` for turn-level mode.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from universal_tap import AGENTDOJO_REVISION
from universal_tap.agent import AgentRunner
from universal_tap.datasets import TaskManifest, load_manifest
from universal_tap.evaluation import Judge, evaluate_trial, summarize_candidate
from universal_tap.llm import LLMClient, ModelConfig, build_clients
from universal_tap.records import RunLogger
from universal_tap.schemas import Candidate, Template

ROLES = ("attacker", "target", "judge")


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads(Path(args.config).read_text(encoding="utf-8"))


def model_configs(config: dict[str, Any]) -> dict[str, ModelConfig]:
    """Per-role settings layered over the shared ``llm`` block."""
    shared = config.get("llm", {})
    roles = {role: config.get(role, {}) for role in ROLES}
    roles["mutator"] = config.get("mutator", roles["attacker"])
    roles["evaluator"] = config.get("evaluator", roles["judge"])
    return {role: ModelConfig(**{**shared, **settings}) for role, settings in roles.items()}


def make_clients(args: argparse.Namespace, config: dict[str, Any], roles: tuple[str, ...]) -> dict[str, LLMClient]:
    models = model_configs(config)
    return build_clients({role: models[role] for role in roles}, args.env_file or config.get("env_file"))


@dataclass
class Run:
    config: dict[str, Any]
    clients: dict[str, LLMClient]
    manifest: TaskManifest
    runner: AgentRunner
    logger: RunLogger


def start_run(args: argparse.Namespace, config: dict[str, Any], roles: tuple[str, ...],
              *, task_manifest: TaskManifest | None = None, **extra: Any) -> Run:
    """Clients, manifest, agent runner and run directory, shared by baselines/eval/search."""
    clients = make_clients(args, config, roles)
    manifest = task_manifest or load_manifest(args.manifest or config["manifest"])
    family = "turn_search" if args.command.startswith("turn-") else "universal"
    run_dir = args.run_dir or f"runs/{family}/{args.command}-{time.strftime('%Y%m%d-%H%M%S')}"
    logger = RunLogger(run_dir, {
        "mode": args.command,
        "benchmark_version": manifest.benchmark_version,
        "agentdojo_revision": AGENTDOJO_REVISION,
        "models": {role: client.config.as_dict() for role, client in clients.items()},
        "manifest": manifest.name,
        "manifest_hash": manifest.manifest_hash,
        "cases": [list(case) for case in manifest.cases],
        "max_tool_iters": config.get("max_tool_iters", 15),
        "system_message_name": config.get("system_message_name"),
        "injection_strategy": "read_vectors",
        **extra,
    }, resume=getattr(args, "resume", False))
    runner = AgentRunner(
        clients["target"],
        benchmark_version=manifest.benchmark_version,
        max_tool_iters=config.get("max_tool_iters", 15),
        system_message_name=config.get("system_message_name"),
    )
    return Run(config, clients, manifest, runner, logger)


def concurrency(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """How many rollouts / model requests run at once (``--concurrency`` or the config)."""
    value = args.concurrency if args.concurrency is not None else config.get("concurrency", 1)
    if value < 1:
        raise ValueError("concurrency must be >= 1")
    return value


def search_params(args: argparse.Namespace, config: dict[str, Any]):
    from universal_tap.search import SearchParams

    overrides = {key: getattr(args, key, None) for key in ("roots", "branching_factor", "width", "depth", "repeats")}
    return SearchParams(**{**config.get("search", {}), **{k: v for k, v in overrides.items() if v is not None}})


def cmd_smoke(args: argparse.Namespace) -> int:
    client = make_clients(args, load_config(args), ("target",))["target"]
    print(f"target model: {client.config.model}  (provider={client.config.provider}, base_url={client.base_url})")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }]
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": "What is the weather in Zurich? Call the tool."},
    ]
    choice = client.chat(messages, tools=tools).choices[0]
    tool_calls = choice.message.tool_calls or []
    print(f"finish_reason={choice.finish_reason}  tool_calls={len(tool_calls)}")
    for call in tool_calls:
        print(f"  -> {call.function.name}({call.function.arguments})  id={call.id}")
    print(f"usage: {client.usage.as_dict()}")
    if not tool_calls:
        print("WARNING: no tool call returned; check the server's tool-calling support.")
        return 1
    print("OK: connectivity, model and structured tool calling verified.")
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    run = start_run(args, load_config(args), ("target",), repeats=args.repeats, injection_strategy="none")
    target = run.clients["target"]
    jobs: list[tuple[str, str, str]] = []  # (kind, suite, task_id), each distinct task once
    for suite_name, user_task_id, injection_task_id in run.manifest.cases:
        for job in (("benign", suite_name, user_task_id), ("standalone", suite_name, injection_task_id)):
            if job not in jobs:
                jobs.append(job)

    results: dict[str, list[bool]] = {"benign": [], "standalone": []}
    n_failed = 0
    for kind, suite_name, task_id in jobs:
        for repeat_id in range(args.repeats):
            start = time.time()
            with target.track() as usage:
                out = run.runner.run_baseline(suite_name=suite_name, kind=kind, task_id=task_id)
            run.logger.log_baseline({
                "kind": kind, "suite": suite_name, "task_id": task_id, "repeat_id": repeat_id,
                "status": out.status.value, "success": out.utility, "error": out.error,
                "n_requests": usage.n_requests, "total_tokens": usage.total_tokens,
                "duration": time.time() - start,
            }, out.messages)
            if out.status.value == "failed":
                n_failed += 1
            else:
                results[kind].append(out.utility)
            failure = f"  ({out.error})" if out.error else ""
            print(f"{kind:10s} {suite_name}/{task_id} r{repeat_id}: status={out.status.value} "
                  f"success={out.utility}{failure}")

    summary = {kind: {"n": len(v), "success_rate": sum(v) / len(v) if v else None} for kind, v in results.items()}
    print("\n=== baselines ===")
    print(json.dumps({**summary, "n_failed": n_failed, "usage": target.usage.as_dict()}, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    config = load_config(args)
    if args.template:
        frozen = json.loads(Path(args.template).read_text(encoding="utf-8"))
        template = Template(frozen["prefix"], frozen["suffix"])
    else:
        template = Template(args.prefix, args.suffix)
    run = start_run(args, config, ("target", "judge"), repeats=args.repeats,
                    template_source=args.template, template_hash=template.template_hash)
    judge = Judge(run.clients["judge"])
    candidate = Candidate("fixed", template, "(fixed template, no search)", parent_id=None, root_id="fixed", depth=0)
    run.logger.log_candidate(candidate)

    trials = []
    with ThreadPoolExecutor(max_workers=concurrency(args, config)) as pool:
        futures = [
            pool.submit(evaluate_trial, runner=run.runner, judge=judge, candidate=candidate, case=case,
                        repeat_id=repeat_id)
            for case in run.manifest.cases
            for repeat_id in range(args.repeats)
        ]
        for future in futures:  # log and print in task order from this thread
            trial, trace = future.result()
            trials.append(trial)
            run.logger.log_trial(trial, trace)
            failure = f"  ({trial.error})" if trial.error else ""
            print(f"{trial.case_id} r{trial.repeat_id}: status={trial.status.value} seen={trial.injection_seen} "
                  f"judge={trial.judge_score} attack_success={trial.attack_success} utility={trial.utility}{failure}")

    summary = summarize_candidate(candidate, run.manifest.cases, trials,
                                  proxy_threshold=search_params(args, config).proxy_threshold)
    run.logger.log_summary(summary, depth=0)
    print("\n=== summary ===")
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from universal_tap.search import AttackerLLM, UniversalTAPSearch

    config = load_config(args)
    params = search_params(args, config)
    run = start_run(args, config, ROLES, search=params.as_dict())
    manifest = run.manifest
    print(f"Starting universal TAP search: {params.as_dict()}")
    print(f"Training manifest: {manifest.name} ({len(manifest.cases)} task pairs, hash={manifest.manifest_hash})")
    result = UniversalTAPSearch(
        attacker=AttackerLLM(run.clients["attacker"], manifest, max_attempts=params.max_generation_attempts),
        judge=Judge(run.clients["judge"]),
        runner=run.runner,
        manifest=manifest,
        params=params,
        logger=run.logger,
        concurrency=concurrency(args, config),
    ).run()

    print("\n=== search finished ===")
    print(f"stop_reason={result.stop_reason}  candidates_evaluated={result.n_candidates_evaluated}  "
          f"generation_failures={result.n_generation_failures}")
    best, summary = result.best_candidate, result.best_summary
    if best is None or summary is None:
        print("No fully evaluated candidate.")
        return 1

    path = run.logger.save_artifact({
        **best.as_dict(),
        "render_rule": "prefix + attacker_goal + suffix",
        "train_manifest": manifest.name,
        "train_manifest_hash": manifest.manifest_hash,
        "selection_reason": "highest mean judge score among fully evaluated candidates",
        "summary": summary.as_dict(),
        "run_config": run.logger.run_config,
    })
    # Final summary in the style of the original TAP's print_final_summary_stats.
    print(f"\n{'=' * 8} FINAL SUMMARY STATISTICS {'=' * 8}")
    print(f"Training manifest: {manifest.name} ({len(manifest.cases)} task pairs)")
    print(f"Best candidate: {best.candidate_id} (depth {best.depth}, root {best.root_id}, parent {best.parent_id})")
    print(f"mean_judge={summary.mean_judge_score}  real_asr={summary.real_asr}  utility={summary.utility_rate}  "
          f"proxy_pass={summary.proxy_pass_rate}  failed={summary.n_failed}")
    print(f"Best template PREFIX:\n\n{best.template.prefix}\n")
    print(f"Best template SUFFIX:\n\n{best.template.suffix}\n")
    for role, client in run.clients.items():
        print(f"usage[{role}]: {client.usage.as_dict()}")
    print(f"frozen template saved to {path}")
    return 0


def turn_params(args, config):
    from universal_tap.search import TurnSearchParams
    overrides = {key: getattr(args, key, None) for key in
        ("roots", "branching_factor", "width", "depth", "max_turns", "max_candidates", "repeats", "renderer")}
    return TurnSearchParams(**{**config.get("turn_search", {}),
                               **{key: value for key, value in overrides.items() if value is not None}})


def guard_turn_output(args):
    if getattr(args, "resume", False) and not args.run_dir:
        raise ValueError("--resume requires --run-dir")
    if args.run_dir and not getattr(args, "resume", False):
        if any((Path(args.run_dir) / name).exists() for name in
               ("run_config.json", "checkpoint.json", "trials.jsonl")):
            raise ValueError("run directory already has results; use --resume or a new --run-dir")


def cmd_turn_search(args):
    import hashlib
    from universal_tap.evaluation import TurnJudge
    from universal_tap.prompts import TURN_JUDGE_VERSION, TURN_MUTATOR_VERSION
    from universal_tap.search import TurnMutator, TurnTAPSearch, load_tool_catalog
    from universal_tap.turns import TurnTemplate, root_template

    guard_turn_output(args)
    config = load_config(args)
    params = turn_params(args, config)
    roots = [root_template(params.renderer)]
    roots_file = args.roots_file or config.get("roots_file")
    if roots_file:
        seeds = json.loads(Path(roots_file).read_text(encoding="utf-8"))
        roots.extend(TurnTemplate.from_dict(seed, renderer=params.renderer, max_turns=params.max_turns)
                     for seed in seeds["roots"][:params.roots - 1])
    if len(roots) != params.roots:
        raise ValueError("roots > 1 needs enough manual seeds in --roots-file")
    catalog = load_tool_catalog(args.tool_catalog or config.get("tool_catalog", "data/tool_catalog"))
    catalog_hash = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()[:16]
    run = start_run(args, config, ("target", "evaluator", "mutator"), turn_search=params.as_dict(),
                    injection_strategy="one_turn_boundaries", judge_version=TURN_JUDGE_VERSION,
                    mutator_version=TURN_MUTATOR_VERSION,
                    tool_catalog_hash=catalog_hash, root_hashes=[root.template_hash for root in roots])
    print(f"Turn search: {params.as_dict()} concurrency={concurrency(args, config)}")
    print(f"Training: {run.manifest.name} ({len(run.manifest.cases)} cases)")
    search = TurnTAPSearch(mutator=TurnMutator(run.clients["mutator"], tool_catalog=catalog,
        max_turns=params.max_turns, max_attempts=params.max_generation_attempts),
        judge=TurnJudge(run.clients["evaluator"]), runner=run.runner, manifest=run.manifest,
        params=params, logger=run.logger, roots=roots, concurrency=concurrency(args, config), resume=args.resume)
    best = search.run()
    for role, client in run.clients.items():
        print(f"usage[{role}]: {client.usage.as_dict()}")
    if best is None:
        print("No fully evaluated candidate; inspect trials.jsonl for infrastructure failures.")
        return 1
    print(f"Best: {best.candidate_id} {search.summaries[best.candidate_id].as_dict()}")
    print(f"Training payload: {run.logger.run_dir / 'best_payload.json'}")
    return 0


def evaluate_turn_candidates(run, candidates, *, repeats, workers, phase="eval", resume=False):
    from universal_tap.evaluation import TurnJudge, evaluate_turn_trial, summarize_turn_candidate
    from universal_tap.search import _read_jsonl
    from universal_tap.schemas import TurnTrialResult
    judge = TurnJudge(run.clients["evaluator"])
    cached = _read_jsonl(run.logger.run_dir / "trials.jsonl") if resume else []
    summaries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for candidate in candidates:
            existing = [TurnTrialResult.from_dict(row) for row in cached if row["candidate_id"] == candidate.candidate_id]
            done = {(trial.case_id, trial.repeat_id) for trial in existing}
            futures = [pool.submit(evaluate_turn_trial, runner=run.runner, judge=judge,
                candidate=candidate, case=case, repeat_id=repeat)
                for case in run.manifest.cases for repeat in range(repeats)
                if ("/".join(case), repeat) not in done]
            trials = existing
            for future in futures:
                trial, trace = future.result()
                run.logger.log_trial(trial, trace)
                trials.append(trial)
                print(f"{phase} {candidate.candidate_id} {trial.case_id} r{trial.repeat_id}: "
                      f"{trial.status.value} grade={trial.grade} security={trial.attack_success} "
                      f"utility={trial.utility} termination={trial.termination}")
            summary = summarize_turn_candidate(candidate, run.manifest.cases, trials, repeats=repeats)
            run.logger.log_summary(summary, candidate.depth)
            summaries.append(summary)
    return summaries


def cmd_turn_eval(args):
    from universal_tap.prompts import TURN_JUDGE_VERSION
    from universal_tap.turns import TurnCandidate, TurnTemplate
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1")
    guard_turn_output(args)
    data = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    template = TurnTemplate.from_dict(data, max_turns=len(data["messages"]))
    run = start_run(args, load_config(args), ("target", "evaluator"),
                    repeats=args.repeats, payload_hash=template.template_hash,
                    renderer=template.renderer, judge_version=TURN_JUDGE_VERSION,
                    injection_strategy="one_turn_boundaries")
    candidate = TurnCandidate("fixed", template)
    if not args.resume:
        run.logger.log_candidate(candidate)
    summary = evaluate_turn_candidates(run, [candidate], repeats=args.repeats,
        workers=concurrency(args, run.config), resume=args.resume)[0]
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 0 if summary.fully_evaluated else 1


def cmd_turn_select(args):
    """Held-out selection runs separately; no mutator client or feedback is involved."""
    from universal_tap.prompts import TURN_JUDGE_VERSION
    from universal_tap.evaluation import rank_turn_candidates
    from universal_tap.schemas import TurnSummary
    from universal_tap.turns import TurnCandidate
    if args.repeats < 3 or args.top_k < 1:
        raise ValueError("selection requires repeats >= 3 and top_k >= 1")
    guard_turn_output(args)
    source = Path(args.source_run)
    checkpoint = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    if checkpoint["phase"] != "done":
        raise ValueError("finish training before held-out selection")
    candidates = {cid: TurnCandidate.from_dict(value) for cid, value in checkpoint["candidates"].items()}
    summaries = {cid: TurnSummary(**value) for cid, value in checkpoint["summaries"].items()}
    eligible = rank_turn_candidates(candidates, summaries)
    chosen = [candidates[cid] for cid in eligible[:args.top_k]]
    if not chosen:
        raise ValueError("no fully evaluated training candidates")
    config = load_config(args)
    validation = args.validation_manifest
    test_manifest = load_manifest(args.test_manifest)
    training_manifest = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    args.manifest = validation
    valid_manifest = load_manifest(validation)
    if set(valid_manifest.cases) & set(test_manifest.cases):
        raise ValueError("validation/test case overlap")
    for manifest in (valid_manifest, test_manifest):
        training_cases = {tuple(case) for case in training_manifest.get("cases", [])}
        if training_cases & set(manifest.cases) or manifest.manifest_hash == training_manifest["manifest_hash"]:
            raise ValueError("held-out cases overlap training data")
    run = start_run(args, config, ("target", "evaluator"), repeats=args.repeats,
        selected_train_candidates=[c.as_dict() for c in chosen], judge_version=TURN_JUDGE_VERSION,
        test_manifest_hash=test_manifest.manifest_hash, injection_strategy="one_turn_boundaries")
    validation_dir = run.logger.run_dir
    if not args.resume:
        for candidate in chosen:
            run.logger.log_candidate(candidate)
    scores = evaluate_turn_candidates(run, chosen, repeats=args.repeats,
        workers=concurrency(args, config), phase="validation", resume=args.resume)
    good = [summary for summary in scores if summary.fully_evaluated]
    if not good:
        print("Validation incomplete; resume after inspecting errors.")
        return 1
    # Final selection uses actual environment outcomes; proxy grades cannot break ties.
    winner = max(good, key=lambda s: (s.security_rate, s.utility_rate, -s.n_turns))
    best = candidates[winner.candidate_id]
    run.logger.save_artifact({**best.as_dict(), "selection": "validation",
        "summary": winner.as_dict(), "judge_version": TURN_JUDGE_VERSION,
        "validation_manifest_hash": run.manifest.manifest_hash}, "best_payload.json")
    # A dedicated subdirectory keeps test and validation trials disjoint on resume.
    args.manifest, args.run_dir = args.test_manifest, str(validation_dir / "test")
    args.resume = (Path(args.run_dir) / "run_config.json").exists()
    test = start_run(args, config, ("target", "evaluator"), repeats=1,
        payload_hash=best.template.template_hash, renderer=best.template.renderer,
        judge_version=TURN_JUDGE_VERSION, injection_strategy="one_turn_boundaries")
    if not args.resume:
        test.logger.log_candidate(best)
    score = evaluate_turn_candidates(test, [best], repeats=1, workers=concurrency(args, config),
        phase="test", resume=args.resume)[0]
    test.logger.save_artifact({"validation": winner.as_dict(), "test": score.as_dict(),
        "best_candidate": best.candidate_id}, "selection_report.json")
    print(f"Selected payload: {validation_dir / 'best_payload.json'}")
    return 0 if score.fully_evaluated else 1


def cmd_turn_rejudge(args):
    from universal_tap.agent import RunOutput
    from universal_tap.evaluation import (TurnJudge, judge_turn_output, rank_turn_candidates,
                                          summarize_turn_candidate)
    from universal_tap.prompts import TURN_JUDGE_VERSION
    from universal_tap.schemas import RunStatus, TurnTrialResult
    from universal_tap.search import _read_jsonl
    from universal_tap.turns import TurnCandidate

    guard_turn_output(args)
    source = Path(args.source_run)
    source_config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    if checkpoint["phase"] != "done":
        raise ValueError("finish training before rejudging all candidates")
    candidates = {cid: TurnCandidate.from_dict(data) for cid, data in checkpoint["candidates"].items()}
    records = _read_jsonl(source / "trials.jsonl")
    manifest = None
    if not args.manifest and source_config.get("cases"):
        manifest = TaskManifest(source_config["manifest"], source_config["benchmark_version"],
                                [tuple(case) for case in source_config["cases"]])
    run = start_run(args, load_config(args), ("target", "evaluator"), task_manifest=manifest,
        source_fingerprint=source_config["fingerprint"], judge_version=TURN_JUDGE_VERSION,
        repeats=source_config.get("turn_search", {}).get("repeats", source_config.get("repeats", 1)),
        injection_strategy="saved_trajectory")
    if run.manifest.manifest_hash != source_config["manifest_hash"]:
        raise ValueError("rejudge manifest differs from the original training cases")
    judge = TurnJudge(run.clients["evaluator"])
    cached = _read_jsonl(run.logger.run_dir / "trials.jsonl") if args.resume else []
    done = {(row["candidate_id"], row["suite"], row["user_task_id"], row["injection_task_id"], row["repeat_id"])
            for row in cached}
    results = [TurnTrialResult.from_dict(row) for row in cached]
    jobs = []
    with ThreadPoolExecutor(max_workers=concurrency(args, run.config)) as pool:
        for row in records:
            old = TurnTrialResult.from_dict(row)
            key = (old.candidate_id, old.suite, old.user_task_id, old.injection_task_id, old.repeat_id)
            if key in done:
                continue
            candidate = candidates[old.candidate_id]
            suite = run.runner.suite(old.suite)
            messages = row.get("messages")
            if messages is None:
                messages = json.loads((source / old.trace_path).read_text(encoding="utf-8"))["messages"]
            original_status = row.get("rollout_status", "ok" if old.judge_error and
                (old.error or "").startswith("judge failed:") else old.status.value)
            output = RunOutput(RunStatus(original_status), attack_success=old.attack_success,
                utility=old.utility, injection_seen=old.injection_seen, messages=messages,
                first_step_response=old.first_step_response, first_step_tool_calls=old.first_step_tool_calls,
                injection_message_id=old.injection_message_id, termination=old.termination,
                user_prompt=suite.get_user_task_by_id(old.user_task_id).PROMPT,
                attacker_goal=suite.get_injection_task_by_id(old.injection_task_id).GOAL,
                injection_text=candidate.template.render(suite.get_injection_task_by_id(old.injection_task_id).GOAL),
                error=row.get("rollout_error"))
            jobs.append(pool.submit(judge_turn_output, run=output, judge=judge, candidate=candidate,
                case=(old.suite, old.user_task_id, old.injection_task_id), repeat_id=old.repeat_id))
        for future in jobs:
            trial, trace = future.result()
            run.logger.log_trial(trial, trace)
            results.append(trial)
            print(f"rejudge {trial.candidate_id}: {trial.status.value} grade={trial.grade} "
                  f"security={trial.attack_success} judge_error={trial.judge_error}")
    summaries = {}
    for cid, candidate in candidates.items():
        if not args.resume:
            run.logger.log_candidate(candidate)
        summary = summarize_turn_candidate(candidate, run.manifest.cases,
            [trial for trial in results if trial.candidate_id == cid], repeats=run.logger.run_config["repeats"])
        run.logger.log_summary(summary, candidate.depth)
        summaries[cid] = summary
    ranked = rank_turn_candidates(candidates, summaries)
    run.logger.save_artifact({**checkpoint, "fingerprint": run.logger.fingerprint,
        "judge_version": TURN_JUDGE_VERSION, "summaries": {cid: s.as_dict() for cid, s in summaries.items()}},
        "checkpoint.json")
    if ranked:
        best = candidates[ranked[0]]
        run.logger.save_artifact({**best.as_dict(), "summary": summaries[best.candidate_id].as_dict(),
            "selection": "training_only", "judge_version": TURN_JUDGE_VERSION}, "best_payload.json")
    print(f"target requests: {run.clients['target'].usage.n_requests}; evaluator: {run.clients['evaluator'].usage.as_dict()}")
    return 0 if ranked else 1


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.json")
    common.add_argument("--env-file", default=None, help="vLLM endpoint file (default: env_file in the config)")
    run_opts = argparse.ArgumentParser(add_help=False, parents=[common])
    run_opts.add_argument("--manifest", default=None, help="task manifest (default: manifest in the config)")
    run_opts.add_argument("--run-dir", default=None,
                          help="default: runs/{universal,turn_search}/<command>-<time>, selected by command")
    run_opts.add_argument("--concurrency", type=int, default=None,
                          help="rollouts run at once (default: concurrency in the config)")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke", parents=[common], help="check connectivity and tool calling").set_defaults(func=cmd_smoke)

    p = sub.add_parser("baselines", parents=[run_opts], help="no-injection baselines")
    p.add_argument("--repeats", type=int, default=1)
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("eval", parents=[run_opts], help="evaluate one fixed template")
    p.add_argument("--template", default=None, help="a best_template.json to evaluate")
    p.add_argument("--prefix", default="")
    p.add_argument("--suffix", default="")
    p.add_argument("--repeats", type=int, default=1)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("search", parents=[run_opts], help="run the universal TAP search")
    for flag in ("--roots", "--branching-factor", "--width", "--depth", "--repeats"):
        p.add_argument(flag, type=int, default=None, help="override the config's search value")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("turn-search", parents=[copy.deepcopy(run_opts)], help="search a shared forged multi-turn dialogue")
    p.set_defaults(config="configs/turn_train.json", func=cmd_turn_search)
    for flag in ("--roots", "--branching-factor", "--width", "--depth", "--repeats", "--max-turns", "--max-candidates"):
        p.add_argument(flag, type=int, default=None)
    p.add_argument("--renderer", choices=("plain_roles", "minimax_m3_chat"), default=None)
    p.add_argument("--roots-file", default=None)
    p.add_argument("--tool-catalog", default=None)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("turn-eval", parents=[copy.deepcopy(run_opts)], help="evaluate a frozen multi-turn payload")
    p.set_defaults(config="configs/turn_train.json", func=cmd_turn_eval)
    p.add_argument("--payload", required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("turn-select", parents=[copy.deepcopy(run_opts)], help="top-k validation selection, then one held-out test")
    p.set_defaults(config="configs/turn_train.json", func=cmd_turn_select)
    p.add_argument("--source-run", required=True)
    p.add_argument("--validation-manifest", default="data/universal/validation.json")
    p.add_argument("--test-manifest", default="data/universal/test.json")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("turn-rejudge", parents=[copy.deepcopy(run_opts)], help="rejudge every saved training trajectory")
    p.set_defaults(config="configs/turn_train.json", func=cmd_turn_rejudge)
    p.add_argument("--source-run", required=True)
    p.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)  # flush per line even when piped
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
