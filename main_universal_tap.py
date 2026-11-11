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
``runs/universal/<command>-<time>``.
"""

from __future__ import annotations

import argparse
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
    return {role: ModelConfig(**{**shared, **config.get(role, {})}) for role in ROLES}


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


def start_run(args: argparse.Namespace, config: dict[str, Any], roles: tuple[str, ...], **extra: Any) -> Run:
    """Clients, manifest, agent runner and run directory, shared by baselines/eval/search."""
    clients = make_clients(args, config, roles)
    manifest = load_manifest(args.manifest or config["manifest"])
    run_dir = args.run_dir or f"runs/universal/{args.command}-{time.strftime('%Y%m%d-%H%M%S')}"
    logger = RunLogger(run_dir, {
        "mode": args.command,
        "benchmark_version": manifest.benchmark_version,
        "agentdojo_revision": AGENTDOJO_REVISION,
        "models": {role: client.config.as_dict() for role, client in clients.items()},
        "manifest": manifest.name,
        "manifest_hash": manifest.manifest_hash,
        "max_tool_iters": config.get("max_tool_iters", 15),
        "system_message_name": config.get("system_message_name"),
        "injection_strategy": "read_vectors",
        **extra,
    })
    runner = AgentRunner(
        clients["target"],
        benchmark_version=manifest.benchmark_version,
        max_tool_iters=config.get("max_tool_iters", 15),
        system_message_name=config.get("system_message_name"),
    )
    return Run(config, clients, manifest, runner, logger)


def concurrency(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """How many rollouts / model requests run at once (``--concurrency`` or the config)."""
    return args.concurrency or config.get("concurrency", 1)


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


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.json")
    common.add_argument("--env-file", default=None, help="vLLM endpoint file (default: env_file in the config)")
    run_opts = argparse.ArgumentParser(add_help=False, parents=[common])
    run_opts.add_argument("--manifest", default=None, help="task manifest (default: manifest in the config)")
    run_opts.add_argument("--run-dir", default=None, help="default: runs/universal/<command>-<time>")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)  # flush per line even when piped
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
