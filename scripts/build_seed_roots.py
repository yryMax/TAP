"""Build search roots by truncating benign logs after every injectable tool response.

A tool response is injectable when it contains at least one AgentDojo injection vector. This is
checked the same way as AgentDojo's ``get_injection_candidates``: the recorded tool calls are
replayed in a fresh environment whose injection vectors are replaced by unique canaries. Replay is
local tool execution only; no model is called.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def repo_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), ROOT)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tool_boundaries(messages: list[dict]) -> list[dict]:
    pending = {}
    responded = set()
    recorded = []
    boundaries = []
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = call["id"]
                if not call_id or call_id in pending or call_id in responded:
                    raise ValueError(f"Invalid or duplicate call ID at message {index}")
                pending[call_id] = copy.deepcopy(call)
        elif message["role"] == "tool":
            call_id = message["tool_call_id"]
            if call_id not in pending:
                raise ValueError(f"Tool response without preceding call at message {index}")
            call = pending.pop(call_id)
            if message.get("tool_call") is not None and message["tool_call"] != call:
                raise ValueError(f"Tool response call metadata mismatch at message {index}")
            responded.add(call_id)
            recorded.append({"call": call, "response_message_index": index})
            boundaries.append({
                "message_index": index,
                "tool_response_ordinal": len(boundaries),
                "tool_name": call["function"],
                "tool_call_id": call_id,
                "tool_error": message.get("error"),
                "recorded_tool_calls": copy.deepcopy(recorded),
                "pending_tool_calls": copy.deepcopy(list(pending.values())),
            })
    if not boundaries:
        raise ValueError("Benign trajectory contains no tool responses")
    return boundaries


def message_text(message: dict) -> str:
    return "".join(part.get("content") or "" for part in message.get("content") or [])


def injection_slots(suite_name: str, version: str, messages: list[dict], boundaries: list[dict]) -> list[list[str]]:
    """Return the injection vectors visible in each boundary's tool response."""
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(version, suite_name)
    canaries = {vector: f"<INJECTION_CANARY::{vector}>" for vector in suite.get_injection_vector_defaults()}
    replays = {}
    for mode, injections in (("default", {}), ("canary", canaries)):
        environment = suite.load_and_inject_default_environment(injections)
        runtime = FunctionsRuntime(suite.tools)
        outputs = []
        for boundary in boundaries:
            call = boundary["recorded_tool_calls"][-1]["call"]
            result, error = runtime.run_function(environment, call["function"], call["args"])
            outputs.append((tool_result_to_str(result) if error is None else "", error))
        replays[mode] = outputs

    slots = []
    for boundary, (text, error), (canary_text, _) in zip(boundaries, replays["default"], replays["canary"]):
        recorded = messages[boundary["message_index"]]
        faithful = (error is not None) if recorded.get("error") else (error is None and text == message_text(recorded))
        if not faithful:
            raise ValueError(f"Replay does not reproduce recorded tool response at message {boundary['message_index']}")
        slots.append([vector for vector, canary in canaries.items() if canary in canary_text])
    return slots


def build_roots(manifest_path: Path, benign_dir: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest["benchmark_version"]
    cases = manifest["cases"]
    case_keys = [(c["suite"], c["user_task_id"], c["injection_task_id"]) for c in cases]
    if len(case_keys) != len(set(case_keys)):
        raise ValueError("Duplicate task pairs in manifest")

    # Preflight every required log before writing output. Tool replay only; no model calls.
    sources = {}
    for case in cases:
        key = (case["suite"], case["user_task_id"])
        if key in sources:
            continue
        path = benign_dir / key[0] / key[1] / "none" / "none.json"
        raw = path.read_bytes()
        trace = json.loads(raw)
        if (trace["suite_name"], trace["user_task_id"], trace["benchmark_version"]) != (*key, version):
            raise ValueError(f"Source metadata does not match case: {path}")
        if trace.get("injections") or trace.get("attack_type") or trace.get("injection_task_id"):
            raise ValueError(f"Expected a no-injection benign trajectory: {path}")
        boundaries = tool_boundaries(trace["messages"])
        for boundary, vectors in zip(boundaries, injection_slots(key[0], version, trace["messages"], boundaries)):
            boundary["injection_vectors"] = vectors
        sources[key] = {
            "messages": trace["messages"],
            "boundaries": boundaries,
            "metadata": {
                "suite": key[0], "user_task_id": key[1],
                "path": repo_path(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "benchmark_version": version,
                "tool_response_count": sum(m["role"] == "tool" for m in trace["messages"]),
            },
        }

    roots = []
    case_index = []
    excluded_cases = []
    candidate_count = 0
    for case in cases:
        source = sources[(case["suite"], case["user_task_id"])]
        case_id = "/".join(case[k] for k in ("suite", "user_task_id", "injection_task_id"))
        entries = []
        for boundary in source["boundaries"]:
            candidate_count += 1
            if not boundary["injection_vectors"]:
                continue
            index = boundary["message_index"]
            messages = copy.deepcopy(source["messages"][:index + 1])
            prefix_hash = digest(messages)
            root_id = case_id.replace("/", "__") + f"__tool{boundary['tool_response_ordinal']:03d}__m{index:04d}"
            path = output_dir / "roots" / case["suite"] / case["user_task_id"] / case["injection_task_id"] / f"tool_{boundary['tool_response_ordinal']:03d}_m{index:04d}.json"
            root = {
                "format_version": 1,
                "root_id": root_id, "context_id": prefix_hash[:16],
                "case_id": case_id, "case": case,
                "benchmark_version": version,
                "paths_relative_to": "repository_root",
                "source": source["metadata"],
                "cutoff": {k: boundary[k] for k in (
                    "message_index", "tool_response_ordinal", "tool_name", "tool_call_id", "tool_error",
                    "injection_vectors",
                )},
                "messages_sha256": prefix_hash,
                "messages": messages,
                "recorded_tool_calls": boundary["recorded_tool_calls"],
                "pending_tool_calls": boundary["pending_tool_calls"],
                "environment_status": "not_restored; replay recorded prefix calls before target continuation",
                "continuation": {"stage": "A", "seed_messages": [], "operation": "append"},
                "tool_catalog": f"data/tool_catalog/{case['suite']}.json",
            }
            roots.append((path, root))
            entries.append({
                "root_id": root_id, "context_id": root["context_id"], "path": repo_path(path),
                "message_index": index, "tool_name": boundary["tool_name"],
                "injection_vectors": boundary["injection_vectors"],
            })
        if entries:
            case_index.append({"case_id": case_id, **case, "root_count": len(entries), "roots": entries})
        else:
            excluded_cases.append({"case_id": case_id, **case, "reason": "no injectable tool response in benign trajectory"})

    output_dir.mkdir(parents=True, exist_ok=True)
    # roots/ is fully generated by this script; clear it so filtered-out roots do not linger.
    shutil.rmtree(output_dir / "roots", ignore_errors=True)
    for path, root in roots:
        write_json(path, root)
    jsonl_path = output_dir / "roots.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(root, ensure_ascii=False) + "\n" for _, root in roots), encoding="utf-8"
    )
    summary = {
        "format_version": 1, "name": manifest["name"], "benchmark_version": version,
        "paths_relative_to": "repository_root",
        "source_manifest": repo_path(manifest_path), "source_manifest_sha256": digest(manifest),
        "benign_dir": repo_path(benign_dir),
        "root_policy": "inclusive prefix after every tool response that contains an AgentDojo injection vector "
                       "(canary replay of recorded calls); no utility filtering",
        "source_case_count": len(cases), "case_count": len(case_index),
        "trajectory_count": len({(c["suite"], c["user_task_id"]) for c in case_index}),
        "candidate_tool_response_count": candidate_count,
        "excluded_non_injectable_root_count": candidate_count - len(roots),
        "root_count": len(roots),
        "unique_context_count": len({root["context_id"] for _, root in roots}),
        "tool_error_root_count": sum(bool(root["cutoff"]["tool_error"]) for _, root in roots),
        "pending_call_root_count": sum(bool(root["pending_tool_calls"]) for _, root in roots),
        "roots_jsonl": repo_path(jsonl_path),
        "source_trajectories": [source["metadata"] for source in sources.values()],
        "cases": case_index,
        "excluded_cases": excluded_cases,
    }
    write_json(output_dir / "manifest.json", summary)
    write_json(output_dir / "initial_template.json", {
        "name": f"{manifest['name']}_initial", "stage": "A", "seed_messages": [],
        "operation": "append", "paths_relative_to": "repository_root",
        "source_pool": repo_path(jsonl_path), "root_manifest": repo_path(output_dir / "manifest.json"),
        "case_count": len(case_index), "root_count": len(roots),
        "helper_reference_files": [f"data/tool_catalog/{suite}.json" for suite in (
            "workspace", "travel", "banking", "slack"
        )],
        "helper_runtime_access": False,
        "root_context_visibility": "target/evaluator only; helper receives static catalog and aggregate feedback",
    })
    rows = "\n".join(
        f"| {c['case_id']} | {c['root_count']} | {', '.join(sorted({v for r in c['roots'] for v in r['injection_vectors']}))} |"
        for c in case_index
    )
    excluded_rows = "\n".join(f"- `{c['case_id']}`：{c['reason']}" for c in excluded_cases) or "- 无"
    (output_dir / "README.md").write_text(f"""# train18 初始搜索 roots

来源：`{repo_path(benign_dir)}`；benchmark：`{version}`。

源 manifest 共 {len(cases)} 个 case。只保留截断点 tool response 含 AgentDojo 注入字段的 root：{candidate_count} 个候选截断点中保留 {len(roots)} 个，去掉 {candidate_count - len(roots)} 个；剩余 {len(case_index)} 个 case、{summary['trajectory_count']} 条 benign 轨迹；去掉不同目标复用的相同上下文后为 {summary['unique_context_count']} 个前缀。

- [initial_template.json](initial_template.json)：初始模板，共享新增消息为空，从各 root 后开始追加。
- [manifest.json](manifest.json)：{len(case_index)} 个 case 的 root 索引、注入字段及源文件哈希；被整体排除的 case 记录在 `excluded_cases`。
- [roots.jsonl](roots.jsonl)：完整 root 集合，每行一条；[roots/](roots/) 下也提供独立 JSON 文件。
- 每条 `messages` 严格等于 benign 原轨迹在指定 tool response 后的截断前缀；索引和 ordinal 从 0 开始。
- 注入位置判定：在独立环境中把每个 injection vector 替换为唯一 canary，按原调用重放 benign 轨迹，截断点回复含 canary 才保留（同 AgentDojo `get_injection_candidates`）。重放同时用默认环境校验与原记录逐字一致。`cutoff.injection_vectors` 记录命中的字段。不按 benign 最终 utility 筛选。
- 不同 injection task 共用同一 benign 轨迹，但拥有独立 root_id；context_id 标识共享前缀。
- 保留原始 content、调用 ID、工具参数和 error；不补齐、不改写，也不包含截断后的消息或新目标请求。
- 若在批量回复中间截断，保留 pending_tool_calls，不延后截断或补入未来回复。当前数据中这类 root 数：{summary['pending_call_root_count']}。
- 构建时仅为判定注入位置在本地重放工具，未调用模型；产物不含环境状态。后续 target runner 根据 recorded_tool_calls 在独立环境恢复状态；搜索 helper 不能访问 root 真实工具结果或 runtime。

| Case | Roots | 注入字段 |
|---|---:|---|
{rows}

被整体排除的 case（benign 轨迹中没有任何注入位置）：

{excluded_rows}

重新生成：

```bash
../structural_attack/.venv/bin/python scripts/build_seed_roots.py
```

更换轨迹来源可指定 `--benign-dir`，另指定 `--output-dir` 保存不同来源的数据，避免混用。
""", encoding="utf-8")
    return {k: summary[k] for k in (
        "source_case_count", "case_count", "trajectory_count", "candidate_tool_response_count",
        "excluded_non_injectable_root_count", "root_count", "unique_context_count", "tool_error_root_count",
        "pending_call_root_count"
    )}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/universal/train.json")
    parser.add_argument("--benign-dir", type=Path, default=ROOT / "runs/universal/m3_benign/openai-compatible")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/seed_roots/train18")
    args = parser.parse_args()
    print(json.dumps(build_roots(args.manifest, args.benign_dir, args.output_dir), ensure_ascii=False))
