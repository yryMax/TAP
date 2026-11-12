"""Export static AgentDojo tool definitions without executing tools or loading environments."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import textwrap
from typing import get_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docstring_parser import parse
from pydantic import BaseModel, TypeAdapter

from universal_tap import AGENTDOJO_REVISION, BENCHMARK_VERSION

SUITES = ("workspace", "travel", "banking", "slack")


def response_definition(tool) -> dict:
    source = textwrap.dedent(inspect.getsource(tool.run))
    returns = [node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Return)]
    expressions = [ast.unparse(node) if node is not None else "None" for node in returns]
    annotation = tool.return_type
    if annotation is not None:
        schema = TypeAdapter(annotation).json_schema(mode="serialization")
        schema_source = "return_annotation"
    elif returns and all(isinstance(node, ast.JoinedStr) for node in returns):
        schema = {"type": "string"}
        schema_source = "static_fstring_returns"
    else:
        schema = {}
        schema_source = "unannotated; consult static_return_expressions"

    is_model = isinstance(annotation, type) and issubclass(annotation, BaseModel)
    wire_format = "yaml" if is_model or get_origin(annotation) is list else "python_str"
    docs = parse(tool.full_docstring)
    return {
        "python_type": str(annotation) if annotation is not None else "unannotated",
        "schema": schema,
        "schema_source": schema_source,
        "description": docs.returns.description if docs.returns else None,
        "wire_format": wire_format,
        "static_return_expressions": expressions or ["None (implicit return)"],
    }


def export_catalog(output_dir: Path) -> dict[str, int]:
    # Import definitions only: no FunctionsRuntime, environments or tool invocations.
    from agentdojo.task_suite.load_suites import get_suite

    distribution = importlib.metadata.distribution("agentdojo")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    revision = direct_url.get("vcs_info", {}).get("commit_id")
    if revision is not None and revision != AGENTDOJO_REVISION:
        raise ValueError(f"AgentDojo revision differs from project pin: {revision}")
    metadata = {
        "catalog_version": 1,
        "benchmark_version": BENCHMARK_VERSION,
        "configured_agentdojo_revision": AGENTDOJO_REVISION,
        "installed_agentdojo_revision": revision,
        "installed_package_version": distribution.version,
        "generation": "static introspection only; no environments or tools executed",
    }
    catalogs = {}
    for suite_name in SUITES:
        suite = get_suite(BENCHMARK_VERSION, suite_name)
        tools = []
        for tool in suite.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_json_schema(),
                "response": response_definition(tool),
                "docstring": tool.full_docstring,
                "source": {"module": tool.run.__module__, "function": tool.run.__name__},
            })
        catalogs[suite_name] = {**metadata, "suite": suite_name, "tool_count": len(tools), "tools": tools}

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, catalog in catalogs.items():
        (output_dir / f"{name}.json").write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        counts[name] = catalog["tool_count"]
    rows = "\n".join(f"| [{name}]({name}.json) | {count} |" for name, count in counts.items())
    readme = f"""# AgentDojo 静态工具目录

Benchmark：`{BENCHMARK_VERSION}`；代码 revision：`{AGENTDOJO_REVISION}`。

| Suite | 工具数 |
|---|---:|
{rows}
| 合计 | {sum(counts.values())} |

## 字段

- `name`、`description`、`parameters`：目标请求使用的工具名、说明及参数 JSON Schema，包含必填项、默认值和参数描述。
- `response.schema`：返回值的数据结构，保留嵌套字段、枚举及 `$defs` 引用；这是类型描述，不是实际运行结果。
- `response.wire_format`：该返回值经当前 AgentDojo `tool_result_to_str()` 处理后的文本格式。
- `response.static_return_expressions`：从工具源码提取的返回表达式。用于查看通用 dict schema 没有写出的字段、动态键和字符串模板；表达式未经执行。
- `docstring`：完整工具说明，包括源码已有的返回值及错误说明。

## Response 文本格式

当前 runner 使用 AgentDojo 默认工具结果格式：

- Pydantic 对象：`model_dump()` 后用 `yaml.safe_dump(...).strip()` 转成 YAML。
- list：对象元素先转 dict，字符串或整数元素先转字符串，再将整个列表转成 YAML。
- 其他值：`str(result)`。因此 dict 是 Python 字典文本，字符串直接输出，空返回为 `None`，不是统一的 JSON。
- 工具执行错误单独记录为 `error`；转换为目标 API 消息时优先使用非空错误文本。

类型注解不保证涵盖全部内容约束。响应字段含义参照 schema、docstring 和静态返回表达式；唯一未注解的 f-string 返回可由静态语法识别为字符串。模型对象 schema 的日期等字段描述归一化类型，实际文本按 YAML 序列化规则呈现。

## 搜索 LLM 的使用规则

四份文件均作为只读参考资料，不注册为搜索 LLM 可执行的函数。搜索 LLM 不得调用 AgentDojo runtime、执行工具或读取真实环境来获取 response。候选中的 tool call 和 tool response 均由它自行编写；编造的 response 不改变评估环境，也不作为真实成功证据。目标模型续跑时的真实工具执行仍由评估 runner 负责。

目录仅包含静态定义，无任务上下文、环境数据或真实 response。重新导出：

```bash
../structural_attack/.venv/bin/python scripts/export_tool_catalog.py
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "tool_catalog")
    args = parser.parse_args()
    print(json.dumps(export_catalog(args.output_dir), ensure_ascii=False))
