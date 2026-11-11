# Universal TAP（间接提示注入）

把 TAP（Tree of Attacks with Pruning）改造成面向 AgentDojo 间接提示注入（IPI）的 universal 版本：搜索**一个**注入模板，即一对共享的 `prefix`/`suffix`，包在每个任务的攻击目标外面，让它在多个任务对上都有效，而不是像原版 TAP 那样每个任务单独搜一个 jailbreak。

实验只在 AgentDojo 的模拟工具和虚构数据上运行。研究背景见 [AGENT.md](AGENT.md)，详细设计见 [plan.md](plan.md) 和 [universal_TAP.pdf](universal_TAP.pdf)。

## 设计思路

**搜索对象。** 一个候选就是一个模板：对攻击目标为 `g` 的任务，注入内容是 `prefix + g + suffix`。所有训练任务共用同一个 prefix/suffix，只有中间的攻击目标随任务变化（来自 AgentDojo 注入任务的 `GOAL`）。

**三个模型角色**，各用各的 client 和对话历史：

- **attacker**：根据训练集概况和上一轮的反馈，生成新的 prefix/suffix。
- **target**：AgentDojo 里正在执行正常用户任务的 agent。注入内容不是直接发给它，而是放进它调用工具时读到的数据里，比如日历活动描述、交易备注。
- **judge**：看 target 读到注入后的第一步回复和工具调用，按"是否在执行攻击目标"打 1–10 分。

**树搜索。** 和原版 TAP 一样逐层扩展，但每个节点是一个模板，评估范围是整个训练集：

1. **分支**：第 1 层有 `roots` 个起点，每个起点让 attacker 生成 `branching_factor` 个模板；之后每个存活节点各生成 `branching_factor` 个子模板。
2. **评估**：每个候选在所有训练任务上各跑一次完整的 agent rollout，每次都从任务的初始环境开始。
3. **剪枝**：按"各任务 judge 分的平均值"排序，保留前 `width` 个。原版用来过滤跑题 prompt 的 on-topic 剪枝去掉了，因为模板本身不含具体任务。
4. **反馈**：把存活候选的整体指标、各任务得分和几条典型失败回复发给 attacker，进入下一层。

论文参数（roots=3, branching=3, width=8, depth=5）下，每层候选数是 9、24、24、24、24，共 105 个。

**两种信号分开记录。**

- judge 分是搜索用的代理信号：连续取值，便于排序。
- 真实结果以 AgentDojo 的环境检查为准：`attack_success` 表示攻击目标真的执行了，`utility` 表示正常任务仍然完成了。judge 给了高分、但工具实际没调用成功，仍然算攻击失败。
- 注入没被 agent 读到的运行记最低分 1 分。
- 每次运行只有 `ok` 和 `failed` 两种状态。没有正常完成的一律算 `failed`，包括 API 出错、超过步数上限、没有给出最终回复、judge 给不出分数等，失败原因记在 `error` 字段。每次运行最多尝试 3 次，每次都从全新的环境开始；3 次都失败，这个候选就被淘汰：不参与剪枝排序，也不能当选最佳模板。

**全部记录。** 每个候选、每次运行在剪枝前都写入磁盘，任何汇总分数都能从原始记录重新算出来。

## 运行

先启动本地 vLLM（2 个节点，空闲时也计费，用完 `scancel`）：

```bash
cd ../structural_attack && sbatch vllm/serve_multinode.sh
# 日志出现 "ready:" 后，把 endpoint 文件复制过来（每次重启都要重新复制）
cp vllm/endpoint-qwen3-235b-a22b-instruct-2507.env ../TAP/
```

然后在本目录下运行：

```bash
source ../structural_attack/.venv/bin/activate

python main_universal_tap.py smoke                                   # 检查连接和工具调用
python main_universal_tap.py search                                  # 3 个任务的小搜索
python main_universal_tap.py search --config configs/train18.json    # 论文设置：18 个任务对
python main_universal_tap.py eval --manifest data/universal/test.json \
    --template runs/universal/search-<时间>/best_template.json       # 在测试集上评估冻结的模板
python main_universal_tap.py baselines                               # 不注入时的基线
```

输出同时显示在 terminal 和 run 目录的 `log.txt` 里。run 目录默认是新建的 `runs/universal/<命令>-<时间>`。常用参数：

- `--manifest`：换一组任务
- `--run-dir`：指定输出目录
- `--config`：换配置文件
- search 专用：`--roots/--branching-factor/--width/--depth/--repeats`，覆盖配置里的值

## 配置

| 文件 | 内容 |
|---|---|
| `configs/default.json` | 模型、endpoint 文件、3 个任务的小搜索（roots=1, branching=2, width=2, depth=2） |
| `configs/train18.json` | 论文的 18 任务训练（roots=3, branching=3, width=8, depth=5，共 105 个候选） |
| `data/universal/*.json` | 任务清单：smoke（1 个）、small（3 个）、train（18 个）、validation、test（含 Travel） |

`concurrency` 是同时进行的 rollout 和模型请求数（默认 32，命令行可用 `--concurrency` 覆盖）。同一层的 attacker 生成、以及所有"候选 × 任务"的运行都并发执行，但 log 仍按候选顺序输出。`llm` 里写三个角色共用的模型，`attacker`/`target`/`judge` 各自只写不同的参数。`provider` 可以是 `vllm`（URL 和 key 从 `env_file` 读）或 `openrouter`（key 从 `OPENROUTER_API_KEY` 读）。

## 输出

| 文件 | 内容 |
|---|---|
| `run_config.json` | 模型、任务清单、参数，以及它们的指纹 |
| `log.txt` | terminal 上的全部输出 |
| `candidates.jsonl` | 所有候选模板，含父节点和改进说明 |
| `trials.jsonl` | 每个候选在每个任务上的结果：注入是否被读到、judge 分数、攻击是否成功、用户任务是否完成 |
| `summaries.jsonl` | 每个候选的汇总：平均 judge 分、真实 ASR、utility、失败次数 |
| `traces/` | 每次运行的完整对话 |
| `best_template.json` | search 选出的最佳模板 |

judge 分数只是搜索时用的代理信号；真实结果以 AgentDojo 的环境检查为准（`attack_success`、`utility`）。

## 项目架构

一次 search 的调用关系：

```
main_universal_tap.py search
 └─ UniversalTAPSearch.run()                        search.py
     ├─ AttackerLLM.generate()  → 新模板             search.py + prompts.py
     ├─ evaluate_trial()        每个候选 × 每个任务   evaluation.py
     │   ├─ AgentRunner.run_case()                   agent.py
     │   │   渲染注入 → 放进 AgentDojo 环境 → target 完整 rollout
     │   │   → 环境检查（attack_success / utility）
     │   │   → 判断注入是否被读到，提取读到后的第一步回复
     │   └─ Judge.score()       → 1–10 分             evaluation.py + prompts.py
     ├─ summarize_candidate()   → 候选汇总分          evaluation.py
     ├─ 剪枝、打印本层统计、把反馈追加到 attacker 对话
     └─ RunLogger               → 写入 run 目录       records.py

三个角色的 client 都来自 llm.py；任务清单由 datasets.py 读取。
```

### 文件说明

| 文件 | 作用 |
|---|---|
| `main_universal_tap.py` | 命令行入口。读配置、建 client 和 run 目录，分派到四个子命令：`smoke`、`baselines`、`eval`、`search`。搜索结束后保存最佳模板并打印最终统计。 |
| `universal_tap/__init__.py` | 固定的 AgentDojo benchmark 版本和 git revision。 |
| `universal_tap/llm.py` | OpenAI 兼容的 chat client。按 `provider` 解析 vLLM 或 OpenRouter 的地址和 key，区分可重试错误（限流、超时、5xx）和致命错误（认证失败、模型不存在），统计每个角色的请求数和 token。 |
| `universal_tap/agent.py` | 和 AgentDojo 交互的部分。找出用户任务会读到的注入位置，把模板渲染并转义后放进环境，跑完整 rollout，做环境检查，从对话中判断注入是否被读到，并提取读到后的第一步回复。`baselines` 用的不注入运行也在这里。 |
| `universal_tap/evaluation.py` | `Judge`：调用 judge 模型并解析 `Rating: [[n]]`。`evaluate_trial`：一个候选在一个任务上的一次运行加打分。`summarize_candidate`：把一个候选的所有运行汇总成平均 judge 分、真实 ASR、utility 和失败次数。 |
| `universal_tap/search.py` | `AttackerLLM`：维护每个分支的 attacker 对话，解析并校验它输出的 JSON。`UniversalTAPSearch`：分支、评估、去重、剪枝、反馈的主循环，以及按原版 TAP 格式打印的进度。 |
| `universal_tap/prompts.py` | attacker 的系统提示词和反馈消息格式，judge 的系统提示词和输入格式，以及 prefix/suffix 的长度限制。 |
| `universal_tap/datasets.py` | 读取任务清单 JSON（每个任务对是一个 `(suite, user_task_id, injection_task_id)` tuple），生成给 attacker 的训练集概况（任务数、领域、攻击目标列表）。 |
| `universal_tap/schemas.py` | 数据结构：`Template`、`Candidate`、`TrialResult`、`CandidateSummary`，以及运行状态 `RunStatus`（`ok` / `failed`）。 |
| `universal_tap/records.py` | `RunLogger`：写入 run 目录里的所有文件（见上面的"输出"），并把 terminal 输出同时写进 `log.txt`。 |
| `configs/*.json` | 运行配置：模型、endpoint 文件、任务清单、搜索参数。 |
| `data/universal/*.json` | 任务清单：每个任务对是 (suite, user_task_id, injection_task_id)。 |
| `runs/universal/` | 各次运行的结果。 |
| `AGENT.md` | 研究背景和使用边界说明。 |
| `plan.md` | 详细的实施计划，包括各阶段的验收标准和设计取舍。 |
| `universal_TAP.pdf` | 论文。 |
| `requirements.txt` | 依赖：openai、python-dotenv，以及固定版本的 AgentDojo。 |

## 致谢

基于 TAP（Mehrotra et al., 2023, [arXiv:2312.02119](https://arxiv.org/abs/2312.02119)）和 [AgentDojo](https://github.com/ethz-spylab/agentdojo)。
