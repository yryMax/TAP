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
cd ../structural_attack && sbatch -J vllm-minimax-m3 vllm/serve_multinode.sh minimax-m3
# 日志出现 "ready:" 后，把 endpoint 文件复制过来（每次重启都要重新复制）
cp vllm/endpoint-minimax-m3.env ../TAP/
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

输出同时显示在 terminal 和 run 目录的 `log.txt` 里。旧 prefix/suffix 模式默认保存到 `runs/universal/<命令>-<时间>`，轮级模式默认保存到 `runs/turn_search/<命令>-<时间>`。常用参数：

- `--manifest`：换一组任务
- `--run-dir`：指定输出目录
- `--config`：换配置文件
- search 专用：`--roots/--branching-factor/--width/--depth/--repeats`，覆盖配置里的值

## 配置

### 轮级搜索

`turn-search` 实现 [turn_search_plan.md](turn_search_plan.md)：每个子节点只执行一个
insert / rewrite / delete，保留父节点参与排序，先比较真实 security 成功率，再比较
0–4 平均等级，最后优先选择更少的轮数。默认根是 `user: {goal}`，在深度 0 评估。
`--roots 2` 会额外评估 [人工 seed](data/turn_seeds.json)。

`target`、`evaluator`、`mutator` 使用独立 client。mutator 看静态工具目录和匿名反馈；
每次附简短 `reason`，并读取同父节点此前全部已接受变体的完整消息及思路。
同父串行、不同父并发；历史保存在 checkpoint，`previous_variant_ids` 记录生成依赖。

evaluator 只判断注入后的真实轨迹：A 原任务让位（含暂停或先新后旧），B 注入交互
生效（含 `?` 引起的真实澄清），C 新 goal 的执行。三项独立，具体判据见
[方法说明](turn_search_plan.md#4-评估)。真实 security 成功记等级 4，优先于代理信号。
协议版本见 [prompts.py](universal_tap/prompts.py)；跨版本使用新 run 目录，旧轨迹用
`turn-rejudge` 重判，无需重跑 target。

```bash
# 既有环境；默认读取 endpoint-minimax-m3.env
source ../structural_attack/.venv/bin/activate
python -m unittest discover -s tests -v
python main_universal_tap.py turn-search --config configs/turn_smoke.json \
  --run-dir runs/turn_search/turn-smoke

# 相同配置恢复；已记录的 trial 不再调用 target / evaluator
python main_universal_tap.py turn-search --config configs/turn_smoke.json \
  --run-dir runs/turn_search/turn-smoke --resume

# 默认训练集搜索（configs/turn_train.json → data/universal/train.json，也是 turn-search 的默认配置）
python main_universal_tap.py turn-search --config configs/turn_train.json
# 默认 plain_roles，可固定为 M3 原生标签
python main_universal_tap.py turn-search --config configs/turn_train.json --renderer minimax_m3_chat

# 在冻结 payload 上评估；renderer 从 JSON 读取
python main_universal_tap.py turn-eval --config configs/turn_smoke.json \
  --payload runs/turn_search/turn-smoke/best_payload.json

# 修改 judge 后，对所有已有轨迹重新判级，target 请求数为 0
python main_universal_tap.py turn-rejudge --source-run runs/turn_search/turn-smoke \
  --run-dir runs/turn_search/turn-rejudged

# 10 条固定轨迹校准新 A/B/C 定义；只调用 evaluator，不运行 target
# 先加 --dry-run 可以仅保存轨迹和预期标签，模型请求数为 0
python -m scripts.calibrate_priority_judge --config configs/turn_train.json

# 训练结束后，top-k 在验证集 repeats>=3 选择，再在 test 上评估一次
# 默认 --top-k 5、验证集 data/universal/validation.json、测试集 data/universal/test.json
python main_universal_tap.py turn-select --source-run runs/turn_search/<那次 run>
```

`configs/turn_smoke.json` 为单 case、width=2、branching=2、depth=5、max_turns=13、max_candidates=7；
预算包含根，因此通常在深度 2 用完。需要走完深度 5 时加 `--max-candidates 19`。
`configs/turn_train.json`（turn-search / turn-eval / turn-select / turn-rejudge 的默认配置）
使用 `data/universal/train.json`、width=16、branching=4、depth=5、repeats=2、max_turns=13、max_candidates=256。
`--concurrency` 同时控制 mutation 和 rollout 并发。

训练 / 验证 / 测试三份 manifest 的 goal（suite×injection_task）两两不相交，
所以验证集和测试集衡量的是对**没见过的 goal** 的泛化。case 以 qwen3 全量跑分为难度
oracle 挑选：只保留无攻击 baseline（dummy）失败、且至少被一种强攻击攻破的 case，
去掉送分题、保留梯度信号，四个 suite（含 travel）均有覆盖。规模为
`train.json` 64、`validation.json` 44、`test.json` 24。旧的 18-case 训练集仍记录在
`runs/turn_search/turn-train18/run_config.json` 里（其 `cases` 内嵌，可据此复现）。

轮级 run 另存 `checkpoint.json`、`best_payload.json`、`search_report.json` 和
`mutations.jsonl`。`trials.jsonl` 包含完整轨迹、等级、环境结果、错误、token 和耗时。
非法候选及原因记入 `candidates.jsonl`，不消耗评估预算；缺失 trial 不缩小分母，
评估不完整的候选不参与排序。恢复时核对配置、任务、seed、工具目录和 judge 版本。
搜索输出的 payload 标记为 `training_only`；`turn-select` 的最终选择只比较真实
security、utility 及轮数，不使用代理等级。验证和测试结果不会进入 mutator。

在 AgentDojo 的其它命令中使用相同注入实现：

```bash
export MULTITURN_FROM_SEARCH_PAYLOAD="$PWD/runs/turn_search/turn-smoke/best_payload.json"
# 在既有 AgentDojo 命令中加：-ml my_attacks --attack multiturn_from_search
```

新 attack 继承 `OneTurnAttack`，复用其工具边界和 YAML 转义。默认 `plain_roles`
直接复用 `_render_generic`；`minimax_m3_chat` 使用 M3 角色及工具调用标签。

### 原 prefix/suffix 模式

| 文件 | 内容 |
|---|---|
| `configs/default.json` | 模型、endpoint 文件、3 个任务的小搜索（roots=1, branching=2, width=2, depth=2） |
| `configs/train18.json` | 旧 prefix/suffix `search` 模式的配置（roots=3, branching=3, width=8, depth=5），现指向新的 `train.json` |
| `configs/turn_train.json` | 轮级搜索默认配置：`train.json`、width=16、branching=4、depth=5、repeats=2、max_candidates=256 |
| `data/universal/*.json` | 任务清单：smoke（1）、small（3）、train（64）、validation（44）、test（24）；三者 goal 两两不相交，均覆盖 4 个 suite（含 Travel），只收录 dummy 失败且可被强攻击攻破的 case |

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
| `runs/universal/` | 原 Universal TAP（prefix/suffix）的运行结果。 |
| `runs/turn_search/` | 轮级搜索、评估、重判及验证集选择的运行结果。 |
| `AGENT.md` | 研究背景和使用边界说明。 |
| `plan.md` | 详细的实施计划，包括各阶段的验收标准和设计取舍。 |
| `universal_TAP.pdf` | 论文。 |
| `requirements.txt` | 依赖：openai、python-dotenv，以及固定版本的 AgentDojo。 |

## 致谢

基于 TAP（Mehrotra et al., 2023, [arXiv:2312.02119](https://arxiv.org/abs/2312.02119)）和 [AgentDojo](https://github.com/ethz-spylab/agentdojo)。
