**Universal TAP 面向 IPI 的实施计划**

当前状态：已阅读 TAP 源码、`universal_TAP.pdf` 和 `/home/renyi/structural_attack/` 的相关基建；本文记录后续实施方案。尚未实现新功能、读取 API key、调用 OpenRouter 或运行实验。

目标是在 AgentDojo 的模拟环境中，将现有单任务 jailbreak TAP 改为跨任务共享注入模板的 universal TAP。先用 OpenRouter 上的 Qwen 235B 跑通单任务执行和评估，再接入小规模搜索，最后扩展到论文的 18 个训练任务对。

论文依据：[universal_TAP.pdf](universal_TAP.pdf) §5.2.3、§6.1、附录 C.2/C.3。实验仅使用 AgentDojo 的模拟工具、虚构数据和环境状态。

1. **明确搜索对象与实验边界。**

   搜索变量为一组共享的 `prefix` 和 `suffix`。对第 i 个任务，注入内容为：

   `injection_i = prefix + separator + attacker_goal_i + separator + suffix`

   分隔规则作为固定配置保存。所有训练任务使用完全相同的 prefix/suffix，任务目标来自该任务的 `injection_task.GOAL`。模板不能逐任务重新生成，也不能根据测试结果继续修改。

   一个节点代表一个模板及其搜索历史；一个候选需要在当前训练集的所有任务对上评估。18 是论文的训练集大小，不是算法必须的固定值。通用性首先指跨任务和目标的泛化；跨模型迁移属于后续独立实验。

   保留原 `main_TAP.py` 作为历史实现和对照，新增 universal 入口。第一版保持论文的平均 judge 分数排序、移除 on-topic 剪枝；小批量初筛、ASR 优先排序等改动作为后续实验，不混入基础版本。

2. **复用已经确认存在的 AgentDojo 基建。**

   | 来源 | 复用内容 | 必要适配 |
   |---|---|---|
   | `/home/renyi/structural_attack/run_agentdojo.py` | 单任务运行、进程隔离、失败重试、任务结果路径、恢复运行、system 角色处理 | 增加显式任务对清单；按候选和重复试验隔离结果；使用明确的 OpenRouter 配置 |
   | `/home/renyi/structural_attack/ui/chateditor.py` 的 `_client()` | OpenRouter 的 base URL 和 key 解析方式 | 提取成轻量客户端，不导入 UI 服务 |
   | `/home/renyi/structural_attack/reference_attacks.py` | `BaseAttack` 注册、注入位置映射、已有注入的固定回放 | 接受冻结模板产物并按目标渲染，不在测试阶段启动搜索 |
   | `/home/renyi/structural_attack/my_attacks.py` | 注入占位符定位、YAML 字符串转义等基础处理 | 只复用注入基础设施；模板内容由 TAP 生成 |
   | `/home/renyi/structural_attack/metrics/view_one.py` | 结果读取与 ASR/utility 字段含义 | 适配选定任务清单及重复试验；单独统计缺失、失败和不完整记录 |
   | `/home/renyi/structural_attack/pyproject.toml` | 已固定的 AgentDojo 来源与兼容依赖参考 | universal 环境与原 TAP 的旧依赖分开 |

   已确认该基建使用 benchmark version `v1.2.2`，AgentDojo Git revision 为 `089ed468cf3ed0322acc66b0211f26d9d90dbf60`。这是基准数据版本和代码 revision，不能将二者混同为 Python 包版本。

   基建中的 `load_endpoint()` 会读取 vLLM endpoint 文件并覆盖环境配置。OpenRouter 路径应显式选择 provider，避免自动继承旧的本地服务地址。第一版不启用基建的 `--recap` 系统提示修改，以免引入额外实验变量。

   复制需要的基础代码时保留来源说明及适用的许可证信息；TAP 运行不依赖修改源项目，也不需要源项目的 Gmail/Calendar 账户连接。

3. **接通 OpenRouter，并处理依赖差异。**

   现有 TAP 固定 `openai==0.28.0`，参考项目使用 OpenAI SDK 1.x 风格客户端。新增 universal 专用依赖文件及独立环境，沿用参考项目的兼容版本；不要把旧 `language_models.py` 的调用方式与新 SDK 直接混用。API-only 入口不应强制加载旧 TAP 的本地模型、Torch 或 FastChat。

   OpenRouter base URL 沿用参考客户端：`https://openrouter.ai/api/v1`。凭据从用户已有 `.env` 加载，优先使用 `OPENROUTER_API_KEY`；仅在 compatible base URL 明确属于 OpenRouter 时，兼容 `OPENAI_COMPATIBLE_API_KEY`。支持显式指定 env 文件路径，不输出或复制 key 到代码、计划、日志和实验配置中。

   参考项目已使用模型 ID `qwen/qwen3-235b-a22b-2507`，暂定为本项目 Qwen 235B 的目标配置；真正运行时再验证它在当前 OpenRouter 账户下的可用性及工具调用支持。若不可用，报告具体错误，不静默切换模型。

   attacker、target、judge 使用独立配置。最小接入测试可暂时都使用该 Qwen 模型，但请求、消息历史和角色配置相互独立；这属于工程连通性配置，不视为复现论文的模型组合。

   客户端必须保留结构化 `tool_calls`、调用 ID、工具结果和 finish reason，支持 assistant 仅返回工具调用、文本为空的情况。沿用并验证参考 runner 的 system 角色适配，确保系统提示实际进入请求。

   配置请求超时、有限重试、最大 agent 步数、输出 token 限制和并发上限。限流及瞬时服务故障可以重试；认证失败或模型不存在应直接报告。生成格式错误、judge 解析失败、API 故障和攻击未成功使用不同状态。

4. **定义任务、候选与执行结果。**

   | 对象 | 必须记录的信息 |
   |---|---|
   | TaskCase | benchmark revision、suite、user_task_id、injection_task_id、注入位置策略；从框架加载合法任务、攻击目标、工具和环境 |
   | Candidate | candidate_id、parent_id、prefix、suffix、improvement、模板哈希、分支会话历史 |
   | TrialResult | candidate_id、任务 ID、repeat_id、seed、运行状态、注入是否被读到、注入后首步响应、工具调用及执行结果、完整轨迹路径、judge 分数、真实攻击成功、utility、请求和 token 统计 |
   | CandidateSummary | 训练任务覆盖数、平均 judge 分数、代理达标率、真实 ASR、utility、错误数、评估成本 |
   | TemplateArtifact | 冻结 prefix/suffix、渲染规则、训练任务清单哈希、模型与运行配置、选择依据 |

   第一版使用框架支持的注入占位符。在固定、可复现的注入位置策略下，修改环境中的相应字段；保留真实 system/user/tool 消息的角色结构。候选文本通过工具返回内容到达 agent，而不是直接作为用户指令发送。

   验证转义发生在正确的一层，特别是引号、反斜线、换行和花括号。保存原始模板、渲染后的注入文本和 agent 实际看到的工具内容，防止重复转义造成实验对象变化。

   每次候选评估使用独立环境。第一版从任务初始状态运行，避免过早引入状态恢复错误；后续可缓存注入前状态，但缓存必须同时覆盖会话、工具状态及环境，并验证与从头运行等价。

   注入位置存在不代表目标 agent 必定读取它。若完整任务运行结束而没有遇到注入，应记录 `injection_seen=false`，真实攻击结果仍按环境检查计入；首步 judge 分数不凭空生成。搜索配置明确将未读到注入的正常运行视作最低代理分，并与基础设施错误区分。

5. **把搜索反馈与真实成功拆开。**

   judge 输入包括合法用户任务、攻击目标、注入文本、注入后的第一步 assistant 响应及结构化工具调用；如果框架提供目标工具调用或检查依据，可作为评估上下文。避免把 agent 自述的“已执行”当成工具真的执行过。

   论文以首步响应的 1–10 分 judge 结果提供连续搜索信号。第一版候选得分定义为：先对每个任务的有效重复试验求平均，再对固定训练集中的任务求平均。所有候选使用同一任务清单和重复次数，API 故障不能通过减少分母使候选分数提高。

   代理达标率单独命名，例如 `proxy_pass_rate`；真实 ASR 仅使用完整执行后的环境检查。在参考项目中 `security=true` 表示攻击目标完成，不能反向理解为防御成功；适配层统一映射为 `attack_success`。`utility` 表示合法用户任务完成。

   为便于首次调试，最小版本可对每个任务完整运行，同时从轨迹中提取注入后首步响应给 judge。这样既有首步搜索信号，也能立即核对实际执行结果；成本高于只做首步探测。后续可拆成首步评估和完整 rollout 两种模式，完整 rollout 至少用于候选验证与最终测试。

   解析失败记为错误并有限重试，不能默认判成功。任务在 agent 步数上限停止属于已执行但未完成；网络、认证等基础设施错误单独统计。候选训练评估未完成时，不以缺少任务的平均分参加正常排序。

6. **接入 universal TAP 的搜索循环。**

   每个搜索深度执行以下步骤：

   - 初始化或读取各分支的模板、历史和聚合反馈；attacker 接收训练集概况，不接收测试反馈。
   - 每个存活节点生成 branching_factor 个候选，输出 `prefix`、`suffix`、`improvement`。
   - 校验字段、类型、长度与模板重复；取消原 on-topic 评分及剪枝。
   - 将候选模板应用到当前训练集的所有任务对，逐任务收集结果。
   - 计算平均 judge 分数，保留 width 个候选，并保存历史最佳候选和全部已评估记录。
   - 给存活分支反馈聚合指标、各任务表现及少量代表性失败轨迹，继续下一层。

   反馈需保留失败类型与必要工具参数，同时控制上下文长度。保留最近若干次反馈和简洁历史摘要，不能只让每个分支看见最后一个任务的结果。

   冻结模板的 `BaseAttack.attack()` 只做渲染和返回注入映射；训练搜索应由单独的 corpus-level orchestrator 执行。否则 benchmark 每调用一个任务对就重新搜索，会退化为单任务攻击。

   默认以深度或预算上限停止。论文还给出“80% 训练任务达到 0.7 阈值”，但没有充分说明与 1–10 原始分的映射。第一版禁用该提前停止条件；若后续启用，必须显式保存归一化公式与覆盖率阈值，并标明这是实现约定。单个任务得到 10 分不能结束 universal 搜索。

7. **修复会影响搜索的现有问题。**

   - `clean_attacks_and_convs()` 全部生成失败时解包报错；改为明确的空候选结果，并采用有限重试或分支终止。
   - `prune()` 的全零分回退混用了分数和下标；新实现按候选对象排序，明确空列表、少于 width、同分和缺失评估的处理。
   - 当前 tuple 排序还会以 index 作为同分次序，抵消先前 shuffle 的目的；新实现明确使用 seed 控制同分处理。
   - GPT 分支生成后调用 `update_last_message()` 会覆盖用户反馈；新 API 路径应正确追加 assistant 回复。
   - 当前输出解析截到第一个 `}`，还删除换行；改用可靠的 JSON 解析和字段校验，保留合法字符串内容。
   - 生成 None 的失败检测表达式无法报告实际失败；新路径显式统计生成失败与重试次数。
   - 旧日志只统计剪枝后保留的响应，会漏算已发生的评估；新日志在剪枝前记录每个候选、任务、重复试验的开销。

   修复优先落在 universal 新路径；需要修改共享旧模块时，维持原单任务入口的行为，并运行对应回归检查。

8. **文件组织与日志设计。**

   以下是拟新增的文件，不代表目前已经实现：

   | 文件 | 责任 |
   |---|---|
   | `main_universal_tap.py` | 搜索入口与参数解析 |
   | `universal_tap/openrouter_client.py` | OpenRouter 请求、结构化输出和重试 |
   | `universal_tap/agent_runner.py` | AgentDojo 运行与结果提取 |
   | `universal_tap/attacks.py` | 冻结模板的 AgentDojo 注入适配 |
   | `universal_tap/datasets.py` | 任务清单加载、分组和拆分检查 |
   | `universal_tap/schemas.py` | 候选、任务和评估记录 |
   | `universal_tap/prompts.py` | universal attacker 与 judge 提示 |
   | `universal_tap/search.py` | 分支生成、聚合、剪枝与停止 |
   | `universal_tap/evaluation.py` | 首步评分、真实 ASR、utility 与报告 |
   | `universal_tap/logging.py` | 本地结果、检查点、成本与错误日志 |
   | `configs/` | 单任务、小规模搜索和 18 任务配置 |
   | `data/universal/` | 明确的 train/validation/test 任务清单 |
   | `requirements-universal.txt` | 独立 API/AgentDojo 依赖 |

   第一版以本地 JSON/JSONL 为主要产物，WandB 可选。记录所有候选，不能只保留最终胜者。

   原 runner 主要按 suite/user_task/attack/injection_task 判断任务是否已经完成；在搜索场景中同一任务会被不同模板反复运行，所以不能直接复用这一跳过条件。结果标识至少包含 run_id、候选模板哈希、任务 ID、repeat_id 和执行配置指纹，配置指纹覆盖模型、provider、系统提示、benchmark revision、注入策略及采样参数。

   独立重复试验必须产生新执行，不能把缓存响应当作多次采样。候选共享的 injection-task standalone 基线可缓存，但其模型、环境和执行配置必须一致，写入时保持单写者或原子提交。

   完成标志必须同时满足结果字段完整和终止状态有效；中断产生的部分 JSON 不能视为完成。断点恢复时核对任务清单与配置指纹，防止混入另一模型或另一模板的结果。

9. **按可验收的阶段实施。**

   | 阶段 | 工作 | 验收标准 |
   |---|---|---|
   | A：OpenRouter 连通 | 配置环境、加载 `.env`、验证 Qwen 模型、完成模拟工具调用往返 | 真实 API 请求成功；system 提示、tool_calls 和 tool results 正确保留 |
   | B：单任务 IPI | 从框架选择一个明确可注入的任务对，分别跑 benign、攻击目标 standalone 和固定注入 | 日志证明注入经工具内容被读取；完整任务结束；获得环境验证的 security/utility |
   | C：单候选多任务 | 固定同一模板，先评估 2–3 个任务对 | 每个任务使用相同 prefix/suffix；环境独立；聚合分可由原始记录重算 |
   | D：最小搜索闭环 | roots=1、branching=2、width=2、depth=2，使用小训练集 | 新候选生成、评估、反馈、第二轮生成和剪枝均跑通；导出最佳模板及其来源记录 |
   | E：18 任务训练 | 固定正式训练清单，扩大树搜索配置 | 所有候选按相同清单评估；可恢复运行；真实结果与成本完整记录 |
   | F：冻结与泛化测试 | 加载最终模板，在独立任务与 Travel 域运行 | 测试不调用 attacker、不改模板；报告 ASR、utility、错误及注入可达率 |

   “跑通”意味着流程、注入路径和真实评估结果可信；不要求某一次攻击必须成功。单个成功案例也不足以证明模板具备 universal 效果。

   最小搜索先用并发 1–2、单次评估和有限步数，观察 API 限流及工具执行稳定性后再扩大。实现过程中应持续留下可复现命令及对应产物路径，而不是只报告终端打印成功。

10. **正式数据划分、规模和成本。**

    论文训练规模是 Workspace、Banking、Slack 各选 3 个用户任务与 2 个攻击目标，形成 18 个任务对；Travel 不参加训练。具体任务 ID 尚未确定，实施时需要从固定版本的框架中核实并写入 manifest，不能把任意 18 个任务称为论文原始拆分。

    保存用户任务、攻击目标以及环境实体的重叠情况。区分未见任务组合、未见用户任务、未见攻击目标和整个域留出；仅任务对不重合不足以声称所有组成部分都未见。若增加验证集进行模板选择，应明确它是本项目的额外设置，并与最终测试隔离。

    论文树参数可作为后续配置：roots=3、branching=3、width=8、depth=5。无提前停止、候选全部有效且没有去重时，每轮候选数为 9、24、24、24、24，总计 105；18 任务下对应 1,890 次候选与任务组合评估。

    每次任务评估可能包含多次模型调用和工具执行，judge 也有额外请求；不能把 1,890 视为 API 请求总数。重复试验次数还会进一步乘上去，benign 和 standalone 基线另外计费。记录实际请求数、token、耗时及 provider 返回的可用成本信息。

    若后续采用“小批训练任务初筛，再在 18 任务上复评”，应明确作为另一个搜索配置，并让候选使用相同初筛任务；最终选择仍需在统一任务集上比较。

11. **必要检查与最终交付。**

    离线检查集中在影响结论的地方：模板渲染与转义正确、全失败和同分剪枝正常、角色历史无覆盖、聚合分母正确、API 错误独立记录、重复试验不复用缓存、恢复运行不会串候选。

    AgentDojo 集成检查确认环境相互隔离、注入实际位于低信任工具内容中、首步响应定位正确，以及 judge 高分但工具执行失败时仍得到 `attack_success=false`。检查 security 字段映射与框架真实判定一致。

    最终报告同时给出无攻击 utility、攻击下 utility、真实 ASR、代理分数、完成/错误/缺失数量、注入读取比例、任务拆分及模型配置。完整实验可再按论文的 4 个独立优化种子、每个产物 6 次评估组织结果；Success@N 的“尝试”定义与重复评估的聚合方式须单独写清楚。

    预期交付物为可运行入口、依赖与配置、任务清单、冻结模板、逐任务完整轨迹、评估报告和复现命令。所有实现和实验阶段目前均待执行。

实施进度：

- [x] 阅读现有 TAP、论文及参考 AgentDojo/OpenRouter 基建。
- [x] 保存本实施计划。
- [x] 实现 `universal_tap/` 全部模块、`main_universal_tap.py` 入口、`configs/`、`data/universal/` 任务清单与 `requirements-universal.txt`（见下「代码实现状态」）。
- [~] A：OpenRouter 与 Qwen 235B 工具调用连通。代码已打通：从 `.env` 读 key、构造 OpenRouter 客户端、发真实请求；当前 `.env` 中的 key 已过期，OpenRouter 返回 401 `API key expired`，被正确判为 `FatalAPIError` 直接报告、不静默切换。更新有效 key 后即可通过验收。
- [ ] B：单任务 IPI 的完整执行与环境验证（入口 `eval-template --manifest data/universal/smoke.json`，待有效 key）。
- [ ] C：固定模板的多任务评估（`eval-template --manifest data/universal/small.json`，待有效 key）。
- [ ] D：小规模 universal TAP 搜索闭环（`search --config configs/small_search.json`，待有效 key）。
- [ ] E：18 任务训练及断点恢复（`search --config configs/train18.json`）。
- [ ] F：冻结模板的泛化测试与报告（加载 `best_template.json`，在 `data/universal/test.json` 含 Travel 域运行）。

代码实现状态（2026-11-11）：

- 离线验证已通过（无需联网）：全部模块导入、注入渲染/转义经 `str.format`+YAML 往返与 AgentDojo 实际加载路径等价（含引号/反斜线/花括号压力测试）、attacker JSON 解析保留字符串内换行与嵌套括号、18/6/7 任务清单加载且 train 与 validation/test 任务对不重合、Travel 为未见域。
- 运行方式：须在参考项目虚拟环境下运行，并把仓库加入 `PYTHONPATH`：
  `PYTHONPATH=/home/renyi/TAP /home/renyi/structural_attack/.venv/bin/python main_universal_tap.py smoke --config configs/single_task.json`
- 计划 §7 的既有问题已在新路径规避：生成失败显式计数与有限重试、剪枝按候选对象排序并用 seed 控制同分、attacker 回复正确追加且反馈单独成轮、JSON 可靠解析、剪枝前记录全部候选/任务/重复试验。
- 第一版约定：对每个任务完整 rollout 并从轨迹提取注入后首步响应给 judge；未读到注入的正常运行记 `injection_seen=false` 且代理分取最低；禁用 0.7 提前停止；并发 1、repeats=1。
