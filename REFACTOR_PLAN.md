# 评测模块重构方案（eval/）

> 状态：**方案已定稿，尚未开始实现**
> 用途：与评测脚本作者对齐，重点是第 4 节「会改变评测结果的改动」
> 日期：2026-08-13

---

## 1. 概况

把 `test/` 里 8 个各自为战的评测脚本，重构成一个可以交给其他同事在自有算力上无人值守跑完整矩阵的评测框架。

**评测矩阵规模**

| 维度 | 数量 | 说明 |
| --- | --- | --- |
| 模型 | 15 | 10 个开源（本地权重）+ 5 个闭源（API） |
| 任务族 | 20 | stack_cubes / tea / wash / … 每族一套 QA 数据 |
| 每族运行次数 | 9 | 8 个任务类型，其中 trajectory 拆成 2D / 3D |
| 每族题量 | 2,700 题 | 对应 2,450 次模型调用（time EQA 按视频分组，300 题合并成 50 次调用） |
| **总调用量** | **735,000** | 本地 490,000 + API 245,000 |

**模型分类**

```
开源（本地权重，10 个）
  Qwen3-VL-8B-Instruct           Qwen3-VL-30B-A3B-Instruct     Qwen3-VL-235B-A22B-Instruct
  InternVL3.5-8B                 InternVL3.5-30B-A3B           SenseNova-SI-1.1-InternVL3-2B
  RynnBrain-2B                   RynnBrain1.1-2B
  Cosmos3-Edge-2B                Cosmos-Reason2-2B

闭源（API，5 个）
  qwen3.8-max    Seed2.0-Lite    GLM-5V-Turbo    Gemini-3.1-Pro    Gemini-3.5-Flash
```

**每个任务的主指标**（最终报表里每格一个数）

| 任务 | 主指标 |
| --- | --- |
| understanding / left_right / planning / planning_2 / step_order / image_in_video | `accuracy` |
| trajectory_2D / trajectory_3D | `score`（Hausdorff / Fréchet / Chamfer 三距离映射后的 0–100 综合分） |
| time | `mean_tIoU` |

---

## 2. 为什么必须重构

不是「加个批量脚本」就能解决的，现有设计有三处硬伤会让 15 × 20 的规模跑不完。

### 2.1 本地模型每个任务重载一次权重

现状是一个脚本 = 一个进程 = 一次 `from_pretrained`。10 个本地模型 × 9 次运行 = **90 次权重加载**，30B 级别每次几分钟。

改成 model-major 调度后降到 **10 次**。

### 2.2 完全串行，规模上跑不完

735,000 次调用，串行按 5 秒/次算需要 **1,020 小时（约 42 天）**。不加并发和分片不可能完成。

### 2.3 checkpoint 每答一题重写整个结果文件

`trajectory_qa_3d` 每题约 50 KB，一个 300 题任务累计写入 **2.2 GB**；整个矩阵约 **0.8 TB 无谓 IO**。改成 JSONL 追加后归零。

### 2.4 代码层面

对 8 个脚本做 AST 比对的结果：

| 指标 | 数值 |
| --- | --- |
| 总行数 | 4,335 行 |
| 逐字重复的样板 | 约 1,600 行 |
| 在 7 个脚本里完全一致的函数 | 6 个（`load_json` / `save_json` / `checkpoint_path_for` / `strip_json_fence` / `load_existing_results` / `is_finished`） |
| 定义了但从未被调用的死代码 | 13 处（迁移到 `vlm_api.call_vlm` 之前的遗留 `call_glm` / `extract_message_text` 等） |

8 个脚本真正不同的只有 **4 个钩子**：取媒体、组 prompt、解析输出、算分。其余全部可共享。

### 2.5 其他已知问题

- `planning` 和 `planning_2` 抢同一个 config key（`tasks.planning`），且两者输入格式不兼容（视频 vs 单帧图）
- QA JSON 里存在三种路径风格：绝对路径 `/home/llm/...`、相对路径、Windows 反斜杠
- 没有任何跨任务、跨模型的汇总工具
- `__pycache__` 被误提交进 git

---

## 3. 新的目录结构

**`test/` 永久冻结，只读，作为回归比对基线。所有新代码写在 `eval/`。**

```
benchmark/
├── data/                        生成流水线，本次不动
├── test/                        【冻结】原 8 个脚本，仅用于回归对照
└── eval/                        【新建】
    ├── robochrono/              框架代码
    │   ├── cli.py               preflight | estimate | run | report | pack
    │   ├── matrix.py            models × datasets 展开成 run 列表，含稀疏规则与分片
    │   ├── engine.py            单个 run 的执行循环（原 main() 的公共骨架）
    │   ├── scheduler.py         model-major 调度、GPU worker 池、并发限流、熔断
    │   ├── store.py             JSONL 追加存储、断点恢复、兼容格式导出
    │   ├── mediapath.py         统一路径解析（三种风格 + dataset_root 变量）
    │   ├── preflight.py         环境 / 数据 / 密钥 / 显存自检
    │   ├── report.py            汇总成对比表
    │   ├── vlm_api.py           从 test/ 搬运，仅改多 GPU 相关部分
    │   └── tasks/
    │       ├── base.py          Task 协议
    │       ├── choice.py        6 个选择题任务共用
    │       ├── time_eqa.py      分组提问 + tIoU
    │       └── trajectory.py    距离指标 + 越界重试
    ├── configs/
    │   ├── models.yaml          15 个模型的接入配置
    │   ├── datasets.yaml        任务族 → 数据路径映射
    │   └── plan.yaml            矩阵定义、稀疏规则、并发、预算
    ├── datasets/                【gitignore】
    │   ├── QA/{planning,understanding}/stack_cubes/
    │   ├── json/stack_cubes/
    │   └── README-HF.md         数据集 license（CC-BY-NC-4.0）
    ├── models/                  【gitignore】本地权重
    ├── results/                 【gitignore】运行产出
    ├── requirements.txt
    ├── run.sh
    └── RUNBOOK.md               给同事的操作手册
```

`datasets/` 内部保持 HuggingFace 数据集的原结构，以后补另外 19 族时直接往里放，配置里只有 `dataset_root: ./datasets` 一行需要改。

---

## 4. ⚠️ 会改变评测结果的改动

**这一节是需要和评测脚本作者逐条确认的部分。** 每条都有编号，方便讨论时引用。

| 编号 | 改动 | 是否改变主指标 |
| --- | --- | --- |
| BC-01 | 推理参数统一 | **是** |
| BC-02 | 剥离 thinking 块 + JSON 兜底解析 | **是** |
| BC-03 | 新增解析失败率指标 | 否（仅新增字段） |
| BC-04 | 结果文件分层存储 | 否（仅改格式） |
| BC-05 | `--limit` 语义拆分 | 否（仅改 CLI） |
| BC-06 | `planning` / `planning_2` 配置分离 | 否（但改变默认输入文件） |
| BC-07 | SenseNova 模型命名更正 | 否（仅改配置名） |
| BC-08 | QA JSON 路径规范化 | 否（但会改写数据文件） |
| BC-09 | 抽帧策略统一 | **是** —— 但**方案未定，暂时挂起**，见下 |
| BC-10 | 修复 `text: null` 选项导致的误判 | **是** —— 新发现的评分 bug，默认关闭待确认 |

### BC-10 · `text: null` 选项导致「none」被误判为选项 A

阶段 1 的解析等价性测试挖出来的**真实评分 bug**。

`left_right` 和 `image_in_video` 的选项是图片，`text` 字段全为 `null`。
冻结脚本取选项文本时写的是：

```python
option_text = normalize_text(str(option.get("text", "")))   # None -> "None" -> "none"
```

`str(None)` 得到字符串 `"None"`，归一化后成为 `"none"`。于是 `extract_choice`
的文本匹配环节里，**任何包含 "none" 的模型输出都会匹配上第一个选项**：

```
模型输出 "none of these"  →  判为 A  →  实际应为 E（"All other options are wrong."）
```

而这两个任务恰好都设了 none-option，「none of these」正是模型表达该选项最自然的
说法之一。也就是说，**模型选对了 none-option，却被系统性记成选了 A**。

打开 BC-10 后，`text` 为 `null` 的选项不参与文本匹配。

**影响范围：** 仅 `left_right`（600 题/族）与 `image_in_video`（300 题/族）。
具体有多少题受影响取决于各模型实际输出中 "none" 的出现频率，
可以用 replay 在已有结果上离线统计，不需要重跑模型。

默认关闭以保证 replay 回归通过；是否启用请团队确认。

### BC-09 · 抽帧策略统一 —— 已挂起

Time EQA 的抽帧策略是本次重构中唯一悬而未决的破坏性变更。调查已完成，
方案待团队确认后接续。完整记录见 **[抽帧问题调查](eval/docs/frame_sampling_investigation.md)**，
官方推荐值调研见 **[官方推荐配置调研](eval/docs/official_config_survey.md)**。

一句话概括：15 个模型在 Time EQA 上看到的帧数相差 2 倍以上，其中 6 个不可知；
当前配置下 InternVL 系还会直接 OOM 跑不完。已定的 fps=1 / fps=2 双档方案
在长 episode 的任务族上对 InternVL 系不可行，需要补一个上限规则。

**挂起期间仍要做的三件事**（与协议选择无关，不阻塞）：

1. `num_segments` 改为**逐视频运行时计算**（`round(时长 × fps)`），不再是静态配置值
2. 配置 schema 用 `frames: {mode: fps|uniform|native, value: N}` 并**强制互斥** ——
   同时给出多个抽帧参数直接报错，杜绝当前 `video_sample_fps` 静默架空 `num_segments` 这种情况
3. 结果文件记录**实际使用的帧数**，以及 API 响应的 `usage`
   （现在 `raw, model_text = call_vlm(...)` 拿到原始响应后直接丢弃）

这样等协议定下来，改一行配置即可生效，不用返工。其余八个任务不受影响 ——
它们要么用 4 秒切片（帧数在 4~8 之间，差异不足以影响结论），要么只送静态图。

---

### BC-01 · 推理参数统一 —— 会改变分数

**现状不一致：**

| 参数 | 现状 |
| --- | --- |
| `temperature` | 名单内的 14 个 provider 都是 `0.0`（配置里 `kimi` 是 `1`，但 Kimi 不在测试名单内） |
| `thinking` | 只有 `glm` 配了 `disabled` 并实际发送，其余 provider 不发送 |
| `max_new_tokens` | 从 `256`（`local_internvl_8b`、全局 defaults）到 `1024`（其余本地 provider）不等 |

**改为：** 全部锁定 `temperature=0.0`、`thinking=disabled`、`max_new_tokens=1024`，并把每个 run 的实际参数完整写进结果文件头。

**影响范围：**

- **`max_new_tokens` 从 256 提到 1024 是唯一实际生效的改动**，影响 `InternVL3.5-8B`、`InternVL3.5-30B-A3B`、`SenseNova-SI-1.1-InternVL3-2B` 三个走 `local_internvl` 的模型。原本被截断的长输出（尤其 trajectory 要吐 10~20 个坐标点）现在能完整返回，**这三个模型的 trajectory 分数大概率上升**。
- `temperature` 统一对名单内的模型是空操作（本来就都是 0）。配置里 `kimi` 的 `temperature=1` 仍会被改掉，但 Kimi 不在 15 个模型名单内，不影响本轮结果。
- `thinking=disabled` 对未显式发送该字段的 provider 是新增行为，具体影响取决于各家服务端默认值，需要在首跑时观察 `parse_recovered` 的比例（见 BC-02）。

**理由：** 跨 15 个模型横向比较必须统一口径，否则分数差异里混着采样参数的差异。

---

### BC-02 · 剥离 thinking 块 + JSON 兜底解析 —— 会改变分数

**问题：** 即使设了 `thinking=disabled`，部分模型（GLM-5V-Turbo 已知，推理型模型普遍如此）仍会把思考过程混在 `content` 里返回：

```
<think>让我看看这段视频……</think>{"choice":"B","reason":"..."}
```

现有的 `strip_json_fence` 只剥 markdown 代码围栏（` ```json `），剥不掉 `<think>` 标签 → `json.loads` 失败 → 走 `extract_choice` 的正则兜底 → 经常兜不住 → `pred_choice=None` → **按答错计分**。

**改为：** 在解析前统一加一道预处理

1. 剥离 `<think>` / `<thinking>` / `<reasoning>` 标签块
2. 剥离 markdown 代码围栏（保持现有行为）
3. 兜底：从文本中提取最后一个合法的 JSON 对象

**影响范围：** 原本因格式问题被判错的题，现在可能被正确解析 → **受影响模型的分数会上升**。新增 `parse_recovered` 布尔字段，标记哪些题是靠兜底救回来的，方便量化这条改动的影响面。

**理由：** 现状会系统性惩罚推理型模型 —— 它答对了，但因为输出格式被算成答错。这不是我们想测的能力。

---

### BC-03 · 新增解析失败率指标 —— 不改变现有指标

**现状：** 模型输出格式不对（`pred_choice=None`）和模型答错，混在同一个 `accuracy` 里，无法区分。

**改为：** **保持现有 `accuracy` 定义完全不变**（未答仍计为错），额外增加两个字段：

| 新字段 | 含义 |
| --- | --- |
| `parse_failure_rate` | 未能解析出有效答案的题目占比 |
| `accuracy_answered` | 只在成功解析的题目上计算的准确率 |

**理由：** 报表主指标仍用原口径，保证与历史结果可比；新字段供分析时判断一个低分是「不会做」还是「不会按格式答」。

---

### BC-04 · 结果文件分层存储 —— 不改变指标，改变格式

**现状：** `result = {**item, ...}` 把整条原始 item 复制进结果。全矩阵最终体积约 10 GB，且每答一题重写整个文件（见 2.3）。

**改为三层：**

| 层 | 内容 | 全矩阵体积 | 默认生成 | 去向 |
| --- | --- | --- | --- | --- |
| 1 汇总 | 每个 (模型, 族, 任务) 一行：主指标、题数、错误数、解析失败率、耗时 | 几百 KB | 是 | 回传 |
| 2 逐题精简 | id / 预测 / 真值 / 对错 / 解析状态 / **原始模型输出** / 耗时 | 约 370 MB | 是 | 回传 |
| 3 完整记录 | 现有格式：原始 item 全字段 + prompt + 输出 | 约 10 GB | 否，需 `--keep-full` | 留服务器 |

运行时逐行追加 JSONL；`report --export` 时可以还原成与现有结果文件**完全同构**的 JSON，现有分析脚本零改动。

**第 2 层保留原始模型输出是关键设计** —— 有了它，指标口径将来要调整（新增指标、改 trajectory 容差等）可以**离线重算，不必重新调模型**。这也是 `replay` 回归的数据来源。

---

### BC-05 · `--limit` 语义拆分 —— 不改变指标

**现状：** `time_eqa_glm_test_multi.py` 的 `--limit` 限制的是**视频数**，其余脚本限制的是**题数**。批量运行时这个不一致会让「每族抽 20 题冒烟」这类需求出错。

**改为：** `--limit-items`（题数）与 `--limit-groups`（分组数）两个显式参数；`--limit` 保留为别名，映射到各任务原本的语义，旧命令行为不变。

---

### BC-06 · `planning` / `planning_2` 配置分离 —— 会改变默认输入文件

**现状：** 两个脚本都调 `task_config(args.config, "planning")`，而 `config_test.json` 的 `tasks` 里只有 `planning` 一节，当前指向的是 `planning_2_vqa.json`。

后果：

- 跑 `planning_2_glm_test.py` 不带 `--input` —— 正常
- 跑 `planning_glm_test.py` 不带 `--input` —— 读到 planning_2 的数据，因为里面只有 `image_*` 字段而没有 `clip_path`，`video_paths_for_item` 抛 `KeyError` 直接崩溃
- 两者不带 `--output` 时会写到同一个文件，且断点续跑机制不校验结果来源，会把另一个任务的结果当成「已完成」跳过

**改为：** 配置里 `planning` 与 `planning_2` 各占一节，各自指向正确的输入输出；结果文件头记录 `task` 字段，断点恢复时校验，不匹配则视为无效。

---

### BC-07 · SenseNova 模型命名更正 —— 仅改配置名

repo 里三处写作 `SenseNova-SI-1.1-InternVL32B`（provider 名、权重路径、README 表格），少了一个连字符，读起来像 **32B**。实际模型是 HuggingFace 上的 `sensenova/SenseNova-SI-1.1-InternVL3-2B`，是 **InternVL3 架构的 2B**，权重 4.18 GB。

照现有 README 去找 32B 权重会找不到，或者按 32B 规划显存（约 64 GB）造成资源浪费。三处统一更正为 `SenseNova-SI-1.1-InternVL3-2B`。

---

### BC-08 · QA JSON 路径规范化 —— 会改写数据文件

现有 9 个 VQA JSON 有三种互不兼容的路径风格：

| 文件 | 路径风格 | 现状 |
| --- | --- | --- |
| `planning_vqa` / `planning_2_vqa` / `trajectory_qa_2d` / `trajectory_qa_3d` | 相对路径 | 需进程 cwd 恰在 JSON 所在目录 |
| `step_order_vqa` | 相对路径 | 可用（该脚本有 `resolve_media_path` 兜底） |
| `time_vqa` / `understanding_vqa` / `left_right_vqa` | 绝对路径 `/home/llm/yyywv/...` | 失效，本机无此路径且无兜底 |
| `image_in_video_vqa` | 相对路径 + **Windows 反斜杠** | 失效，Linux 下 `Path()` 不拆反斜杠 |

**改为：** 统一的 `mediapath` 解析层，运行时按 `dataset_root` 解析，三种风格都能吃。数据文件本身做一次性规范化，**原文件备份为 `.orig`**。

---

## 5. 保证不变的部分

**评分逻辑原样搬运，不重写、不「顺手优化」。** 具体是这些函数从现有脚本逐字复制到 `eval/robochrono/tasks/`：

- `build_prompt`（7 个各不相同 —— 任务语义）
- `score_prediction`（6 种实现 —— 指标定义）
- `summarize`（5 种实现 —— 指标定义）

**三道保险：**

1. **`replay` provider** —— 新增一个假 provider，从已有结果文件里读回 `model_output` 当作模型响应。用同一份数据让新旧两套代码算分，**逐字节 diff `summary`**，相同才算通过。这个回归不需要 GPU、不需要 API key、覆盖全部 9 个任务。
2. **`test/` 永久冻结** —— 任何时候都能拿原脚本跑对照。
3. **BC-01 / BC-02 单独开关** —— 这两条是唯一会改变分数的改动，做成配置项。关掉它们时，新框架的输出必须与旧脚本逐字节一致；这就是回归测试的判据。

---

## 6. 架构

### 6.1 Task 协议

8 个脚本的差异收敛到 4 个钩子：

```python
class Task(Protocol):
    name: str
    unit: Literal["item", "video_group"]        # time_eqa 用 video_group
    def media_parts(self, item) -> list[Part]   # 统一的 text / image / video
    def build_prompt(self, item) -> str         # ← 原样搬运
    def parse(self, text, item) -> Prediction   # ← 原样搬运（+ BC-02 预处理）
    def score(self, item, pred) -> dict         # ← 原样搬运
    def summarize(self, rows, elapsed) -> dict  # ← 原样搬运（+ BC-03 新字段）
    def retry_prompt(self, ...) -> str | None   # 仅 trajectory 用
```

约 1,600 行重复样板收敛成约 250 行共享代码。

### 6.2 调度

**本地模型 —— GPU worker 池**

```
主进程 (scheduler)
  ├─ worker-0  CUDA_VISIBLE_DEVICES=0  ┐
  ├─ worker-1  CUDA_VISIBLE_DEVICES=1  │ 每个 worker 加载当前模型的一份副本，
  ├─ ...                               │ 从共享队列领 (family, task, item) 任务
  └─ worker-N  CUDA_VISIBLE_DEVICES=N  ┘
```

同一时刻全部 GPU 跑**同一个模型**，把 20 族 × 9 任务的题目摊给各 worker；该模型全部做完再整体换下一个。既只加载一次权重，又吃满所有卡。

需要改动 `vlm_api.py`：现在 `local_internvl` 硬编码 `device_map={"":0}`、`local_qwen` 用 `device_map="auto"`，都要改成由 worker 的 `CUDA_VISIBLE_DEVICES` 决定。

**API 模型** —— 线程池 + 每 provider 独立并发度 + 令牌桶限流 + 成本累计。

**多机分片** —— `--shard i/N`，按 `(model, family, task)` 稳定哈希，机器间互不重叠；机器内再由 GPU worker 池二次拆分。结果各自成文件，最后 `report --merge` 合并。

### 6.3 推理后端：本期不引入 vLLM

**已决定：纯 transformers 交付。** 理由：

- **自定义模型不支持** —— RynnBrain 系列和 Cosmos3-Edge 靠 `trust_remote_code` 加载自定义代码，vLLM 大概率跑不了，无论如何都得混合部署
- **换后端会破坏可比性** —— 现有 `vlm_api` 是自己做视觉预处理的（`internvl_frame_indices()` 决定抽哪几帧、`dynamic_preprocess_internvl()` 切图块）。改走 vLLM 服务后由 vLLM 自己采样，帧数、位置、切图策略全都不同，**同一模型同一道题会得到不同分数**

代价是 30B / 235B 的吞吐只有 vLLM 的 1/5 到 1/10。留作后续可选增强：backend 做成 per-model 配置项，默认 transformers，开启时在结果文件头记录 `backend` 字段并在报告里加脚注。

### 6.4 容错

| 机制 | 行为 |
| --- | --- |
| 数据缺失 | preflight 在开跑前列出所有将被跳过的格子供确认，报告里标 `skipped` |
| 单题失败 | 记为一条 error 记录，不中断 |
| 熔断 | 单个 run 连续 20 题失败（服务挂了 / key 失效 / OOM）则中止该 run 标记 `aborted`，不影响其他 run |
| 断点续跑 | 读已有 JSONL 的 id 集合，跳过已完成项，不需要全量反序列化 |
| 预算闸门 | API provider 累计用量触顶即停该 provider，报告标注「预算中止」 |

---

## 7. 运行与交付

**交付方式：** 推到现有 `yyywv/benchmark` 仓库的 `feature/eval-harness` 分支，走 PR。同事 `git clone` 后拿到 `data/`（生成器）+ `test/`（冻结基线）+ `eval/`（新框架）。`datasets/`、`models/`、`results/` 全部 gitignore，由同事自行准备。

**运行：**

```bash
# 0. 环境
conda env create -f eval/environment.yml && conda activate robochrono

# 1. 自检（数据是否齐、媒体能否打开、key 是否配好、显存够不够）
python -m robochrono preflight --config eval/configs/plan.yaml

# 2. 估算 API 成本（不调模型，只统计调用数与媒体字节）
python -m robochrono estimate --config eval/configs/plan.yaml

# 3. 跑
./eval/run.sh --shard 1/2 --only local     # A 机（多 GPU）
./eval/run.sh --shard 2/2 --only local     # B 机
./eval/run.sh --only api --concurrency 16  # 任意机器，不需要 GPU

# 4. 汇总 + 打包回传（scp）
python -m robochrono report --merge results-*/
python -m robochrono pack -o robochrono-results.tar.zst
```

`pack` 默认只含第 1、2 层（约 370 MB），`--full` 才带第 3 层。

**RUNBOOK 里会额外写清楚：** 权重下载的两条路径（直连 HF / `HF_ENDPOINT=https://hf-mirror.com`），以及这次踩到的坑 —— 代理会拖慢国内镜像、`hf download --include` 的通配符会被 shell 展开、镜像并发过高会 429 限流。另外 `nvidia/Cosmos-Reason2-2B` 是 gated 模型，需要先在 HuggingFace 上接受 NVIDIA 许可并配 token。

---

## 8. 验证计划

**验证配置：** `SenseNova-SI-1.1-InternVL3-2B`（15 个模型里最小的开源模型，权重 4.18 GB）× `stack_cubes` 一个任务族 × 全部 9 个任务 × 8 × RTX 4090（24 GB/卡）。

选它还有一个原因：它走的 `local_internvl` 是四条本地 adapter 里**最复杂的一条**（自己用 decord 抽帧、dynamic tiling 切图、调 `model.chat()` 而不是标准 `generate()`，且硬编码了单卡 `device_map`），最容易在多 GPU 改造中出问题。

**验收标准：**

1. `replay` 回归：9 个任务的 summary 与旧脚本逐字节相同（关闭 BC-01 / BC-02 的前提下）
2. 8 张卡确实并行工作
3. 整轮只加载一次权重
4. 中断后重跑能正确跳过已完成项
5. 报告能生成完整的 9 格对比表

**这个验证覆盖不到、需要在同事首跑时才能确认的部分：**

| 缺口 | 补偿手段 |
| --- | --- |
| 另外三条本地 adapter（`local_qwen` / `local_transformers_vlm` / `local_cosmos_transformers`） | preflight 对它们做更严格的加载前检查；RUNBOOK 标注为「未经真实验证」 |
| 30B / 235B 的多卡切分 | 4090 显存放不下，交付 `--dry-run --check-vram` 供首跑前自检 |
| 其他 19 个任务族的数据形态差异 | loader 对字段做宽松匹配，preflight 报告检测到的数据版本特征 |
| API 路径（并发 / 限流 / 预算 / 成本统计） | 用一个 Qwen key 跑 `--limit 2` 做真实验证 |
| 调度器 / 分片 / 断点 / 报告合并 | `mock` provider，几秒内跑完一个假的 15 × 20 × 9 矩阵 |

**关键点：评分正确性完全不依赖真实模型** —— `replay` 回归就能覆盖全部 9 个任务。真实模型那一遍主要验证工程链路。

---

## 9. 落地顺序

| 阶段 | 内容 | 门禁 |
| --- | --- | --- |
| 0 ✅ | 建 `eval/` 骨架、迁数据、建 conda 环境、拉验证模型、路径规范化（BC-08） | 全部 9 个任务的媒体在本机可解析 |
| 1 | 抽公共层 + Task 协议 + BC-01～BC-07 + BC-09 的三项管道改造 | **`replay` 回归：关闭 BC-01/02 时新旧 summary 逐字节相同** |
| 2 | JSONL 分层存储 + matrix + `estimate` 成本表 | stack_cubes × 1 本地 + 1 API 模型全 9 任务跑通 |
| 3 | GPU worker 池 + 两级分片 + preflight 显存实测 | 8 卡吃满，单模型全族跑通 |
| 4 | report / pack / `run.sh` / RUNBOOK / `requirements.txt` | 同事零沟通成本上手 |

阶段 1 结束前，`test/` 下的旧命令行为完全不变，随时可回退。

### 阶段 0 完成记录（2026-08-13）

- 分支 `feature/eval-harness` 已建。
- `eval/` 骨架就位；`QA/`、`json/` 迁入 `eval/datasets/`（同文件系统 rename，未复制 4.4 GB）。
- 验证模型 `SenseNova-SI-1.1-InternVL3-2B` 下载完成：4.19 GB / 20 文件 / 3.2 分钟 / 21.98 MB/s / 零失败。
- BC-08 路径规范化已执行：**22,300 条路径重写**，原文件备份为 `*.orig`。
  另有 9,600 条溯源路径与 5,500 条未发布媒体（`time_joined_videos/`、原始 LeRobot 视频）原样保留 —— 经确认均非评测关键字段。
- **验收通过**：`eval/tools/check_media.py` 复用 `test/` 下九个脚本自身的取媒体函数，
  对 2,700 道题的 8,450 个媒体引用逐一检查，**零缺失、零报错**。

| task | items | media refs | missing | errors |
| --- | ---: | ---: | ---: | ---: |
| time | 300 | 300 | 0 | 0 |
| understanding | 300 | 300 | 0 | 0 |
| left_right | 600 | 3,600 | 0 | 0 |
| image_in_video | 300 | 1,800 | 0 | 0 |
| planning | 250 | 250 | 0 | 0 |
| planning_2 | 300 | 300 | 0 | 0 |
| step_order | 50 | 100 | 0 | 0 |
| trajectory_2D | 300 | 900 | 0 | 0 |
| trajectory_3D | 300 | 900 | 0 | 0 |

- **端到端冒烟通过**：`eval/tools/smoke_all.sh` 用冻结的 `test/` 脚本 + 真实权重
  （SenseNova-SI-1.1-InternVL3-2B，单卡 4090）把九个任务各跑一遍，**9/9 通过、零 error**。

```
task             status   answered  errors
time             OK       6         0
understanding    OK       1         0
left_right       OK       1         0
image_in_video   OK       1         0
planning         OK       1         0
planning_2       OK       1         0
step_order       OK       1         0
trajectory_2D    OK       1         0
trajectory_3D    OK       1         0
```

- conda 环境 `robochrono` 建好：torch 2.6.0+cu124（cuda=True，8 GPU）、transformers 4.56.2、
  decord、opencv-headless。见 `eval/setup_env.sh`。
- 模型实测：2.09B 参数、bf16、**单卡显存占用 4.18 GB** —— 24 GB 的 4090 上开 8 副本毫无压力。

注：`test/` 目录未作任何修改（含误提交的 `__pycache__`，按「永久冻结」原则原样保留）。

#### 阶段 0 踩到的三个坑（都要写进 RUNBOOK）

**① transformers 版本下限是 4.56。** 冻结代码在 `load_local_internvl()` 里用
`AutoModel.from_pretrained(..., dtype=...)`，而 `dtype=` 是 4.56 才引入的
`torch_dtype` 新名字。在 4.51.3 上这个 kwarg 会被透传给模型构造函数，报
`InternVLChatModel.__init__() got an unexpected keyword argument 'dtype'`。
已锁 4.56.2。**原始跑分用的 transformers 版本未知** —— 如果你们知道，告诉我改锁那个版本。

**② `auto_map` 跨仓库引用遇上带点号的 repo id 会崩。** SenseNova 的 `config.json` 写的是
`sensenova/SenseNova-SI-1.1-InternVL3-2B--modeling_internvl_chat.InternVLChatModel`。
这个 `repo_id--module.Class` 语法让 transformers 按 **HF repo id** 建动态模块目录，
而 id 里的 `1.1` 含点号，Python 当成包分隔符：

```
ModuleNotFoundError: No module named 'transformers_modules.sensenova.SenseNova-SI-1'
```

改本地目录名无效（模块路径来自 `auto_map` 不是传入路径）。修法是去掉 `repo_id--` 前缀，
已做成工具 `eval/tools/patch_local_model.py`（默认 dry-run，`--apply` 才写，备份 `.orig`）。
本地目录同时改名为 `SenseNova-SI-1_1-InternVL3-2B` 以避免其它带点号路径的问题。

**③ Time EQA 会 OOM，而且抽帧策略是基准参数不是运维旋钮。** 见下面「新增开放问题」。

---

## 10. 决策记录

| # | 决策 | 结论 |
| --- | --- | --- |
| 1 | 矩阵规模 | 20 个任务族（场景），每族跑适用的 9 次运行 |
| 2 | 算力形态 | 本地 GPU 跑开源权重 + API 调闭源 + 多机分片；同事服务器约 8 × H100 |
| 3 | 结果格式 | 内部 JSONL，导出时转成现有格式 |
| 4 | 最小验证模型 | SenseNova-SI-1.1-InternVL3-2B（4.18 GB） |
| 5 | 任务计数 | 8 个任务类型 = 9 次运行（trajectory 拆 2D / 3D） |
| 6 | vLLM / SGLang | 本期不引入，纯 transformers 交付 |
| 7 | 解析失败与答错分离 | 采纳（BC-03） |
| 8 | thinking 剥离 | 采纳（BC-02） |
| 9 | 推理参数统一 | 采纳（BC-01） |
| 10 | `--limit` 语义 | 拆分（BC-05） |
| 11 | `time` 主指标 | `mean_tIoU` |
| 12 | `test/` 定位 | 永久冻结 |
| 13 | 交付方式 | 现有仓库开分支走 PR |
| 14 | 结果回传 | scp |
| 15 | 交付节奏 | 不急，一次性交付完整版 |
| 16 | 验证范围 | 一个模型 + 一个任务族 |
| 17 | 权重准备 | 同事自行处理 |

---

## 11. 未决与风险

### 新增开放问题：Time EQA 的抽帧策略（阶段 0 发现，需要拍板）

这是阶段 0 挖出来的最重要的问题，**会直接决定 Time EQA 的分数是否有意义**。

`internvl_frame_indices()` 里 `video_sample_fps > 0` 时直接 return，**`num_segments` 成为死参数**：

```python
if sample_fps > 0:
    frame_count = max(1, int(duration * sample_fps))
    return [...]          # ← num_segments 永远走不到
```

当前 config 里 InternVL 系 provider 是 `num_segments: 4` + `video_sample_fps: 1.0`。
Time EQA 送的是**整段 episode**（stack_cubes 每段 73.2 秒 @ 20 fps），于是实际抽 **73 帧**：

| 抽帧数 | 结果 |
| --- | --- |
| 73 帧（`video_sample_fps=1.0`，当前配置） | 24 GB 卡上 **CUDA OOM**（单次分配 8.84 GiB） |
| 8 帧（`video_sample_fps=0` + `num_segments=8`） | 跑通，但 **tIoU 全为 0**，起止时间平均偏差 38.6 / 47.4 秒 |

8 帧摊在 73 秒上，相当于每 9 秒一帧 —— 模型不可能把动作定位到秒级。这不是模型能力问题，
是输入信息量不足。**换句话说，抽帧数直接决定 Time EQA 测的是什么。**

更麻烦的是**跨模型不一致**：InternVL 系走 `internvl_frame_indices()`，Qwen 系走
`qwen_vl_utils` 自己的采样逻辑，API 模型则是各家服务端自己决定。当前配置下不同模型
看到的帧数根本不同，Time EQA 的横向对比是被混淆的。

**需要决定：**

1. Time EQA 统一用多少帧？（要在「信息量足够定位」和「显存放得下」之间取平衡）
2. 能不能对所有模型强制统一帧数？API 模型做不到的话，是否在报告里对 Time EQA 加限定说明？
3. 30B / 235B 模型 + 长视频的显存怎么算？preflight 需要按 (模型, 任务) 预估峰值显存，
   而不是运行时才发现 OOM。

在此之前，`eval/configs/config_smoke.json` 用的 8 帧只是为了让链路跑通，**不是评测配置**。

**待补充信息**

- 另外 19 个任务族的 QA 数据目前不在手上。HuggingFace 上的 `yyyyywv/ROBOCHRONO` 只有 7 个族（airpods / express / gift_inhand / stack_cubes / tea / tea2 / wash，共约 61 GB），本地只下了 stack_cubes（4.40 GB）。全量铺开前需要确认其余数据的来源与生成器版本。

**部署层面的阻塞项（新增，2026-08-13）**

调研官方推荐配置时发现，各模型要求的 transformers 版本互不兼容：

```
RynnBrain-2B      官方要求 transformers==4.57.1
RynnBrain1.1-2B   官方要求 transformers==5.2.0
InternVL 系        4.56.2 实测可用
冻结代码下限        ≥4.56（因为用了 dtype= 而非 torch_dtype=）
```

**单一 conda 环境无法同时满足。** 要么放弃「遵循官方版本」，要么按模型分环境部署 ——
后者会把交付形态从「一个环境」变成「一套环境矩阵」，同事的部署成本明显上升。
这一条需要在阶段 3 之前定下来，因为它决定 GPU worker 池怎么组织。

**模型定位存疑**

`Cosmos3-Edge-2B` 官方定位是 **omnimodal world model**（输出文本、图像、视频、动作），
`library_name: cosmos`、tags 含 `diffusers`，README 中**没有任何问答式推理示例**，
全部是视频生成。我们用 `AutoModelForImageTextToText` 加载属于非官方用法，
它能否稳定产出可解析的选择题答案**未经验证**。建议在正式跑之前先单独验证，
否则可能白跑 20 个族。

**已知风险**

| 风险 | 说明 |
| --- | --- |
| 任务族数据高度异构 | stack_cubes 是 20 fps / 45–81 秒，tea 是 **25 fps / 119.8 秒**。只抽查了两族，其余 18 族的时长与帧率分布未知。任何按 fps 定的策略都会因族而异，preflight 必须先扫一遍全部族的媒体特征。 |
| 数据与生成器版本不一致 | 手上的 `time_vqa.json` 含 `time_view` 字段且实际是**单视角**（`view_order: ['left_eye']`），而 repo 里当前的 `data/time_unders_workflow.py` 根本没有这个字段。说明这批 QA 是比 repo 更新的一版生成器产出的。README 描述的「多视角拼接后送入」与实际数据不符，解读 Time EQA 结果时需注意。 |
| 其他族的 trajectory 首帧可能抽错 | 最近的提交 `daee5cc` 把 trajectory 首帧抽取从硬编码 `trajectory_fps=20.0` 改为读视频实际帧率。stack_cubes 实测就是 20 fps，已下载的 1,200 张首帧正确；但其他族若不是 20 fps，旧代码产出的 QA 数据首帧是错位的，2D 真值会叠在时间上对不齐的图上。用之前需要先验帧率。 |
| API 成本 | 245,000 次带视频 / 图片的付费调用，单次媒体可达数 MB。必须先跑 `estimate` 并设预算上限，建议先用一个族验真实单价再全量铺开。 |
| `step_order` 样本量小 | 每族仅 50 题、6 选项、随机基线 16.7%，单个 accuracy 置信区间较宽。报告对 n < 100 的格子加标注。 |
| 三条 adapter 与多卡切分未经真实验证 | 见第 8 节缺口表。 |
