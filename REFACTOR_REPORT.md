# 评测模块重构 · 结项报告

> 分支：`feature/eval-harness`（16 个提交，基于 `3f62ddd`）
> 日期：2026-08-13 ~ 08-14
> 相关文档：[重构方案](REFACTOR_PLAN.md) ｜ [操作手册](eval/RUNBOOK.md) ｜
> [抽帧调查](eval/docs/frame_sampling_investigation.md) ｜ [官方配置调研](eval/docs/official_config_survey.md)

---

## 一、结论先行

九条互不相同的命令收敛成一条，评分逻辑逐字节不变。

```bash
./run.sh                       # 自检 → 估算 → 执行 → 汇总 → 打包
./run.sh --shard 1/4           # 多机分工
./run.sh --only api            # 只跑 API，不需要 GPU
```

**过程中挖出 4 个真实缺陷、1 个方法论问题。** 其中两个（评分 bug、
API 请求体上限）会直接影响榜单数字，一个（抽帧策略）会让 Time EQA 的
横向对比不成立。这些是本次工作里比代码本身更值得记录的部分。

`test/` 目录**未改动任何 `.py` 源码**（只从 git 里移除了误提交的 `__pycache__`），
作为回归基线长期保留。

---

## 二、交付内容

| | 规模 |
| --- | --- |
| 框架代码 `eval/robochrono/` | 4,833 行 |
| 工具与测试 `eval/tools/`、`eval/tests/` | 887 行 |
| 文档 | 1,531 行 |
| 提交 | 16 个 |

```
eval/
├── robochrono/          框架：cli / matrix / matrix_run / engine / pool /
│                        store / preflight / report / media_prep / parsing /
│                        vlm_api / tasks{base,choice,time_eqa,trajectory}
├── configs/             providers.json（模型接入）、plan.json（矩阵定义）
├── tools/               路径规范化、媒体校验、模型补丁、BC-10 影响分析、冒烟
├── tests/               三套回归，均不需要 GPU 与 API key
├── datasets/ models/    数据与权重（gitignore）
├── run.sh RUNBOOK.md requirements.txt setup_env.sh
```

**八个冻结脚本 4,335 行里约 1,600 行是逐字重复的样板。** AST 比对确认：
六个选择题任务的解析行为**本就完全一致**，三种写法的差异都是无效的
（`planning` 有个未使用的局部变量，`extract_choice` 里的
`option_id in valid_ids` 恒为真）。真正的差异只有 `step_order` 用 `choices`
且保留连字符。

---

## 三、验证

### 三套回归，全部不依赖 GPU 与 API key

| 测试 | 覆盖 | 结果 |
| --- | --- | --- |
| `test_parsing_equivalence` | 6 任务 × 40 题 × 20 种输出形态 | **4,800 条比对，零不一致** |
| `test_replay_regression` | 9 任务的打分与汇总 | **130 条，逐字段零差异** |
| `test_engine_replay` | engine + store + 断点，全链路 | **9 任务零差异，断点正确** |

回归数据是**冻结脚本跑真实模型录下来的输出**，所以任何改动都能立刻验证
有没有动到评分，不用重新占卡。

### 真实运行

```
全矩阵：2 模型 × 9 任务 = 18 个 run，零错误

| model                 | time  | underst | left_r | img_in_v | plan | plan_2 | step | traj2D | traj3D |
| SenseNova-2B (local)  | 0     | 0.25    | 0      | 0        | 0    | 0.25   | 0.25 | 0.373  | 0.226  |
| qwen3.8-max (api)     | 0.708 | 0.75    | 1.0    | 0.75     | 0.5  | 0.75   | 1.0  | 27.06  | 2.55   |
```

（样本量小，仅用于验证链路。trajectory 的 `mean_score` 是 0–100 分。）

### 工程指标

| | 冻结版 | 现在 |
| --- | --- | --- |
| 权重加载次数（10 本地模型 × 9 任务） | 90 | **10**（model-major） |
| 多卡并行 | 无 | **4 卡实测 236s → 114s** |
| 结果行体积 | ~7,600 B | **1,621 B** |
| checkpoint 累计写入（单个 300 题 trajectory_3D） | 2.2 GB | **追加写，≈0** |
| 断点续跑 | 全量反序列化 | 扫 id |
| 跨模型汇总 | 无 | `report` 出表 + CSV |

---

## 四、发现的缺陷

### 4.1 评分 bug：`text: null` 选项导致「none」被判成选项 A（BC-10）

`left_right` 与 `image_in_video` 的选项是图片，`text` 为 `null`。
冻结脚本写的是 `str(option.get("text", ""))` —— **`str(None)` 得到 `"None"`**，
归一化后是 `"none"`，于是任何含 "none" 的回答都会命中**第一个选项**：

```
模型输出 "none of these"  →  判为 A  →  这两个任务恰好都有 none-option E
```

这是解析等价性测试挖出来的。两点更正之前的表述：修复后映射到
`None`（解析失败）而**不是** E；且它是**偏向 A 的噪声**，不是单向惩罚 ——
正确答案恰好是 A 时反而蒙对。

**实测影响面：240 条真实输出（SenseNova-2B），零触发** ——
该模型始终输出干净的 `{"choice": "X"}`，从不落到第三级文本匹配。
但这不能外推到其他模型，因为触发条件正是「不守输出格式」。

已做成 `eval/tools/bc10_impact.py`，**离线重放即可统计，不必重跑模型**。
默认关闭。

### 4.2 配置 bug：`planning` 与 `planning_2` 抢同一个 config key（BC-06）

两个脚本都读 `tasks.planning`，而那一节当前指向 `planning_2_vqa.json`。
跑 `planning` 不带 `--input` 会直接崩（取媒体在 `try` 之外）。
更隐蔽的是两者不带 `--output` 时会写同一个文件，而断点续跑**不校验结果来源**，
会把另一个任务的结果当成「已完成」跳过。

### 4.3 命名陷阱：SenseNova 实际是 2B 不是 32B（BC-07）

repo 里三处写作 `SenseNova-SI-1.1-InternVL32B`，少了个连字符，读起来像 32B。
实际是 HuggingFace 上的 `sensenova/SenseNova-SI-1.1-InternVL3-2B`，
**InternVL3 架构的 2B，权重 4.18 GB**。照原名会找不到权重，或按 32B
规划 64 GB 显存。

### 4.4 API 请求体上限：Time EQA 完全跑不了（BC-11）

二分探测出边界：**base64 10.32 MB 通过、10.99 MB 返回 413**，约 10 MiB。
而 Time EQA 视频 base64 后中位 **10.47 MB** —— 不处理的话
**5 个 API 模型的 Time EQA 一题都跑不了**。

`media_prep.py` 在超预算时降分辨率重编码。**关键验证：这不改变模型
实际获得的视觉预算** —— 取一条本来就在预算内的 clip，压缩前后各发一次，
服务端报告的 `video_tokens` **完全相同（2902）**，说明服务端本就会降到
自己的目标分辨率。

效果：Time EQA 从「跑不了」变成 **18/18 作答、`mean_tIoU = 0.716`**。

---

## 五、方法论问题：抽帧策略（BC-09，未决）

这是本次工作里最重要的发现，**它决定 Time EQA 的数字是否有意义**。

### 现状：15 个模型看到的东西各不相同

| 组 | 抽帧由谁决定 | 模型数 |
| --- | --- | --- |
| A `local_internvl` | **我们的 config**（`video_sample_fps=1.0`） | 3 |
| B `local_qwen` / `local_transformers_vlm` | **`qwen_vl_utils` 的默认值**（架构巧合） | 5 |
| C `local_cosmos_transformers` | 模型自带 processor | 1 |
| D `openai_compatible` | **服务端** | 6 |

调研官方推荐后发现更尴尬的事实：**9 个开源模型里只有 1 个用的是自己的默认值**。
5 个用了别人家配套库的默认值，3 个用了当初写 config 的人随手填的值。

而且 `video_sample_fps > 0` 时 `internvl_frame_indices` 直接 return，
**config 里那两处 `num_segments: 4` 从未生效过，代码不报警**。

### 两堵墙，都实测过

**显存墙。** 4090 单卡扫描（SenseNova-2B，`use_flash_attn=false`）：

```
32 帧  12.55 GiB  OK
48 帧  OOM，单次申请 7.24 GiB
64 帧  OOM，单次申请 12.85 GiB
```

真因不是 ViT，是 **LLM 注意力矩阵被物化成 fp32**，公式精确吻合：

```
num_heads(12) × seq² × 4 bytes
48 帧: seq=12,728 → 7.24 GiB   实测 7.24 ✓
64 帧: seq=16,824 → 12.7 GiB   实测 12.85 ✓
```

**所以多卡无效** —— `device_map="auto"` 聚合的是权重容量，失败的是单个连续张量。
实测把权重摊到 4 张卡后，仍在 48 帧 OOM，**申请量一字不差**。

`use_flash_attn: false` 是我们 config 里的选择，不是硬件限制，
而 InternVL 和 Qwen 官方**都推荐开启**。这应该在讨论换 H100 之前先试。

**上下文墙。** 与显存独立，换卡解决不了：

| 模型 | 上下文 | 每帧 token | 可装帧数 |
| --- | ---: | ---: | ---: |
| Qwen 系 / RynnBrain | 262,144 | ~225 | ≈453 |
| InternVL3.5-8B / 30B | 14,588 ~ 40,960 ⚠ 两数矛盾 | 256 | **56 ~ 160** |
| SenseNova-2B | 16,384 ~ 32,768 ⚠ | 256 | **64 ~ 128** |

### 这不是理论担忧，有实测证据

同一份 stack_cubes 数据上：

```
本地 InternVL，8 帧 / 2,048 视觉 token    →  mean_tIoU = 0.0
qwen3.8-max，服务端采样 / ~30,000 token  →  mean_tIoU = 0.716
```

**视觉预算差一个数量级，成绩从 0 到 0.72。** 也就是说 Time EQA 目前
在很大程度上测的是各 adapter 的默认视觉预算，而不是模型的时间推理能力。

### 团队已定的协议与它的可行性问题

已定：所有模型统一测 **fps=1** 与 **fps=2**，`num_segments` 型按
`round(时长 × fps)` 逐视频换算；闭源模型无法调整就照测。

**但长 episode 的族上对 InternVL 系不可行**（tea 是 25 fps / 119.8 秒）：

| 场景 | 帧数 | 估算峰值 | 4090 | H100 |
| --- | ---: | ---: | :---: | :---: |
| stack_cubes fps=1 | 54 | ~28 G | ✗ | ✓ |
| stack_cubes fps=2 | 108 | ~96 G | ✗ | ✗ |
| tea fps=1 | 120 | ~117 G | ✗ | ✗ |
| tea fps=2 | 240 | ~449 G | ✗ | ✗ |

InternVL3.5-30B-A3B 权重就占 60 GB，单张 H100 一档都跑不了。

**管道已就位，只差取值。** `frames` 改成互斥声明（旧写法降级为**告警**，
新旧混用直接报错，`strict_frames` 下告警升为错误）；
`frames_for_video()` 能按 `round(时长 × fps)` 逐视频换算，
打开 `align_fps_to_segments` 即可，不用改代码。

### 一个可能更好的可比指标

服务端会返回 `prompt_tokens_details.video_tokens` —— 闭源模型**不再是黑盒**，
是读数不是推算。但它拆不开帧数与分辨率。

建议改用**「每秒视频的视觉 token 数」**：本地与 API 都可观测，
且帧数 × 分辨率合起来才是模型真实获得的信息量。

```
qwen3.8-max（服务端决定）   542 token/秒
InternVL（num_segments=8）  383 token/秒
```

同一段素材上两者其实在同一量级 —— 差距主要出现在整段长视频上。

---

## 六、环境与部署

### transformers 版本冲突：结果比预想的好

各模型官方要求互不兼容（RynnBrain-2B 要 4.57.1，RynnBrain1.1-2B 要 5.2.0）。
在克隆环境里跑 9 项检查实测：

| 版本 | 结果 |
| --- | --- |
| 4.51.3 | ✗ 冻结代码的 `dtype=` 被透传给模型构造函数 |
| 4.56.2 | ✓ 可用，但低于 RynnBrain-2B 的 4.57.1 |
| **4.57.6** | **✓ 9/9 全过，且满足 ≥4.57.1** |
| 5.2.0 | ✗ InternVL 自定义代码在 meta device 下调 `.item()` 即崩，改掉那一行仍有同类问题 |

**不是「每模型一套环境」，而是一主一副** —— 4.57.6 覆盖 InternVL ×3、
Qwen ×3、RynnBrain-2B、Cosmos-Reason2；只有 RynnBrain1.1-2B 需要单独环境。

升级 4.56.2 → 4.57.6 后**分数逐位不变**。

### 数据侧核查

有说法称为规避 OOM 对 Time EQA 数据做过帧间压缩。**核查结论：没有。**
stack_cubes（20 fps）与 tea（25 fps）的全部视频都保持源帧率、
连续帧零重复。唯一的压缩是空间裁剪（去掉顶部 10% 时间戳条）与重编码。
推测该说法指的是推理侧的 `video_sample_fps=1.0`，**建议与相关同事确认**。

### 数据异构

| 族 | fps | episode 时长 |
| --- | ---: | ---: |
| stack_cubes | 20 | 45–81 秒 |
| **tea** | **25** | **119.8 秒** |

只抽查了两族，**其余 18 族未知**。任何按 fps 定的策略都会因族而异，
preflight 应先扫一遍全部族的媒体特征再定档位。

### 路径规范化（BC-08）

九个 QA JSON 有三种互不兼容的路径风格（生成机的绝对路径、相对路径、
Windows 反斜杠）。已统一重写 **22,300 条**，原文件备份为 `*.orig`。
另有 9,600 条溯源路径与 5,500 条未发布媒体原样保留 —— 经确认均非评测关键字段。

验收用 `check_media.py`：复用九个冻结脚本**自身**的取媒体函数，
对 2,700 题的 8,450 个媒体引用逐一检查，零缺失零报错。

---

## 七、待决事项

| # | 事项 | 影响 |
| --- | --- | --- |
| 1 | **抽帧协议取值**（BC-09） | Time EQA 的横向对比在此之前不成立 |
| 2 | 是否先试 FlashAttention | 可能一次性解除显存墙，且两家官方都推荐 |
| 3 | 是否启用 BC-10 修复 | 建议正式跑完后按模型用 `bc10_impact.py` 判断 |
| 4 | 是否启用 BC-01/BC-02 | 目前默认关闭；BC-01 的实际影响只剩 `max_new_tokens` 256→1024 |
| 5 | 官方推荐配置的 9 处分歧 | 已全部记录未下结论，见调研文档 |
| 6 | `Cosmos3-Edge` 是否该进榜 | 官方定位是世界模型，无问答示例，我们的用法非官方 |
| 7 | 其余 19 个任务族的数据 | 目前只有 stack_cubes |
| 8 | 闭源模型的时间可复现性 | 服务端策略会漂移且无版本号，建议在报告加限定说明 |

---

## 八、成本预估

`estimate` 实测（每模型每族）：**2,450 次调用、3.05 GB 媒体**。

```
全矩阵 15 模型 × 20 族：
  调用    735,000 次
  媒体    ≈ 915 GB 上传
  其中付费 API（5 模型）  ≈ 245,000 次调用 / 305 GB
```

另需注意：实测该端点的 `reasoning_tokens` 占 completion 的 **91%**
（640/703）—— 模型默认在思考，即使我们没要求。按 token 计费的话这是笔
可观成本，建议确认能否关闭。

**建议先用一个族跑真实账单反推单价，再决定是否全量铺开。**

---

## 九、过程中的几个教训

**回归测试挖出的 bug 比重构本身更有价值。** BC-10 那个评分 bug 不是靠读代码
发现的，是靠「用 20 种输出形态 × 40 道真题跑 4,800 次比对」撞出来的。
如果直接重写而不做等价性验证，这个 bug 会被静默地「修掉」或「换个方式保留」，
两种都糟糕。

**不要用推理代替实测。** 「多卡能否解决 OOM」「能否升级到 transformers 5.2」
「压缩视频会不会改变模型输入」这三个问题，我最初的推理有两个方向对、
一个方向错，但**都需要实测才能确定量级和边界**。尤其「压缩不改变
`video_tokens`」这个结论，纯推理是得不出来的。

**几个数字被我先算错过，实测后更正了。** Qwen 的每帧 token（578 → 约 225，
漏了 `temporal_patch_size=2`）、媒体总体积（3.37 → 3.05 GB，误算了溯源路径）、
BC-10 的影响方向（「系统性判错」→「偏向 A 的噪声」）。已在相应文档更正。

**API key 差点进了 git。** 配置里的 `api_key_env` 是环境变量名，很容易被
误当成填 key 的地方。已改为从 `~/.config/robochrono/keys.env`（仓库外，600）
读取，并在 preflight 里加了检测。
