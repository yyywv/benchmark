# Test Project

本项目用于评估机器人 pick/place 场景中的视觉语言模型能力。当前测试围绕 5 类能力展开：`Understand`、`Plan`、`Timestamp`、`Trajectory`、`Step Order`。

## 1. Understand

`Understand` 测试关注模型是否能够理解当前视觉输入中的状态、视角和正在发生的动作。

### 1.1 Viewpoint Identification

测试目标：判断模型是否能够根据主视角或任务上下文，识别对应的夹爪视角、左右手视角或多视角关系。

适用范围：

- 只对双手均参与的任务进行测试。
- 当前双手任务覆盖：`pick cube`。
- 对应脚本：`left_right_glm_test.py`。

说明：该测试依赖左右手或双夹爪之间的视角差异。如果任务只涉及单手操作，则不纳入 `Viewpoint Identification` 测试。

### 1.2 State Understand

测试目标：判断模型是否能够理解当前场景或视频片段中的物体状态、手部状态和动作状态。

适用范围：

- 可用于具备明确状态变化的任务。
- 当前主要覆盖：`pick cube`、`pick flower`、`gift hand`。
- 对应脚本：`understanding_glm_test.py`。

## 2. Plan

`Plan` 测试关注模型是否能够基于当前状态推理下一步应该执行什么动作。

### 2.1 Goal-conditioned Planning

测试目标：在给定任务目标的情况下，判断当前状态下最合理的下一步动作。

适用范围：

- 适用于需要结合目标进行动作选择的任务。
- 当前主要覆盖：`pick cube`、`pick flower`、`gift hand`。
- 对应脚本：`planning_glm_test.py`。

## 3. Timestamp

`Timestamp` 测试关注模型是否能够定位动作发生的时间区间。

测试目标：

- 给定一个任务或问题，预测对应动作在视频中的开始时间和结束时间。
- 评估预测时间段与标注时间段之间的重叠程度。
- 对应脚本：`time_eqa_glm_test_multi.py`。

适用范围：

- 30 个任务都可以进行 `Timestamp` 测试。
- 该测试不依赖任务是否为单手或双手操作。

## 4. Trajectory

`Trajectory` 测试关注模型是否能够根据视觉输入预测左右夹爪的后续运动轨迹。

测试目标：

- 输出左右夹爪的有序 2D 或 3D 轨迹点。
- 使用 Hausdorff、discrete Frechet、Chamfer 和综合 score 评估预测轨迹。
- 对应脚本：`trajectory_glm_test.py`。

适用范围：

- 适用于有轨迹标注、相机参数或投影结果的任务。
- 当前配置默认使用 `trajectory_qa_2d_first50_6move.json`。

## 5. Step Order

`Step Order` 测试关注模型是否能够根据初始状态和打乱顺序的结果图，恢复正确操作顺序。

测试目标：

- 输入初始状态图和 shuffled result-state montage 图。
- 从多选项中选择正确的 chronological operation order。
- 对应脚本：`step_order_glm_test.py`。

适用范围：

- 适用于有多阶段状态变化且可抽取状态图的任务。
- 当前 config 中 `step_order.input` 是示例路径，运行前需要改成本机数据路径或用 `--input` 覆盖。

## 当前测试覆盖矩阵

| 测试大类 | 小类 | 当前适用任务范围 | 当前覆盖 | 脚本 |
| --- | --- | --- | --- | --- |
| `Understand` | `Viewpoint Identification` | 双手均参与的任务 | `pick cube` | `left_right_glm_test.py` |
| `Understand` | `State Understand` | 具备明确状态变化的任务 | `pick cube`, `pick flower`, `gift hand` | `understanding_glm_test.py` |
| `Plan` | `Goal-conditioned Planning` | 需要结合目标推理下一步的任务 | `pick cube`, `pick flower`, `gift hand` | `planning_glm_test.py` |
| `Timestamp` | 动作时间定位 | 全部 30 个任务 | 全部 30 个任务 | `time_eqa_glm_test_multi.py` |
| `Trajectory` | 夹爪轨迹预测 | 有轨迹标注的任务 | 当前 trajectory QA 数据 | `trajectory_glm_test.py` |
| `Step Order` | 操作顺序恢复 | 有多阶段状态图的任务 | 当前 step-order QA 数据 | `step_order_glm_test.py` |

## 模型测试范围

模型在配置中按 provider 分组。运行时用 `--provider` 选择接口来源，用 `--model` 切换具体模型。

| Provider | 模型范围 |
| --- | --- |
| `qwen` | `Qwen3-VL-8B-Instruct`, `Qwen3-VL-32B-Instruct`, `Qwen3-VL-30B-A3B-Instruct`, `Qwen3-VL-235B-A22B-Instruct`, `qwen3.5-omni-plus`, `qwen3.7-plus` |
| `kimi` | `Kimi-K2.6` |
| `seed` | `Seed2.1`, `Seed2.0-Lite` |
| `internvl` | `InternVL3.5-8B`, `InternVL3.5-30B-A3B` |
| `glm` | `GLM-5V-Turbo` |
| `gemini` | `Gemini-3.1-Pro`, `Gemini-3.5-Flash` |
| `local_qwen` | 本地 Qwen-VL 系权重 |

推荐命令形式：

```bash
python pickplace/workflow/test/<script>.py \
  --config pickplace/workflow/test/config_test.json \
  --provider qwen \
  --model Qwen3-VL-32B-Instruct
```

## 计划测试模型清单

- `Qwen3-VL-8B-Instruct`
- `Qwen3-VL-32B-Instruct`
- `Qwen3-VL-30B-A3B-Instruct`
- `Qwen3-VL-235B-A22B-Instruct`
- `qwen3.5-omni-plus`
- `qwen3.7-plus`
- `Kimi-K2.6`
- `Seed2.1`
- `Seed2.0-Lite`
- `InternVL3.5-8B`
- `InternVL3.5-30B-A3B`
- `GLM-5V-Turbo`
- `Gemini-3.1-Pro`
- `Gemini-3.5-Flash`

## 当前模型测试进度

| 任务 | 已覆盖测试 | 已测试模型 | 状态 |
| --- | --- | --- | --- |
| `pick cube` | 当前任务对应的已配置测试 | `qwen3.7-plus`, `glm-5v-turbo` | 已测试 |
| `pick flower` | 当前任务对应的已配置测试 | 待补充 | 待测试 |
| `gift hand` | 当前任务对应的已配置测试 | 待补充 | 待测试 |
| trajectory QA | `Trajectory` | 待补充 | 待测试 |
| step-order QA | `Step Order` | 待补充 | 待测试 |

## 任务覆盖规则

- 所有 30 个任务都可以用于 `Timestamp` 测试。
- `Viewpoint Identification` 只测试双手均参与的任务。
- 当前满足双手均参与条件的任务有 1 个：`pick cube`。
- 除 `Viewpoint Identification` 外，当前已有测试对 `pick cube`、`pick flower`、`gift hand` 三个任务全覆盖。
- `Trajectory` 和 `Step Order` 依赖额外生成的数据文件，不按 30 个原始任务自动全覆盖。
- 当前已完成 `pick cube` 在 `qwen3.7-plus` 和 `glm-5v-turbo` 上的对应测试；其他模型和任务结果后续补充。
- 新增任务、小类或模型时，需要同步更新本文件和 `TESTING.md` 中的运行说明。
