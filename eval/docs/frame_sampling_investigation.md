# 抽帧问题调查记录

> 状态：**调查完成，方案未定，Time EQA 相关工作暂时挂起。**
> 目的：把已查清的事实固化下来，等团队定了协议后可直接接续。
> 日期：2026-08-13
> 相关文档：[官方推荐配置调研](official_config_survey.md)

---

## 一、问题的边界

九个任务里只有四个送视频，媒体长度差一个数量级：

| 任务 | 送什么 | stack_cubes 实测 |
| --- | --- | --- |
| **time** | **整段 episode** | 中位 53.9 s（范围 45.1–81.0），20 fps，960×486 |
| understanding | 动作切片（多视角拼接） | 中位 4.0 s，20 fps，2880×540 |
| image_in_video | 动作切片（单视角） | 中位 4.0 s，20 fps，960×486 |
| planning | 动作切片（多视角拼接） | 中位 4.0 s，20 fps，2880×540 |

其余五个任务只送静态图。**问题集中在 Time EQA。**

动作段时长分布（300 段）：最短 1.30 s ｜ P10 2.35 s ｜ 中位 4.00 s ｜ P90 11.05 s ｜ 最长 22.25 s。
63% 短于 5 秒，17% 短于 3 秒。要在 54 秒视频里定位中位 4 秒的动作。

### 数据是异构的

| 任务族 | fps | episode 时长 | 帧数 |
| --- | ---: | ---: | ---: |
| stack_cubes | 20 | 53.9 s（45–81） | 1078–1620 |
| **tea** | **25** | **119.8 s** | **2994** |

只抽查了这两族，**其余 18 族的时长分布未知**。任何按 fps 定的策略都会因族而异，必须先扫一遍全部族的时长再定档位。

---

## 二、代码：两条抽帧路径

### 路径 A — 我们自己的代码（`test/vlm_api.py:756`）

```python
def internvl_frame_indices(video_path, num_segments, sample_fps=0.0):
    video = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    max_frame = len(video) - 1

    if sample_fps > 0:                      # 分支一：固定帧率，从 t=0 起等间隔
        source_fps  = float(video.get_avg_fps())
        duration    = (max_frame + 1) / source_fps
        frame_count = max(1, int(duration * sample_fps))
        return [min(max_frame, int(round(index * source_fps / sample_fps)))
                for index in range(frame_count)]

    if num_segments <= 1:
        return [max_frame // 2]

    seg_size = max_frame / num_segments     # 分支二：均匀分段取中点
    return [min(max_frame, int((seg_size / 2) + round(seg_size * idx)))
            for idx in range(num_segments)]
```

参数取值链（`vlm_api.py:177-178`）：`providers.<name>` → `defaults` → 硬编码默认
（`video_sample_fps=0.0`，`num_segments=16`）。**八个评测脚本都没有对应的命令行参数。**

只有 `local_internvl` 类型的 provider 会读这两个参数，而 `config_test.json` 里这类
provider 恰好只有两个，且**都设了 `video_sample_fps: 1.0`** —— 所以那两处
`num_segments: 4` **从未生效过，且代码不报警**。

每帧切块：`max_video_tiles=1` → 1×1 网格 → 每帧 1 块、无缩略图，
960×486 被强行拉伸成 448×448（宽高比 1.98 压到 1.0）。
每块 448/14=32 → 1024 patch → pixel shuffle（`downsample_ratio 0.5`）→ **256 LLM token**。

### 路径 B — `qwen_vl_utils` 0.0.14（外部库）

我们调 `process_vision_info(messages)` 时**没有传任何覆盖参数**，全部走库默认：

```python
fps        = ele.get("fps", 2.0)                   # FPS = 2.0
min_frames = ceil_by_factor(4, 2)                  # FPS_MIN_FRAMES = 4
max_frames = floor_by_factor(min(768, total), 2)   # FPS_MAX_FRAMES = 768
nframes    = total_frames / video_fps * fps
nframes    = min(min(max(nframes, min_frames), max_frames), total_frames)
nframes    = floor_by_factor(nframes, 2)           # 必须偶数
```

---

## 三、15 个模型分四组

| 组 | 抽帧由谁决定 | 模型 |
| --- | --- | --- |
| **A** `local_internvl` | 我们的代码（`video_sample_fps=1.0`） | InternVL3.5-8B、InternVL3.5-30B-A3B、SenseNova-SI-1.1-InternVL3-2B |
| **B** `local_qwen` / `local_transformers_vlm` | `qwen_vl_utils` 默认值 | Qwen3-VL-8B、Qwen3-VL-30B-A3B、RynnBrain-2B、RynnBrain1.1-2B、Cosmos-Reason2-2B |
| **C** `local_cosmos_transformers` | 模型自带 processor | Cosmos3-Edge-2B |
| **D** `openai_compatible` | 服务端，**不可知不可控** | Qwen3-VL-235B、qwen3.8-max、Seed2.0-Lite、GLM-5V-Turbo、Gemini-3.1-Pro、Gemini-3.5-Flash |

---

## 四、实测：各组拿到多少帧

用两边真实函数算出（`internvl_frame_indices` 与 `qwen_vl_utils.smart_nframes`）：

| 任务 | 时长 | A 组帧数 | A 组 token | B 组帧数 |
| --- | ---: | ---: | ---: | ---: |
| time | 53.9 s | 53 | 13,568 | **106** |
| understanding | 4.0 s | 4 | 1,024 | 8 |
| image_in_video | 4.0 s | 4 | 1,024 | 8 |
| planning | 4.0 s | 4 | 1,024 | 8 |

> **更正记录：** 早期版本报「B 组每帧 578 token、合计 61,268」是错的。
> 578 是 `qwen_vl_utils` 内部按 `patch=14` 算出的 **resize 网格**，不是模型的 token 数。
> Qwen3-VL 真实参数为 `patch_size=16`、`merge_size=2`、**`temporal_patch_size=2`**
> （视频两帧合一个时间 patch），实际约 23,850 token。
> **帧数 2 倍的差距成立，token 差距是 1.8 倍而非 4.5 倍。**

---

## 五、实测：显存墙

硬件 8 × RTX 4090（24 GiB/张），模型 SenseNova-SI-1.1-InternVL3-2B，`use_flash_attn=false`。

### 单卡扫描（权重常驻 3.89 GiB）

| 帧数 | 视觉 token | 峰值显存 | 增量 | 结果 |
| ---: | ---: | ---: | ---: | --- |
| 8 | 2,048 | 4.55 G | 0.66 G | OK |
| 16 | 4,096 | 6.20 G | 2.31 G | OK |
| 24 | 6,144 | 8.87 G | 4.98 G | OK |
| 32 | 8,192 | 12.55 G | 8.66 G | OK |
| 48 | 12,288 | — | — | **OOM** |
| 64 / 73 / 120 | — | — | — | **OOM** |

### 多卡 `device_map="auto"` 扫描

权重被切到 4 张卡（每张仅 ~1 GiB，单卡时 3.89 GiB）：

| 帧数 | 单卡峰值 | 全卡合计 | 结果 |
| ---: | ---: | ---: | --- |
| 32 | 9.59 G | 31.64 G | OK |
| 48 | — | — | **OOM，单次申请 7.24 GiB** |
| 64 | — | — | **OOM，单次申请 12.85 GiB** |
| 96 | — | — | **OOM，单次申请 14.42 GiB** |
| 120 | — | — | **OOM，单次申请 22.54 GiB** |

**结论：多卡按层切分（pipeline sharding）不解决这个 OOM。** 它聚合的是权重容量，
而失败的是**单个连续张量的分配**，必须在一张卡上放下。8 张卡变不出一张 189 GiB 的卡。

对 30B / 235B 这类权重放不下的模型，多卡是必需的；但对本问题无效。

### 分块 ViT 前向的尝试 —— 失败，但暴露了真因

设想：ViT 沿 batch 维处理各 tile，帧间无交互，分块前向应当等价。
结果：**OOM 的单次申请量与未分块时一字不差**（7.24 / 12.85 / 14.42 / 22.54），
说明 ViT 根本不是瓶颈。

### 真因：LLM 的注意力矩阵被物化成 fp32

```
注意力张量 = num_heads(12) × seq² × 4 bytes

48 帧: seq = 48×256+440 = 12,728  →  12 × 12728² × 4 = 7.24 GiB   实测 7.24 ✓
64 帧: seq = 64×256+440 = 16,824  →  12 × 16824² × 4 = 12.7 GiB   实测 12.85 ✓
```

公式与实测精确吻合。因为 `use_flash_attn: false`，注意力矩阵被完整物化，
**显存随视觉 token 数的平方增长**。

### 按此公式外推到 H100

总峰值 ≈ 权重 + 2.6 × 注意力张量（系数由 32 帧实测反推）：

| 场景 | 帧数 | 注意力张量 | 估算总峰值 | 4090 24G | H100 80G |
| --- | ---: | ---: | ---: | :---: | :---: |
| stack_cubes fps=1 | 54 | 9.1 G | ~28 G | ✗ | **✓** |
| stack_cubes fps=2 | 108 | 35 G | ~96 G | ✗ | ✗ |
| tea fps=1 | 120 | 43 G | ~117 G | ✗ | ✗ |
| tea fps=2 | 240 | 171 G | ~449 G | ✗ | ✗ |

**这还只是 2B 模型。** InternVL3.5-30B-A3B 权重占 60 GB，单张 H100 只剩 20 GB，
连 stack_cubes 的 fps=1 都跑不动。

---

## 六、上下文墙（尚未测到）

| 模型 | `max_position_embeddings` | tokenizer `model_max_length` | 每帧 token | 可装帧数 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL 8B/30B/235B | 262,144 | 262,144 | ~225 | **≈ 453** |
| RynnBrain-2B / 1.1-2B | 262,144 | 262,144 | ~225 | ≈ 453 |
| Cosmos3-Edge-2B | 131,072 | 262,144 ⚠ 矛盾 | ? | ? |
| **InternVL3.5-8B / 30B-A3B** | **40,960** | **14,588** | 256 | **56 ~ 160** ⚠ |
| **SenseNova-SI-1.1-InternVL3-2B** | **32,768** | **16,384** | 256 | **64 ~ 128** ⚠ |

**InternVL 系的两个数字自相矛盾，哪个真正生效仍未确定。** 实测时见过
`Token indices sequence length is longer than ... (16951 > 16384)` 警告但推理照常返回，
说明 tokenizer 那个上限只警告不截断 —— 真实上限可能是 `max_position_embeddings`。
但因为显存 OOM 先发生，**始终没能测到 64 帧以上，这个问题悬而未决**。

Qwen 系根本不受上下文约束：453 帧的预算远大于库默认算出的 106 帧，
**限制 Qwen 的是库的默认值，不是模型能力。**

---

## 七、Double check：数据侧没有帧间压缩

有说法称为了规避 OOM，对 Time EQA 用的数据做过帧间压缩。**核查结论：没有。**

| 数据 | fps | 帧数 | 时长 | 前 N 帧重复检测 |
| --- | ---: | ---: | ---: | --- |
| stack_cubes time_video_crop_top | 20.0 | 1464 | 73.2 s | 唯一 120 / 重复 0 / 最长串 1 |
| stack_cubes 其余三类视频 | 20.0 | 354 | 17.7 s | 零重复 |
| tea time_video_crop_top | 25.0 | 2994 | 119.8 s | 唯一 150 / 重复 0 / 最长串 1 |
| tea time_joined_videos | 25.0 | 2994 | 119.8 s | 零重复 |

所有视频保持源帧率，连续帧全部互不相同。生成侧 `crop_video_top` 也只做空间裁剪
（`crop` filter + libx264 crf 18，无 `-r`、无 fps filter）。

唯一的「压缩」是**空间裁剪**（去掉顶部 10% 的时间戳条，540→486）和**重编码**。

推测该说法指的是**推理侧**的 `video_sample_fps=1.0`（每秒取 1 帧而非全部 20 帧），
这确实是时间抽稀且确实是为了 OOM，但发生在推理时、未写进数据。**待与相关同事确认。**

---

## 八、已决定的协议（可行性待验证）

团队讨论结论：

1. 所有模型统一测 **fps=1** 和 **fps=2** 两档
2. 按**实际帧数**对齐而非按参数名：fps 型 adapter 直接设 fps；
   `num_segments` 型用 `num_segments = round(时长 × fps)` 换算，**逐视频计算**
3. 闭源模型无法调整，用服务端默认照测，标注清楚

**可行性问题（见第五节）：** 该方案在长 episode 的族上对 InternVL 系不可行。
stack_cubes fps=2（108 帧）与 tea 的两档（120 / 240 帧）在 H100 上都跑不动，
InternVL3.5-30B-A3B 更是一档都跑不了。需要补一个决定：

- (a) 加上限截断 `帧数 = min(时长 × fps, N_max)` —— 超过阈值后帧数又不一致
- (b) 跑不动就记 N/A 并标注原因 —— InternVL 系在长族大面积缺数据
- (c) 按族分别定 fps —— 跨族不可比

---

## 九、下一步的最高优先级：先试 FlashAttention

`use_flash_attn: false` **是我们 config 里的选择，不是硬件限制**。开启后注意力矩阵
不再物化，显存从 O(seq²) 降到 O(seq)，第五节那张表会整体改写。

而且两家官方都推荐开：

- InternVL 模型代码原生支持 `use_flash_attn` 参数
- Qwen3-VL 官方 README：「We recommend enabling flash_attention_2 for better
  acceleration and memory saving, **especially in multi-image and video scenarios**」

**在讨论换 H100 之前应先做这件事。** 若显存墙塌掉：

- fps=1/fps=2 双档方案可能在现有 4090 上就能跑
- 剩下的只有上下文这一堵墙（终于能测到 64 帧以上）
- H100 的必要性需要重新评估

---

## 十、挂起的待办

| 项 | 阻塞在 |
| --- | --- |
| 安装 FlashAttention 并重跑显存扫描 | 编译耗时/版本兼容风险 |
| 测出 InternVL 真实上下文上限（16384 还是 40960/32768） | 需先解决显存 |
| tIoU 随帧数的曲线（找饱和点/排名稳定点） | 需先解决显存 |
| 全部 20 个族的 episode 时长分布 | 需下载各族样本 |
| B 组（Qwen 路径）的显存实测 | 需要 Qwen 权重 |
| C 组（Cosmos3-Edge）抽帧行为 | 需要权重；且该模型定位存疑（见调研 #7） |
| 闭源模型帧数的 `usage` 反推 | 需要 API key；需先把 `usage` 记录进结果 |
| 确认「帧间压缩」说法的实际所指 | 需与相关同事确认 |

---

## 十一、无论协议如何都要做的（不阻塞）

这三件事与协议选择无关，直接做进重构：

1. **`num_segments` 改为逐视频运行时计算**，不再是静态配置值
2. **配置 schema 强制互斥**：`frames: {mode: fps|uniform|native, value: N}`，
   同时给出多个抽帧参数直接报错，杜绝当前这种静默覆盖
3. **结果文件记录实际帧数**，以及 API 响应的 `usage`
   （现在 `raw, model_text = call_vlm(...)` 拿到原始响应后直接丢弃）
