#!/usr/bin/env python3
# coding: utf-8
"""验证新代码发给模型的东西与冻结脚本完全一致。

replay 回归证明的是「给定相同模型输出，算出相同分数」，它跳过了 prompt 比对，
也没有验证媒体。但真实跑的时候，如果 prompt 或媒体不同，模型的回答就会不同 ——
分数逻辑一致也救不回来。

这里逐题比对两件事：
  prompt   完整字符串，逐字节
  媒体     类型、路径、顺序

九个任务全部覆盖，不需要 GPU 与 API key。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "test"))
sys.path.insert(0, str(REPO / "eval"))

from robochrono import tasks  # noqa: E402
from robochrono.tasks.base import load_items  # noqa: E402

QA = REPO / "eval/datasets/QA"
SAMPLE = 12


def frozen_parts(run: str, item: dict[str, Any], qa_path: Path,
                 group: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """用冻结脚本自己的函数组装请求内容。"""
    if run == "understanding":
        import understanding_glm_test as m
        prompt = m.build_prompt(str(item.get("Q") or item.get("question")))
        paths = m.media_paths_for_item(item, "clip_path")
        return m.video_parts_for_item(paths, prompt)

    if run == "planning":
        import planning_glm_test as m
        prompt = m.build_prompt(str(item.get("Q") or item.get("question")))
        return m.video_parts_for_item(m.video_paths_for_item(item), prompt)

    if run == "planning_2":
        import planning_2_glm_test as m
        prompt = m.build_prompt(str(item.get("Q") or item.get("question")))
        return m.image_parts_for_item(m.image_paths_for_item(item), prompt)

    if run == "left_right":
        import left_right_glm_test as m
        return m.content_for_item(item, m.build_prompt(item))

    if run == "image_in_video":
        import image_in_video_glm_test as m
        return m.content_for_item(item, m.build_prompt(item), qa_path)

    if run == "step_order":
        import step_order_glm_test as m
        paths = m.image_paths_for_item(item, qa_path)
        return m.image_parts_for_item(paths, m.build_prompt(item))

    if run.startswith("trajectory"):
        import trajectory_glm_test as m
        inputs = m.image_inputs_for_item(item)
        prompt = m.build_prompt(str(item.get("Q") or item.get("question")), item)
        return m.image_parts_for_item(inputs, prompt)

    if run == "time":
        import time_eqa_glm_test_multi as m
        assert group is not None
        prompt = m.build_multi_prompt(group)
        exts = (".mp4", ".webm", ".mov", ".mkv", ".avi")
        paths = m.video_paths_for_item(group[0], qa_path.parent, exts)
        return m.video_parts_for_group(paths, prompt)

    raise ValueError(run)


def normalize(parts: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """归一成 (类型, 内容) 序列。路径统一成绝对路径以消除写法差异。"""
    out: list[tuple[str, str]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            out.append(("text", str(part.get("text", ""))))
        else:
            out.append((str(kind), str(Path(str(part["path"])).resolve())))
    return out


def main() -> int:
    print(f"{'run':<16} {'units':>6} {'prompt 不一致':>13} {'媒体不一致':>11}  status")
    print("-" * 68)
    failures = 0

    for run in tasks.ALL_RUNS:
        qa_path = tasks.qa_path(QA.parent, "stack_cubes", run)
        if not qa_path.exists():
            print(f"{run:<16} {'-':>6} {'-':>13} {'-':>11}  QA 缺失")
            failures += 1
            continue

        items = load_items(qa_path)
        task = tasks.build(run)
        units = task.units(items)[:SAMPLE]

        bad_prompt = bad_media = 0
        first = ""
        for unit in units:
            new = normalize(task.parts(unit))
            old = normalize(frozen_parts(
                run, unit.items[0], qa_path,
                group=unit.items if run == "time" else None,
            ))

            new_text = [v for k, v in new if k == "text"]
            old_text = [v for k, v in old if k == "text"]
            new_media = [(k, v) for k, v in new if k != "text"]
            old_media = [(k, v) for k, v in old if k != "text"]

            if new_text != old_text:
                bad_prompt += 1
                if not first:
                    for a, b in zip(old_text, new_text):
                        if a != b:
                            first = f"{unit.key}: 文本差异 …{a[-40:]!r} vs …{b[-40:]!r}"
                            break
                    else:
                        first = f"{unit.key}: 文本段数 {len(old_text)} vs {len(new_text)}"
            if new_media != old_media:
                bad_media += 1
                if not first:
                    first = f"{unit.key}: 媒体 {old_media[:2]} vs {new_media[:2]}"

        ok = bad_prompt == 0 and bad_media == 0
        failures += 0 if ok else 1
        print(f"{run:<16} {len(units):>6} {bad_prompt:>13} {bad_media:>11}  "
              f"{'OK' if ok else first[:40]}")

    print("-" * 68)
    print("请求内容完全一致" if failures == 0 else f"{failures} 个任务存在差异")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
