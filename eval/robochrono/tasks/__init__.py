#!/usr/bin/env python3
# coding: utf-8
"""任务注册表。

九个 run 对应八个任务类型 —— trajectory 的 2D 与 3D 是两份独立输入、
两次独立运行、两组独立分数，所以按 run 建模。
"""

from __future__ import annotations

from typing import Any

from . import choice, time_eqa, trajectory

# run 名 -> 该 run 在生成产物里对应的 QA 文件名
QA_FILENAME: dict[str, str] = {
    "time": "time_vqa.json",
    "understanding": "understanding_vqa.json",
    "left_right": "left_right_vqa.json",
    "image_in_video": "image_in_video_vqa.json",
    # BC-06：planning 与 planning_2 各自指向自己的输入。
    # 冻结版两个脚本都读 config 的 tasks.planning，而那一节当前指向
    # planning_2_vqa.json —— 跑 planning 不带 --input 会直接崩。
    "planning": "planning_vqa.json",
    "planning_2": "planning_2_vqa.json",
    "step_order": "step_order_vqa.json",
    "trajectory_2D": "trajectory_qa_2d.json",
    "trajectory_3D": "trajectory_qa_3d.json",
}

# run 名 -> 数据所在的组目录（生成流水线把产物分成两组）
QA_GROUP: dict[str, str] = {
    "time": "understanding",
    "understanding": "understanding",
    "left_right": "understanding",
    "image_in_video": "understanding",
    "planning": "planning",
    "planning_2": "planning",
    "step_order": "planning",
    "trajectory_2D": "planning",
    "trajectory_3D": "planning",
}

# run 名 -> 报表里的主指标
PRIMARY_METRIC: dict[str, str] = {
    "understanding": "accuracy",
    "left_right": "accuracy",
    "planning": "accuracy",
    "planning_2": "accuracy",
    "step_order": "accuracy",
    "image_in_video": "accuracy",
    "trajectory_2D": "mean_score",
    "trajectory_3D": "mean_score",
    "time": "mean_tIoU",
}

ALL_RUNS: tuple[str, ...] = tuple(QA_FILENAME)


def build(name: str, **flags: Any):
    """按 run 名构造任务实例。"""
    if name == "time":
        return time_eqa.build(**flags)
    if name.startswith("trajectory"):
        return trajectory.build(name, **flags)
    if name in choice.SPECS:
        return choice.build(name, **flags)
    raise ValueError(f"unknown run {name!r}; known runs: {list(ALL_RUNS)}")


def qa_path(datasets_root: Any, family: str, run: str) -> Any:
    """<datasets_root>/QA/<group>/<family>/<file>"""
    from pathlib import Path

    return Path(datasets_root) / "QA" / QA_GROUP[run] / family / QA_FILENAME[run]
