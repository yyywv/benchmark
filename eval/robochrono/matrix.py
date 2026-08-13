#!/usr/bin/env python3
# coding: utf-8
"""矩阵展开：把 (模型 × 任务族 × 任务) 展成一张 run 列表。

冻结版没有这个概念 —— 每个组合都是手敲一条命令。15 模型 × 20 族 × 9 任务
意味着最多 2700 条命令，必须由程序展开。

三件事在这里完成：
  稀疏规则   不是每个任务都适用于每个族（比如视角识别只测双手任务）
  分片       多机分工，按稳定哈希切分，机器之间互不重叠也不遗漏
  排序       本地模型按 model-major 排，让同一个模型的所有 run 连在一起，
             权重只加载一次
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import tasks


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model: str | None = None
    kind: str = "local"          # local | api
    note: str = ""

    @property
    def is_local(self) -> bool:
        return self.kind == "local"


@dataclass(frozen=True)
class RunSpec:
    model: ModelSpec
    family: str
    run: str

    @property
    def key(self) -> str:
        return f"{self.model.name}__{self.family}__{self.run}"

    def qa_path(self, datasets_root: Path) -> Path:
        return tasks.qa_path(datasets_root, self.family, self.run)


@dataclass
class Plan:
    models: list[ModelSpec] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    family_attrs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_rules: list[dict[str, Any]] = field(default_factory=list)


def load_plan(path: Path) -> Plan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    models = [
        ModelSpec(
            name=str(m["name"]),
            provider=str(m["provider"]),
            model=m.get("model"),
            kind=str(m.get("kind", "local")),
            note=str(m.get("note", "")),
        )
        for m in raw.get("models", [])
    ]
    return Plan(
        models=models,
        families=[str(f) for f in raw.get("families", [])],
        runs=[str(r) for r in raw.get("runs", tasks.ALL_RUNS)],
        family_attrs=raw.get("family_attrs", {}),
        skip_rules=raw.get("skip_rules", []),
    )


def _skipped(plan: Plan, family: str, run: str) -> str | None:
    """返回跳过原因，None 表示不跳过。"""
    attrs = plan.family_attrs.get(family, {})
    for rule in plan.skip_rules:
        if rule.get("run") and rule["run"] != run:
            continue
        unless = rule.get("unless")
        if unless and not attrs.get(unless):
            return f"{run} requires family attribute `{unless}`"
        only_if = rule.get("only_if")
        if only_if and not attrs.get(only_if):
            return f"{run} requires family attribute `{only_if}`"
    return None


def shard_of(key: str, shards: int) -> int:
    """稳定哈希分片。同一个 key 在任何机器上都落到同一片。"""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % shards


def expand(
    plan: Plan,
    datasets_root: Path,
    *,
    shard: tuple[int, int] | None = None,
    only_kind: str | None = None,
) -> tuple[list[RunSpec], list[tuple[str, str]]]:
    """展开矩阵。返回 (要跑的 run 列表, [(key, 跳过原因)])。

    ``shard`` 形如 (1, 4) 表示「四台机器里的第一台」。
    """
    selected: list[RunSpec] = []
    skipped: list[tuple[str, str]] = []

    for model in plan.models:
        if only_kind and model.kind != only_kind:
            continue
        for family in plan.families:
            for run in plan.runs:
                spec = RunSpec(model=model, family=family, run=run)

                reason = _skipped(plan, family, run)
                if reason:
                    skipped.append((spec.key, reason))
                    continue
                if not spec.qa_path(datasets_root).exists():
                    skipped.append((spec.key, f"QA 文件缺失 {spec.qa_path(datasets_root)}"))
                    continue
                if shard is not None:
                    index, total = shard
                    if shard_of(spec.key, total) != index - 1:
                        continue
                selected.append(spec)

    # model-major：同一个模型的所有 run 连在一起，本地权重只加载一次。
    # 15 模型 × 9 任务原本要加载 135 次，排序后降到 15 次。
    selected.sort(key=lambda s: (s.model.name, s.family, s.run))
    return selected, skipped
