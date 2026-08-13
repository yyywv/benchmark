#!/usr/bin/env python3
# coding: utf-8
"""执行引擎：八个冻结脚本 ``main()`` 的公共骨架。

那八个 ``main()`` 结构完全相同 —— 读输入、断点恢复、逐条调用、异常兜底、
写 checkpoint、汇总 —— 只是各自复制了一份。这里收敛成一个循环，
任务相关的部分全部通过 Task 协议下沉。

相对冻结版的行为改动：
  - 结果走 JSONL 追加而非每题重写整个文件（BC-04）
  - 单个 run 连续失败达到阈值就熔断，避免一个坏配置白烧几小时
  - 记录每次调用实际用了多少帧、以及服务端返回的 usage（BC-09 管道改造）
其余（打分、汇总、失败时的占位行）与冻结版逐字节一致。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .store import ResultStore
from .tasks.base import CallContext, Unit
from .vlm_api import call_vlm

DEFAULT_CIRCUIT_BREAKER = 20


def limit_units(units: list[Unit], limit_items: int | None, limit_groups: int | None) -> list[Unit]:
    """BC-05：``--limit`` 语义拆分。

    冻结版里 ``time`` 的 ``--limit`` 限的是视频数，其余脚本限的是题数，
    做批量时这个不一致会让「每族抽 N 题冒烟」出错。改成两个显式参数。
    """
    if limit_groups is not None:
        if limit_groups < 0:
            raise ValueError("--limit-groups must be non-negative")
        units = units[:limit_groups]
    if limit_items is not None:
        if limit_items < 0:
            raise ValueError("--limit-items must be non-negative")
        kept: list[Unit] = []
        seen = 0
        for unit in units:
            if seen >= limit_items:
                break
            kept.append(unit)
            seen += len(unit.items)
        units = kept
    return units


def run(
    task: Any,
    items: list[dict[str, Any]],
    runtime: dict[str, Any],
    store: ResultStore,
    *,
    limit_items: int | None = None,
    limit_groups: int | None = None,
    overwrite: bool = False,
    circuit_breaker: int = DEFAULT_CIRCUIT_BREAKER,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """跑完一个 (模型, 任务) 组合，返回汇总。"""
    units = limit_units(task.units(items), limit_items, limit_groups)

    if overwrite and store.path.exists():
        store.path.unlink()
    store.open()

    done = set() if overwrite else store.completed_ids()
    pending = [u for u in units if not all(str(i.get("id")) in done for i in u.items)]

    total_items = sum(len(u.items) for u in units)
    print(f"  units={len(units)} pending={len(pending)} items={total_items}", flush=True)

    started = time.perf_counter()
    consecutive_failures = 0
    aborted: str | None = None

    for index, unit in enumerate(pending, 1):
        unit_started = time.perf_counter()
        meta: dict[str, Any] = {}
        runtime["replay_key"] = unit.key          # replay provider 用
        try:
            _, text = call_vlm(runtime, task.parts(unit), meta)

            # trajectory 的 2D 越界重问。其余任务没有这个钩子。
            retry_hook = getattr(task, "retry_parts", None)
            attempt = 0
            while retry_hook is not None:
                retry = retry_hook(unit, text, attempt)
                if retry is None:
                    break
                attempt += 1
                print(f"    retry {attempt}: out-of-bounds 2D points", flush=True)
                _, text = call_vlm(runtime, retry, meta)

            ctx = CallContext(
                frames_used=meta.get("frames_used", {}),
                usage=meta.get("usage", {}),
                media_transforms=meta.get("media_transforms", []),
            )
            rows = task.rows(unit, text, ctx)
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001
            rows = task.error_rows(unit, f"{type(exc).__name__}: {exc}")
            consecutive_failures += 1
            print(f"    error: {type(exc).__name__}: {exc}", flush=True)

        seconds = round(time.perf_counter() - unit_started, 3)
        for row in rows:
            row["timing"] = {"seconds": seconds}
            if on_row is not None:
                on_row(row)
        store.append(rows)

        if index % 10 == 0 or index == len(pending):
            print(f"    [{index}/{len(pending)}] {unit.key}", flush=True)

        if circuit_breaker and consecutive_failures >= circuit_breaker:
            aborted = f"circuit breaker: {consecutive_failures} consecutive failures"
            print(f"  ABORTED — {aborted}", flush=True)
            break

    all_rows = list(store.rows())
    summary = task.summarize(all_rows, time.perf_counter() - started)
    if aborted:
        summary["aborted"] = aborted
    return summary
