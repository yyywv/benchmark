#!/usr/bin/env python3
# coding: utf-8
"""矩阵执行：model-major 调度，本地模型用 GPU worker 池。

把 (模型 × 任务族 × 任务) 的矩阵按模型分组依次执行。对本地模型，
该模型下所有任务族、所有任务的 unit 汇成一个队列摊给各卡；
对 API 模型，走串行路径（并发留待后续，先保证正确）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import engine, pool, tasks
from .matrix import ModelSpec, Plan, RunSpec
from .store import ResultStore
from .tasks.base import load_items
from .vlm_api import runtime_config


def _store_for(spec: RunSpec, results_root: Path, runtime_meta: dict[str, Any]) -> ResultStore:
    # 抽帧档位单独分目录，fps=1 与 fps=2 的结果不会互相覆盖
    out_dir = results_root / spec.model.name / spec.family / spec.variant
    return ResultStore(out_dir / f"{spec.run}.jsonl", meta=runtime_meta)


def _apply_frames(runtime: dict[str, Any], spec: RunSpec) -> dict[str, Any]:
    """把该 run 的抽帧档位写进 runtime。

    frames 只影响预处理、不影响权重，所以同一次模型加载可以服务多个档位，
    不需要为 fps=1/fps=2 各加载一遍。
    """
    if spec.frames_fps is None:
        return runtime
    runtime = dict(runtime)
    runtime["frames"] = {
        "mode": "fps",
        "value": spec.frames_fps,
        "video_sample_fps": spec.frames_fps,
        "num_segments": 1,
    }
    # 团队定的口径：按实际帧数对齐，num_segments 型换算成 round(时长 × fps)
    runtime["align_fps_to_segments"] = True
    return runtime


def _meta(spec: RunSpec, runtime: dict[str, Any], qa_path: Path, flags: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": spec.model.name,
        "provider": runtime["provider"],
        "model": runtime["model"],
        "api_url": runtime["api_url"],
        "family": spec.family,
        "run": spec.run,
        "input": str(qa_path),
        "frames": runtime["frames"],
        "generation": {
            "temperature": runtime["temperature"],
            "thinking": runtime["thinking"],
            "max_new_tokens": runtime["max_new_tokens"],
        },
        "flags": dict(flags),
        "frames_variant": spec.variant,
    }


def run_matrix(
    plan: Plan,
    specs: list[RunSpec],
    *,
    config_path: Path,
    datasets_root: Path,
    results_root: Path,
    gpus: list[int],
    flags: dict[str, Any],
    limit_items: int | None = None,
    limit_groups: int | None = None,
    overwrite: bool = False,
) -> int:
    by_model: dict[str, list[RunSpec]] = {}
    for spec in specs:
        by_model.setdefault(spec.model.name, []).append(spec)

    failures = 0
    for model_name, model_specs in by_model.items():
        model: ModelSpec = model_specs[0].model
        print(f"\n{'='*70}\n模型 {model_name}（{model.kind}，{len(model_specs)} 个 run）\n{'='*70}")

        use_pool = model.is_local and len(gpus) > 1
        if use_pool:
            failures += _run_local_pool(
                model, model_specs, config_path=config_path, datasets_root=datasets_root,
                results_root=results_root, gpus=gpus, flags=flags,
                limit_items=limit_items, limit_groups=limit_groups, overwrite=overwrite,
            )
        else:
            failures += _run_serial(
                model, model_specs, config_path=config_path, datasets_root=datasets_root,
                results_root=results_root, flags=flags,
                limit_items=limit_items, limit_groups=limit_groups, overwrite=overwrite,
            )
    return failures


def _prepare(spec: RunSpec, datasets_root: Path, config_path: Path, model: ModelSpec,
             flags: dict[str, Any], results_root: Path, overwrite: bool):
    qa_path = spec.qa_path(datasets_root)
    runtime = runtime_config(config_path=config_path, provider_name=model.provider,
                             default_model="", cli_model=model.model)
    runtime = _apply_frames(runtime, spec)
    store = _store_for(spec, results_root, _meta(spec, runtime, qa_path, flags))
    if overwrite and store.path.exists():
        store.path.unlink()
    store.open()
    items = load_items(qa_path)
    task_flags = flags if spec.run in pool._CHOICE_RUNS else {}
    return qa_path, runtime, store, items, tasks.build(spec.run, **task_flags)


def _run_serial(model, model_specs, *, config_path, datasets_root, results_root,
                flags, limit_items, limit_groups, overwrite) -> int:
    failures = 0
    for spec in model_specs:
        print(f"\n--- {spec.family} × {spec.run} ---")
        try:
            _, runtime, store, items, task = _prepare(
                spec, datasets_root, config_path, model, flags, results_root, overwrite)
            summary = engine.run(task, items, runtime, store,
                                 limit_items=limit_items, limit_groups=limit_groups,
                                 overwrite=False)
            _write_summary(store, summary, spec)
        except Exception as exc:  # noqa: BLE001
            print(f"  RUN FAILED: {type(exc).__name__}: {exc}")
            failures += 1
    return failures


def _run_local_pool(model, model_specs, *, config_path, datasets_root, results_root,
                    gpus, flags, limit_items, limit_groups, overwrite) -> int:
    """该模型的所有 run 汇成一个队列，摊给各卡。权重每卡只加载一次。"""
    work: list[pool.WorkItem] = []
    stores: dict[str, ResultStore] = {}
    contexts: dict[str, tuple[RunSpec, Any]] = {}

    for spec in model_specs:
        try:
            _, _, store, items, task = _prepare(
                spec, datasets_root, config_path, model, flags, results_root, overwrite)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {spec.family}/{spec.run}: {type(exc).__name__}: {exc}")
            continue

        stores[spec.key] = store
        contexts[spec.key] = (spec, task)
        done = store.completed_ids()
        units = engine.limit_units(task.units(items), limit_items, limit_groups)
        for unit in units:
            if all(str(i.get("id")) in done for i in unit.items):
                continue
            work.append(pool.WorkItem(spec.key, spec.run, unit.key, unit.items,
                                      frames_fps=spec.frames_fps))

    if not work:
        print("  全部已完成，无需执行")
    else:
        print(f"  {len(work)} 个 unit 摊给 {len(gpus)} 张卡")

        def on_result(result: pool.WorkResult) -> None:
            store = stores.get(result.spec_key)
            if store is not None:
                store.append(result.rows)

        stats = pool.run_pool(
            work, gpus=gpus, config_path=config_path, provider=model.provider,
            model_override=model.model, task_flags=flags, on_result=on_result,
        )
        print(f"  完成 {stats['done']}，错误 {stats['errors']}")

    for key, (spec, task) in contexts.items():
        store = stores[key]
        summary = task.summarize(list(store.rows()), 0.0)
        _write_summary(store, summary, spec)
    return 0


def _write_summary(store: ResultStore, summary: dict[str, Any], spec: RunSpec) -> None:
    path = store.path.with_name(f"{spec.run}.summary.json")  # 与 jsonl 同目录，已按 variant 分开
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metric = tasks.PRIMARY_METRIC[spec.run]
    print(f"  {spec.family}/{spec.run}: {metric} = {summary.get(metric)}  "
          f"(answered {summary.get('answered')}/{summary.get('total')}, "
          f"errors {summary.get('errors')})")
