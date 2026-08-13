#!/usr/bin/env python3
# coding: utf-8
"""命令行入口。

    python -m robochrono list
    python -m robochrono run --provider <name> --family stack_cubes --runs understanding
    python -m robochrono export --results-dir <dir>

冻结版是八条互不相同的命令、每条一套参数；这里是一条命令、一套参数、
一个 (模型 × 任务族 × 任务) 的矩阵。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import engine, tasks
from .store import ResultStore
from .tasks.base import load_items
from .vlm_api import runtime_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/providers.json"
DEFAULT_DATASETS = Path(__file__).resolve().parents[1] / "datasets"
DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results"
KEYS_FILE = Path.home() / ".config/robochrono/keys.env"


def load_keys(path: Path = KEYS_FILE) -> None:
    """从仓库之外的 keys.env 读 API key，避免密钥进 git。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip())


def cmd_list(args: argparse.Namespace) -> int:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print("runs:")
    for run in tasks.ALL_RUNS:
        print(f"  {run:<16} {tasks.QA_GROUP[run]:<14} {tasks.QA_FILENAME[run]:<26} "
              f"主指标 {tasks.PRIMARY_METRIC[run]}")
    print("\nproviders:")
    for name, provider in config.get("providers", {}).items():
        print(f"  {name:<38} {provider.get('type')}")
    root = Path(args.datasets_root) / "QA"
    if root.exists():
        families = sorted({p.name for group in root.iterdir() if group.is_dir()
                           for p in group.iterdir() if p.is_dir()})
        print(f"\nfamilies: {', '.join(families) or '(无)'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    load_keys()
    config_path = Path(args.config)
    runs = args.runs or list(tasks.ALL_RUNS)
    results_root = Path(args.results_dir)

    failures = 0
    for family in args.families:
        for run in runs:
            qa_path = tasks.qa_path(args.datasets_root, family, run)
            if not qa_path.exists():
                print(f"[skip] {family}/{run}: QA 文件缺失 {qa_path}")
                continue

            print(f"\n=== {args.provider} × {family} × {run} ===")
            runtime = runtime_config(
                config_path=config_path,
                provider_name=args.provider,
                default_model="",
                cli_model=args.model,
                cli_temperature=args.temperature,
                cli_timeout=args.timeout,
                cli_max_retries=args.max_retries,
            )
            print(f"  provider={runtime['provider']} model={runtime['model']} "
                  f"frames={runtime['frames']['mode']}:{runtime['frames']['value']}")

            task = tasks.build(run, **{k: v for k, v in (
                ("strip_reasoning", args.strip_reasoning),
                ("null_text_fix", args.null_text_fix),
            ) if run in ("understanding", "left_right", "image_in_video",
                         "planning", "planning_2", "step_order")})

            items = load_items(qa_path)
            out_dir = results_root / args.provider / family
            store = ResultStore(
                out_dir / f"{run}.jsonl",
                meta={
                    "provider": runtime["provider"],
                    "model": runtime["model"],
                    "api_url": runtime["api_url"],
                    "family": family,
                    "run": run,
                    "input": str(qa_path),
                    "frames": runtime["frames"],
                    "generation": {
                        "temperature": runtime["temperature"],
                        "thinking": runtime["thinking"],
                        "max_new_tokens": runtime["max_new_tokens"],
                    },
                    "flags": {
                        "strip_reasoning": args.strip_reasoning,
                        "null_text_fix": args.null_text_fix,
                    },
                },
            )

            try:
                summary = engine.run(
                    task, items, runtime, store,
                    limit_items=args.limit_items,
                    limit_groups=args.limit_groups,
                    overwrite=args.overwrite,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  RUN FAILED: {type(exc).__name__}: {exc}")
                failures += 1
                continue

            (out_dir / f"{run}.summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            metric = tasks.PRIMARY_METRIC[run]
            print(f"  {metric} = {summary.get(metric)}  "
                  f"(answered {summary.get('answered')}/{summary.get('total')}, "
                  f"errors {summary.get('errors')})")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robochrono", description="RoboChrono 评测框架")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--datasets-root", default=str(DEFAULT_DATASETS))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出 run、provider 与任务族").set_defaults(func=cmd_list)

    run_parser = sub.add_parser("run", help="跑一个或多个 (任务族 × 任务)")
    run_parser.add_argument("--provider", required=True)
    run_parser.add_argument("--model", default=None)
    run_parser.add_argument("--families", nargs="+", default=["stack_cubes"])
    run_parser.add_argument("--runs", nargs="+", default=None,
                            help=f"默认全部：{' '.join(tasks.ALL_RUNS)}")
    # BC-05：拆分后的两个显式参数
    run_parser.add_argument("--limit-items", type=int, default=None, help="限制题数")
    run_parser.add_argument("--limit-groups", type=int, default=None,
                            help="限制调用组数（time 即视频数）")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--temperature", type=float, default=None)
    run_parser.add_argument("--timeout", type=int, default=None)
    run_parser.add_argument("--max-retries", type=int, default=None)
    # 会改变分数的开关，默认关闭
    run_parser.add_argument("--strip-reasoning", action="store_true",
                            help="BC-02：剥离思考块并启用 JSON 兜底")
    run_parser.add_argument("--null-text-fix", action="store_true",
                            help="BC-10：text 为 null 的选项不参与文本匹配")
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
