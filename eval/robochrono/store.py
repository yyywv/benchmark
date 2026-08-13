#!/usr/bin/env python3
# coding: utf-8
"""结果存储（BC-04）。

冻结版每答完一题就把整个结果文件重写一遍，且每行复制整条原始 item。
trajectory 的 3D 输入每题约 50 KB，一个 300 题任务累计写入约 2.2 GB，
整个 15×20×9 矩阵约 0.8 TB 无谓 IO。

这里改成 JSONL 追加：
  - 每题一行，写完即落盘，进程被杀也不丢已完成的部分
  - 断点恢复只需扫一遍已有行的 id，不必反序列化整个文件
  - 导出时再组装成与冻结版**完全同构**的 JSON，现有分析脚本零改动
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class ResultStore:
    """一个 (模型, 任务族, 任务) 的结果存储。"""

    def __init__(self, path: Path, meta: dict[str, Any] | None = None) -> None:
        self.path = path
        self.meta_path = path.with_suffix(".meta.json")
        self.meta = meta or {}

    # -- 写 ----------------------------------------------------------------

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.meta:
            self.meta_path.write_text(
                json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def append(self, rows: list[dict[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- 读 ----------------------------------------------------------------

    def rows(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def completed_ids(self) -> set[str]:
        """已完成的题目 id。与冻结版 ``is_finished`` 口径一致：
        有 model_output 且没有 error 才算完成，失败的会被重跑。"""
        done: set[str] = set()
        for row in self.rows():
            if row.get("model_output") and not row.get("error"):
                done.add(str(row.get("id")))
        return done

    # -- 导出 --------------------------------------------------------------

    def export(self, output_path: Path, summary: dict[str, Any], items_by_id: dict[str, dict[str, Any]] | None = None) -> None:
        """导出成与冻结版同构的 JSON。

        ``items_by_id`` 给出时，把原始 item 字段合并回每行 —— 这样导出结果与
        冻结版逐字段一致；不给出时导出精简版。
        """
        results = []
        for row in self.rows():
            if items_by_id:
                item = items_by_id.get(str(row.get("id")))
                if item:
                    row = {**item, **row}
            results.append(row)

        payload = {**self.meta, "results": results, "summary": summary}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
