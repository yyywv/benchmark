#!/usr/bin/env python3
# coding: utf-8
"""API 请求体预算适配（BC-11）。

远程 provider 把媒体 base64 内联进请求体，而服务端有请求体大小上限。
实测阿里云 MaaS 端点：base64 10.32 MB 通过，10.99 MB 返回
``413 RequestTooLarge`` —— 上限约 10 MiB。

我们的媒体体积（base64 后）：

    planning_2 单帧        0.31 MB   安全
    left_right 选项图 ×7   0.77 MB   安全
    image_in_video clip    0.76 MB   基本安全
    understanding clip     3.77 MB   中位安全，最大 19.96 MB 会失败
    planning clip          3.64 MB   同上
    time 整段视频          10.47 MB  基本全部失败

也就是说不做任何处理的话，**5 个 API 模型的 Time EQA 完全跑不了**。

本模块在超预算时对视频降分辨率重编码，使其落回预算内。三条原则：

1. **默认不启用。** 只有 provider 配置里给了 ``max_request_bytes`` 才生效，
   否则原样发送、超了就让它 413（失败会被如实记录，不会静默）。
2. **优先降空间分辨率，不动帧率与时长。** 时间信息正是 Time EQA 要测的东西，
   抽帧由下游服务端决定，我们不该在这里再削一刀。
3. **每次变换都记录**（原始/压缩后大小、缩放比例），写进结果行，可审计。

产物按内容缓存，同一个视频只重编码一次。
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

# base64 把 3 字节编成 4 字符
BASE64_RATIO = 4 / 3
# 除媒体外还有 prompt 文本与 JSON 结构，留一点余量
OVERHEAD_BYTES = 64 * 1024

# 依次尝试的缩放比例。先只降分辨率，最后才动画质。
SCALE_LADDER = (0.75, 0.5, 0.375, 0.25, 0.1875)
CRF_LADDER = (23, 28, 32)


def encoded_size(path: Path) -> int:
    return int(path.stat().st_size * BASE64_RATIO)


def _cache_path(source: Path, scale: float, crf: int, cache_dir: Path) -> Path:
    stat = source.stat()
    digest = hashlib.md5(
        f"{source.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{scale}|{crf}".encode()
    ).hexdigest()[:16]
    return cache_dir / f"{source.stem}__{digest}.mp4"


def _reencode(source: Path, dest: Path, scale: float, crf: int) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # trunc(...*scale/2)*2 保证宽高是偶数，libx264 要求
    vf = f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-an", str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def shrink_video(source: Path, budget_bytes: int, cache_dir: Path) -> tuple[Path, dict[str, Any] | None]:
    """把视频压到 base64 后不超过 budget_bytes。返回 (可用路径, 变换记录)。

    压不下去时返回原路径与 ``{"failed": True}``，由调用方决定是放弃还是硬发。
    """
    if encoded_size(source) <= budget_bytes:
        return source, None

    for crf in CRF_LADDER:
        for scale in SCALE_LADDER:
            dest = _cache_path(source, scale, crf, cache_dir)
            if dest.exists() and dest.stat().st_size > 0:
                if encoded_size(dest) <= budget_bytes:
                    return dest, _record(source, dest, scale, crf)
                continue
            if not _reencode(source, dest, scale, crf):
                continue
            if encoded_size(dest) <= budget_bytes:
                return dest, _record(source, dest, scale, crf)
            dest.unlink(missing_ok=True)

    return source, {
        "source": str(source),
        "failed": True,
        "reason": "无法压缩到预算内",
        "source_encoded_bytes": encoded_size(source),
        "budget_bytes": budget_bytes,
    }


def _record(source: Path, dest: Path, scale: float, crf: int) -> dict[str, Any]:
    return {
        "source": str(source),
        "prepared": str(dest),
        "scale": scale,
        "crf": crf,
        "source_encoded_bytes": encoded_size(source),
        "prepared_encoded_bytes": encoded_size(dest),
    }


def prepare_parts(
    parts: list[dict[str, Any]],
    max_request_bytes: int,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按整个请求的预算处理 parts。返回 (新 parts, 变换记录列表)。

    预算是**整个请求**的，不是单个文件的 —— image_in_video 会同时发一个视频
    和六张选项图。图片不动（体积小且是答案本身），只压视频。
    """
    media = [p for p in parts if p.get("type") in {"image", "video"}]
    if not media:
        return parts, []

    budget = max_request_bytes - OVERHEAD_BYTES
    image_bytes = sum(encoded_size(Path(p["path"])) for p in media if p["type"] == "image")
    videos = [p for p in media if p["type"] == "video"]
    total = image_bytes + sum(encoded_size(Path(p["path"])) for p in videos)

    if total <= budget or not videos:
        return parts, []

    # 视频之间平分剩余预算
    per_video = max(1, (budget - image_bytes) // len(videos))
    transforms: list[dict[str, Any]] = []
    replacement: dict[str, str] = {}

    for part in videos:
        source = Path(part["path"])
        prepared, record = shrink_video(source, per_video, cache_dir)
        if record is not None:
            transforms.append(record)
        if prepared != source:
            replacement[part["path"]] = str(prepared)

    if not replacement:
        return parts, transforms

    new_parts = [
        {**p, "path": replacement[p["path"]]} if p.get("path") in replacement else p
        for p in parts
    ]
    return new_parts, transforms


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None
