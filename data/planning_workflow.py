#!/usr/bin/env python3
# coding: utf-8
"""Generic workflow for building Planning and Trajectory VQA JSON.

The input format is a directory of `*_segments.json` files. Each file should
contain a `segments` list, and each segment should include at least:

    id, start, end, narration

Optional fields such as objects, main_verbs, frames, and timestamp strings are
preserved in metadata. Video media is resolved by `<video_id>.<ext>`, where
`video_id` is the segment filename without `_segments.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import requests

DEFAULT_DATA_DIR = Path("/home/llm/yyywv/test_vlm/json_pickplace")
DEFAULT_VIDEO_DIR = Path("/home/llm/NAS/自采数据集/yyywv__pickplace__unversioned/video")
DEFAULT_MULTI_VIEW_VIDEO_ROOT = Path("/home/llm/NAS/自采数据集/internal__lerobot_gripper_hand__2026_06_17/gripper/videos")
DEFAULT_TRAJECTORY_DATASET_ROOT = Path("/home/llm/NAS/自采数据集/internal__lerobot_gripper_hand__2026_06_17/gripper")
DEFAULT_TRAJECTORY_VIEWS = "observation.images.left_eye,observation.images.right_eye,observation.images.left_wrist"
DEFAULT_TRAJECTORY_PRIMARY_VIEW = "observation.images.left_eye"
DEFAULT_VIEWS = "left_eye=observation.images.left_eye,left_wrist=observation.images.left_wrist,right_wrist=observation.images.right_wrist"
DEFAULT_OUTPUT_DIR = Path("/home/llm/yyywv/test_vlm/workflow_outputs")
DEFAULT_TIME_CROPPED_VIDEO_DIR = Path("/home/llm/yyywv/test_vlm/workflow_outputs/planning/step_order")
DEFAULT_CATEGORY_LABEL_PATH = Path("/home/llm/yyywv/test_vlm/workflow/stack_all_cubes.txt")
DEFAULT_VIDEO_EXTS = ".mp4,.webm,.mov,.mkv,.avi"
DEFAULT_TASKS = "planning,planning_2,trajectory"
DEFAULT_NUM_OPTIONS = 6

DEFAULT_TIME_QUESTION = 'When did the action "{action}" happen?'
DEFAULT_PLANNING_QUESTION = "Based on the current visual state, what should happen next?"
DEFAULT_PLANNING_2_QUESTION = (
    "The overall task is {task_name}. Based on the current visual state, what should happen next?"
)
DEFAULT_STEP_ORDER_QUESTION = (
    "Given the initial image, the numbered images are shuffled result states from "
    "the same robot episode. Which option arranges the numbered images in the "
    "correct operation order?"
)
NONE_OPTION_TEXT = "All other options are wrong."
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]
CONFIG_PATH_KEYS = {
    "data_dir",
    "video_dir",
    "prejoined_video_dir",
    "multi_view_video_root",
    "output_dir",
    "time_cropped_video_dir",
    "trajectory_dataset_root",
    "trajectory_image_dir",
    "llm_distractor_cache_path",
    "category_label_path",
}
CONFIG_LIST_OR_STRING_KEYS = {"tasks", "video_exts"}
CONFIG_KEYS = {
    "data_dir",
    "video_dir",
    "prejoined_video_dir",
    "multi_view_video_root",
    "views",
    "output_dir",
    "video_exts",
    "tasks",
    "file_limit",
    "num_options",
    "nearby_distractors_per_question",
    "generated_distractors_per_question",
    "llm_distractors_per_label",
    "use_llm_distractors",
    "llm_distractor_api_url",
    "llm_distractor_api_key_env",
    "llm_distractor_api_key",
    "llm_distractor_model",
    "llm_distractor_timeout",
    "llm_distractor_max_retries",
    "llm_distractor_cache_path",
    "category_label_path",
    "window_mode",
    "pick_before_window",
    "place_before_window",
    "default_before_window",
    "after_window",
    "planning_timestamp_key",
    "planning_question",
    "planning_2_question",
    "step_order_question",
    "step_order_view",
    "step_order_initial_frame",
    "step_order_end_offset",
    "step_order_end_offset_ratio",
    "step_order_cell_width",
    "step_order_jpeg_quality",
    "step_order_seed",
    "trajectory_dataset_root",
    "trajectory_views",
    "trajectory_primary_view",
    "trajectory_image_dir",
    "trajectory_internal",
    "trajectory_num_keypoints",
    "trajectory_left_xyz_indices",
    "trajectory_right_xyz_indices",
    "trajectory_decimals",
    "trajectory_fps",
    "trajectory_use_base_to_camera_extrinsic",
    "trajectory_base_to_camera_xyz",
    "trajectory_base_to_camera_rpy",
    "trajectory_overwrite_images",
    "trajectory_skip_errors",
    "trajectory_prompt_template_2d",
    "trajectory_prompt_template_3d",
    "no_media",
}


def default_config() -> dict[str, Any]:
    return {
        "data_dir": DEFAULT_DATA_DIR,
        "video_dir": DEFAULT_VIDEO_DIR,
        "prejoined_video_dir": None,
        "multi_view_video_root": DEFAULT_MULTI_VIEW_VIDEO_ROOT,
        "views": DEFAULT_VIEWS,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "video_exts": DEFAULT_VIDEO_EXTS,
        "tasks": DEFAULT_TASKS,
        "file_limit": None,
        "num_options": DEFAULT_NUM_OPTIONS,
        "nearby_distractors_per_question": 2,
        "generated_distractors_per_question": 2,
        "llm_distractors_per_label": 6,
        "use_llm_distractors": False,
        "llm_distractor_api_url": "",
        "llm_distractor_api_key_env": "OPENAI_API_KEY",
        "llm_distractor_api_key": "",
        "llm_distractor_model": "",
        "llm_distractor_timeout": 120,
        "llm_distractor_max_retries": 3,
        "llm_distractor_cache_path": None,
        "category_label_path": DEFAULT_CATEGORY_LABEL_PATH,
        "window_mode": "raw",
        "pick_before_window": 10.0,
        "place_before_window": 2.0,
        "default_before_window": 2.0,
        "after_window": 1.0,
        "planning_timestamp_key": "start",
        "time_question": DEFAULT_TIME_QUESTION,
        "planning_question": DEFAULT_PLANNING_QUESTION,
        "planning_2_question": DEFAULT_PLANNING_2_QUESTION,
        "step_order_question": DEFAULT_STEP_ORDER_QUESTION,
        "step_order_view": "left_eye",
        "step_order_initial_frame": 0,
        "step_order_end_offset": 10,
        "step_order_end_offset_ratio": 0.05,
        "step_order_cell_width": 320,
        "step_order_jpeg_quality": 95,
        "step_order_seed": 42,
        "trajectory_dataset_root": DEFAULT_TRAJECTORY_DATASET_ROOT,
        "trajectory_views": DEFAULT_TRAJECTORY_VIEWS,
        "trajectory_primary_view": DEFAULT_TRAJECTORY_PRIMARY_VIEW,
        "trajectory_image_dir": None,
        "trajectory_internal": "gripper",
        "trajectory_num_keypoints": 10,
        "trajectory_left_xyz_indices": "0:3",
        "trajectory_right_xyz_indices": "7:10",
        "trajectory_decimals": 6,
        "trajectory_fps": 20.0,
        "trajectory_use_base_to_camera_extrinsic": True,
        "trajectory_base_to_camera_xyz": [0.093353689, 0.033, 1.260691643],
        "trajectory_base_to_camera_rpy": [-2.3562, 0.0, -1.5708],
        "trajectory_overwrite_images": False,
        "trajectory_skip_errors": False,
        "trajectory_prompt_template_2d": (
            "{category}. You are given synchronized images from three camera views: {views}. "
            "Use all views as context, but predict the key trajectory points **in the main-view image** "
            "({primary_view}) needed to complete this task from the main viewpoint onward."
        ),
        "trajectory_prompt_template_3d": (
            "{category}. You are given synchronized images from three camera views: {views}. "
            "Use all views as context, but predict the key **3D** trajectory points (in meters) "
            "needed to complete this task from the main viewpoint ({primary_view}) onward."
        ),
        "crop_time_video_top": False,
        "time_crop_top_fraction": 0.1,
        "time_cropped_video_dir": None,
        "overwrite_time_crop": False,
        "no_media": False,
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def output_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def relative_path_text(value: str | os.PathLike[str], base_dir: Path) -> str:
    text = os.fspath(value)
    if not Path(text).is_absolute():
        return text
    try:
        return os.path.relpath(text, base_dir)
    except ValueError:
        return text


def relativize_paths_for_json(data: Any, base_dir: Path) -> Any:
    if isinstance(data, dict):
        return {key: relativize_paths_for_json(value, base_dir) for key, value in data.items()}
    if isinstance(data, list):
        return [relativize_paths_for_json(value, base_dir) for value in data]
    if isinstance(data, tuple):
        return [relativize_paths_for_json(value, base_dir) for value in data]
    if isinstance(data, Path):
        return relative_path_text(data, base_dir)
    if isinstance(data, str) and Path(data).is_absolute():
        return relative_path_text(data, base_dir)
    return data


def save_json(path: Path, data: Any, skip_existing: bool = False) -> bool:
    if skip_existing and output_exists(path):
        print(f"Skip existing JSON: {path}", flush=True)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    data = relativize_paths_for_json(data, path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    config_values = {key: value for key, value in config.items() if not key.startswith("_")}
    unknown_keys = sorted(set(config_values) - CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {path}: {unknown_keys}")
    return config_values


def normalize_config_value(key: str, value: Any) -> Any:
    if key in CONFIG_PATH_KEYS:
        if value is None:
            return None
        text = str(value).strip()
        return Path(text) if text else None
    if key in CONFIG_LIST_OR_STRING_KEYS and isinstance(value, list):
        return ",".join(str(part).strip() for part in value if str(part).strip())
    if key == "views" and isinstance(value, dict):
        return ",".join(f"{view_name}={view_dir}" for view_name, view_dir in value.items())
    return value


def merge_config(config_path: Path | None, cli_values: dict[str, Any]) -> dict[str, Any]:
    merged = default_config()
    for key, value in load_config(config_path).items():
        merged[key] = normalize_config_value(key, value)
    for key, value in cli_values.items():
        merged[key] = normalize_config_value(key, value)
    return merged


def clean_text(text: str) -> str:
    cleaned = " ".join(str(text).strip().split()).rstrip(".")
    if cleaned:
        cleaned = cleaned[0].lower() + cleaned[1:]
    return cleaned


def seconds_to_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, rem = divmod(milliseconds, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def normalize_option_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "item"


def segment_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**12, path.name)


def sorted_segment_files(data_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(data_dir.glob("*_segments.json"), key=segment_sort_key)
    if not paths and (data_dir / "json").is_dir():
        paths = sorted((data_dir / "json").glob("*_segments.json"), key=segment_sort_key)
    return paths[:limit] if limit is not None else paths


def video_id_from_path(path: Path) -> str:
    return path.name.removesuffix("_segments.json")


def action_text_for_segment(segment: dict[str, Any]) -> str:
    narration = clean_text(str(segment.get("narration") or ""))
    if narration:
        return narration
    verbs = [clean_text(str(verb)) for verb in segment.get("main_verbs", []) if str(verb).strip()]
    objects = [clean_text(str(obj)) for obj in segment.get("objects", []) if str(obj).strip()]
    parts = verbs + objects
    return " ".join(parts) if parts else "the action"


def segment_verb(segment: dict[str, Any]) -> str:
    verbs = segment.get("main_verbs", [])
    for verb in verbs:
        text = clean_text(str(verb))
        if text:
            return text
    action = action_text_for_segment(segment)
    return action.split()[0] if action.split() else "unknown"


def segment_metadata(segment: dict[str, Any], start: float, end: float, window_type: str) -> dict[str, Any]:
    return {
        "narration": segment.get("narration"),
        "objects": segment.get("objects", []),
        "main_verbs": segment.get("main_verbs", []),
        "original_start": float(segment["start"]),
        "original_end": float(segment["end"]),
        "original_start_time": segment.get("start_time"),
        "original_end_time": segment.get("end_time"),
        "window_start": start,
        "window_end": end,
        "window_type": window_type,
        "start_frame": segment.get("start_frame"),
        "end_frame": segment.get("end_frame"),
    }


def window_for_segment(
    segment: dict[str, Any],
    mode: str,
    pick_before_window: float,
    place_before_window: float,
    default_before_window: float,
    after_window: float,
) -> tuple[float, float, str]:
    segment_start = float(segment["start"])
    segment_end = float(segment["end"])
    verb = segment_verb(segment)

    if mode == "raw":
        return segment_start, segment_end, "raw_segment"

    if mode == "legacy_pickplace":
        if verb == "pick":
            return min(segment_end, segment_start + pick_before_window), segment_end + after_window, "pick_start_offset"
        if verb == "place":
            return max(0.0, segment_end - place_before_window), segment_end + after_window, "place_transition"
        return max(0.0, segment_end - default_before_window), segment_end + after_window, "default_transition"

    if mode == "transition":
        return max(0.0, segment_end - default_before_window), segment_end + after_window, "transition"

    raise ValueError(f"Unknown window mode: {mode}")


def load_segments(data_dir: Path, file_limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted_segment_files(data_dir, file_limit):
        data = load_json(path)
        segments = data.get("segments", [])
        if not isinstance(segments, list):
            raise ValueError(f"{path} must contain a list field `segments`")
        video_id = video_id_from_path(path)
        for index, segment in enumerate(segments):
            if not all(key in segment for key in ("id", "start", "end")):
                raise ValueError(f"{path} segment #{index} must contain id/start/end")
            rows.append({"video_id": video_id, "segment": segment, "source_path": str(path)})
    return rows


def build_time_items(
    segment_rows: list[dict[str, Any]],
    question_template: str,
    window_mode: str,
    pick_before_window: float,
    place_before_window: float,
    default_before_window: float,
    after_window: float,
    prejoined_video_dir: Path | None,
    multi_view_video_root: Path | None,
    views: dict[str, str],
    video_exts: tuple[str, ...],
    crop_time_video_top: bool,
    time_cropped_video_dir: Path,
    time_crop_top_fraction: float,
    overwrite_time_crop: bool,
    no_media: bool,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    prejoined_video_cache: dict[str, Path] = {}
    video_cache: dict[tuple[str, str], Path] = {}
    cropped_video_cache: dict[tuple[str, str], Path] = {}
    joined_video_cache: dict[str, dict[str, Any]] = {}
    for row in segment_rows:
        segment = row["segment"]
        video_id = row["video_id"]
        action_text = action_text_for_segment(segment)
        start, end, window_type = window_for_segment(
            segment,
            window_mode,
            pick_before_window,
            place_before_window,
            default_before_window,
            after_window,
        )
        answer = f"{seconds_to_timestamp(start)}-{seconds_to_timestamp(end)}"
        question = question_template.format(action=action_text, video_id=video_id)
        input_data: dict[str, Any] = {
            "start": start,
            "end": end,
        }
        if prejoined_video_dir is not None:
            prejoined_video = prejoined_video_cache.get(video_id)
            if prejoined_video is None:
                prejoined_video = prejoined_video_path_for(video_id, prejoined_video_dir, video_exts)
                prejoined_video_cache[video_id] = prejoined_video
            input_data.update(
                {
                    "video_path": str(prejoined_video),
                    "video_paths": [str(prejoined_video)],
                    "prejoined_video_path": str(prejoined_video),
                    "source_video_paths": None,
                    "view_order": list(views) if multi_view_video_root is not None else None,
                }
            )
        elif multi_view_video_root is not None and not no_media:
            video_paths = multiview_video_paths_for(video_id, multi_view_video_root, views, video_exts, video_cache)
            original_video_paths = video_paths
            if crop_time_video_top:
                video_paths = maybe_crop_time_videos(
                    video_id=video_id,
                    video_paths=original_video_paths,
                    output_dir=time_cropped_video_dir,
                    top_fraction=time_crop_top_fraction,
                    overwrite=overwrite_time_crop,
                    cache=cropped_video_cache,
                )
            joined_video = joined_video_cache.get(video_id)
            if joined_video is None:
                joined_video = join_multiview_video_paths(
                    video_id=video_id,
                    item_id=video_id,
                    start=0.0,
                    end=None,
                    output_dir=time_cropped_video_dir.parent / "time_joined_videos",
                    video_paths=video_paths,
                    output_suffix="time_joined_views",
                )
                joined_video_cache[video_id] = joined_video
            input_data.update(
                {
                    "video_path": joined_video["video_path"],
                    "video_paths": [joined_video["video_path"]],
                    "joined_video": joined_video,
                    "source_video_paths": joined_video["source_video_paths"],
                    "view_order": joined_video["view_order"],
                    "videos": {
                        view_name: {
                            "view": view_name,
                            "video_path": str(path),
                            "original_video_path": str(original_video_paths[view_name]),
                            "crop_top_applied": crop_time_video_top,
                            "crop_top_fraction": time_crop_top_fraction if crop_time_video_top else None,
                        }
                        for view_name, path in video_paths.items()
                    },
                    "original_video_paths": [str(path) for path in original_video_paths.values()],
                    "crop_top_applied": crop_time_video_top,
                    "crop_top_fraction": time_crop_top_fraction if crop_time_video_top else None,
                }
            )
        items.append(
            {
                "id": str(segment["id"]),
                "source_id": str(segment["id"]),
                "video_id": video_id,
                "type": "time",
                "Q": question,
                "A": answer,
                "question": question,
                "answer": answer,
                "answer_text": answer,
                "answer_seconds": {"start": start, "end": end},
                "answer_action": segment_verb(segment),
                "answer_objects": segment.get("objects", []),
                "input": input_data,
                "evidence": [
                    {
                        "segment_id": str(segment["id"]),
                        "source_path": row["source_path"],
                        "start": start,
                        "end": end,
                        "start_time": seconds_to_timestamp(start),
                        "end_time": seconds_to_timestamp(end),
                        "window_type": window_type,
                    }
                ],
                "metadata": segment_metadata(segment, start, end, window_type),
            }
        )
    return items


def deterministic_option_texts(item_id: str, texts: list[str]) -> list[str]:
    return sorted(texts, key=lambda text: hashlib.md5(f"{item_id}|{text}".encode("utf-8")).hexdigest())


def deterministic_sample(seed: str, texts: list[str], count: int) -> list[str]:
    return deterministic_option_texts(seed, texts)[:count]


def add_unique(texts: list[str], text: str) -> None:
    if normalize_option_text(text) not in {normalize_option_text(existing) for existing in texts}:
        texts.append(text)


def object_text(objects: list[Any] | Any | None) -> str:
    if objects is None:
        return "the object"
    if isinstance(objects, list):
        parts = [clean_text(str(obj)) for obj in objects if str(obj).strip()]
        text = " and ".join(parts) if parts else "the object"
        return option_object_phrase(text)
    text = clean_text(str(objects))
    return option_object_phrase(text or "the object")


def option_object_phrase(text: str) -> str:
    text = clean_text(text)
    if not text:
        return "the object"
    if text.startswith(("the ", "a ", "an ", "this ", "that ")):
        return text
    return f"the {text}"


def fallback_option_texts(
    correct_text: str,
    action: str | None,
    objects: list[Any] | Any | None,
    all_actions: list[str],
    all_objects: list[str],
) -> list[str]:
    action = clean_text(str(action or ""))
    obj = object_text(objects)
    fallback_actions = [verb for verb in all_actions if verb and verb != action]
    for verb in ("pick", "place", "move", "put", "remove", "open", "close"):
        if verb != action and verb not in fallback_actions:
            fallback_actions.append(verb)

    object_candidates = [candidate for candidate in all_objects if normalize_option_text(candidate) != normalize_option_text(obj)]
    for candidate in ("the object", "the target object", "the next object"):
        if normalize_option_text(candidate) != normalize_option_text(obj):
            object_candidates.append(candidate)

    candidates: list[str] = []
    for verb in fallback_actions:
        if verb == "move":
            candidates.append(f"move {obj}")
        else:
            candidates.append(f"{verb} {obj}")

    if action:
        for candidate_obj in object_candidates:
            candidates.append(f"{action} {candidate_obj}")

    for verb in fallback_actions:
        for candidate_obj in object_candidates[:4]:
            candidates.append(f"{verb} {candidate_obj}")

    return [
        text
        for text in candidates
        if normalize_option_text(text) != normalize_option_text(correct_text)
    ]


def labels_from_time_items(time_items: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for item in time_items:
        label = clean_text(str(item["metadata"].get("narration") or item.get("answer_action") or "the action"))
        add_unique(labels, label)
    return labels


def load_category_labels(path: Path | None, fallback_items: list[dict[str, Any]]) -> list[str]:
    if path is None or not path.exists():
        return labels_from_time_items(fallback_items)
    labels: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line or "：" in line:
            _, line = re.split(r"[:：]", line, maxsplit=1)
        add_unique(labels, clean_text(line))
    return labels or labels_from_time_items(fallback_items)


def nearby_action_texts(item: dict[str, Any], time_items: list[dict[str, Any]], count: int) -> list[str]:
    if count <= 0:
        return []
    video_id = str(item["video_id"])
    item_id = str(item["id"])
    correct = clean_text(str(item["metadata"].get("narration") or item.get("answer_action") or "the action"))
    video_items = sorted(
        [row for row in time_items if str(row["video_id"]) == video_id],
        key=lambda row: (float(row["answer_seconds"]["start"]), str(row["id"])),
    )
    index = next((idx for idx, row in enumerate(video_items) if str(row["id"]) == item_id), -1)
    ordered: list[dict[str, Any]] = []
    if index >= 0:
        for offset in range(1, len(video_items)):
            if index - offset >= 0:
                ordered.append(video_items[index - offset])
            if index + offset < len(video_items):
                ordered.append(video_items[index + offset])
    else:
        ordered = deterministic_sample(item_id, [json.dumps(row, sort_keys=True) for row in video_items], len(video_items))
        ordered = [json.loads(row) for row in ordered]

    labels: list[str] = []
    for row in ordered:
        label = clean_text(str(row["metadata"].get("narration") or row.get("answer_action") or "the action"))
        if normalize_option_text(label) != normalize_option_text(correct):
            add_unique(labels, label)
        if len(labels) >= count:
            break
    return labels


def next_time_item(item: dict[str, Any], time_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    video_id = str(item["video_id"])
    item_id = str(item["id"])
    video_items = sorted(
        [row for row in time_items if str(row["video_id"]) == video_id],
        key=lambda row: (float(row["answer_seconds"]["start"]), str(row["id"])),
    )
    index = next((idx for idx, row in enumerate(video_items) if str(row["id"]) == item_id), -1)
    if index < 0 or index + 1 >= len(video_items):
        return None
    return video_items[index + 1]


def item_label(item: dict[str, Any]) -> str:
    return clean_text(str(item["metadata"].get("narration") or item.get("answer_action") or "the action"))


def task_name_from_category_label_path(path: Path | None) -> str:
    if path is None:
        return "unknown"
    return clean_text(path.stem.replace("_", " ").replace("-", " ")) or path.stem


def strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_llm_distractor_response(text: str) -> dict[str, list[str]]:
    cleaned = strip_json_fence(text)
    data = json.loads(cleaned)
    if isinstance(data, dict) and isinstance(data.get("distractors"), dict):
        data = data["distractors"]
    if not isinstance(data, dict):
        raise ValueError(f"LLM distractor response must be a JSON object: {text[:300]}")
    output: dict[str, list[str]] = {}
    for label, values in data.items():
        if not isinstance(values, list):
            continue
        output[clean_text(str(label))] = [clean_text(str(value)) for value in values if str(value).strip()]
    return output


def call_llm_distractor_api(
    labels: list[str],
    per_label: int,
    api_url: str,
    api_key: str,
    model: str,
    timeout: int,
    max_retries: int,
) -> dict[str, list[str]]:
    if not api_url or not api_key or not model:
        raise ValueError("LLM distractor API requires api_url, api_key, and model")
    prompt = f"""Generate wrong action-category labels for robot VQA multiple-choice distractors.

Correct category labels for this task:
{json.dumps(labels, ensure_ascii=False, indent=2)}

For every correct label, generate exactly {per_label} plausible but wrong labels.
Rules:
- Each wrong label must be different from every correct category label listed above.
- Prefer visually plausible robot manipulation actions.
- Vary action verb, object, or target state when possible.
- Do not include numbering or explanations.

Return JSON only with this schema:
{{
  "distractors": {{
    "<correct label>": ["<wrong label 1>", "..."]
  }}
}}
"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(min(60, 2**attempt))
                continue
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return parse_llm_distractor_response(str(content))
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(60, 2**attempt))
                continue
    raise RuntimeError(f"LLM distractor generation failed: {last_error}")


def generated_label_fallbacks(
    correct_text: str,
    action: str | None,
    objects: list[Any] | Any | None,
    all_actions: list[str],
    all_objects: list[str],
    correct_labels: list[str],
    count: int,
) -> list[str]:
    correct_norms = {normalize_option_text(label) for label in correct_labels}
    candidates = [
        text
        for text in fallback_option_texts(correct_text, action, objects, all_actions, all_objects)
        if normalize_option_text(text) not in correct_norms
    ]
    generic = [
        "move toward the target object",
        "move away from the target object",
        "hold the target object",
        "release the target object",
        "place the target object near the workspace center",
        "pick the wrong object",
        "wait without manipulating any object",
        "inspect the workspace without moving the object",
    ]
    for text in generic:
        if normalize_option_text(text) not in correct_norms:
            add_unique(candidates, text)
    return candidates[:count]


def build_llm_distractor_pool(
    category_labels: list[str],
    use_llm: bool,
    per_label: int,
    api_url: str,
    api_key_env: str,
    api_key: str,
    model: str,
    timeout: int,
    max_retries: int,
    cache_path: Path | None,
) -> dict[str, list[str]]:
    cache: dict[str, Any] = {}
    if cache_path and cache_path.exists():
        loaded = load_json(cache_path)
        if isinstance(loaded, dict):
            cache = loaded

    cache_key = "task_category_distractors"
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return {
            clean_text(str(label)): [clean_text(str(value)) for value in values if str(value).strip()]
            for label, values in cached.items()
            if isinstance(values, list)
        }

    pool: dict[str, list[str]] = {}
    resolved_api_key = api_key or os.getenv(api_key_env, "")
    labels = category_labels
    if not use_llm:
        if cache_path:
            save_json(cache_path, cache)
        return pool
    try:
        generated = call_llm_distractor_api(
            labels=labels,
            per_label=per_label,
            api_url=api_url,
            api_key=resolved_api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
        correct_norms = {normalize_option_text(label) for label in labels}
        pool = {
            label: [
                value
                for value in generated.get(label, generated.get(normalize_option_text(label), []))
                if normalize_option_text(value) not in correct_norms
                and normalize_option_text(value) != normalize_option_text(label)
            ][:per_label]
            for label in labels
        }
        cache[cache_key] = pool
    except Exception as exc:
        print(f"Warning: {exc}. Using rule-based generated distractors.", flush=True)
        pool = {}

    if cache_path:
        save_json(cache_path, cache)
    return pool


def build_options(
    item_id: str,
    correct_text: str,
    pool: list[str],
    num_options: int,
    include_none: bool,
    action: str | None,
    objects: list[Any] | Any | None,
    all_actions: list[str],
    all_objects: list[str],
) -> list[dict[str, Any]]:
    if num_options < 2:
        raise ValueError("--num-options must be at least 2")

    option_texts = [correct_text]
    for text in pool:
        if len(option_texts) >= num_options - int(include_none):
            break
        if normalize_option_text(text) != normalize_option_text(correct_text):
            add_unique(option_texts, text)

    if include_none:
        add_unique(option_texts, NONE_OPTION_TEXT)

    for text in fallback_option_texts(correct_text, action, objects, all_actions, all_objects):
        if len(option_texts) >= num_options:
            break
        add_unique(option_texts, text)

    if len(option_texts) < num_options:
        raise ValueError(
            f"Only built {len(option_texts)} options for {item_id}; "
            f"need {num_options}. Add more segment actions or reduce --num-options."
        )

    shuffled = deterministic_option_texts(item_id, option_texts[:num_options])
    return [
        {
            "id": chr(ord("A") + index),
            "text": text,
            "is_none_option": text == NONE_OPTION_TEXT,
        }
        for index, text in enumerate(shuffled)
    ]


def build_planning_options(
    item_id: str,
    correct_text: str,
    nearby_texts: list[str],
    llm_distractor_pool: dict[str, list[str]],
    num_options: int,
    generated_count: int,
    action: str | None,
    objects: list[Any] | Any | None,
    all_actions: list[str],
    all_objects: list[str],
    category_labels: list[str],
) -> list[dict[str, Any]]:
    if num_options < 4:
        raise ValueError("--num-options must be at least 4 for planning")
    correct_labels = category_labels
    correct_norms = {normalize_option_text(label) for label in correct_labels}
    option_rows: list[dict[str, Any]] = [{"text": correct_text, "distractor_type": "correct"}]

    for text in nearby_texts:
        if len([row for row in option_rows if row["distractor_type"] == "nearby_action"]) >= 2:
            break
        if normalize_option_text(text) != normalize_option_text(correct_text):
            if normalize_option_text(text) not in {normalize_option_text(row["text"]) for row in option_rows}:
                option_rows.append({"text": text, "distractor_type": "nearby_action"})

    generated_pool = llm_distractor_pool.get(clean_text(correct_text), [])
    generated_pool = [
        text
        for text in generated_pool
        if normalize_option_text(text) not in correct_norms
        and normalize_option_text(text) != normalize_option_text(correct_text)
    ]
    if len(generated_pool) < generated_count:
        for text in generated_label_fallbacks(
            correct_text,
            action,
            objects,
            all_actions,
            all_objects,
            correct_labels,
            generated_count * 4,
        ):
            add_unique(generated_pool, text)

    existing_norms = {normalize_option_text(row["text"]) for row in option_rows}
    for text in deterministic_sample(f"{item_id}|generated_distractors", generated_pool, len(generated_pool)):
        if len([row for row in option_rows if row["distractor_type"] == "generated_wrong_label"]) >= generated_count:
            break
        norm = normalize_option_text(text)
        if norm not in existing_norms and norm not in correct_norms:
            option_rows.append({"text": text, "distractor_type": "generated_wrong_label"})
            existing_norms.add(norm)

    # If nearby actions are unavailable, fill their slots from real labels in other videos.
    if len(option_rows) < num_options - 1:
        all_real_labels = [
            label
            for label in category_labels
            if normalize_option_text(label) != normalize_option_text(correct_text)
        ]
        for text in deterministic_sample(f"{item_id}|real_label_fill", all_real_labels, len(all_real_labels)):
            if len(option_rows) >= num_options - 1:
                break
            norm = normalize_option_text(text)
            if norm not in {normalize_option_text(row["text"]) for row in option_rows}:
                option_rows.append({"text": text, "distractor_type": "nearby_action_fallback"})

    if len(option_rows) < num_options - 1:
        for text in generated_label_fallbacks(
            correct_text,
            action,
            objects,
            all_actions,
            all_objects,
            correct_labels,
            num_options * 4,
        ):
            if len(option_rows) >= num_options - 1:
                break
            norm = normalize_option_text(text)
            if norm not in {normalize_option_text(row["text"]) for row in option_rows}:
                option_rows.append({"text": text, "distractor_type": "rule_fallback"})

    option_rows.append({"text": NONE_OPTION_TEXT, "distractor_type": "none"})
    if len(option_rows) < num_options:
        raise ValueError(f"Only built {len(option_rows)} options for {item_id}; need {num_options}.")

    shuffled = deterministic_option_texts(item_id, [json.dumps(row, sort_keys=True) for row in option_rows[:num_options]])
    row_by_json = {json.dumps(row, sort_keys=True): row for row in option_rows[:num_options]}
    return [
        {
            "id": chr(ord("A") + index),
            "text": row_by_json[row]["text"],
            "is_none_option": row_by_json[row]["text"] == NONE_OPTION_TEXT,
            "distractor_type": row_by_json[row]["distractor_type"],
        }
        for index, row in enumerate(shuffled)
    ]


def video_path_for(video_id: str, video_dir: Path, video_exts: tuple[str, ...]) -> Path:
    for ext in video_exts:
        candidate = video_dir / f"{video_id}{ext}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    for ext in video_exts:
        matches = sorted(video_dir.rglob(f"{video_id}{ext}"))
        for match in matches:
            if match.is_file() and match.stat().st_size > 0:
                return match
    raise FileNotFoundError(f"Cannot find video for video_id={video_id} under {video_dir}")


def prejoined_video_path_for(video_id: str, video_dir: Path, video_exts: tuple[str, ...]) -> Path:
    try:
        return video_path_for(video_id, video_dir, video_exts)
    except FileNotFoundError:
        pass
    for ext in video_exts:
        matches = sorted(video_dir.rglob(f"*{video_id}*{ext}"))
        for match in matches:
            if match.is_file() and match.stat().st_size > 0:
                return match
    raise FileNotFoundError(f"Cannot find prejoined video for video_id={video_id} under {video_dir}")


def parse_view_specs(spec: str) -> dict[str, str]:
    views: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            view_name, view_dir = part.split("=", 1)
        else:
            view_name = Path(part).name.removeprefix("observation.images.")
            view_dir = part
        view_name = view_name.strip()
        view_dir = view_dir.strip()
        if not view_name or not view_dir:
            raise ValueError(f"Invalid view spec: {part!r}")
        views[view_name] = view_dir
    if not views:
        raise ValueError("--views must contain at least one view")
    return views


def multiview_video_path_for(
    video_id: str,
    multi_view_video_root: Path,
    view_dir: str,
    video_exts: tuple[str, ...],
) -> Path:
    base_dir = multi_view_video_root / view_dir
    for ext in video_exts:
        for candidate in (base_dir / "chunk-000" / f"{video_id}{ext}", base_dir / f"{video_id}{ext}"):
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
    for ext in video_exts:
        matches = sorted(base_dir.rglob(f"{video_id}{ext}"))
        for match in matches:
            if match.is_file() and match.stat().st_size > 0:
                return match
    raise FileNotFoundError(f"Cannot find {video_id} for view={view_dir} under {multi_view_video_root}")


def multiview_video_paths_for(
    video_id: str,
    multi_view_video_root: Path,
    views: dict[str, str],
    video_exts: tuple[str, ...],
    video_cache: dict[tuple[str, str], Path],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for view_name, view_dir in views.items():
        cache_key = (view_name, video_id)
        video_path = video_cache.get(cache_key)
        if video_path is None:
            video_path = multiview_video_path_for(
                video_id=video_id,
                multi_view_video_root=multi_view_video_root,
                view_dir=view_dir,
                video_exts=video_exts,
            )
            video_cache[cache_key] = video_path
        paths[view_name] = video_path
    return paths


def crop_video_top(input_path: Path, output_path: Path, top_fraction: float, overwrite: bool) -> None:
    if not 0 < top_fraction < 1:
        raise ValueError("--time-crop-top-fraction must be between 0 and 1")
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop_filter = f"crop=iw:trunc(ih*(1-{top_fraction})/2)*2:0:trunc(ih*{top_fraction}/2)*2"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vf",
        crop_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-an",
        "-y" if overwrite else "-n",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg crop failed for {input_path}: {result.stderr}")


def maybe_crop_time_videos(
    video_id: str,
    video_paths: dict[str, Path],
    output_dir: Path,
    top_fraction: float,
    overwrite: bool,
    cache: dict[tuple[str, str], Path],
) -> dict[str, Path]:
    cropped: dict[str, Path] = {}
    for view_name, video_path in video_paths.items():
        cache_key = (view_name, video_id)
        output_path = cache.get(cache_key)
        if output_path is None:
            output_path = output_dir / view_name / f"{video_id}{video_path.suffix}"
            crop_video_top(video_path, output_path, top_fraction, overwrite)
            cache[cache_key] = output_path
        cropped[view_name] = output_path
    return cropped


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata for {video_path}")

    duration = frame_count / fps
    actual_timestamp = min(max(timestamp, 0.0), max(0.0, duration - 1.0 / fps))
    frame_index = min(max(int(round(actual_timestamp * fps)), 0), frame_count - 1)
    if output_exists(output_path):
        cap.release()
        return frame_index, frame_index / fps

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write frame: {output_path}")
    return frame_index, frame_index / fps


def extract_multiview_frames(
    video_id: str,
    item_id: str,
    timestamp_key: str,
    timestamp: float,
    frames_dir: Path,
    multi_view_video_root: Path,
    views: dict[str, str],
    video_exts: tuple[str, ...],
    video_cache: dict[tuple[str, str], Path],
) -> dict[str, dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    video_paths = multiview_video_paths_for(video_id, multi_view_video_root, views, video_exts, video_cache)
    for view_name, video_path in video_paths.items():
        image_name = f"{safe_filename(video_id)}_{safe_filename(item_id)}_{timestamp_key}_{timestamp:.3f}s_{safe_filename(view_name)}.jpg"
        frame_path = frames_dir / view_name / video_id / image_name
        frame_index, actual_timestamp = extract_frame(video_path, timestamp, frame_path)
        images[view_name] = {
            "view": view_name,
            "image_path": str(frame_path),
            "video_path": str(video_path),
            "timestamp": timestamp,
            "actual_timestamp": actual_timestamp,
            "frame_index": frame_index,
        }
    return images


def extract_clip(video_path: Path, start: float, end: float, output_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata for {video_path}")

    duration = frame_count / fps
    clip_start = min(max(start, 0.0), max(0.0, duration - 1.0 / fps))
    clip_end = min(max(end, clip_start + 1.0 / fps), duration)
    start_frame = min(max(int(round(clip_start * fps)), 0), frame_count - 1)
    end_frame = min(max(int(round(clip_end * fps)), start_frame + 1), frame_count)
    if output_exists(output_path):
        existing_metadata = video_metadata(output_path)
        cap.release()
        existing_frames = int(existing_metadata["frame_count"])
        return {
            "start": clip_start,
            "end": clip_start + existing_frames / float(existing_metadata["fps"]),
            "start_frame": start_frame,
            "end_frame": start_frame + existing_frames - 1,
            "fps": float(existing_metadata["fps"]),
            "frames": existing_frames,
            "skipped_existing": True,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create clip: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for _ in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    if written == 0:
        raise RuntimeError(f"No frames written for clip: {output_path}")
    return {
        "start": clip_start,
        "end": clip_end,
        "start_frame": start_frame,
        "end_frame": start_frame + written - 1,
        "fps": fps,
        "frames": written,
    }


def video_metadata(video_path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata for {video_path}")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps,
    }


def extract_joined_multiview_clip(
    video_id: str,
    item_id: str,
    start: float,
    end: float,
    clips_dir: Path,
    multi_view_video_root: Path,
    views: dict[str, str],
    video_exts: tuple[str, ...],
    video_cache: dict[tuple[str, str], Path],
) -> dict[str, Any]:
    video_paths = multiview_video_paths_for(video_id, multi_view_video_root, views, video_exts, video_cache)
    return join_multiview_video_paths(
        video_id=video_id,
        item_id=item_id,
        start=start,
        end=end,
        output_dir=clips_dir,
        video_paths=video_paths,
        output_suffix="joined_views",
    )


def join_multiview_video_paths(
    video_id: str,
    item_id: str,
    start: float,
    end: float | None,
    output_dir: Path,
    video_paths: dict[str, Path],
    output_suffix: str,
) -> dict[str, Any]:
    metadata_by_view = {view_name: video_metadata(path) for view_name, path in video_paths.items()}
    min_duration = min(float(metadata["duration"]) for metadata in metadata_by_view.values())
    clip_start = min(max(start, 0.0), max(0.0, min_duration))
    requested_end = min_duration if end is None else end
    clip_end = min(max(requested_end, clip_start), min_duration)
    if clip_end <= clip_start:
        first_fps = float(next(iter(metadata_by_view.values()))["fps"])
        clip_end = min(min_duration, clip_start + 1.0 / first_fps)

    ordered_views = list(video_paths)
    output_fps = float(metadata_by_view[ordered_views[0]]["fps"])
    output_frames = max(1, int(round((clip_end - clip_start) * output_fps)))
    target_height = min(int(metadata_by_view[view_name]["height"]) for view_name in ordered_views)
    widths = [
        max(1, int(round(int(metadata_by_view[view_name]["width"]) * target_height / int(metadata_by_view[view_name]["height"]))))
        for view_name in ordered_views
    ]
    output_size = (sum(widths), target_height)
    end_text = "full" if end is None else f"{end:.3f}"
    clip_name = f"{safe_filename(video_id)}_{safe_filename(item_id)}_{start:.3f}_{end_text}_{safe_filename(output_suffix)}.mp4"
    clip_path = output_dir / video_id / clip_name
    if output_exists(clip_path):
        existing_metadata = video_metadata(clip_path)
        return {
            "clip_path": str(clip_path),
            "video_path": str(clip_path),
            "view_order": ordered_views,
            "source_video_paths": {view_name: str(path) for view_name, path in video_paths.items()},
            "start": clip_start,
            "end": clip_start + float(existing_metadata["duration"]),
            "fps": float(existing_metadata["fps"]),
            "frames": int(existing_metadata["frame_count"]),
            "width": int(existing_metadata["width"]),
            "height": int(existing_metadata["height"]),
            "skipped_existing": True,
        }
    clip_path.parent.mkdir(parents=True, exist_ok=True)

    caps = {view_name: cv2.VideoCapture(str(video_paths[view_name])) for view_name in ordered_views}
    try:
        for view_name, cap in caps.items():
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {video_paths[view_name]}")
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, output_size)
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create joined clip: {clip_path}")
        written = 0
        for frame_offset in range(output_frames):
            timestamp = min(clip_start + frame_offset / output_fps, max(0.0, clip_end - 1.0 / output_fps))
            frames = []
            for view_name, width in zip(ordered_views, widths):
                cap = caps[view_name]
                view_fps = float(metadata_by_view[view_name]["fps"])
                frame_count = int(metadata_by_view[view_name]["frame_count"])
                frame_index = min(max(int(round(timestamp * view_fps)), 0), frame_count - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Failed to read frame {frame_index} from {video_paths[view_name]}")
                frames.append(cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA))
            writer.write(cv2.hconcat(frames))
            written += 1
        writer.release()
    finally:
        for cap in caps.values():
            cap.release()

    if written == 0:
        raise RuntimeError(f"No frames written for joined clip: {clip_path}")
    return {
        "clip_path": str(clip_path),
        "video_path": str(clip_path),
        "view_order": ordered_views,
        "source_video_paths": {view_name: str(path) for view_name, path in video_paths.items()},
        "start": clip_start,
        "end": clip_start + written / output_fps,
        "fps": output_fps,
        "frames": written,
        "width": output_size[0],
        "height": output_size[1],
    }


def build_planning_items(
    time_items: list[dict[str, Any]],
    video_dir: Path,
    prejoined_video_dir: Path | None,
    clips_dir: Path,
    video_exts: tuple[str, ...],
    question: str,
    num_options: int,
    timestamp_key: str,
    no_media: bool,
    multi_view_video_root: Path | None,
    views: dict[str, str],
    llm_distractor_pool: dict[str, list[str]],
    category_labels: list[str],
    nearby_distractors_per_question: int,
    generated_distractors_per_question: int,
) -> list[dict[str, Any]]:
    all_actions = sorted({clean_text(str(item.get("answer_action") or "")) for item in time_items if item.get("answer_action")})
    all_objects = sorted({object_text(item.get("answer_objects")) for item in time_items if item.get("answer_objects")})
    video_cache: dict[str, Path] = {}
    prejoined_video_cache: dict[str, Path] = {}
    multiview_video_cache: dict[tuple[str, str], Path] = {}
    output_items: list[dict[str, Any]] = []

    for item in time_items:
        next_item = next_time_item(item, time_items)
        if next_item is None:
            continue
        item_id = str(item["id"])
        video_id = str(item["video_id"])
        start = float(item["answer_seconds"]["start"])
        end = float(item["answer_seconds"]["end"])
        answer_text = item_label(next_item)
        options = build_planning_options(
            item_id=f"{item_id}_plan_next",
            correct_text=answer_text,
            nearby_texts=nearby_action_texts(next_item, time_items, nearby_distractors_per_question),
            llm_distractor_pool=llm_distractor_pool,
            num_options=num_options,
            generated_count=generated_distractors_per_question,
            action=str(next_item.get("answer_action") or ""),
            objects=next_item.get("answer_objects", []),
            all_actions=all_actions,
            all_objects=all_objects,
            category_labels=category_labels,
        )
        correct_option = next(option for option in options if normalize_option_text(option["text"]) == normalize_option_text(answer_text))
        option_lines = "\n".join(f"{option['id']}. {option['text']}" for option in options)
        full_question = f"{question}\nOptions:\n{option_lines}"

        input_data: dict[str, Any] = {
            "video_path": None,
            "start": start,
            "end": end,
        }
        if prejoined_video_dir is not None:
            video_path = prejoined_video_cache.get(video_id)
            if video_path is None:
                video_path = prejoined_video_path_for(video_id, prejoined_video_dir, video_exts)
                prejoined_video_cache[video_id] = video_path
            input_data.update(
                {
                    "video_path": str(video_path),
                    "video_paths": [str(video_path)],
                    "prejoined_video_path": str(video_path),
                }
            )
        if not no_media:
            if prejoined_video_dir is not None:
                video_path = prejoined_video_cache[video_id]
                clip_path = clips_dir / video_id / f"{safe_filename(video_id)}_{safe_filename(item_id)}_{start:.3f}_{end:.3f}_prejoined.mp4"
                clip_info = extract_clip(video_path, start, end, clip_path)
                input_data.update(
                    {
                        "clip_path": str(clip_path),
                        "clip_paths": [str(clip_path)],
                        "clips": {
                            "prejoined": {
                                "view": "prejoined",
                                "clip_path": str(clip_path),
                                "video_path": str(video_path),
                                **clip_info,
                            }
                        },
                        "video_path": str(clip_path),
                        "video_paths": [str(clip_path)],
                        "source_video_path": str(video_path),
                        "prejoined_video_path": str(video_path),
                        **clip_info,
                    }
                )
            elif multi_view_video_root is not None:
                joined_clip = extract_joined_multiview_clip(
                    video_id=video_id,
                    item_id=item_id,
                    start=start,
                    end=end,
                    clips_dir=clips_dir,
                    multi_view_video_root=multi_view_video_root,
                    views=views,
                    video_exts=video_exts,
                    video_cache=multiview_video_cache,
                )
                input_data.update(
                    {
                        "clip_path": joined_clip["clip_path"],
                        "clip_paths": [joined_clip["clip_path"]],
                        "joined_clip": joined_clip,
                        "video_path": joined_clip["video_path"],
                        "video_paths": [joined_clip["video_path"]],
                        "source_video_paths": joined_clip["source_video_paths"],
                        "view_order": joined_clip["view_order"],
                        "fps": joined_clip["fps"],
                        "frames": joined_clip["frames"],
                    }
                )
            else:
                video_path = video_cache.get(video_id)
                if video_path is None:
                    video_path = video_path_for(video_id, video_dir, video_exts)
                    video_cache[video_id] = video_path
                clip_path = clips_dir / video_id / f"{safe_filename(video_id)}_{safe_filename(item_id)}_{start:.3f}_{end:.3f}.mp4"
                clip_info = extract_clip(video_path, start, end, clip_path)
                input_data.update(
                    {
                        "clip_path": str(clip_path),
                        "clip_paths": [str(clip_path)],
                        "clips": {"default": {"view": "default", "clip_path": str(clip_path), "video_path": str(video_path), **clip_info}},
                        "video_path": str(video_path),
                        "video_paths": [str(video_path)],
                        **clip_info,
                    }
                )

        output_items.append(
            {
                "id": f"{item_id}_plan_next",
                "source_id": item_id,
                "next_source_id": str(next_item["id"]),
                "video_id": video_id,
                "type": "planning",
                "input": input_data,
                "Q": full_question,
                "A": correct_option["id"],
                "question": full_question,
                "answer": correct_option["id"],
                "answer_text": answer_text,
                "answer_action": next_item.get("answer_action"),
                "answer_objects": next_item.get("answer_objects", []),
                "options": options,
                "correct_option": correct_option,
                "source_time_eqa": item,
                "next_time_eqa": next_item,
            }
        )
    return output_items


def build_planning_2_items(
    time_items: list[dict[str, Any]],
    video_dir: Path,
    prejoined_video_dir: Path | None,
    frames_dir: Path,
    video_exts: tuple[str, ...],
    question_template: str,
    task_name: str,
    num_options: int,
    timestamp_key: str,
    no_media: bool,
    multi_view_video_root: Path | None,
    views: dict[str, str],
    llm_distractor_pool: dict[str, list[str]],
    category_labels: list[str],
    nearby_distractors_per_question: int,
    generated_distractors_per_question: int,
) -> list[dict[str, Any]]:
    all_actions = sorted({clean_text(str(item.get("answer_action") or "")) for item in time_items if item.get("answer_action")})
    all_objects = sorted({object_text(item.get("answer_objects")) for item in time_items if item.get("answer_objects")})
    video_cache: dict[str, Path] = {}
    prejoined_video_cache: dict[str, Path] = {}
    multiview_video_cache: dict[tuple[str, str], Path] = {}
    output_items: list[dict[str, Any]] = []

    for item in time_items:
        next_item = next_time_item(item, time_items)
        item_id = str(item["id"])
        video_id = str(item["video_id"])
        timestamp = float(item["answer_seconds"][timestamp_key])
        answer_text = item_label(item)
        options = build_planning_options(
            item_id=f"{item_id}_plan2_{timestamp_key}",
            correct_text=answer_text,
            nearby_texts=nearby_action_texts(item, time_items, nearby_distractors_per_question),
            llm_distractor_pool=llm_distractor_pool,
            num_options=num_options,
            generated_count=generated_distractors_per_question,
            action=str(item.get("answer_action") or ""),
            objects=item.get("answer_objects", []),
            all_actions=all_actions,
            all_objects=all_objects,
            category_labels=category_labels,
        )
        correct_option = next(option for option in options if normalize_option_text(option["text"]) == normalize_option_text(answer_text))
        question = question_template.format(task_name=task_name, video_id=video_id, item_id=item_id)
        option_lines = "\n".join(f"{option['id']}. {option['text']}" for option in options)
        full_question = f"{question}\nOptions:\n{option_lines}"

        input_data: dict[str, Any] = {
            "video_path": None,
            "timestamp": timestamp,
            "timestamp_key": timestamp_key,
            "task_name": task_name,
        }
        if prejoined_video_dir is not None:
            video_path = prejoined_video_cache.get(video_id)
            if video_path is None:
                video_path = prejoined_video_path_for(video_id, prejoined_video_dir, video_exts)
                prejoined_video_cache[video_id] = video_path
            input_data.update(
                {
                    "video_path": str(video_path),
                    "video_paths": [str(video_path)],
                    "prejoined_video_path": str(video_path),
                }
            )
        if not no_media:
            if prejoined_video_dir is not None:
                video_path = prejoined_video_cache[video_id]
                frame_path = frames_dir / "prejoined" / video_id / f"{safe_filename(video_id)}_{safe_filename(item_id)}_{timestamp_key}_{timestamp:.3f}s_prejoined.jpg"
                frame_index, actual_timestamp = extract_frame(video_path, timestamp, frame_path)
                input_data.update(
                    {
                        "image_path": str(frame_path),
                        "image_paths": [str(frame_path)],
                        "images": {
                            "prejoined": {
                                "view": "prejoined",
                                "image_path": str(frame_path),
                                "video_path": str(video_path),
                                "timestamp": timestamp,
                                "actual_timestamp": actual_timestamp,
                                "frame_index": frame_index,
                            }
                        },
                        "actual_timestamp": actual_timestamp,
                        "frame_index": frame_index,
                    }
                )
            elif multi_view_video_root is not None:
                images = extract_multiview_frames(
                    video_id=video_id,
                    item_id=item_id,
                    timestamp_key=timestamp_key,
                    timestamp=timestamp,
                    frames_dir=frames_dir,
                    multi_view_video_root=multi_view_video_root,
                    views=views,
                    video_exts=video_exts,
                    video_cache=multiview_video_cache,
                )
                primary_image = next(iter(images.values()))
                input_data.update(
                    {
                        "image_path": primary_image["image_path"],
                        "image_paths": [image["image_path"] for image in images.values()],
                        "images": images,
                        "video_path": primary_image["video_path"],
                        "video_paths": [image["video_path"] for image in images.values()],
                        "actual_timestamp": primary_image["actual_timestamp"],
                        "frame_index": primary_image["frame_index"],
                    }
                )
            else:
                video_path = video_cache.get(video_id)
                if video_path is None:
                    video_path = video_path_for(video_id, video_dir, video_exts)
                    video_cache[video_id] = video_path
                frame_path = frames_dir / video_id / f"{safe_filename(video_id)}_{safe_filename(item_id)}_{timestamp_key}_{timestamp:.3f}s.jpg"
                frame_index, actual_timestamp = extract_frame(video_path, timestamp, frame_path)
                input_data.update(
                    {
                        "image_path": str(frame_path),
                        "image_paths": [str(frame_path)],
                        "images": {
                            "default": {
                                "view": "default",
                                "image_path": str(frame_path),
                                "video_path": str(video_path),
                                "timestamp": timestamp,
                                "actual_timestamp": actual_timestamp,
                                "frame_index": frame_index,
                            }
                        },
                        "video_path": str(video_path),
                        "video_paths": [str(video_path)],
                        "actual_timestamp": actual_timestamp,
                        "frame_index": frame_index,
                    }
                )

        output_items.append(
            {
                "id": f"{item_id}_plan2_{timestamp_key}",
                "source_id": item_id,
                "next_source_id": str(next_item["id"]) if next_item is not None else None,
                "video_id": video_id,
                "type": "planning_2",
                "task_name": task_name,
                "input": input_data,
                "Q": full_question,
                "A": correct_option["id"],
                "question": full_question,
                "answer": correct_option["id"],
                "answer_text": answer_text,
                "answer_action": item.get("answer_action"),
                "answer_objects": item.get("answer_objects", []),
                "options": options,
                "correct_option": correct_option,
                "source_time_eqa": item,
                "next_time_eqa": next_item,
            }
        )
    return output_items


def step_order_frame_index(
    segment: dict[str, Any],
    fps: float,
    end_offset: int,
    end_offset_ratio: float,
) -> int:
    if segment.get("start_frame") is not None or segment.get("end_frame") is not None:
        start = int(segment.get("start_frame", 0) or 0)
        end = int(segment.get("end_frame", start) or start)
    else:
        start = int(round(float(segment.get("start", 0.0)) * fps))
        end = int(round(float(segment.get("end", segment.get("start", 0.0))) * fps))
    length = max(1, end - start + 1)
    ratio_offset = max(1, int(round(length * float(end_offset_ratio))))
    offset = min(max(0, int(end_offset)), ratio_offset)
    return max(start, end - offset)


def read_frame_by_index(cap: cv2.VideoCapture, frame_index: int) -> Any:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        frame_index = max(0, min(int(frame_index), total_frames - 1))
    else:
        frame_index = max(0, int(frame_index))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if ok:
        return frame

    for delta in (1, -1, 2, -2, 5, -5):
        probe = max(0, frame_index + delta)
        cap.set(cv2.CAP_PROP_POS_FRAMES, probe)
        ok, frame = cap.read()
        if ok:
            return frame
    raise RuntimeError(f"Could not read frame {frame_index}")


def write_jpeg(path: Path, image: Any, jpeg_quality: int) -> None:
    if output_exists(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    if not cv2.imwrite(str(path), image, params):
        raise RuntimeError(f"Failed to write image: {path}")


def resize_keep_aspect(frame: Any, width: int) -> Any:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    height = max(1, int(round(h * (width / w))))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def put_montage_label(frame: Any, label: str) -> Any:
    label_h = 42
    h, w = frame.shape[:2]
    canvas = cv2.copyMakeBorder(
        frame,
        label_h,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(18, 18, 18),
    )
    cv2.putText(
        canvas,
        label,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (0, 0), (w - 1, h + label_h - 1), (225, 225, 225), 1)
    return canvas


def hconcat_images_same_height(frames: list[Any]) -> Any:
    target_height = min(frame.shape[0] for frame in frames)
    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        width = max(1, int(round(w * target_height / h)))
        resized.append(cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA))
    return cv2.hconcat(resized)


def make_step_order_multiview_frame(frames_by_view: dict[str, Any], cell_width: int) -> Any:
    labeled = [
        put_montage_label(resize_keep_aspect(frame, cell_width), view_name)
        for view_name, frame in frames_by_view.items()
    ]
    return hconcat_images_same_height(labeled)


def make_step_order_montage(
    frames: list[Any],
    labels: list[str],
    output_path: Path,
    cell_width: int,
    jpeg_quality: int,
) -> None:
    labeled = [
        put_montage_label(resize_keep_aspect(frame, cell_width), label)
        for frame, label in zip(frames, labels)
    ]
    max_h = max(image.shape[0] for image in labeled)
    padded = []
    for image in labeled:
        pad_h = max_h - image.shape[0]
        if pad_h:
            image = cv2.copyMakeBorder(
                image,
                0,
                pad_h,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(245, 245, 245),
            )
        padded.append(image)
    write_jpeg(output_path, cv2.hconcat(padded), jpeg_quality)


def step_order_to_choice(source_order: list[int], source_to_display: dict[int, int]) -> str:
    return "-".join(str(source_to_display[source_index]) for source_index in source_order)


def candidate_step_orders(num_steps: int, rng: random.Random) -> list[list[int]]:
    correct = list(range(1, num_steps + 1))
    orders = [correct]
    if num_steps == 4:
        orders.extend(
            [
                [2, 1, 3, 4],
                [1, 2, 4, 3],
                [3, 4, 1, 2],
                [1, 3, 2, 4],
                [1, 4, 2, 3],
            ]
        )
    else:
        for index in range(0, num_steps - 1):
            swapped = correct[:]
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            orders.append(swapped)

    seen = {tuple(order) for order in orders}
    attempts = 0
    while len(orders) < len(CHOICE_LABELS) and attempts < 200:
        attempts += 1
        shuffled = correct[:]
        rng.shuffle(shuffled)
        key = tuple(shuffled)
        if key not in seen:
            seen.add(key)
            orders.append(shuffled)
    return orders


def make_step_order_options(
    num_steps: int,
    source_to_display: dict[int, int],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], str, str]:
    if math.factorial(num_steps) < len(CHOICE_LABELS):
        raise ValueError(f"need at least {len(CHOICE_LABELS)} unique step-order options, but only {math.factorial(num_steps)} exist")
    source_orders = candidate_step_orders(num_steps, rng)
    choice_values: list[str] = []
    seen: set[str] = set()
    for source_order in source_orders:
        value = step_order_to_choice(source_order, source_to_display)
        if value not in seen:
            seen.add(value)
            choice_values.append(value)
        if len(choice_values) == len(CHOICE_LABELS):
            break

    correct_value = step_order_to_choice(list(range(1, num_steps + 1)), source_to_display)
    if correct_value not in choice_values:
        choice_values[-1] = correct_value
    if len(choice_values) < len(CHOICE_LABELS):
        raise ValueError(f"only built {len(choice_values)} step-order options; need {len(CHOICE_LABELS)}")

    rng.shuffle(choice_values)
    options = [
        {"id": label, "text": value, "is_correct": value == correct_value}
        for label, value in zip(CHOICE_LABELS, choice_values)
    ]
    answer = next(option["id"] for option in options if option["is_correct"])
    return options, answer, correct_value


def step_order_step_text(segment: dict[str, Any]) -> str:
    verbs = segment.get("main_verbs") or []
    objects = segment.get("objects") or []
    verb = clean_text(str(verbs[0])) if verbs else "step"
    obj = clean_text(str(objects[0])) if objects else "object"
    return f"{verb} {obj}".strip()


def step_order_video_path(
    video_id: str,
    video_dir: Path,
    multi_view_video_root: Path | None,
    views: dict[str, str],
    step_order_view: str,
    video_exts: tuple[str, ...],
    single_view_cache: dict[str, Path],
    multi_view_cache: dict[tuple[str, str], Path],
) -> Path:
    if multi_view_video_root is None:
        video_path = single_view_cache.get(video_id)
        if video_path is None:
            video_path = video_path_for(video_id, video_dir, video_exts)
            single_view_cache[video_id] = video_path
        return video_path

    if step_order_view not in views:
        raise ValueError(f"step_order_view {step_order_view!r} is not present in --views")
    cache_key = (step_order_view, video_id)
    video_path = multi_view_cache.get(cache_key)
    if video_path is None:
        video_path = multiview_video_path_for(
            video_id=video_id,
            multi_view_video_root=multi_view_video_root,
            view_dir=views[step_order_view],
            video_exts=video_exts,
        )
        multi_view_cache[cache_key] = video_path
    return video_path


def step_order_video_paths(
    video_id: str,
    video_dir: Path,
    multi_view_video_root: Path | None,
    views: dict[str, str],
    step_order_view: str,
    video_exts: tuple[str, ...],
    single_view_cache: dict[str, Path],
    multi_view_cache: dict[tuple[str, str], Path],
) -> dict[str, Path]:
    if multi_view_video_root is None:
        return {
            "default": step_order_video_path(
                video_id=video_id,
                video_dir=video_dir,
                multi_view_video_root=None,
                views=views,
                step_order_view=step_order_view,
                video_exts=video_exts,
                single_view_cache=single_view_cache,
                multi_view_cache=multi_view_cache,
            )
        }

    paths: dict[str, Path] = {}
    for view_name in views:
        paths[view_name] = step_order_video_path(
            video_id=video_id,
            video_dir=video_dir,
            multi_view_video_root=multi_view_video_root,
            views=views,
            step_order_view=view_name,
            video_exts=video_exts,
            single_view_cache=single_view_cache,
            multi_view_cache=multi_view_cache,
        )
    return paths


def build_step_order_items(
    data_dir: Path,
    file_limit: int | None,
    video_dir: Path,
    output_dir: Path,
    video_exts: tuple[str, ...],
    question: str,
    step_order_view: str,
    initial_frame: int,
    end_offset: int,
    end_offset_ratio: float,
    cell_width: int,
    jpeg_quality: int,
    seed: int,
    no_media: bool,
    multi_view_video_root: Path | None,
    views: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if no_media:
        raise ValueError("step_order task requires media extraction. Set no_media=false in config or pass --extract-media.")

    rng = random.Random(seed)
    single_view_cache: dict[str, Path] = {}
    multi_view_cache: dict[tuple[str, str], Path] = {}
    items: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for segment_path in sorted_segment_files(data_dir, file_limit):
        video_id = video_id_from_path(segment_path)
        try:
            raw = load_json(segment_path)
            segments = raw.get("segments", [])
            if not isinstance(segments, list) or not segments:
                raise ValueError("missing or empty segments list")
            segments = sorted(
                segments,
                key=lambda item: (
                    int(item.get("start_frame", 0) or 0),
                    float(item.get("start", 0.0) or 0.0),
                    int(item.get("end_frame", 0) or 0),
                ),
            )
            if len(segments) < 3:
                raise ValueError("need at least three segments for six step_order options")

            video_paths = step_order_video_paths(
                video_id=video_id,
                video_dir=video_dir,
                multi_view_video_root=multi_view_video_root,
                views=views,
                step_order_view=step_order_view,
                video_exts=video_exts,
                single_view_cache=single_view_cache,
                multi_view_cache=multi_view_cache,
            )
            caps = {view_name: cv2.VideoCapture(str(path)) for view_name, path in video_paths.items()}
            try:
                for view_name, cap in caps.items():
                    if not cap.isOpened():
                        raise RuntimeError(f"Could not open {view_name} video: {video_paths[view_name]}")
                primary_view_name = next(iter(caps))
                fps = float(caps[primary_view_name].get(cv2.CAP_PROP_FPS)) or 30.0
                states = [
                    {
                        "source_order": 0,
                        "role": "initial",
                        "frame_index": int(initial_frame),
                        "description": "initial state",
                        "segment_id": None,
                    }
                ]
                for index, segment in enumerate(segments, start=1):
                    states.append(
                        {
                            "source_order": index,
                            "role": "result",
                            "frame_index": step_order_frame_index(
                                segment=segment,
                                fps=fps,
                                end_offset=end_offset,
                                end_offset_ratio=end_offset_ratio,
                            ),
                            "description": step_order_step_text(segment),
                            "segment_id": segment.get("id"),
                            "segment": segment,
                        }
                    )
                source_frames = [
                    make_step_order_multiview_frame(
                        {
                            view_name: read_frame_by_index(cap, state["frame_index"])
                            for view_name, cap in caps.items()
                        },
                        cell_width,
                    )
                    for state in states
                ]
            finally:
                for cap in caps.values():
                    cap.release()

            view_suffix = "multiview" if multi_view_video_root is not None else "default"
            initial_image_path = output_dir / "step_order" / "initial_images" / f"{safe_filename(video_id)}_{view_suffix}_initial.jpg"
            write_jpeg(initial_image_path, source_frames[0], jpeg_quality)

            display_source_order = list(range(1, len(states)))
            rng.shuffle(display_source_order)
            source_to_display = {
                source_index: display_index + 1
                for display_index, source_index in enumerate(display_source_order)
            }

            sample_image_dir = output_dir / "step_order" / "images" / video_id
            displayed_frames: list[Any] = []
            displayed_metadata: list[dict[str, Any]] = []
            for display_index, source_index in enumerate(display_source_order, start=1):
                frame = source_frames[source_index]
                image_path = sample_image_dir / f"{safe_filename(video_id)}_{view_suffix}_image_{display_index}.jpg"
                write_jpeg(image_path, frame, jpeg_quality)
                displayed_frames.append(frame)

                state = dict(states[source_index])
                state.pop("segment", None)
                state["display_label"] = f"Image {display_index}"
                state["display_index"] = display_index
                state["image_path"] = str(image_path)
                displayed_metadata.append(state)

            montage_path = output_dir / "step_order" / "montages" / f"{safe_filename(video_id)}_{view_suffix}_step_order.jpg"
            make_step_order_montage(
                displayed_frames,
                [f"Image {index}" for index in range(1, len(displayed_frames) + 1)],
                montage_path,
                cell_width * len(video_paths),
                jpeg_quality,
            )

            options, answer, answer_order = make_step_order_options(len(segments), source_to_display, rng)
            choices = {option["id"]: option["text"] for option in options}
            option_lines = "\n".join(f"{option['id']}. {option['text']}" for option in options)
            full_question = f"{question}\nOptions:\n{option_lines}"
            initial_state = states[0]
            items.append(
                {
                    "id": f"{video_id}_step_order",
                    "source_id": video_id,
                    "video_id": video_id,
                    "type": "step_order",
                    "task": "step_order_with_initial_state",
                    "video_path": str(next(iter(video_paths.values()))),
                    "video_paths": {view_name: str(path) for view_name, path in video_paths.items()},
                    "segments_path": str(segment_path),
                    "view": "multiview" if multi_view_video_root is not None else "default",
                    "views": list(video_paths),
                    "input": {
                        "initial_image": str(initial_image_path),
                        "image": str(montage_path),
                        "image_path": str(montage_path),
                        "image_paths": [str(initial_image_path), str(montage_path)],
                        "images": displayed_metadata,
                        "video_path": str(next(iter(video_paths.values()))),
                        "video_paths": {view_name: str(path) for view_name, path in video_paths.items()},
                        "views": list(video_paths),
                        "segments_path": str(segment_path),
                    },
                    "initial_image": str(initial_image_path),
                    "image": str(montage_path),
                    "images": displayed_metadata,
                    "Q": full_question,
                    "A": answer,
                    "question": full_question,
                    "answer": answer,
                    "answer_text": answer_order,
                    "answer_order": answer_order,
                    "options": options,
                    "choices": choices,
                    "num_choices_to_order": len(segments),
                    "chronological_states": [
                        dict(initial_state, display_index=None),
                        *[
                            {
                                "source_order": state["source_order"],
                                "role": state["role"],
                                "frame_index": state["frame_index"],
                                "description": state["description"],
                                "segment_id": state["segment_id"],
                                "display_index": source_to_display[state["source_order"]],
                            }
                            for state in states[1:]
                        ],
                    ],
                }
            )
        except Exception as exc:
            skipped.append({"video_id": video_id, "segments_path": str(segment_path), "reason": str(exc)})

    return items, skipped


def step_order_vqa_pair(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "initial_image": item["initial_image"],
        "image": item["image"],
        "images": [
            {
                "label": image["display_label"],
                "image_path": image["image_path"],
            }
            for image in item["images"]
        ],
        "question": item["question"],
        "choices": item["choices"],
        "answer": item["answer"],
    }


# ---------------------------------------------------------------------------
# Inlined trajectory QA generation.
#
# This code is intentionally kept inside planning_workflow(1).py so the
# planning workflow does not dynamically import ../trajactory.py for the
# trajectory task. Names are prefixed to avoid collisions with the rest of this
# workflow file.
# ---------------------------------------------------------------------------
TRAJECTORY_DEFAULT_INPUT = Path("/home/kewei/YWC/egodata/pickplace/time_eqa_first50_6move.json")
TRAJECTORY_DEFAULT_DATASET_ROOT = Path("/home/kewei/NAS/lerobot_datasets-26-06-17-17-25-20/gripper")
TRAJECTORY_DEFAULT_OUTPUT = Path("/home/kewei/YWC/egodata/pickplace/trajectory_qa_first50_6move.json")
TRAJECTORY_DEFAULT_OUTPUT_2D = Path("/home/kewei/YWC/egodata/pickplace/trajectory_qa_2d_first50_6move.json")
TRAJECTORY_DEFAULT_OUTPUT_3D = Path("/home/kewei/YWC/egodata/pickplace/trajectory_qa_3d_first50_6move.json")
TRAJECTORY_DEFAULT_IMAGE_DIR = Path("/home/kewei/YWC/egodata/pickplace/trajectory_first_frames_left_eye")
TRAJECTORY_DEFAULT_VIEWS = "observation.images.left_eye,observation.images.left_wrist,observation.images.right_wrist"
TRAJECTORY_DEFAULT_PRIMARY_VIEW = "observation.images.left_eye"
TRAJECTORY_DEFAULT_PROMPT = (
    "{category}. Please predict the key trajectory points needed to complete "
    "this task from this viewpoint onward"
)
TRAJECTORY_DEFAULT_PROMPT_2D = (
    "{category}. You are given synchronized images from three camera views: {views}. "
    "Use all views as context, but predict the key trajectory points **in the main-view image** "
    "({primary_view}) "
    "needed to complete this task from the main viewpoint onward."
)
TRAJECTORY_DEFAULT_PROMPT_3D = (
    "{category}. You are given synchronized images from three camera views: {views}. "
    "Use all views as context, but predict the key **3D** trajectory points (in meters) "
    "needed to complete this task from the main viewpoint ({primary_view}) onward."
)
TRAJECTORY_DEFAULT_NUM_KEYPOINTS = 10
TRAJECTORY_DEFAULT_FPS = 20.0
TRAJECTORY_DEFAULT_INTERNAL = "gripper"
TRAJECTORY_DEFAULT_LEFT_XYZ_INDICES = "0:3"
TRAJECTORY_DEFAULT_RIGHT_XYZ_INDICES = "7:10"
TRAJECTORY_DEFAULT_USE_BASE_TO_CAMERA_EXTRINSIC = True
TRAJECTORY_DEFAULT_BASE_TO_CAMERA_XYZ = [0.093353689, 0.033, 1.260691643]
TRAJECTORY_DEFAULT_BASE_TO_CAMERA_RPY = [-2.3562, 0.0, -1.5708]

TRAJECTORY_CAMERA_INTRINSICS = {
    "gripper": {
        "name": "K_rgb_960x540",
        "width": 960,
        "height": 540,
        "K": [
            [681.3442, 0.0, 490.6951],
            [0.0, 680.3921, 286.6483],
            [0.0, 0.0, 1.0],
        ],
        "distortion_model": None,
        "distortion_coefficients": [],
    },
}

TRAJECTORY_TRAJECTORY_INTERNAL_PROFILES = {
    "gripper": {
        "left_xyz_indices": TRAJECTORY_DEFAULT_LEFT_XYZ_INDICES,
        "right_xyz_indices": TRAJECTORY_DEFAULT_RIGHT_XYZ_INDICES,
        "left_point_name": "left_gripper",
        "right_point_name": "right_gripper",
        "state_point_description": "left/right gripper XYZ positions",
        "coordinate_assumption": (
            "observation.state XYZ values are gripper positions in world coordinates. "
            "The configured camera pose is T_cam_world, mapping camera coordinates to world coordinates; "
            "its inverse maps world coordinates to camera coordinates before projection."
        ),
    },
}


@dataclass(frozen=True)
class TrajectoryVideoRef:
    chunk_index: int
    file_index: int


@dataclass(frozen=True)
class TrajectoryEpisodeRef:
    episode_index: int
    data_chunk_index: int
    data_file_index: int
    videos: dict[str, TrajectoryVideoRef]


@dataclass(frozen=True)
class TrajectoryCameraIntrinsics:
    profile: str
    width: int | None
    height: int | None
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str | None
    distortion_coefficients: list[float]

    @property
    def K(self) -> list[list[float]]:
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]


def trajectory_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def trajectory_save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = relativize_paths_for_json(data, path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def trajectory_parse_video_id(video_id: str) -> int:
    match = re.fullmatch(r"file-(\d+)", str(video_id).strip())
    if not match:
        raise ValueError(f"Cannot parse episode index from video_id={video_id!r}")
    return int(match.group(1))


def trajectory_clean_category(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    cleaned = cleaned.strip("\"' .")
    return cleaned[0].lower() + cleaned[1:] if cleaned else "task"


def trajectory_category_from_item(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    narration = metadata.get("narration")
    if narration:
        return trajectory_clean_category(narration)

    question = str(item.get("Q") or item.get("question") or "")
    match = re.search(r'"([^"]+)"', question)
    if match:
        return trajectory_clean_category(match.group(1))
    question = re.sub(r"^When did the robot\s+", "", question, flags=re.IGNORECASE)
    return trajectory_clean_category(question)


def trajectory_parse_views(text: str) -> list[str]:
    views = [part.strip() for part in str(text).split(",") if part.strip()]
    if not views:
        raise ValueError("At least one view is required")
    return views


def trajectory_view_label(view: str) -> str:
    return view.rsplit(".", 1)[-1]


def trajectory_camera_intrinsics_from_name(name: str) -> TrajectoryCameraIntrinsics:
    key = str(name).strip().lower()
    if key not in TRAJECTORY_CAMERA_INTRINSICS:
        raise ValueError(f"Unknown internal camera {name!r}; choose one of {sorted(TRAJECTORY_CAMERA_INTRINSICS)}")
    row = TRAJECTORY_CAMERA_INTRINSICS[key]
    K = row["K"]
    return TrajectoryCameraIntrinsics(
        profile=key,
        width=row["width"],
        height=row["height"],
        fx=float(K[0][0]),
        fy=float(K[1][1]),
        cx=float(K[0][2]),
        cy=float(K[1][2]),
        distortion_model=row["distortion_model"],
        distortion_coefficients=list(row["distortion_coefficients"]),
    )


def trajectory_project_camera_point(point_xyz_opencv: list[float], intrinsics: TrajectoryCameraIntrinsics, decimals: int) -> dict[str, Any]:
    x, y, z = [float(value) for value in point_xyz_opencv]
    if z <= 0:
        return {
            "u": None,
            "v": None,
            "valid": False,
            "reason": "z must be positive for perspective projection",
        }
    u = intrinsics.fx * x / z + intrinsics.cx
    v = intrinsics.fy * y / z + intrinsics.cy
    in_image = True
    if intrinsics.width is not None:
        in_image = in_image and 0 <= u < intrinsics.width
    if intrinsics.height is not None:
        in_image = in_image and 0 <= v < intrinsics.height
    return {
        "u": round(u, decimals),
        "v": round(v, decimals),
        "valid": True,
        "in_image": in_image,
    }


def trajectory_rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def trajectory_transform_matrix_from_xyz_rpy(xyz: list[float], rpy: list[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = trajectory_rotation_matrix_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    transform[:3, 3] = np.array([float(value) for value in xyz])
    return transform


def trajectory_transform_point(transform: np.ndarray, point_xyz: list[float], decimals: int) -> list[float]:
    point = np.array([float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0])
    output = transform @ point
    return [round(float(value), decimals) for value in output[:3]]


def trajectory_camera_rgb_link_to_opencv_camera(point_xyz: list[float], decimals: int) -> list[float]:
    x_link, y_link, z_link = [float(value) for value in point_xyz]
    return [
        round(-x_link, decimals),
        round(-y_link, decimals),
        round(z_link, decimals),
    ]


def trajectory_source_forward_left_up_to_opencv_camera(point_xyz: list[float], decimals: int) -> list[float]:
    x_forward, y_left, z_up = [float(value) for value in point_xyz]
    return [
        round(-y_left, decimals),
        round(-z_up, decimals),
        round(x_forward, decimals),
    ]


def trajectory_camera_trajectory_from_dataset_points(
    trajectory: list[dict[str, Any]],
    intrinsics: TrajectoryCameraIntrinsics,
    decimals: int,
    base_to_camera_transform: np.ndarray | None,
) -> list[dict[str, Any]]:
    camera_points: list[dict[str, Any]] = []
    world_to_camera_transform = np.linalg.inv(base_to_camera_transform) if base_to_camera_transform is not None else None
    for point in trajectory:
        left_xyz_source = point["left_gripper_xyz"]
        right_xyz_source = point["right_gripper_xyz"]
        if world_to_camera_transform is not None:
            left_xyz_camera_link = trajectory_transform_point(world_to_camera_transform, left_xyz_source, decimals)
            right_xyz_camera_link = trajectory_transform_point(world_to_camera_transform, right_xyz_source, decimals)
        else:
            left_xyz_camera_link = left_xyz_source
            right_xyz_camera_link = right_xyz_source
        left_xyz_camera = left_xyz_camera_link
        right_xyz_camera = right_xyz_camera_link
        row = {
            "timestamp": point["timestamp"],
            "frame_index": point["frame_index"],
            "sampling": point.get("sampling"),
            "left_gripper_xyz_base": left_xyz_source,
            "right_gripper_xyz_base": right_xyz_source,
            "left_gripper_xyz_camera_link": left_xyz_camera_link,
            "right_gripper_xyz_camera_link": right_xyz_camera_link,
            "left_gripper_xyz_camera": left_xyz_camera,
            "right_gripper_xyz_camera": right_xyz_camera,
            "left_gripper_uv": trajectory_project_camera_point(left_xyz_camera, intrinsics, decimals),
            "right_gripper_uv": trajectory_project_camera_point(right_xyz_camera, intrinsics, decimals),
        }
        if "source_frame_indices" in point:
            row["source_frame_indices"] = point["source_frame_indices"]
            row["source_timestamps"] = point["source_timestamps"]
        camera_points.append(row)
    return camera_points


def trajectory_valid_uv_points(camera_trajectory: list[dict[str, Any]], key: str) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for row in camera_trajectory:
        uv = row.get(key) or {}
        if uv.get("valid") and uv.get("in_image") and uv.get("u") is not None and uv.get("v") is not None:
            points.append((int(round(float(uv["u"]))), int(round(float(uv["v"])))))
    return points


def trajectory_catmull_rom_point(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float,
) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def trajectory_smooth_points(points: list[tuple[int, int]], samples_per_segment: int = 24) -> list[tuple[int, int]]:
    if len(points) <= 2:
        return points
    arrays = [np.array(point, dtype=np.float32) for point in points]
    smoothed: list[tuple[int, int]] = []
    for index in range(len(arrays) - 1):
        p0 = arrays[max(index - 1, 0)]
        p1 = arrays[index]
        p2 = arrays[index + 1]
        p3 = arrays[min(index + 2, len(arrays) - 1)]
        for sample_index in range(samples_per_segment):
            t = sample_index / samples_per_segment
            point = trajectory_catmull_rom_point(p0, p1, p2, p3, t)
            smoothed.append((int(round(float(point[0]))), int(round(float(point[1])))))
    smoothed.append(points[-1])
    return smoothed


def trajectory_sparse_control_points(points: list[tuple[int, int]], max_middle_points: int = 2) -> list[tuple[int, int]]:
    if len(points) <= max_middle_points + 2:
        return points
    last_index = len(points) - 1
    middle_indices = [
        round((index + 1) * last_index / (max_middle_points + 1))
        for index in range(max_middle_points)
    ]
    indices = [0, *middle_indices, last_index]
    unique_indices = sorted(set(max(0, min(last_index, index)) for index in indices))
    return [points[index] for index in unique_indices]


def trajectory_draw_direction_arrows(
    image: Any,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    spacing_pixels: float = 48.0,
) -> None:
    if len(points) < 2:
        return
    distance_since_arrow = 0.0
    for start, end in zip(points, points[1:]):
        segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
        distance_since_arrow += segment_length
        if distance_since_arrow >= spacing_pixels:
            cv2.arrowedLine(image, start, end, color, 2, line_type=cv2.LINE_AA, tipLength=0.35)
            distance_since_arrow = 0.0
    cv2.arrowedLine(image, points[-2], points[-1], color, 2, line_type=cv2.LINE_AA, tipLength=0.35)


def trajectory_draw_polyline_with_points(
    image: Any,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
) -> None:
    if not points:
        return
    control_points = trajectory_sparse_control_points(points, max_middle_points=2)
    path_points = trajectory_smooth_points(control_points)
    if len(path_points) >= 2:
        cv2.polylines(image, [np.array(path_points, dtype=np.int32)], False, color, 2, lineType=cv2.LINE_AA)
        trajectory_draw_direction_arrows(image, path_points, color)
    for index, point in enumerate(control_points):
        radius = 5 if index in {0, len(control_points) - 1} else 3
        cv2.circle(image, point, radius, color, -1, lineType=cv2.LINE_AA)


def trajectory_draw_trajectory_overlay(
    image_path: Path,
    output_path: Path,
    camera_trajectory: list[dict[str, Any]],
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image for trajectory overlay: {image_path}")
    trajectory_draw_polyline_with_points(image, trajectory_valid_uv_points(camera_trajectory, "left_gripper_uv"), (255, 128, 0))
    trajectory_draw_polyline_with_points(image, trajectory_valid_uv_points(camera_trajectory, "right_gripper_uv"), (0, 64, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Cannot write trajectory overlay: {output_path}")
    return output_path


def trajectory_load_episode_index(dataset_root: Path, views: list[str]) -> dict[int, TrajectoryEpisodeRef]:
    meta_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No episode metadata parquet files under {dataset_root / 'meta' / 'episodes'}")

    refs: dict[int, TrajectoryEpisodeRef] = {}
    needed = {
        "episode_index",
        "data/chunk_index",
        "data/file_index",
    }
    for view in views:
        needed.add(f"videos/{view}/chunk_index")
        needed.add(f"videos/{view}/file_index")

    for path in meta_files:
        df = pd.read_parquet(path)
        missing = needed - set(df.columns)
        if missing:
            raise KeyError(f"{path} is missing metadata columns: {sorted(missing)}")
        for row in df.to_dict("records"):
            episode_index = int(row["episode_index"])
            videos: dict[str, TrajectoryVideoRef] = {}
            for view in views:
                chunk_value = row[f"videos/{view}/chunk_index"]
                file_value = row[f"videos/{view}/file_index"]
                if not pd.isna(chunk_value) and not pd.isna(file_value):
                    videos[view] = TrajectoryVideoRef(chunk_index=int(chunk_value), file_index=int(file_value))
            refs[episode_index] = TrajectoryEpisodeRef(
                episode_index=episode_index,
                data_chunk_index=int(row["data/chunk_index"]),
                data_file_index=int(row["data/file_index"]),
                videos=videos,
            )
    return refs


def trajectory_read_episode_data(
    dataset_root: Path,
    episode_ref: TrajectoryEpisodeRef,
    cache: dict[tuple[int, int], pd.DataFrame],
) -> pd.DataFrame:
    key = (episode_ref.data_chunk_index, episode_ref.data_file_index)
    if key not in cache:
        path = dataset_root / "data" / f"chunk-{key[0]:03d}" / f"file-{key[1]:03d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing data parquet: {path}")
        cache[key] = pd.read_parquet(
            path,
            columns=["observation.state", "timestamp", "frame_index", "episode_index"],
        )
    df = cache[key]
    episode_df = df[df["episode_index"] == episode_ref.episode_index].copy()
    if episode_df.empty:
        raise ValueError(f"No rows for episode {episode_ref.episode_index} in data file {key}")
    return episode_df.sort_values("timestamp")


def trajectory_parse_index_range(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", text)
    if not match:
        raise ValueError(f"Index range must look like start:end, got {text!r}")
    start, end = int(match.group(1)), int(match.group(2))
    if start >= end:
        raise ValueError(f"Invalid index range {text!r}: start must be less than end")
    return start, end


def trajectory_trajectory_internal_profile(internal: str) -> dict[str, str]:
    key = str(internal).strip().lower()
    if key not in TRAJECTORY_TRAJECTORY_INTERNAL_PROFILES:
        raise ValueError(f"Unknown trajectory internal {internal!r}; choose one of {sorted(TRAJECTORY_TRAJECTORY_INTERNAL_PROFILES)}")
    return TRAJECTORY_TRAJECTORY_INTERNAL_PROFILES[key]


def trajectory_resolve_xyz_index_ranges(
    internal: str,
    left_xyz_indices: str | None,
    right_xyz_indices: str | None,
) -> tuple[tuple[int, int], tuple[int, int], dict[str, str]]:
    internal_key = str(internal).strip().lower()
    profile = trajectory_trajectory_internal_profile(internal_key)
    left_text = left_xyz_indices or profile["left_xyz_indices"]
    right_text = right_xyz_indices or profile["right_xyz_indices"]

    return trajectory_parse_index_range(left_text), trajectory_parse_index_range(right_text), {
        "left_xyz_indices": left_text,
        "right_xyz_indices": right_text,
        "left_point_name": profile["left_point_name"],
        "right_point_name": profile["right_point_name"],
        "state_point_description": profile["state_point_description"],
        "coordinate_assumption": profile["coordinate_assumption"],
    }


def trajectory_vector_slice(values: Any, index_range: tuple[int, int], decimals: int) -> list[float]:
    start, end = index_range
    return [round(float(v), decimals) for v in list(values)[start:end]]


def trajectory_point_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def trajectory_path_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(trajectory_point_distance(points[index], points[index + 1]) for index in range(len(points) - 1))


def trajectory_active_gripper_from_source_3d(
    trajectory: list[dict[str, Any]],
    decimals: int,
    static_threshold_m: float = 0.01,
    dominance_ratio: float = 2.0,
) -> dict[str, Any]:
    left_points = [
        row["left_gripper_xyz"]
        for row in trajectory
        if isinstance(row.get("left_gripper_xyz"), list)
    ]
    right_points = [
        row["right_gripper_xyz"]
        for row in trajectory
        if isinstance(row.get("right_gripper_xyz"), list)
    ]
    left_length = trajectory_path_length(left_points)
    right_length = trajectory_path_length(right_points)

    if left_length < static_threshold_m and right_length < static_threshold_m:
        active_gripper = "unknown"
    elif left_length >= static_threshold_m and left_length >= right_length * dominance_ratio:
        active_gripper = "left_gripper"
    elif right_length >= static_threshold_m and right_length >= left_length * dominance_ratio:
        active_gripper = "right_gripper"
    else:
        active_gripper = "both"

    return {
        "active_gripper": active_gripper,
        "source": "source_3d_state_path_length",
        "left_path_length_m": round(left_length, decimals),
        "right_path_length_m": round(right_length, decimals),
        "static_threshold_m": static_threshold_m,
        "dominance_ratio": dominance_ratio,
    }


def trajectory_uniform_sample_times(start_time: float, end_time: float, num_keypoints: int) -> list[float]:
    if num_keypoints <= 0:
        return []
    if num_keypoints == 1 or end_time <= start_time:
        return [start_time]
    step = (end_time - start_time) / (num_keypoints - 1)
    return [start_time + step * index for index in range(num_keypoints)]


def trajectory_interpolate_values(left_values: Any, right_values: Any, ratio: float) -> list[float]:
    left_list = list(left_values)
    right_list = list(right_values)
    if len(left_list) != len(right_list):
        raise ValueError("Cannot interpolate observation.state rows with different lengths")
    return [
        float(left_value) + (float(right_value) - float(left_value)) * ratio
        for left_value, right_value in zip(left_list, right_list)
    ]


def trajectory_interpolate_episode_row(episode_df: pd.DataFrame, timestamp: float) -> dict[str, Any]:
    if episode_df.empty:
        raise ValueError("Cannot interpolate an empty episode dataframe")
    rows = episode_df.sort_values("timestamp").reset_index(drop=True)
    timestamps = [float(value) for value in rows["timestamp"].tolist()]

    if timestamp <= timestamps[0]:
        row = rows.iloc[0]
        return {
            "timestamp": timestamp,
            "frame_index": int(row["frame_index"]),
            "observation.state": list(row["observation.state"]),
            "interpolation": "clamped_start",
        }
    if timestamp >= timestamps[-1]:
        row = rows.iloc[-1]
        return {
            "timestamp": timestamp,
            "frame_index": int(row["frame_index"]),
            "observation.state": list(row["observation.state"]),
            "interpolation": "clamped_end",
        }

    upper_index = next(index for index, value in enumerate(timestamps) if value >= timestamp)
    lower_index = max(0, upper_index - 1)
    lower_row = rows.iloc[lower_index]
    upper_row = rows.iloc[upper_index]
    lower_time = timestamps[lower_index]
    upper_time = timestamps[upper_index]
    if upper_time == lower_time:
        ratio = 0.0
    else:
        ratio = (timestamp - lower_time) / (upper_time - lower_time)
    lower_frame = float(lower_row["frame_index"])
    upper_frame = float(upper_row["frame_index"])
    return {
        "timestamp": timestamp,
        "frame_index": int(round(lower_frame + (upper_frame - lower_frame) * ratio)),
        "observation.state": trajectory_interpolate_values(
            lower_row["observation.state"],
            upper_row["observation.state"],
            ratio,
        ),
        "interpolation": "linear",
        "source_frame_indices": [int(lower_row["frame_index"]), int(upper_row["frame_index"])],
        "source_timestamps": [round(lower_time, 6), round(upper_time, 6)],
    }


def trajectory_trajectory_from_window(
    episode_df: pd.DataFrame,
    start_time: float,
    end_time: float,
    left_range: tuple[int, int],
    right_range: tuple[int, int],
    num_keypoints: int,
    decimals: int,
) -> list[dict[str, Any]]:
    window_df = episode_df[(episode_df["timestamp"] >= start_time) & (episode_df["timestamp"] <= end_time)]
    if window_df.empty:
        nearest_index = (episode_df["timestamp"] - start_time).abs().idxmin()
        window_df = episode_df.loc[[nearest_index]]

    points: list[dict[str, Any]] = []
    if num_keypoints <= 0:
        sampled_rows = [
            {
                "timestamp": float(row["timestamp"]),
                "frame_index": int(row["frame_index"]),
                "observation.state": row["observation.state"],
                "interpolation": "observed",
            }
            for row in window_df.to_dict("records")
        ]
    else:
        sampled_rows = [
            trajectory_interpolate_episode_row(episode_df, sample_time)
            for sample_time in trajectory_uniform_sample_times(start_time, end_time, num_keypoints)
        ]

    for row in sampled_rows:
        points.append(
            {
                "timestamp": round(float(row["timestamp"]), 3),
                "frame_index": int(row["frame_index"]),
                "left_gripper_xyz": trajectory_vector_slice(row["observation.state"], left_range, decimals),
                "right_gripper_xyz": trajectory_vector_slice(row["observation.state"], right_range, decimals),
                "sampling": row["interpolation"],
            }
        )
        if "source_frame_indices" in row:
            points[-1]["source_frame_indices"] = row["source_frame_indices"]
            points[-1]["source_timestamps"] = row["source_timestamps"]
    return points


def trajectory_extract_first_frame(
    dataset_root: Path,
    episode_ref: TrajectoryEpisodeRef,
    view: str,
    timestamp: float,
    output_path: Path,
    fps: float,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path
    video_ref = episode_ref.videos.get(view)
    if video_ref is None:
        raise ValueError(f"Episode {episode_ref.episode_index} has no video metadata for {view}")

    video_path = (
        dataset_root
        / "videos"
        / view
        / f"chunk-{video_ref.chunk_index:03d}"
        / f"file-{video_ref.file_index:03d}.mp4"
    )
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        frame_index = max(0, int(math.floor(timestamp * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame from {video_path}")
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Cannot write frame image: {output_path}")
    finally:
        cap.release()
    return output_path


def trajectory_build_trajectory_qa(args: argparse.Namespace) -> dict[str, Any]:
    source = trajectory_load_json(args.input)
    items = source.get("items") if isinstance(source, dict) else source
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list or an object with an 'items' list")

    views = trajectory_parse_views(args.views)
    if args.primary_view not in views:
        views.insert(0, args.primary_view)
    episode_refs = trajectory_load_episode_index(args.dataset_root, views)
    data_cache: dict[tuple[int, int], pd.DataFrame] = {}
    left_range, right_range, trajectory_profile = trajectory_resolve_xyz_index_ranges(
        args.internal,
        args.left_xyz_indices,
        args.right_xyz_indices,
    )
    intrinsics = trajectory_camera_intrinsics_from_name(args.internal)
    base_to_camera_transform = (
        trajectory_transform_matrix_from_xyz_rpy(args.base_to_camera_xyz, args.base_to_camera_rpy)
        if args.use_base_to_camera_extrinsic
        else None
    )

    qa_items_2d: list[dict[str, Any]] = []
    qa_items_3d: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for item in items:
        try:
            video_id = item["video_id"]
            episode_index = trajectory_parse_video_id(video_id)
            episode_ref = episode_refs[episode_index]
            answer_seconds = item.get("answer_seconds") or {}
            start_time = float(answer_seconds["start"])
            end_time = float(answer_seconds["end"])
            category = trajectory_category_from_item(item)
            prompt_context = {
                "category": category,
                "views": ", ".join(trajectory_view_label(view) for view in views),
                "primary_view": trajectory_view_label(args.primary_view),
            }
            prompt_2d = args.prompt_template_2d.format(**prompt_context)
            prompt_3d = args.prompt_template_3d.format(**prompt_context)

            episode_df = trajectory_read_episode_data(args.dataset_root, episode_ref, data_cache)
            trajectory = trajectory_trajectory_from_window(
                episode_df=episode_df,
                start_time=start_time,
                end_time=end_time,
                left_range=left_range,
                right_range=right_range,
                num_keypoints=args.num_keypoints,
                decimals=args.decimals,
            )
            active_gripper_metadata = trajectory_active_gripper_from_source_3d(
                trajectory=trajectory,
                decimals=args.decimals,
            )
            active_gripper = active_gripper_metadata["active_gripper"]
            images: dict[str, str] = {}
            for view in views:
                label = trajectory_view_label(view)
                image_path = trajectory_extract_first_frame(
                    dataset_root=args.dataset_root,
                    episode_ref=episode_ref,
                    view=view,
                    timestamp=start_time,
                    output_path=args.image_dir / label / f"{item['id']}_{label}.jpg",
                    fps=args.fps,
                    overwrite=args.overwrite_images,
                )
                images[label] = str(image_path)
            primary_image = images[trajectory_view_label(args.primary_view)]
            camera_trajectory = trajectory_camera_trajectory_from_dataset_points(
                trajectory=trajectory,
                intrinsics=intrinsics,
                decimals=args.decimals,
                base_to_camera_transform=base_to_camera_transform,
            )
            trajectory_overlay = trajectory_draw_trajectory_overlay(
                image_path=Path(primary_image),
                output_path=args.image_dir
                / "trajectory_overlay"
                / trajectory_view_label(args.primary_view)
                / f"{item['id']}_{trajectory_view_label(args.primary_view)}_trajectory.jpg",
                camera_trajectory=camera_trajectory,
                overwrite=args.overwrite_images,
            )

            common_answer_metadata = {
                "extrinsic": {
                    "enabled": bool(args.use_base_to_camera_extrinsic),
                    "applied": bool(args.use_base_to_camera_extrinsic),
                    "parent": "world",
                    "child": "camera",
                    "xyz": args.base_to_camera_xyz,
                    "rpy": args.base_to_camera_rpy,
                    "matrix_camera_to_world": base_to_camera_transform.tolist()
                    if base_to_camera_transform is not None
                    else None,
                    "matrix_world_to_camera": np.linalg.inv(base_to_camera_transform).tolist()
                    if base_to_camera_transform is not None
                    else None,
                },
                "camera_profile": intrinsics.profile,
                "camera_intrinsics": {
                    "K": intrinsics.K,
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "cx": intrinsics.cx,
                    "cy": intrinsics.cy,
                    "width": intrinsics.width,
                    "height": intrinsics.height,
                    "distortion_model": intrinsics.distortion_model,
                    "distortion_coefficients": intrinsics.distortion_coefficients,
                },
                "sampling_method": (
                    "uniform_time_linear_interpolation"
                    if args.num_keypoints > 0
                    else "observed_rows_in_time_window"
                ),
                "input_views": views,
                "input_view_labels": [trajectory_view_label(view) for view in views],
                "prediction_view": args.primary_view,
                "prediction_view_label": trajectory_view_label(args.primary_view),
                "state_indices": {
                    "left_gripper_xyz": list(range(left_range[0], left_range[1])),
                    "right_gripper_xyz": list(range(right_range[0], right_range[1])),
                },
                "state_point_names": {
                    "left_gripper_xyz": trajectory_profile["left_point_name"],
                    "right_gripper_xyz": trajectory_profile["right_point_name"],
                },
                "state_point_description": trajectory_profile["state_point_description"],
                "active_gripper": active_gripper,
                "active_gripper_metadata": active_gripper_metadata,
            }
            answer_2d = {
                **common_answer_metadata,
                "coordinate_frame": "image_pixels",
                "source_coordinate_frame": "world",
                "camera_coordinate_frame": "camera_opencv",
                "axis_convention": "2D image pixels with u increasing right and v increasing down",
                "projection": "u = fx * x / z + cx, v = fy * y / z + cy",
                "coordinate_assumption": trajectory_profile["coordinate_assumption"],
                "trajectory_2d_pixels": [
                    {
                        "timestamp": row["timestamp"],
                        "frame_index": row["frame_index"],
                        "sampling": row.get("sampling"),
                        "left_gripper_uv": row["left_gripper_uv"],
                        "right_gripper_uv": row["right_gripper_uv"],
                    }
                    for row in camera_trajectory
                ],
                "trajectory": [
                    {
                        "timestamp": row["timestamp"],
                        "frame_index": row["frame_index"],
                        "sampling": row.get("sampling"),
                        "left_gripper_uv": row["left_gripper_uv"],
                        "right_gripper_uv": row["right_gripper_uv"],
                    }
                    for row in camera_trajectory
                ],
            }
            answer_3d = {
                **common_answer_metadata,
                "coordinate_frame": "camera_opencv",
                "source_coordinate_frame": "world",
                "camera_link_frame": "camera",
                "axis_convention": "3D camera coordinates in meters; x right, y down, z forward",
                "axis_conversion": None,
                "coordinate_assumption": trajectory_profile["coordinate_assumption"],
                "trajectory_3d_camera": [
                    {
                        "timestamp": row["timestamp"],
                        "frame_index": row["frame_index"],
                        "sampling": row.get("sampling"),
                        "left_gripper_xyz": row["left_gripper_xyz_camera"],
                        "right_gripper_xyz": row["right_gripper_xyz_camera"],
                    }
                    for row in camera_trajectory
                ],
                "trajectory_3d_base": [
                    {
                        "timestamp": row["timestamp"],
                        "frame_index": row["frame_index"],
                        "sampling": row.get("sampling"),
                        "left_gripper_xyz": row["left_gripper_xyz_base"],
                        "right_gripper_xyz": row["right_gripper_xyz_base"],
                    }
                    for row in camera_trajectory
                ],
                "trajectory_3d_camera_link": [
                    {
                        "timestamp": row["timestamp"],
                        "frame_index": row["frame_index"],
                        "sampling": row.get("sampling"),
                        "left_gripper_xyz": row["left_gripper_xyz_camera_link"],
                        "right_gripper_xyz": row["right_gripper_xyz_camera_link"],
                    }
                    for row in camera_trajectory
                ],
                "trajectory": [
                    {
                        "timestamp": row["timestamp"],
                        "frame_index": row["frame_index"],
                        "sampling": row.get("sampling"),
                        "left_gripper_xyz": row["left_gripper_xyz_camera"],
                        "right_gripper_xyz": row["right_gripper_xyz_camera"],
                    }
                    for row in camera_trajectory
                ],
            }
            common_item_metadata = {
                "video_id": video_id,
                "category": category,
                "image": primary_image,
                "main_image": primary_image,
                "prediction_image": primary_image,
                "images": images,
                "view": args.primary_view,
                "primary_view": args.primary_view,
                "prediction_view": args.primary_view,
                "views": views,
                "internal": args.internal,
                "active_gripper": active_gripper,
                "active_gripper_metadata": active_gripper_metadata,
                "answer_seconds": {"start": start_time, "end": end_time},
                "source_item_id": item.get("id"),
                "metadata": {
                    "source_question": item.get("Q") or item.get("question"),
                    "source_answer": item.get("A") or item.get("answer"),
                    "source_metadata": item.get("metadata", {}),
                },
            }
            qa_items_2d.append(
                {
                    **common_item_metadata,
                    "id": f"{item['id']}_trajectory_2d",
                    "type": "trajectory_prediction_2d",
                    "trajectory_overlay_image": str(trajectory_overlay),
                    "Q": prompt_2d,
                    "question": prompt_2d,
                    "A": answer_2d,
                    "answer": answer_2d,
                    "answer_text": json.dumps(answer_2d, ensure_ascii=False),
                }
            )
            qa_items_3d.append(
                {
                    **common_item_metadata,
                    "id": f"{item['id']}_trajectory_3d",
                    "type": "trajectory_prediction_3d",
                    "Q": prompt_3d,
                    "question": prompt_3d,
                    "A": answer_3d,
                    "answer": answer_3d,
                    "answer_text": json.dumps(answer_3d, ensure_ascii=False),
                }
            )
        except Exception as exc:  # keep batch generation usable and report exact misses
            skipped.append({"id": str(item.get("id", "")), "error": str(exc)})
            if not args.skip_errors:
                raise

    common_output_metadata = {
        "input": str(args.input),
        "dataset_root": str(args.dataset_root),
        "primary_view": args.primary_view,
        "prediction_view": args.primary_view,
        "prediction_view_label": trajectory_view_label(args.primary_view),
        "views": views,
        "view_labels": [trajectory_view_label(view) for view in views],
        "num_keypoints": args.num_keypoints,
        "internal": args.internal,
        "state_indices": {
            "left_gripper_xyz": list(range(left_range[0], left_range[1])),
            "right_gripper_xyz": list(range(right_range[0], right_range[1])),
        },
        "state_point_names": {
            "left_gripper_xyz": trajectory_profile["left_point_name"],
            "right_gripper_xyz": trajectory_profile["right_point_name"],
        },
        "state_point_description": trajectory_profile["state_point_description"],
        "camera_intrinsics": {
            "K": intrinsics.K,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "width": intrinsics.width,
            "height": intrinsics.height,
            "distortion_model": intrinsics.distortion_model,
            "distortion_coefficients": intrinsics.distortion_coefficients,
        },
        "base_to_camera_extrinsic": {
            "enabled": bool(args.use_base_to_camera_extrinsic),
            "applied": bool(args.use_base_to_camera_extrinsic),
            "parent": "world",
            "child": "camera",
            "xyz": args.base_to_camera_xyz,
            "rpy": args.base_to_camera_rpy,
            "matrix_camera_to_world": base_to_camera_transform.tolist()
            if base_to_camera_transform is not None
            else None,
            "matrix_world_to_camera": np.linalg.inv(base_to_camera_transform).tolist()
            if base_to_camera_transform is not None
            else None,
        },
        "coordinate_source": trajectory_profile["coordinate_assumption"],
        "skipped": skipped,
    }
    return {
        "2d": {
            **common_output_metadata,
            "task": "trajectory_prediction_2d",
            "items": qa_items_2d,
        },
        "3d": {
            **common_output_metadata,
            "task": "trajectory_prediction_3d",
            "items": qa_items_3d,
        },
    }

def build_trajectory_task_outputs(
    time_items: list[dict[str, Any]],
    output_dir: Path,
    dataset_root: Path,
    views: str,
    primary_view: str,
    image_dir: Path,
    num_keypoints: int,
    left_xyz_indices: str,
    right_xyz_indices: str,
    decimals: int,
    fps: float,
    use_base_to_camera_extrinsic: bool,
    base_to_camera_xyz: list[float],
    base_to_camera_rpy: list[float],
    overwrite_images: bool,
    skip_errors: bool,
    prompt_template_2d: str,
    prompt_template_3d: str,
) -> dict[str, Any]:
    input_path = output_dir / "trajectory_time_input.json"
    output_2d_path = output_dir / "trajectory_qa_2d.json"
    output_3d_path = output_dir / "trajectory_qa_3d.json"
    output_combined_path = output_dir / "trajectory_qa.json"
    save_json(
        input_path,
        {
            "task": "time_eqa",
            "source": "planning_workflow_time_items",
            "items": time_items,
        },
    )
    trajectory_args = argparse.Namespace(
        input=input_path,
        dataset_root=dataset_root,
        output=output_combined_path,
        output_2d=output_2d_path,
        output_3d=output_3d_path,
        image_dir=image_dir,
        views=views,
        primary_view=primary_view,
        prompt_template=TRAJECTORY_DEFAULT_PROMPT,
        prompt_template_2d=prompt_template_2d,
        prompt_template_3d=prompt_template_3d,
        internal="gripper",
        num_keypoints=num_keypoints,
        left_xyz_indices=left_xyz_indices,
        right_xyz_indices=right_xyz_indices,
        decimals=decimals,
        fps=fps,
        use_base_to_camera_extrinsic=use_base_to_camera_extrinsic,
        base_to_camera_xyz=base_to_camera_xyz,
        base_to_camera_rpy=base_to_camera_rpy,
        overwrite_images=overwrite_images,
        skip_errors=skip_errors,
    )
    output = trajectory_build_trajectory_qa(trajectory_args)
    save_json(output_2d_path, output["2d"])
    save_json(output_3d_path, output["3d"])
    save_json(output_combined_path, output)
    return {
        "input_path": str(input_path),
        "output_2d_path": str(output_2d_path),
        "output_3d_path": str(output_3d_path),
        "output_combined_path": str(output_combined_path),
        "num_2d_items": len(output["2d"]["items"]),
        "num_3d_items": len(output["3d"]["items"]),
        "num_skipped": len(output["2d"]["skipped"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build generic Time, Planning, and Understanding VQA JSON from segment files.")
    parser.set_defaults(**{})
    parser.add_argument("--config", type=Path, default=None, help="JSON config path. CLI arguments override config values.")
    parser.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--video-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument(
        "--prejoined-video-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help="Directory containing already joined multi-view episode videos named by video_id.",
    )
    parser.add_argument(
        "--multi-view-video-root",
        default=argparse.SUPPRESS,
        help="Root containing per-view video folders. Use an empty string to fall back to --video-dir single-view media.",
    )
    parser.add_argument(
        "--views",
        default=argparse.SUPPRESS,
        help="Comma-separated view specs like left_eye=observation.images.left_eye,left_wrist=...",
    )
    parser.add_argument("--output-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--video-exts", default=argparse.SUPPRESS)
    parser.add_argument(
        "--tasks",
        default=argparse.SUPPRESS,
        help="Comma-separated: planning,planning_2,step_order,trajectory",
    )
    parser.add_argument("--file-limit", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--category-label-path", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--num-options", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--nearby-distractors-per-question", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--generated-distractors-per-question", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--llm-distractors-per-label", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--use-llm-distractors", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--llm-distractor-api-url", default=argparse.SUPPRESS)
    parser.add_argument("--llm-distractor-api-key-env", default=argparse.SUPPRESS)
    parser.add_argument("--llm-distractor-api-key", default=argparse.SUPPRESS)
    parser.add_argument("--llm-distractor-model", default=argparse.SUPPRESS)
    parser.add_argument("--llm-distractor-timeout", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--llm-distractor-max-retries", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--llm-distractor-cache-path", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--window-mode", choices=["raw", "transition", "legacy_pickplace"], default=argparse.SUPPRESS)
    parser.add_argument("--pick-before-window", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--place-before-window", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--default-before-window", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--after-window", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--planning-timestamp-key", choices=["start", "end"], default=argparse.SUPPRESS)
    parser.add_argument("--time-question", default=argparse.SUPPRESS)
    parser.add_argument("--planning-question", default=argparse.SUPPRESS)
    parser.add_argument("--planning-2-question", default=argparse.SUPPRESS)
    parser.add_argument("--step-order-question", default=argparse.SUPPRESS)
    parser.add_argument("--step-order-view", default=argparse.SUPPRESS)
    parser.add_argument("--step-order-initial-frame", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--step-order-end-offset", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--step-order-end-offset-ratio", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--step-order-cell-width", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--step-order-jpeg-quality", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--step-order-seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--left-right-question", default=argparse.SUPPRESS)
    parser.add_argument("--left-right-target-side", choices=["left", "right", "both", "alternate"], default=argparse.SUPPRESS)
    parser.add_argument("--left-right-timestamp-key", choices=["start", "mid", "end"], default=argparse.SUPPRESS)
    parser.add_argument("--left-right-head-view", default=argparse.SUPPRESS)
    parser.add_argument("--left-right-left-wrist-view", default=argparse.SUPPRESS)
    parser.add_argument("--left-right-right-wrist-view", default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-dataset-root", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-views", default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-primary-view", default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-image-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-internal", choices=["gripper"], default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-num-keypoints", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-left-xyz-indices", default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-right-xyz-indices", default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-decimals", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-fps", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--trajectory-use-base-to-camera-extrinsic",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--trajectory-base-to-camera-xyz", type=float, nargs=3, default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-base-to-camera-rpy", type=float, nargs=3, default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-overwrite-images", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-skip-errors", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-prompt-template-2d", default=argparse.SUPPRESS)
    parser.add_argument("--trajectory-prompt-template-3d", default=argparse.SUPPRESS)
    parser.add_argument(
        "--crop-time-video-top",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="For Time EQA only, crop the top fraction from each input video before writing video paths.",
    )
    parser.add_argument(
        "--time-crop-top-fraction",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of video height to remove from the top for Time EQA videos.",
    )
    parser.add_argument(
        "--time-cropped-video-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help="Directory for cropped Time EQA videos. Defaults to <output-dir>/time_video_crop_top.",
    )
    parser.add_argument("--overwrite-time-crop", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument(
        "--no-media",
        dest="no_media",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Generate JSON only; do not extract frames or clips.",
    )
    parser.add_argument(
        "--extract-media",
        dest="no_media",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Extract frames and clips even if config sets no_media=true.",
    )
    raw_args = vars(parser.parse_args())
    config_path = raw_args.pop("config")
    args = argparse.Namespace(**merge_config(config_path, raw_args))

    tasks = {task.strip() for task in str(args.tasks).split(",") if task.strip()}
    unknown_tasks = tasks - {"planning", "planning_2", "step_order", "trajectory"}
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {sorted(unknown_tasks)}")
    if str(args.trajectory_internal).strip().lower() != "gripper":
        raise ValueError("Trajectory generation only supports trajectory_internal='gripper'.")

    video_exts = tuple(ext.strip() for ext in args.video_exts.split(",") if ext.strip())
    multi_view_video_root = args.multi_view_video_root if str(args.multi_view_video_root).strip() else None
    views = parse_view_specs(args.views)
    time_cropped_video_dir = args.time_cropped_video_dir or (args.output_dir / "time_video_crop_top")
    segment_rows = load_segments(args.data_dir, args.file_limit)
    time_items = build_time_items(
        segment_rows,
        question_template=args.time_question,
        window_mode=args.window_mode,
        pick_before_window=args.pick_before_window,
        place_before_window=args.place_before_window,
        default_before_window=args.default_before_window,
        after_window=args.after_window,
        prejoined_video_dir=args.prejoined_video_dir,
        multi_view_video_root=multi_view_video_root,
        views=views,
        video_exts=video_exts,
        crop_time_video_top=args.crop_time_video_top,
        time_cropped_video_dir=time_cropped_video_dir,
        time_crop_top_fraction=args.time_crop_top_fraction,
        overwrite_time_crop=args.overwrite_time_crop,
        no_media=args.no_media,
    )
    category_labels = load_category_labels(args.category_label_path, time_items)
    task_name = task_name_from_category_label_path(args.category_label_path)
    llm_distractor_cache_path = args.llm_distractor_cache_path or (args.output_dir / "llm_distractors.json")
    needs_choice_tasks = bool(tasks & {"planning", "planning_2"})
    llm_distractor_pool = (
        build_llm_distractor_pool(
            category_labels=category_labels,
            use_llm=bool(args.use_llm_distractors),
            per_label=int(args.llm_distractors_per_label),
            api_url=str(args.llm_distractor_api_url or ""),
            api_key_env=str(args.llm_distractor_api_key_env or "OPENAI_API_KEY"),
            api_key=str(args.llm_distractor_api_key or ""),
            model=str(args.llm_distractor_model or ""),
            timeout=int(args.llm_distractor_timeout),
            max_retries=int(args.llm_distractor_max_retries),
            cache_path=llm_distractor_cache_path,
        )
        if needs_choice_tasks
        else {}
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    planning_option_design = {
        "num_options": args.num_options,
        "correct": 1,
        "none": 1,
        "nearby_action_distractors": args.nearby_distractors_per_question,
        "generated_wrong_label_distractors": args.generated_distractors_per_question,
        "llm_distractors_per_label": args.llm_distractors_per_label,
        "use_llm_distractors": bool(args.use_llm_distractors),
        "llm_distractor_model": args.llm_distractor_model or None,
        "llm_distractor_cache_path": str(llm_distractor_cache_path) if needs_choice_tasks else None,
        "category_label_path": str(args.category_label_path) if args.category_label_path else None,
        "num_category_labels": len(category_labels),
        "category_labels": category_labels,
        "task_name": task_name,
        "note": "Planning/Understanding options are correct + nearby real action labels + generated wrong labels + All other options are wrong.",
    }
    common = {
        "source": str(args.data_dir),
        "video_dir": str(args.video_dir),
        "prejoined_video_dir": str(args.prejoined_video_dir) if args.prejoined_video_dir is not None else None,
        "multi_view_video_root": str(multi_view_video_root) if multi_view_video_root is not None else None,
        "views": views if multi_view_video_root is not None else None,
        "time_crop_top_applied": args.crop_time_video_top,
        "time_crop_top_fraction": args.time_crop_top_fraction if args.crop_time_video_top else None,
        "time_cropped_video_dir": str(time_cropped_video_dir) if args.crop_time_video_top else None,
        "window_mode": args.window_mode,
        "num_source_segments": len(segment_rows),
    }

    if "planning" in tasks:
        planning_items = build_planning_items(
            time_items,
            video_dir=args.video_dir,
            prejoined_video_dir=args.prejoined_video_dir,
            clips_dir=args.output_dir / "planning_clips",
            video_exts=video_exts,
            question=args.planning_question,
            num_options=args.num_options,
            timestamp_key=args.planning_timestamp_key,
            no_media=args.no_media,
            multi_view_video_root=multi_view_video_root,
            views=views,
            llm_distractor_pool=llm_distractor_pool,
            category_labels=category_labels,
            nearby_distractors_per_question=args.nearby_distractors_per_question,
            generated_distractors_per_question=args.generated_distractors_per_question,
        )
        save_json(
            args.output_dir / "planning_vqa.json",
            {
                **common,
                "task": "next_action_planning",
                "clips_dir": str(args.output_dir / "planning_clips"),
                "timestamp_key": args.planning_timestamp_key,
                "question": args.planning_question,
                "option_design": planning_option_design,
                "items": planning_items,
            },
            skip_existing=True,
        )

    if "planning_2" in tasks:
        planning_2_items = build_planning_2_items(
            time_items,
            video_dir=args.video_dir,
            prejoined_video_dir=args.prejoined_video_dir,
            frames_dir=args.output_dir / "planning_2_frames",
            video_exts=video_exts,
            question_template=args.planning_2_question,
            task_name=task_name,
            num_options=args.num_options,
            timestamp_key=args.planning_timestamp_key,
            no_media=args.no_media,
            multi_view_video_root=multi_view_video_root,
            views=views,
            llm_distractor_pool=llm_distractor_pool,
            category_labels=category_labels,
            nearby_distractors_per_question=args.nearby_distractors_per_question,
            generated_distractors_per_question=args.generated_distractors_per_question,
        )
        save_json(
            args.output_dir / "planning_2_vqa.json",
            {
                **common,
                "task": "next_action_planning_with_task_prompt",
                "frames_dir": str(args.output_dir / "planning_2_frames"),
                "timestamp_key": args.planning_timestamp_key,
                "task_name": task_name,
                "question": args.planning_2_question,
                "option_design": planning_option_design,
                "items": planning_2_items,
            },
            skip_existing=True,
        )

    if "step_order" in tasks:
        step_order_items, step_order_skipped = build_step_order_items(
            data_dir=args.data_dir,
            file_limit=args.file_limit,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            video_exts=video_exts,
            question=args.step_order_question,
            step_order_view=args.step_order_view,
            initial_frame=int(args.step_order_initial_frame),
            end_offset=int(args.step_order_end_offset),
            end_offset_ratio=float(args.step_order_end_offset_ratio),
            cell_width=int(args.step_order_cell_width),
            jpeg_quality=int(args.step_order_jpeg_quality),
            seed=int(args.step_order_seed),
            no_media=args.no_media,
            multi_view_video_root=multi_view_video_root,
            views=views,
        )
        save_json(
            args.output_dir / "step_order_vqa.json",
            {
                **common,
                "task": "step_order_with_initial_state",
                "question": args.step_order_question,
                "step_order_dir": str(args.output_dir / "step_order"),
                "step_order_view": "multiview" if multi_view_video_root is not None else "default",
                "step_order_views": list(views) if multi_view_video_root is not None else ["default"],
                "initial_frame": int(args.step_order_initial_frame),
                "end_offset": int(args.step_order_end_offset),
                "end_offset_ratio": float(args.step_order_end_offset_ratio),
                "cell_width": int(args.step_order_cell_width),
                "jpeg_quality": int(args.step_order_jpeg_quality),
                "seed": int(args.step_order_seed),
                "num_samples": len(step_order_items),
                "num_skipped": len(step_order_skipped),
                "skipped": step_order_skipped,
                "items": step_order_items,
            },
            skip_existing=True,
        )
        save_json(
            args.output_dir / "step_order_vqa_only.json",
            [step_order_vqa_pair(item) for item in step_order_items],
            skip_existing=True,
        )

    if "trajectory" in tasks:
        trajectory_image_dir = args.trajectory_image_dir or (args.output_dir / "trajectory_first_frames")
        trajectory_outputs = build_trajectory_task_outputs(
            time_items=time_items,
            output_dir=args.output_dir,
            dataset_root=args.trajectory_dataset_root,
            views=args.trajectory_views,
            primary_view=args.trajectory_primary_view,
            image_dir=trajectory_image_dir,
            num_keypoints=int(args.trajectory_num_keypoints),
            left_xyz_indices=args.trajectory_left_xyz_indices,
            right_xyz_indices=args.trajectory_right_xyz_indices,
            decimals=int(args.trajectory_decimals),
            fps=float(args.trajectory_fps),
            use_base_to_camera_extrinsic=bool(args.trajectory_use_base_to_camera_extrinsic),
            base_to_camera_xyz=[float(value) for value in args.trajectory_base_to_camera_xyz],
            base_to_camera_rpy=[float(value) for value in args.trajectory_base_to_camera_rpy],
            overwrite_images=bool(args.trajectory_overwrite_images),
            skip_errors=bool(args.trajectory_skip_errors),
            prompt_template_2d=args.trajectory_prompt_template_2d,
            prompt_template_3d=args.trajectory_prompt_template_3d,
        )
        save_json(
            args.output_dir / "trajectory_task_manifest.json",
            {
                **common,
                "task": "trajectory_prediction",
                "dataset_root": str(args.trajectory_dataset_root),
                "image_dir": str(trajectory_image_dir),
                "views": args.trajectory_views,
                "primary_view": args.trajectory_primary_view,
                "internal": "gripper",
                "num_keypoints": int(args.trajectory_num_keypoints),
                "outputs": trajectory_outputs,
            },
            skip_existing=True,
        )

    print(f"Source segments: {len(segment_rows)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Generated tasks: {', '.join(sorted(tasks))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
