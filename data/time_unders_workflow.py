#!/usr/bin/env python3
# coding: utf-8
"""Generic workflow for building Time, Understanding, Left/Right, and image-in-video VQA JSON.

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
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import requests

if not hasattr(argparse, "BooleanOptionalAction"):
    class _BooleanOptionalAction(argparse.Action):
        def __init__(
            self,
            option_strings: list[str],
            dest: str,
            default: Any = None,
            **kwargs: Any,
        ) -> None:
            option_strings = list(option_strings)
            for option_string in option_strings[:]:
                if option_string.startswith("--"):
                    option_strings.append("--no-" + option_string[2:])
            super().__init__(option_strings=option_strings, dest=dest, nargs=0, default=default, **kwargs)

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: str | None,
            option_string: str | None = None,
        ) -> None:
            setattr(namespace, self.dest, not str(option_string).startswith("--no-"))

    argparse.BooleanOptionalAction = _BooleanOptionalAction  # type: ignore[attr-defined]

DEFAULT_DATA_DIR = Path("/home/kewei/YWC/egodata/pickplace/json")
DEFAULT_VIDEO_DIR = Path("/home/kewei/YWC/egodata/pickplace/video")
DEFAULT_MULTI_VIEW_VIDEO_ROOT = Path("/home/kewei/NAS/lerobot_datasets-26-06-17-17-25-20/gripper/videos")
DEFAULT_VIEWS = "left_eye=observation.images.left_eye,left_wrist=observation.images.left_wrist,right_wrist=observation.images.right_wrist"
DEFAULT_OUTPUT_DIR = Path("/home/kewei/YWC/egodata/pickplace/workflow_outputs")
DEFAULT_TIME_CROPPED_VIDEO_DIR = Path("/home/kewei/YWC/egodata/pickplace/workflow_outputs/time_video_crop_top")
DEFAULT_CATEGORY_LABEL_PATH = Path("/home/kewei/YWC/egodata/pickplace/workflow/cube.txt")
DEFAULT_VIDEO_EXTS = ".mp4,.webm,.mov,.mkv,.avi"
DEFAULT_TASKS = "time,understanding,left_right,image_in_video"
DEFAULT_NUM_OPTIONS = 6

DEFAULT_TIME_QUESTION = 'When did the action "{action}" happen?'
DEFAULT_UNDERSTANDING_QUESTION = (
    "Based on the egocentric video up to now, choose the ONE option that best matches what is happening RIGHT NOW?"
)
DEFAULT_LEFT_RIGHT_QUESTION = (
    "Given the image captured by the head camera, which option shows the {side} "
    "gripper camera's view at this moment?"
)
DEFAULT_IMAGE_IN_VIDEO_QUESTION = (
    "Given this left-eye video clip of an action segment, which option image appeared in the clip?"
)
NONE_OPTION_TEXT = "All other options are wrong."
CONFIG_PATH_KEYS = {
    "data_dir",
    "video_dir",
    "multi_view_video_root",
    "output_dir",
    "time_cropped_video_dir",
    "llm_distractor_cache_path",
    "category_label_path",
}
CONFIG_LIST_OR_STRING_KEYS = {"tasks", "video_exts"}
CONFIG_KEYS = {
    "data_dir",
    "video_dir",
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
    "time_question",
    "understanding_question",
    "left_right_question",
    "image_in_video_question",
    "image_in_video_view",
    "left_right_target_side",
    "left_right_timestamp_key",
    "left_right_head_view",
    "left_right_left_wrist_view",
    "left_right_right_wrist_view",
    "crop_time_video_top",
    "time_crop_top_fraction",
    "time_cropped_video_dir",
    "overwrite_time_crop",
    "no_media",
}


def default_config() -> dict[str, Any]:
    return {
        "data_dir": DEFAULT_DATA_DIR,
        "video_dir": DEFAULT_VIDEO_DIR,
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
        "time_question": DEFAULT_TIME_QUESTION,
        "understanding_question": DEFAULT_UNDERSTANDING_QUESTION,
        "left_right_question": DEFAULT_LEFT_RIGHT_QUESTION,
        "image_in_video_question": DEFAULT_IMAGE_IN_VIDEO_QUESTION,
        "image_in_video_view": "left_eye",
        "left_right_target_side": "both",
        "left_right_timestamp_key": "mid",
        "left_right_head_view": "left_eye",
        "left_right_left_wrist_view": "left_wrist",
        "left_right_right_wrist_view": "right_wrist",
        "crop_time_video_top": False,
        "time_crop_top_fraction": 0.1,
        "time_cropped_video_dir": None,
        "overwrite_time_crop": False,
        "no_media": False,
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = relativize_paths_for_json(data, path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
    suffix = "_segments.json"
    name = path.name
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


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
    multi_view_video_root: Path | None,
    views: dict[str, str],
    video_exts: tuple[str, ...],
    crop_time_video_top: bool,
    time_cropped_video_dir: Path,
    time_crop_top_fraction: float,
    overwrite_time_crop: bool,
    no_media: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    items: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
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
        try:
            if multi_view_video_root is not None and not no_media:
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
        except FileNotFoundError as exc:
            skipped.append(
                {
                    "id": str(segment["id"]),
                    "video_id": str(video_id),
                    "source_path": row["source_path"],
                    "reason": str(exc),
                }
            )
            print(f"[skip missing video] {video_id}: {exc}", flush=True)
            continue
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
    return items, skipped


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


def build_understanding_options(
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
        raise ValueError("--num-options must be at least 4 for understanding")
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(output_path.suffix or ".jpg", frame)
    if not ok:
        raise RuntimeError(f"Failed to write frame: {output_path}")
    encoded.tofile(str(output_path))
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


def left_right_timestamp_for_item(item: dict[str, Any], timestamp_key: str) -> float:
    start = float(item["answer_seconds"]["start"])
    end = float(item["answer_seconds"]["end"])
    if timestamp_key == "start":
        return start
    if timestamp_key == "end":
        return end
    if timestamp_key == "mid":
        return (start + end) / 2
    raise ValueError(f"Unknown left_right_timestamp_key: {timestamp_key}")


def left_right_target_sides_for_item(item: dict[str, Any], target_side: str) -> list[str]:
    if target_side in {"left", "right"}:
        return [target_side]
    if target_side == "both":
        return ["left", "right"]
    if target_side == "alternate":
        digest = hashlib.md5(str(item["id"]).encode("utf-8")).hexdigest()
        return ["left" if int(digest[:8], 16) % 2 == 0 else "right"]
    raise ValueError(f"Unknown left_right_target_side: {target_side}")


def deterministic_rows(seed: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.md5(f"{seed}|{row.get('id')}".encode("utf-8")).hexdigest())


def deterministic_timestamp(seed: str, start: float, end: float) -> float:
    if end <= start:
        return start
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    ratio = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    return start + ratio * (end - start)


def left_right_temporal_items(
    item: dict[str, Any],
    all_items: list[dict[str, Any]],
    timestamp_key: str,
    count: int,
) -> list[dict[str, Any]]:
    video_id = str(item["video_id"])
    item_id = str(item["id"])
    timestamp = left_right_timestamp_for_item(item, timestamp_key)
    candidates = [
        row
        for row in all_items
        if str(row.get("video_id")) == video_id
        and str(row.get("id")) != item_id
        and abs(left_right_timestamp_for_item(row, timestamp_key) - timestamp) > 0.5
    ]
    return deterministic_rows(f"{item_id}|left_right_temporal", candidates)[:count]


def left_right_scene_items(item: dict[str, Any], all_items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    video_id = str(item["video_id"])
    item_id = str(item["id"])
    candidates = [row for row in all_items if str(row.get("video_id")) != video_id]
    return deterministic_rows(f"{item_id}|left_right_scene", candidates)[:count]


def left_right_option_image(
    item: dict[str, Any],
    view_name: str,
    view_dir: str,
    role: str,
    timestamp: float,
    images_dir: Path,
    multi_view_video_root: Path,
    video_exts: tuple[str, ...],
    video_cache: dict[tuple[str, str], Path],
) -> dict[str, Any]:
    video_id = str(item["video_id"])
    item_id = str(item["id"])
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

    image_name = (
        f"{safe_filename(video_id)}_{safe_filename(item_id)}_"
        f"{timestamp:.3f}s_{safe_filename(view_name)}_{safe_filename(role)}.jpg"
    )
    image_path = images_dir / "option_images" / view_name / video_id / image_name
    frame_index, actual_timestamp = extract_frame(video_path, timestamp, image_path)
    return {
        "type": "image",
        "text": None,
        "image_path": str(image_path),
        "view": view_name,
        "role": role,
        "source_item_id": item_id,
        "source_video_id": video_id,
        "video_path": str(video_path),
        "timestamp": timestamp,
        "actual_timestamp": actual_timestamp,
        "frame_index": frame_index,
    }


def build_left_right_options(
    item: dict[str, Any],
    side: str,
    timestamp: float,
    all_items: list[dict[str, Any]],
    timestamp_key: str,
    images_dir: Path,
    multi_view_video_root: Path,
    video_exts: tuple[str, ...],
    views: dict[str, str],
    left_wrist_view: str,
    right_wrist_view: str,
    video_cache: dict[tuple[str, str], Path],
) -> list[dict[str, Any]]:
    item_id = str(item["id"])
    correct_view = left_wrist_view if side == "left" else right_wrist_view
    opposite_side = "right" if side == "left" else "left"
    opposite_view = right_wrist_view if side == "left" else left_wrist_view

    option_rows: list[dict[str, Any]] = [
        {
            **left_right_option_image(
                item=item,
                view_name=correct_view,
                view_dir=views[correct_view],
                role="correct",
                timestamp=timestamp,
                images_dir=images_dir,
                multi_view_video_root=multi_view_video_root,
                video_exts=video_exts,
                video_cache=video_cache,
            ),
            "is_correct": True,
            "distractor_type": None,
        },
        {
            **left_right_option_image(
                item=item,
                view_name=opposite_view,
                view_dir=views[opposite_view],
                role="symmetric_distractor",
                timestamp=timestamp,
                images_dir=images_dir,
                multi_view_video_root=multi_view_video_root,
                video_exts=video_exts,
                video_cache=video_cache,
            ),
            "is_correct": False,
            "distractor_type": "symmetric",
            "target_opposite_side": opposite_side,
        },
    ]

    temporal_items = left_right_temporal_items(item, all_items, timestamp_key, 1)
    if len(temporal_items) < 1:
        raise ValueError(f"Not enough temporal distractors for {item_id}: need 1, got {len(temporal_items)}")
    temporal_item = temporal_items[0]
    temporal_timestamp = left_right_timestamp_for_item(temporal_item, timestamp_key)
    option_rows.append(
        {
            **left_right_option_image(
                item=temporal_item,
                view_name=correct_view,
                view_dir=views[correct_view],
                role="temporal_distractor_1",
                timestamp=temporal_timestamp,
                images_dir=images_dir,
                multi_view_video_root=multi_view_video_root,
                video_exts=video_exts,
                video_cache=video_cache,
            ),
            "is_correct": False,
            "distractor_type": "temporal",
        }
    )

    scene_items = left_right_scene_items(item, all_items, 2)
    if len(scene_items) < 2:
        raise ValueError(f"Not enough scene distractors for {item_id}: need 2, got {len(scene_items)}")
    for index, scene_item in enumerate(scene_items, 1):
        scene_timestamp = left_right_timestamp_for_item(scene_item, timestamp_key)
        scene_view = left_wrist_view if index % 2 else right_wrist_view
        option_rows.append(
            {
                **left_right_option_image(
                    item=scene_item,
                    view_name=scene_view,
                    view_dir=views[scene_view],
                    role=f"scene_distractor_{index}",
                    timestamp=scene_timestamp,
                    images_dir=images_dir,
                    multi_view_video_root=multi_view_video_root,
                    video_exts=video_exts,
                    video_cache=video_cache,
                ),
                "is_correct": False,
                "distractor_type": "scene",
            }
        )

    option_rows.append(
        {
            "type": "none",
            "text": NONE_OPTION_TEXT,
            "image_path": None,
            "view": None,
            "role": "none_option",
            "source_item_id": None,
            "source_video_id": None,
            "is_correct": False,
            "distractor_type": "none",
            "is_none_option": True,
        }
    )

    shuffled = deterministic_option_texts(f"{item_id}|{side}|left_right_options", [json.dumps(row, sort_keys=True) for row in option_rows])
    row_by_json = {json.dumps(row, sort_keys=True): row for row in option_rows}
    return [
        {
            "id": chr(ord("A") + index),
            "is_none_option": row_by_json[row].get("type") == "none",
            **row_by_json[row],
        }
        for index, row in enumerate(shuffled)
    ]


def build_left_right_items(
    time_items: list[dict[str, Any]],
    images_dir: Path,
    multi_view_video_root: Path | None,
    video_exts: tuple[str, ...],
    views: dict[str, str],
    question_template: str,
    target_side: str,
    timestamp_key: str,
    head_view: str,
    left_wrist_view: str,
    right_wrist_view: str,
    no_media: bool,
) -> list[dict[str, Any]]:
    if no_media:
        raise ValueError("left_right task requires media extraction. Set no_media=false in config or pass --extract-media.")
    if multi_view_video_root is None:
        raise ValueError("left_right task requires --multi-view-video-root with head and wrist views.")
    for view_name in (head_view, left_wrist_view, right_wrist_view):
        if view_name not in views:
            raise ValueError(f"left_right view {view_name!r} is not present in --views")
    if target_side not in {"left", "right", "both", "alternate"}:
        raise ValueError("--left-right-target-side must be one of left,right,both,alternate")
    if timestamp_key not in {"start", "mid", "end"}:
        raise ValueError("--left-right-timestamp-key must be one of start,mid,end")

    video_cache: dict[tuple[str, str], Path] = {}
    output_items: list[dict[str, Any]] = []
    for item in time_items:
        item_id = str(item["id"])
        video_id = str(item["video_id"])
        timestamp = left_right_timestamp_for_item(item, timestamp_key)
        head_image = left_right_option_image(
            item=item,
            view_name=head_view,
            view_dir=views[head_view],
            role="question_head",
            timestamp=timestamp,
            images_dir=images_dir,
            multi_view_video_root=multi_view_video_root,
            video_exts=video_exts,
            video_cache=video_cache,
        )
        for side in left_right_target_sides_for_item(item, target_side):
            question = question_template.format(side=side, video_id=video_id, item_id=item_id)
            options = build_left_right_options(
                item=item,
                side=side,
                timestamp=timestamp,
                all_items=time_items,
                timestamp_key=timestamp_key,
                images_dir=images_dir,
                multi_view_video_root=multi_view_video_root,
                video_exts=video_exts,
                views=views,
                left_wrist_view=left_wrist_view,
                right_wrist_view=right_wrist_view,
                video_cache=video_cache,
            )
            correct_option = next(option for option in options if option.get("is_correct"))
            option_lines = [
                f"{option['id']}. {option['text']}"
                if option.get("is_none_option")
                else f"{option['id']}. <image: {option['image_path']}>"
                for option in options
            ]
            full_question = f"{question}\nOptions:\n" + "\n".join(option_lines)
            output_items.append(
                {
                    "id": f"{item_id}_{side}_gripper_view",
                    "source_id": item_id,
                    "video_id": video_id,
                    "type": "left_right_gripper_view",
                    "target_side": side,
                    "timestamp": timestamp,
                    "timestamp_key": timestamp_key,
                    "input": {
                        "image_path": head_image["image_path"],
                        "head_image": head_image,
                    },
                    "Q": full_question,
                    "A": correct_option["id"],
                    "question": full_question,
                    "answer": correct_option["id"],
                    "answer_text": f"{side} gripper camera view",
                    "options": options,
                    "correct_option": correct_option,
                    "source_time_eqa": item,
                }
            )
    return output_items


def image_in_video_category(item: dict[str, Any]) -> str:
    return clean_text(str(item["metadata"].get("narration") or item.get("answer_action") or "the action"))


def image_in_video_candidate_items(
    item: dict[str, Any],
    all_items: list[dict[str, Any]],
    same_video: bool,
    same_category: bool,
    count: int,
    seed_suffix: str,
) -> list[dict[str, Any]]:
    video_id = str(item["video_id"])
    item_id = str(item["id"])
    category = normalize_option_text(image_in_video_category(item))
    candidates = []
    for row in all_items:
        if str(row.get("id")) == item_id:
            continue
        row_same_video = str(row.get("video_id")) == video_id
        row_same_category = normalize_option_text(image_in_video_category(row)) == category
        if row_same_video == same_video and row_same_category == same_category:
            candidates.append(row)
    return deterministic_rows(f"{item_id}|image_in_video|{seed_suffix}", candidates)[:count]


def image_in_video_option_image(
    item: dict[str, Any],
    view_name: str,
    view_dir: str,
    role: str,
    images_dir: Path,
    multi_view_video_root: Path,
    video_exts: tuple[str, ...],
    video_cache: dict[tuple[str, str], Path],
    crop_time_video_top: bool,
    time_cropped_video_dir: Path,
    time_crop_top_fraction: float,
    overwrite_time_crop: bool,
) -> dict[str, Any]:
    video_id = str(item["video_id"])
    item_id = str(item["id"])
    start = float(item["answer_seconds"]["start"])
    end = float(item["answer_seconds"]["end"])
    timestamp = deterministic_timestamp(f"{item_id}|{role}|frame", start, end)
    cache_key = (view_name, video_id)
    video_path = video_cache.get(cache_key)
    if video_path is None:
        video_path = multiview_video_path_for(
            video_id=video_id,
            multi_view_video_root=multi_view_video_root,
            view_dir=view_dir,
            video_exts=video_exts,
        )
        if crop_time_video_top:
            original_video_path = video_path
            video_path = time_cropped_video_dir / view_name / f"{video_id}{original_video_path.suffix}"
            crop_video_top(original_video_path, video_path, time_crop_top_fraction, overwrite_time_crop)
        video_cache[cache_key] = video_path

    image_name = (
        f"{safe_filename(video_id)}_{safe_filename(item_id)}_"
        f"{timestamp:.3f}s_{safe_filename(view_name)}_{safe_filename(role)}.jpg"
    )
    image_path = images_dir / "option_images" / view_name / video_id / image_name
    frame_index, actual_timestamp = extract_frame(video_path, timestamp, image_path)
    return {
        "type": "image",
        "text": None,
        "image_path": str(image_path),
        "view": view_name,
        "role": role,
        "source_item_id": item_id,
        "source_video_id": video_id,
        "source_category": image_in_video_category(item),
        "video_path": str(video_path),
        "crop_top_applied": crop_time_video_top,
        "crop_top_fraction": time_crop_top_fraction if crop_time_video_top else None,
        "timestamp": timestamp,
        "actual_timestamp": actual_timestamp,
        "frame_index": frame_index,
    }


def build_image_in_video_options(
    item: dict[str, Any],
    all_items: list[dict[str, Any]],
    images_dir: Path,
    multi_view_video_root: Path,
    video_exts: tuple[str, ...],
    view_name: str,
    view_dir: str,
    video_cache: dict[tuple[str, str], Path],
    crop_time_video_top: bool,
    time_cropped_video_dir: Path,
    time_crop_top_fraction: float,
    overwrite_time_crop: bool,
) -> list[dict[str, Any]]:
    item_id = str(item["id"])
    option_rows: list[dict[str, Any]] = [
        {
            **image_in_video_option_image(
                item=item,
                view_name=view_name,
                view_dir=view_dir,
                role="correct",
                images_dir=images_dir,
                multi_view_video_root=multi_view_video_root,
                video_exts=video_exts,
                video_cache=video_cache,
                crop_time_video_top=crop_time_video_top,
                time_cropped_video_dir=time_cropped_video_dir,
                time_crop_top_fraction=time_crop_top_fraction,
                overwrite_time_crop=overwrite_time_crop,
            ),
            "is_correct": True,
            "distractor_type": None,
        }
    ]

    same_video_other_category = image_in_video_candidate_items(
        item, all_items, same_video=True, same_category=False, count=2, seed_suffix="same_video_other_category"
    )
    if len(same_video_other_category) < 2:
        raise ValueError(
            f"Not enough same-video different-category distractors for {item_id}: "
            f"need 2, got {len(same_video_other_category)}"
        )
    for index, row in enumerate(same_video_other_category, 1):
        option_rows.append(
            {
                **image_in_video_option_image(
                    item=row,
                    view_name=view_name,
                    view_dir=view_dir,
                    role=f"same_video_other_category_{index}",
                    images_dir=images_dir,
                    multi_view_video_root=multi_view_video_root,
                    video_exts=video_exts,
                    video_cache=video_cache,
                    crop_time_video_top=crop_time_video_top,
                    time_cropped_video_dir=time_cropped_video_dir,
                    time_crop_top_fraction=time_crop_top_fraction,
                    overwrite_time_crop=overwrite_time_crop,
                ),
                "is_correct": False,
                "distractor_type": "same_video_other_category",
            }
        )

    other_video_same_category = image_in_video_candidate_items(
        item, all_items, same_video=False, same_category=True, count=1, seed_suffix="other_video_same_category"
    )
    if len(other_video_same_category) < 1:
        raise ValueError(f"Not enough other-video same-category distractors for {item_id}: need 1, got 0")
    option_rows.append(
        {
            **image_in_video_option_image(
                item=other_video_same_category[0],
                view_name=view_name,
                view_dir=view_dir,
                role="other_video_same_category",
                images_dir=images_dir,
                multi_view_video_root=multi_view_video_root,
                video_exts=video_exts,
                video_cache=video_cache,
                crop_time_video_top=crop_time_video_top,
                time_cropped_video_dir=time_cropped_video_dir,
                time_crop_top_fraction=time_crop_top_fraction,
                overwrite_time_crop=overwrite_time_crop,
            ),
            "is_correct": False,
            "distractor_type": "other_video_same_category",
        }
    )

    other_video_other_category = image_in_video_candidate_items(
        item, all_items, same_video=False, same_category=False, count=1, seed_suffix="other_video_other_category"
    )
    if len(other_video_other_category) < 1:
        raise ValueError(f"Not enough other-video different-category distractors for {item_id}: need 1, got 0")
    option_rows.append(
        {
            **image_in_video_option_image(
                item=other_video_other_category[0],
                view_name=view_name,
                view_dir=view_dir,
                role="other_video_other_category",
                images_dir=images_dir,
                multi_view_video_root=multi_view_video_root,
                video_exts=video_exts,
                video_cache=video_cache,
                crop_time_video_top=crop_time_video_top,
                time_cropped_video_dir=time_cropped_video_dir,
                time_crop_top_fraction=time_crop_top_fraction,
                overwrite_time_crop=overwrite_time_crop,
            ),
            "is_correct": False,
            "distractor_type": "other_video_other_category",
        }
    )

    option_rows.append(
        {
            "type": "none",
            "text": NONE_OPTION_TEXT,
            "image_path": None,
            "view": None,
            "role": "none_option",
            "source_item_id": None,
            "source_video_id": None,
            "source_category": None,
            "is_correct": False,
            "distractor_type": "none",
            "is_none_option": True,
        }
    )

    shuffled = deterministic_option_texts(
        f"{item_id}|image_in_video_options",
        [json.dumps(row, sort_keys=True) for row in option_rows],
    )
    row_by_json = {json.dumps(row, sort_keys=True): row for row in option_rows}
    return [
        {
            "id": chr(ord("A") + index),
            "is_none_option": row_by_json[row].get("type") == "none",
            **row_by_json[row],
        }
        for index, row in enumerate(shuffled)
    ]


def build_image_in_video_items(
    time_items: list[dict[str, Any]],
    clips_dir: Path,
    images_dir: Path,
    multi_view_video_root: Path | None,
    video_exts: tuple[str, ...],
    views: dict[str, str],
    question: str,
    view_name: str,
    no_media: bool,
    crop_time_video_top: bool,
    time_cropped_video_dir: Path,
    time_crop_top_fraction: float,
    overwrite_time_crop: bool,
) -> list[dict[str, Any]]:
    if no_media:
        raise ValueError("image_in_video task requires media extraction. Set no_media=false in config or pass --extract-media.")
    if multi_view_video_root is None:
        raise ValueError("image_in_video task requires --multi-view-video-root.")
    if view_name not in views:
        raise ValueError(f"image_in_video view {view_name!r} is not present in --views")

    view_dir = views[view_name]
    video_cache: dict[tuple[str, str], Path] = {}
    output_items: list[dict[str, Any]] = []
    for item in time_items:
        item_id = str(item["id"])
        video_id = str(item["video_id"])
        start = float(item["answer_seconds"]["start"])
        end = float(item["answer_seconds"]["end"])
        video_path = video_cache.get((view_name, video_id))
        if video_path is None:
            video_path = multiview_video_path_for(
                video_id=video_id,
                multi_view_video_root=multi_view_video_root,
                view_dir=view_dir,
                video_exts=video_exts,
            )
            original_video_path = video_path
            if crop_time_video_top:
                video_path = time_cropped_video_dir / view_name / f"{video_id}{original_video_path.suffix}"
                crop_video_top(original_video_path, video_path, time_crop_top_fraction, overwrite_time_crop)
            video_cache[(view_name, video_id)] = video_path
        clip_path = clips_dir / view_name / video_id / f"{safe_filename(video_id)}_{safe_filename(item_id)}_{start:.3f}_{end:.3f}_{safe_filename(view_name)}.mp4"
        clip_info = extract_clip(video_path, start, end, clip_path)
        options = build_image_in_video_options(
            item=item,
            all_items=time_items,
            images_dir=images_dir,
            multi_view_video_root=multi_view_video_root,
            video_exts=video_exts,
            view_name=view_name,
            view_dir=view_dir,
            video_cache=video_cache,
            crop_time_video_top=crop_time_video_top,
            time_cropped_video_dir=time_cropped_video_dir,
            time_crop_top_fraction=time_crop_top_fraction,
            overwrite_time_crop=overwrite_time_crop,
        )
        correct_option = next(option for option in options if option.get("is_correct"))
        option_lines = [
            f"{option['id']}. {option['text']}"
            if option.get("is_none_option")
            else f"{option['id']}. <image: {option['image_path']}>"
            for option in options
        ]
        full_question = f"{question}\nOptions:\n" + "\n".join(option_lines)
        output_items.append(
            {
                "id": f"{item_id}_image_in_video",
                "source_id": item_id,
                "video_id": video_id,
                "type": "image_in_video",
                "view": view_name,
                "input": {
                    "clip_path": str(clip_path),
                    "clip_paths": [str(clip_path)],
                    "video_path": str(clip_path),
                    "video_paths": [str(clip_path)],
                    "source_video_path": str(video_path),
                    "crop_top_applied": crop_time_video_top,
                    "crop_top_fraction": time_crop_top_fraction if crop_time_video_top else None,
                    "start": start,
                    "end": end,
                    **clip_info,
                },
                "Q": full_question,
                "A": correct_option["id"],
                "question": full_question,
                "answer": correct_option["id"],
                "answer_text": "the option image that appeared in the video clip",
                "answer_category": image_in_video_category(item),
                "answer_seconds": {"start": start, "end": end},
                "options": options,
                "correct_option": correct_option,
                "source_time_eqa": item,
            }
        )
    return output_items


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


def extract_multiview_clips(
    video_id: str,
    item_id: str,
    start: float,
    end: float,
    clips_dir: Path,
    multi_view_video_root: Path,
    views: dict[str, str],
    video_exts: tuple[str, ...],
    video_cache: dict[tuple[str, str], Path],
) -> dict[str, dict[str, Any]]:
    clips: dict[str, dict[str, Any]] = {}
    video_paths = multiview_video_paths_for(video_id, multi_view_video_root, views, video_exts, video_cache)
    for view_name, video_path in video_paths.items():
        clip_name = f"{safe_filename(video_id)}_{safe_filename(item_id)}_{start:.3f}_{end:.3f}_{safe_filename(view_name)}.mp4"
        clip_path = clips_dir / view_name / video_id / clip_name
        clip_info = extract_clip(video_path, start, end, clip_path)
        clips[view_name] = {
            "view": view_name,
            "clip_path": str(clip_path),
            "video_path": str(video_path),
            **clip_info,
        }
    return clips


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


def action_text_items(time_items: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "id": str(item["id"]),
            "answer_action_text": clean_text(str(item["metadata"].get("narration") or item.get("answer_action") or "the action")),
        }
        for item in time_items
    ]


def build_understanding_items(
    time_items: list[dict[str, Any]],
    video_dir: Path,
    clips_dir: Path,
    video_exts: tuple[str, ...],
    question: str,
    num_options: int,
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
    multiview_video_cache: dict[tuple[str, str], Path] = {}
    output_items: list[dict[str, Any]] = []

    for item in time_items:
        item_id = str(item["id"])
        video_id = str(item["video_id"])
        start = float(item["answer_seconds"]["start"])
        end = float(item["answer_seconds"]["end"])
        answer_text = clean_text(str(item["metadata"].get("narration") or item.get("answer_action") or "the action"))
        options = build_understanding_options(
            item_id=f"{item_id}_understand",
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
        option_lines = "\n".join(f"{option['id']}. {option['text']}" for option in options)
        full_question = f"{question}\nOptions:\n{option_lines}"

        input_data: dict[str, Any] = {
            "video_path": None,
            "start": start,
            "end": end,
        }
        if not no_media:
            if multi_view_video_root is not None:
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
                "id": f"{item_id}_understand",
                "source_id": item_id,
                "video_id": video_id,
                "type": "understanding",
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
            }
        )
    return output_items


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Time, Understanding, Left/Right, and image-in-video VQA JSON from segment files.")
    parser.set_defaults(**{})
    parser.add_argument("--config", type=Path, default=None, help="JSON config path. CLI arguments override config values.")
    parser.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--video-dir", type=Path, default=argparse.SUPPRESS)
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
    parser.add_argument("--tasks", default=argparse.SUPPRESS, help="Comma-separated: time,understanding,left_right,image_in_video")
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
    parser.add_argument("--time-question", default=argparse.SUPPRESS)
    parser.add_argument("--understanding-question", default=argparse.SUPPRESS)
    parser.add_argument("--left-right-question", default=argparse.SUPPRESS)
    parser.add_argument("--image-in-video-question", default=argparse.SUPPRESS)
    parser.add_argument("--image-in-video-view", default=argparse.SUPPRESS)
    parser.add_argument("--left-right-target-side", choices=["left", "right", "both", "alternate"], default=argparse.SUPPRESS)
    parser.add_argument("--left-right-timestamp-key", choices=["start", "mid", "end"], default=argparse.SUPPRESS)
    parser.add_argument("--left-right-head-view", default=argparse.SUPPRESS)
    parser.add_argument("--left-right-left-wrist-view", default=argparse.SUPPRESS)
    parser.add_argument("--left-right-right-wrist-view", default=argparse.SUPPRESS)
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
    unknown_tasks = tasks - {"time", "understanding", "left_right", "image_in_video"}
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {sorted(unknown_tasks)}")

    video_exts = tuple(ext.strip() for ext in args.video_exts.split(",") if ext.strip())
    multi_view_video_root = args.multi_view_video_root if str(args.multi_view_video_root).strip() else None
    views = parse_view_specs(args.views)
    time_cropped_video_dir = args.time_cropped_video_dir or (args.output_dir / "time_video_crop_top")
    segment_rows = load_segments(args.data_dir, args.file_limit)
    time_items, missing_media_skipped = build_time_items(
        segment_rows,
        question_template=args.time_question,
        window_mode=args.window_mode,
        pick_before_window=args.pick_before_window,
        place_before_window=args.place_before_window,
        default_before_window=args.default_before_window,
        after_window=args.after_window,
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
    needs_choice_tasks = bool(tasks & {"understanding"})
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
    understanding_option_design = {
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
        "note": "Understanding options are correct + nearby real action labels + generated wrong labels + All other options are wrong.",
    }
    common = {
        "source": str(args.data_dir),
        "video_dir": str(args.video_dir),
        "multi_view_video_root": str(multi_view_video_root) if multi_view_video_root is not None else None,
        "views": views if multi_view_video_root is not None else None,
        "time_crop_top_applied": args.crop_time_video_top,
        "time_crop_top_fraction": args.time_crop_top_fraction if args.crop_time_video_top else None,
        "time_cropped_video_dir": str(time_cropped_video_dir) if args.crop_time_video_top else None,
        "window_mode": args.window_mode,
        "num_source_segments": len(segment_rows),
        "num_missing_media_skipped": len(missing_media_skipped),
        "missing_media_skipped": missing_media_skipped,
    }

    if "time" in tasks:
        save_json(
            args.output_dir / "time_vqa.json",
            {**common, "task": "time_eqa", "question": args.time_question, "items": time_items},
        )

    if "understanding" in tasks:
        understanding_items = build_understanding_items(
            time_items,
            video_dir=args.video_dir,
            clips_dir=args.output_dir / "understanding_clips",
            video_exts=video_exts,
            question=args.understanding_question,
            num_options=args.num_options,
            no_media=args.no_media,
            multi_view_video_root=multi_view_video_root,
            views=views,
            llm_distractor_pool=llm_distractor_pool,
            category_labels=category_labels,
            nearby_distractors_per_question=args.nearby_distractors_per_question,
            generated_distractors_per_question=args.generated_distractors_per_question,
        )
        save_json(
            args.output_dir / "understanding_vqa.json",
            {
                **common,
                "task": "current_action_understanding",
                "clips_dir": str(args.output_dir / "understanding_clips"),
                "question": args.understanding_question,
                "option_design": understanding_option_design,
                "items": understanding_items,
            },
        )

    if "left_right" in tasks:
        left_right_items = build_left_right_items(
            time_items,
            images_dir=args.output_dir / "left_right_vqa",
            multi_view_video_root=multi_view_video_root,
            video_exts=video_exts,
            views=views,
            question_template=args.left_right_question,
            target_side=args.left_right_target_side,
            timestamp_key=args.left_right_timestamp_key,
            head_view=args.left_right_head_view,
            left_wrist_view=args.left_right_left_wrist_view,
            right_wrist_view=args.left_right_right_wrist_view,
            no_media=args.no_media,
        )
        save_json(
            args.output_dir / "left_right_vqa.json",
            {
                **common,
                "task": "left_right_gripper_view_matching",
                "images_dir": str(args.output_dir / "left_right_vqa"),
                "question": args.left_right_question,
                "target_side": args.left_right_target_side,
                "timestamp_key": args.left_right_timestamp_key,
                "head_view": args.left_right_head_view,
                "left_wrist_view": args.left_right_left_wrist_view,
                "right_wrist_view": args.left_right_right_wrist_view,
                "option_design": {
                    "num_options": 6,
                    "correct": 1,
                    "none": 1,
                    "symmetric_distractors": 1,
                    "temporal_distractors": 1,
                    "scene_distractors": 2,
                },
                "items": left_right_items,
            },
        )

    if "image_in_video" in tasks:
        image_in_video_items = build_image_in_video_items(
            time_items,
            clips_dir=args.output_dir / "image_in_video_clips",
            images_dir=args.output_dir / "image_in_video_vqa",
            multi_view_video_root=multi_view_video_root,
            video_exts=video_exts,
            views=views,
            question=args.image_in_video_question,
            view_name=args.image_in_video_view,
            no_media=args.no_media,
            crop_time_video_top=args.crop_time_video_top,
            time_cropped_video_dir=time_cropped_video_dir,
            time_crop_top_fraction=args.time_crop_top_fraction,
            overwrite_time_crop=args.overwrite_time_crop,
        )
        save_json(
            args.output_dir / "image_in_video_vqa.json",
            {
                **common,
                "task": "image_in_video_matching",
                "clips_dir": str(args.output_dir / "image_in_video_clips"),
                "images_dir": str(args.output_dir / "image_in_video_vqa"),
                "question": args.image_in_video_question,
                "view": args.image_in_video_view,
                "option_design": {
                    "num_options": 6,
                    "correct": 1,
                    "none": 1,
                    "same_video_other_category_distractors": 2,
                    "other_video_same_category_distractors": 1,
                    "other_video_other_category_distractors": 1,
                },
                "items": image_in_video_items,
            },
        )

    print(f"Source segments: {len(segment_rows)}")
    if missing_media_skipped:
        print(f"Skipped missing media: {len(missing_media_skipped)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Generated tasks: {', '.join(sorted(tasks))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
