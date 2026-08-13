#!/usr/bin/env python3
# coding: utf-8
"""Trajectory：夹爪轨迹预测（2D 像素坐标 / 3D 相机坐标两套输入）。

九个任务里唯一的回归任务，也是唯一带自适应重试的 —— 2D 预测点越界时会
带着越界信息重问一次。指标是 Hausdorff / discrete Fréchet / Chamfer 三个距离，
经 100/(1+(d/tol)^2) 映射成 0~100 分。

几何函数、容差规则、prompt 与汇总口径均从 test/trajectory_glm_test.py 逐字搬运。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..parsing import first_json_object
from .base import CallContext, Unit, base_row, image_part, one_item_per_unit, text_part

MAX_COORDINATE_RETRIES = 1


# --------------------------------------------------------------------------
# 几何：逐字搬运
# --------------------------------------------------------------------------


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def directed_hausdorff(a: list[list[float]], b: list[list[float]]) -> float | None:
    if not a or not b:
        return None
    return max(min(euclidean(x, y) for y in b) for x in a)


def hausdorff_distance(a: list[list[float]], b: list[list[float]]) -> float | None:
    forward = directed_hausdorff(a, b)
    backward = directed_hausdorff(b, a)
    if forward is None or backward is None:
        return None
    return max(forward, backward)


def chamfer_distance(a: list[list[float]], b: list[list[float]]) -> float | None:
    if not a or not b:
        return None
    a_to_b = sum(min(euclidean(x, y) for y in b) for x in a) / len(a)
    b_to_a = sum(min(euclidean(y, x) for x in a) for y in b) / len(b)
    return a_to_b + b_to_a


def discrete_frechet_distance(a: list[list[float]], b: list[list[float]]) -> float | None:
    if not a or not b:
        return None
    rows, cols = len(a), len(b)
    cache = [[-1.0 for _ in range(cols)] for _ in range(rows)]

    def compute(i: int, j: int) -> float:
        if cache[i][j] >= 0:
            return cache[i][j]
        dist = euclidean(a[i], b[j])
        if i == 0 and j == 0:
            value = dist
        elif i > 0 and j == 0:
            value = max(compute(i - 1, 0), dist)
        elif i == 0 and j > 0:
            value = max(compute(0, j - 1), dist)
        else:
            value = max(min(compute(i - 1, j), compute(i - 1, j - 1), compute(i, j - 1)), dist)
        cache[i][j] = value
        return value

    return compute(rows - 1, cols - 1)


def rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def mean_available(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    return sum(finite) / len(finite) if finite else None


def path_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(euclidean(points[i], points[i + 1]) for i in range(len(points) - 1))


def flatten_points(trajectories: dict[str, list[list[float]]]) -> list[list[float]]:
    return trajectories.get("left_gripper", []) + trajectories.get("right_gripper", [])


def point_cloud_extent(points: list[list[float]]) -> float | None:
    if not points:
        return None
    dim = len(points[0])
    mins = [min(p[i] for p in points) for i in range(dim)]
    maxs = [max(p[i] for p in points) for i in range(dim)]
    return math.sqrt(sum((maxs[i] - mins[i]) ** 2 for i in range(dim)))


# --------------------------------------------------------------------------
# item 读取：逐字搬运
# --------------------------------------------------------------------------


def image_label_from_key(key: Any) -> str:
    return str(key).split(".")[-1]


def infer_dimension(item: dict[str, Any]) -> int:
    answer = item.get("answer") or item.get("A") or {}
    frame = str(answer.get("coordinate_frame", "")).lower() if isinstance(answer, dict) else ""
    return 2 if ("image" in frame or "pixel" in frame) else 3


def expected_trajectory(item: dict[str, Any]) -> dict[str, list[list[float]]]:
    answer = item.get("answer") or item.get("A") or {}
    rows = answer.get("trajectory", []) if isinstance(answer, dict) else []
    left: list[list[float]] = []
    right: list[list[float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "left_gripper_uv" in row:
            l_uv = row.get("left_gripper_uv") or {}
            r_uv = row.get("right_gripper_uv") or {}
            if l_uv.get("valid") and l_uv.get("u") is not None and l_uv.get("v") is not None:
                left.append([float(l_uv["u"]), float(l_uv["v"])])
            if r_uv.get("valid") and r_uv.get("u") is not None and r_uv.get("v") is not None:
                right.append([float(r_uv["u"]), float(r_uv["v"])])
        else:
            if isinstance(row.get("left_gripper_xyz"), list):
                left.append([float(v) for v in row["left_gripper_xyz"]])
            if isinstance(row.get("right_gripper_xyz"), list):
                right.append([float(v) for v in row["right_gripper_xyz"]])
    return {"left_gripper": left, "right_gripper": right}


def active_gripper_for_item(item: dict[str, Any], expected: dict[str, list[list[float]]] | None = None) -> str:
    answer = item.get("answer") or item.get("A") or {}
    for source in (item, answer if isinstance(answer, dict) else {}):
        if not isinstance(source, dict):
            continue
        active = source.get("active_gripper")
        if active in {"left_gripper", "right_gripper", "both", "unknown"}:
            return str(active)
        metadata = source.get("active_gripper_metadata")
        if isinstance(metadata, dict) and metadata.get("active_gripper") in {
            "left_gripper", "right_gripper", "both", "unknown",
        }:
            return str(metadata["active_gripper"])

    trajectories = expected if expected is not None else expected_trajectory(item)
    left_length = path_length(trajectories.get("left_gripper", []))
    right_length = path_length(trajectories.get("right_gripper", []))
    static_threshold = 0.01 if infer_dimension(item) == 3 else 1.0
    dominance_ratio = 2.0
    if left_length < static_threshold and right_length < static_threshold:
        return "unknown"
    if left_length >= static_threshold and left_length >= right_length * dominance_ratio:
        return "left_gripper"
    if right_length >= static_threshold and right_length >= left_length * dominance_ratio:
        return "right_gripper"
    return "both"


def grippers_to_score(active_gripper: str) -> list[str]:
    if active_gripper in {"left_gripper", "right_gripper"}:
        return [active_gripper]
    if active_gripper == "both":
        return ["left_gripper", "right_gripper"]
    return []


def main_view_image_path_for_item(item: dict[str, Any]) -> Path | None:
    for key in ("prediction_image", "main_image", "image"):
        if item.get(key):
            return Path(str(item[key]))
    images = item.get("images")
    if isinstance(images, dict) and images:
        prediction_view = item.get("prediction_view") or item.get("primary_view")
        label = image_label_from_key(prediction_view) if prediction_view else ""
        for key in (prediction_view, label):
            if key and images.get(key):
                return Path(str(images[key]))
    return None


def image_size_from_file(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except (FileNotFoundError, OSError, ValueError):
        return None, None


def image_size_from_intrinsics(item: dict[str, Any]) -> tuple[int | None, int | None]:
    answer = item.get("answer") or item.get("A") or {}
    if not isinstance(answer, dict):
        return None, None
    intrinsics = answer.get("camera_intrinsics") or {}
    if not isinstance(intrinsics, dict):
        return None, None
    try:
        return int(intrinsics.get("width")), int(intrinsics.get("height"))
    except (TypeError, ValueError):
        return None, None


def image_size_for_item(item: dict[str, Any]) -> tuple[int | None, int | None]:
    path = main_view_image_path_for_item(item)
    if path is not None:
        width, height = image_size_from_file(path)
        if width is not None and height is not None:
            return width, height
    return image_size_from_intrinsics(item)


def primary_image_key_for_item(item: dict[str, Any], images: dict[Any, Any]) -> Any | None:
    for key in ("prediction_view_label", "primary_view_label", "view_label"):
        if item.get(key) in images:
            return item[key]
    for key in ("prediction_view", "primary_view", "view"):
        value = item.get(key)
        if value in images:
            return value
        label = image_label_from_key(value) if value else ""
        if label in images:
            return label
    for key in ("prediction_image", "main_image", "image"):
        if item.get(key):
            target = str(item[key])
            for image_key, path in images.items():
                if str(path) == target:
                    return image_key
    return None


def image_inputs_for_item(item: dict[str, Any]) -> list[dict[str, str]]:
    """主视角在前、上下文视角在后。视角名从数据里读，**不硬编码** ——
    生成器已改为让视角跟随 config 的 views，旧数据是
    left_eye/right_eye/left_wrist，新数据是 left_eye/left_wrist/right_wrist。"""
    images = item.get("images")
    if isinstance(images, dict) and images:
        primary_key = primary_image_key_for_item(item, images)
        ordered: list[Any] = []
        if primary_key is not None:
            ordered.append(primary_key)
        ordered.extend(k for k, _ in sorted(images.items()) if k != primary_key)
        return [
            {
                "label": image_label_from_key(k),
                "path": str(images[k]),
                "role": "primary" if k == primary_key else "context",
            }
            for k in ordered
        ]
    if item.get("image"):
        return [{"label": "main_view", "path": str(item["image"]), "role": "primary"}]
    data = item.get("input", {})
    if isinstance(data, dict):
        if isinstance(data.get("image_paths"), list):
            return [
                {
                    "label": "main_view" if i == 0 else f"context_view_{i}",
                    "path": str(p),
                    "role": "primary" if i == 0 else "context",
                }
                for i, p in enumerate(data["image_paths"])
            ]
        if data.get("image_path"):
            return [{"label": "main_view", "path": str(data["image_path"]), "role": "primary"}]
    raise ValueError(f"Cannot find image paths for item {item.get('id')}")


# --------------------------------------------------------------------------
# prompt / 解析：逐字搬运
# --------------------------------------------------------------------------


def build_prompt(question: str, item: dict[str, Any]) -> str:
    dim = infer_dimension(item)
    expected = expected_trajectory(item)
    active_gripper = active_gripper_for_item(item, expected)
    if dim == 2:
        width, height = image_size_for_item(item)
        size_note = ""
        if width is not None and height is not None:
            size_note = (
                f" The main-view image size is {width} pixels wide and {height} pixels high; "
                f"the visible canvas spans 0 <= u < {width} and 0 <= v < {height}. "
                f"Any point with u < 0, u >= {width}, v < 0, or v >= {height} is invalid."
            )
        coordinate_note = (
            "Use image pixel coordinates [u, v] in the main-view image, where u increases right "
            f"and v increases down.{size_note} Coordinates must refer only to the first attached "
            "image / main view, not to a concatenated image, not to a resized image, and not to the "
            "side/wrist views. Do not use normalized coordinates."
        )
        example_point = "[123.4, 256.7]"
    else:
        coordinate_note = "Use 3D camera-frame coordinates in meters [x, y, z]."
        example_point = "[0.12, -0.03, 0.45]"

    scored = grippers_to_score(active_gripper)
    point_count = sum(len(expected[g]) for g in scored)
    if active_gripper in {"left_gripper", "right_gripper"}:
        gripper_instruction = (
            f"Predict ordered key trajectory points only for the active gripper: {active_gripper}."
        )
        count_instruction = f"Return approximately {point_count} {active_gripper} points if visible/available."
        schema_body = f'  "{active_gripper}": [{example_point}]'
    elif active_gripper == "both":
        gripper_instruction = "Both grippers are active. Predict ordered key trajectory points for both grippers."
        count_instruction = (
            f"Return approximately {len(expected['left_gripper'])} left-gripper points and "
            f"{len(expected['right_gripper'])} right-gripper points if visible/available."
        )
        schema_body = f'  "left_gripper": [{example_point}],\n  "right_gripper": [{example_point}]'
    else:
        gripper_instruction = (
            "The active gripper could not be determined from the reference metadata. "
            "Predict the gripper trajectory that is visibly performing the action."
        )
        count_instruction = "Return only the gripper trajectory that is performing the action."
        schema_body = f'  "left_gripper": [{example_point}]'

    return f"""You are evaluating a robot manipulation trajectory prediction task.

Question:
{question}

{gripper_instruction}
{coordinate_note}
{count_instruction}

Output JSON only. Do not use Markdown.
Required schema:
{{
{schema_body}
}}
"""


def coerce_point_list(value: Any, dim: int) -> list[list[float]]:
    if isinstance(value, dict):
        for key in ("points", "trajectory", "coordinates"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list):
        return []
    if len(value) >= dim and all(isinstance(value[i], (int, float)) for i in range(dim)):
        try:
            return [[float(value[i]) for i in range(dim)]]
        except (TypeError, ValueError):
            return []
    points: list[list[float]] = []
    for row in value:
        if isinstance(row, dict):
            values = [row.get("u"), row.get("v")] if dim == 2 else [row.get("x"), row.get("y"), row.get("z")]
        else:
            values = row
        if not isinstance(values, list) or len(values) < dim:
            continue
        try:
            points.append([float(values[i]) for i in range(dim)])
        except (TypeError, ValueError):
            continue
    return points


def parse_model_answer(text: str, dim: int) -> dict[str, Any]:
    data = first_json_object(text)
    left = (
        data.get("left_gripper") or data.get("left")
        or data.get("left_gripper_points") or data.get("left_trajectory") or []
    )
    right = (
        data.get("right_gripper") or data.get("right")
        or data.get("right_gripper_points") or data.get("right_trajectory") or []
    )
    return {
        "parsed": data,
        "trajectory": {
            "left_gripper": coerce_point_list(left, dim),
            "right_gripper": coerce_point_list(right, dim),
        },
    }


def out_of_bounds_points(
    prediction: dict[str, Any], width: int | None, height: int | None, grippers: list[str] | None = None
) -> list[str]:
    if width is None or height is None:
        return []
    trajectory = prediction.get("trajectory", {})
    if not isinstance(trajectory, dict):
        return []
    invalid: list[str] = []
    for gripper in grippers or ["left_gripper", "right_gripper"]:
        points = trajectory.get(gripper, [])
        if not isinstance(points, list):
            continue
        for index, point in enumerate(points):
            if not isinstance(point, list) or len(point) < 2:
                continue
            u, v = point[0], point[1]
            if u < 0 or u >= width or v < 0 or v >= height:
                invalid.append(f"{gripper}[{index}]=[{round(float(u), 3)}, {round(float(v), 3)}]")
    return invalid


def build_retry_prompt(
    original_prompt: str, model_text: str, invalid_points: list[str], width: int, height: int
) -> str:
    examples = ", ".join(invalid_points[:12])
    return f"""{original_prompt}

Your previous JSON answer used invalid 2D pixel coordinates outside the main-view image.
Main-view image size: width={width}, height={height}.
Valid coordinate range: 0 <= u < {width} and 0 <= v < {height}.
Invalid points found: {examples}

Previous answer:
{model_text}

Return a corrected JSON answer only. Every [u, v] point must be inside the valid coordinate range above.
"""


# --------------------------------------------------------------------------
# 打分：逐字搬运
# --------------------------------------------------------------------------


def score_tolerance_for_item(item: dict[str, Any], expected: dict[str, list[list[float]]]) -> float:
    if infer_dimension(item) == 2:
        width, height = image_size_for_item(item)
        if width is not None and height is not None:
            return 0.05 * math.hypot(width, height)
        extent = point_cloud_extent(flatten_points(expected))
        return max(10.0, 0.1 * extent) if extent is not None else 50.0
    extent = point_cloud_extent(flatten_points(expected))
    return max(0.02, 0.1 * extent) if extent is not None else 0.05


def distance_to_score(distance: float | None, tolerance: float) -> float | None:
    if distance is None or tolerance <= 0:
        return None
    ratio = float(distance) / tolerance
    return 100.0 / (1.0 + ratio * ratio)


def score_curve(expected: list[list[float]], predicted: list[list[float]], tolerance: float) -> dict[str, Any]:
    hausdorff = hausdorff_distance(expected, predicted)
    frechet = discrete_frechet_distance(expected, predicted)
    chamfer = chamfer_distance(expected, predicted)
    metric_scores = {
        "hausdorff": rounded(distance_to_score(hausdorff, tolerance)),
        "discrete_frechet": rounded(distance_to_score(frechet, tolerance)),
        "chamfer": rounded(distance_to_score(chamfer, tolerance)),
    }
    return {
        "expected_points": len(expected),
        "predicted_points": len(predicted),
        "hausdorff": rounded(hausdorff),
        "discrete_frechet": rounded(frechet),
        "chamfer": rounded(chamfer),
        "metric_scores": metric_scores,
        "score": rounded(mean_available(list(metric_scores.values()))),
    }


def score_prediction(item: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected = expected_trajectory(item)
    active_gripper = active_gripper_for_item(item, expected)
    scored_grippers = grippers_to_score(active_gripper)
    expected_scored = {g: expected.get(g, []) for g in scored_grippers}
    predicted = prediction.get("trajectory", {})
    if not isinstance(predicted, dict):
        predicted = {}
    tolerance = score_tolerance_for_item(item, expected_scored)
    gripper_metrics = {
        g: score_curve(expected.get(g, []), predicted.get(g, []), tolerance) for g in scored_grippers
    }
    width, height = image_size_for_item(item) if infer_dimension(item) == 2 else (None, None)
    invalid_points = out_of_bounds_points(prediction, width, height, scored_grippers)

    mean_metric_scores = {
        key: rounded(mean_available([row["metric_scores"][key] for row in gripper_metrics.values()]))
        for key in ("hausdorff", "discrete_frechet", "chamfer")
    }
    mean_metrics = {
        key: rounded(mean_available([row[key] for row in gripper_metrics.values()]))
        for key in ("hausdorff", "discrete_frechet", "chamfer")
    }

    return {
        "dimension": infer_dimension(item),
        "active_gripper": active_gripper,
        "scored_grippers": scored_grippers,
        "image_size": {"width": width, "height": height} if width is not None and height is not None else None,
        "score_mapping": {
            "function": "100 / (1 + (distance / tolerance)^2)",
            "tolerance": rounded(tolerance),
            "tolerance_rule": "2D uses 5% of image diagonal; 3D uses max(0.02 m, 10% of GT trajectory extent).",
        },
        "out_of_bounds": {"count": len(invalid_points), "examples": invalid_points[:12]},
        "expected_trajectory": expected,
        "expected_scored_trajectory": expected_scored,
        "predicted_trajectory": predicted,
        "predicted_scored_trajectory": {g: predicted.get(g, []) for g in scored_grippers},
        "gripper_metrics": gripper_metrics,
        "active_gripper_metrics": gripper_metrics.get(active_gripper),
        "mean_metrics": mean_metrics,
        "mean_metric_scores": mean_metric_scores,
        "score": rounded(mean_available(list(mean_metric_scores.values()))),
    }


# --------------------------------------------------------------------------
# 任务
# --------------------------------------------------------------------------


class TrajectoryTask:
    """2D 与 3D 共用同一实现，维度由 item 的 coordinate_frame 决定。"""

    def __init__(self, name: str = "trajectory", **_flags: Any) -> None:
        self.name = name

    def units(self, items: list[dict[str, Any]]) -> list[Unit]:
        return one_item_per_unit(items)

    def parts(self, unit: Unit) -> list[dict[str, Any]]:
        item = unit.items[0]
        prompt = build_prompt(str(item.get("Q") or item.get("question")), item)
        return self._parts_with_prompt(item, prompt)

    @staticmethod
    def _parts_with_prompt(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for image_input in image_inputs_for_item(item):
            if image_input.get("role") == "primary":
                parts.append(text_part("Primary view image:"))
            else:
                parts.append(text_part(f"Context view {image_input['label']}:"))
            parts.append(image_part(image_input["path"]))
        parts.append(text_part(prompt))
        return parts

    def retry_parts(self, unit: Unit, text: str, attempt: int) -> list[dict[str, Any]] | None:
        """2D 预测点越界时重问一次。返回 None 表示不需要重试。"""
        item = unit.items[0]
        if attempt >= MAX_COORDINATE_RETRIES or infer_dimension(item) != 2:
            return None
        width, height = image_size_for_item(item)
        if width is None or height is None:
            return None
        prediction = parse_model_answer(text, 2)
        scored = grippers_to_score(active_gripper_for_item(item))
        invalid = out_of_bounds_points(prediction, width, height, scored)
        if not invalid:
            return None
        prompt = build_prompt(str(item.get("Q") or item.get("question")), item)
        retry_prompt = build_retry_prompt(prompt, text, invalid, width, height)
        return self._parts_with_prompt(item, retry_prompt)

    def rows(self, unit: Unit, text: str, ctx: CallContext) -> list[dict[str, Any]]:
        item = unit.items[0]
        prompt = build_prompt(str(item.get("Q") or item.get("question")), item)
        prediction = parse_model_answer(text, infer_dimension(item))
        row = base_row(item, prompt, text, ctx)
        row["model_prediction"] = prediction.get("parsed")
        row.update(score_prediction(item, prediction))
        row["parse_ok"] = bool(prediction.get("parsed"))
        return [row]

    def error_rows(self, unit: Unit, error: str) -> list[dict[str, Any]]:
        item = unit.items[0]
        prompt = build_prompt(str(item.get("Q") or item.get("question")), item)
        row = base_row(item, prompt, None, None)
        row["model_prediction"] = None
        row.update(score_prediction(item, {"trajectory": {}}))
        row["error"] = error
        row["parse_ok"] = False
        return [row]

    def summarize(self, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        answered = [r for r in rows if r.get("model_output")]
        mean_metrics = [r.get("mean_metrics", {}) for r in answered]
        mean_metric_scores = [r.get("mean_metric_scores", {}) for r in answered]
        return {
            "total": len(rows),
            "answered": len(answered),
            "errors": sum(1 for r in rows if r.get("error")),
            "mean_hausdorff": rounded(mean_available([m.get("hausdorff") for m in mean_metrics])),
            "mean_discrete_frechet": rounded(mean_available([m.get("discrete_frechet") for m in mean_metrics])),
            "mean_chamfer": rounded(mean_available([m.get("chamfer") for m in mean_metrics])),
            "mean_hausdorff_score": rounded(mean_available([m.get("hausdorff") for m in mean_metric_scores])),
            "mean_discrete_frechet_score": rounded(
                mean_available([m.get("discrete_frechet") for m in mean_metric_scores])
            ),
            "mean_chamfer_score": rounded(mean_available([m.get("chamfer") for m in mean_metric_scores])),
            "mean_score": rounded(mean_available([r.get("score") for r in answered])),
            "elapsed_seconds": round(elapsed, 3),
            "parse_failure_rate": (
                sum(1 for r in rows if not r.get("parse_ok")) / len(rows) if rows else 0.0
            ),
        }


def build(name: str = "trajectory", **flags: Any) -> TrajectoryTask:
    return TrajectoryTask(name, **flags)
