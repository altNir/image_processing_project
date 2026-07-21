"""YOLO inference, box conversion, and object-detection metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ..config import COCO_ID_TO_SHARED_CLASS, SHARED_DETECTION_CLASSES
from ..types import Detection
from ..utils.dependencies import cv2_module


DETECTION_EVALUATOR_VERSION = 2


def _predict_kwargs(device: str | None, use_half: bool) -> dict[str, Any]:
    return {
        "device": device,
        "quantize": 16 if use_half and device and device.startswith("cuda") else None,
    }


def yolo_overlay(
    img_pil: Image.Image,
    model: Any,
    conf: float = 0.25,
    device: str | None = None,
    use_half: bool = False,
) -> tuple[np.ndarray, Any]:
    """Run YOLO and return its RGB visualization and raw result."""

    result = model.predict(
        img_pil, conf=conf, verbose=False, **_predict_kwargs(device, use_half)
    )[0]
    return cv2_module().cvtColor(result.plot(), cv2_module().COLOR_BGR2RGB), result


def model_detections(
    image: Image.Image,
    model: Any,
    image_id: str,
    class_mapping: Mapping[int, str],
    confidence: float,
    device: str,
    use_half: bool,
) -> list[Detection]:
    """Convert one model result to project Detection records."""

    result = model.predict(
        image,
        conf=confidence,
        max_det=300,
        verbose=False,
        **_predict_kwargs(device, use_half),
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    return [
        Detection(image_id, class_mapping[int(class_id)], tuple(float(v) for v in bbox), float(score))
        for bbox, score, class_id in zip(boxes, scores, class_ids)
        if int(class_id) in class_mapping
    ]


def yolo_detections(
    img_pil: Image.Image,
    model: Any,
    image_id: str,
    conf: float = 0.001,
    device: str | None = None,
    use_half: bool = False,
) -> list[Detection]:
    """Run a COCO YOLO checkpoint and retain Cityscapes-shared classes."""

    return model_detections(
        img_pil, model, image_id, COCO_ID_TO_SHARED_CLASS, conf, device or "cpu", use_half
    )


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Compute intersection over union for two xyxy boxes."""

    x1, y1 = max(float(first[0]), float(second[0])), max(float(first[1]), float(second[1]))
    x2, y2 = min(float(first[2]), float(second[2])), min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    return float(np.mean([
        float(candidates.max()) if (candidates := precision[recall >= level]).size else 0.0
        for level in np.linspace(0.0, 1.0, 101)
    ]))


def _evaluate_detection_class(
    predictions: Sequence[Detection], ground_truth: Sequence[Detection], iou_threshold: float
) -> dict[str, float]:
    gt_by_image: dict[str, list[Detection]] = defaultdict(list)
    for item in ground_truth:
        gt_by_image[item.image_id].append(item)
    used = {key: np.zeros(len(items), dtype=bool) for key, items in gt_by_image.items()}
    ordered = sorted(predictions, key=lambda item: item.score, reverse=True)
    tp, fp, matched = np.zeros(len(ordered)), np.zeros(len(ordered)), []
    for index, prediction in enumerate(ordered):
        candidates = gt_by_image.get(prediction.image_id, [])
        if not candidates:
            fp[index] = 1.0
            continue
        overlaps = np.asarray([bbox_iou(prediction.bbox, item.bbox) for item in candidates])
        # Match against the best *unused* ground truth. Selecting the global best
        # first can incorrectly reject a valid match when that object was already
        # claimed but another unused object still exceeds the IoU threshold.
        available = ~used[prediction.image_id]
        eligible_overlaps = np.where(available, overlaps, -1.0)
        best_index = int(eligible_overlaps.argmax()) if overlaps.size and np.any(available) else -1
        best_iou = float(overlaps[best_index]) if best_index >= 0 else 0.0
        if best_iou >= iou_threshold:
            tp[index], used[prediction.image_id][best_index] = 1.0, True
            matched.append(best_iou)
        else:
            fp[index] = 1.0
    gt_count = len(ground_truth)
    cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
    recall = cumulative_tp / gt_count if gt_count else np.zeros_like(cumulative_tp)
    precision = np.divide(
        cumulative_tp, cumulative_tp + cumulative_fp,
        out=np.zeros_like(cumulative_tp), where=(cumulative_tp + cumulative_fp) > 0,
    )
    return {
        "ap": _interpolated_ap(recall, precision) if gt_count else float("nan"),
        "precision": float(precision[-1]) if precision.size else 0.0,
        "recall": float(recall[-1]) if recall.size else 0.0,
        "mean_matched_iou": float(np.mean(matched)) if matched else 0.0,
        "gt_count": float(gt_count), "prediction_count": float(len(predictions)),
    }


def evaluate_detections(
    predictions: Sequence[Detection],
    ground_truth: Sequence[Detection],
    classes: Sequence[str] = SHARED_DETECTION_CLASSES,
    operating_confidence: float = 0.25,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    """Compute AP from all scores plus practical metrics at one confidence threshold."""

    if not 0.0 <= operating_confidence <= 1.0:
        raise ValueError("operating_confidence must be between 0 and 1")

    thresholds = np.arange(0.50, 0.96, 0.05)
    rows: list[dict[str, float | str]] = []
    for class_name in classes:
        class_predictions = [item for item in predictions if item.class_name == class_name]
        class_ground_truth = [item for item in ground_truth if item.class_name == class_name]
        evaluations = [
            _evaluate_detection_class(
                class_predictions,
                class_ground_truth,
                float(threshold),
            )
            for threshold in thresholds
        ]
        aps = np.asarray([item["ap"] for item in evaluations])
        at_50 = evaluations[0]
        operating = _evaluate_detection_class(
            [item for item in class_predictions if item.score >= operating_confidence],
            class_ground_truth,
            0.50,
        )
        operating_precision = float(operating["precision"])
        operating_recall = float(operating["recall"])
        operating_f1 = (
            2.0 * operating_precision * operating_recall
            / (operating_precision + operating_recall)
            if operating_precision + operating_recall
            else 0.0
        )
        rows.append({
            "class_name": class_name,
            "map_50_95": float(np.nanmean(aps)) if np.any(~np.isnan(aps)) else float("nan"),
            "ap_50": float(at_50["ap"]), "precision_50": float(at_50["precision"]),
            "recall_50": float(at_50["recall"]),
            "mean_matched_iou_50": float(at_50["mean_matched_iou"]),
            "operating_confidence": float(operating_confidence),
            "precision_50_at_operating_confidence": operating_precision,
            "recall_50_at_operating_confidence": operating_recall,
            "f1_50_at_operating_confidence": operating_f1,
            "gt_count": float(at_50["gt_count"]),
            "prediction_count": float(at_50["prediction_count"]),
            "operating_prediction_count": float(operating["prediction_count"]),
        })
    valid = [row for row in rows if not math.isnan(float(row["map_50_95"]))]
    mean = lambda key: float(np.mean([float(row[key]) for row in valid])) if valid else 0.0
    return {
        "map_50_95": mean("map_50_95"), "map_50": mean("ap_50"),
        "mean_precision_50": mean("precision_50"), "mean_recall_50": mean("recall_50"),
        "mean_matched_iou_50": mean("mean_matched_iou_50"),
        "operating_confidence": float(operating_confidence),
        "mean_precision_50_at_operating_confidence": mean(
            "precision_50_at_operating_confidence"
        ),
        "mean_recall_50_at_operating_confidence": mean(
            "recall_50_at_operating_confidence"
        ),
        "mean_f1_50_at_operating_confidence": mean("f1_50_at_operating_confidence"),
        "ground_truth_objects": float(len(ground_truth)),
        "predicted_objects": float(len(predictions)), "evaluated_classes": float(len(valid)),
        "operating_predicted_objects": float(
            sum(1 for item in predictions if item.score >= operating_confidence)
        ),
    }, rows
