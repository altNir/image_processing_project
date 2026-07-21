"""SegFormer inference and semantic-segmentation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..config import CITYSCAPES_CLASSES


def predict_segmentation(
    img_pil: Image.Image,
    processor: Any,
    model: Any,
    device: str,
    use_half: bool = False,
) -> np.ndarray:
    """Predict a Cityscapes train-ID mask with SegFormer."""

    import torch
    import torch.nn.functional as functional

    inputs = {name: tensor.to(device) for name, tensor in processor(
        images=img_pil, return_tensors="pt"
    ).items()}
    cuda_half = bool(use_half and device.startswith("cuda"))
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=cuda_half
    ):
        logits = model(**inputs).logits
    upsampled = functional.interpolate(
        logits, size=(img_pil.height, img_pil.width), mode="bilinear", align_corners=False
    )
    return upsampled.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.int32)


def compute_ious(pred_0idx: np.ndarray, gt_train_ids: np.ndarray) -> dict[int, float]:
    """Compute image-level IoU for each present semantic class."""

    prediction, ground_truth = np.asarray(pred_0idx), np.asarray(gt_train_ids)
    valid = (ground_truth >= 0) & (ground_truth < len(CITYSCAPES_CLASSES))
    ious: dict[int, float] = {}
    for class_id in range(len(CITYSCAPES_CLASSES)):
        predicted, actual = (prediction == class_id) & valid, (ground_truth == class_id) & valid
        union = int((predicted | actual).sum())
        if union:
            ious[class_id] = float((predicted & actual).sum() / union)
    return ious


class SegmentationAccumulator:
    """Accumulate a confusion matrix and derive split-level metrics."""

    def __init__(self, num_classes: int = len(CITYSCAPES_CLASSES)) -> None:
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction: np.ndarray, ground_truth: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.int64)
        ground_truth = np.asarray(ground_truth, dtype=np.int64)
        valid = (
            (ground_truth >= 0) & (ground_truth < self.num_classes)
            & (prediction >= 0) & (prediction < self.num_classes)
        )
        encoded = self.num_classes * ground_truth[valid] + prediction[valid]
        self.confusion += np.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def results(self) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
        matrix = self.confusion.astype(np.float64)
        true_positive = np.diag(matrix)
        gt_count, pred_count = matrix.sum(axis=1), matrix.sum(axis=0)
        union = gt_count + pred_count - true_positive
        iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
        accuracy = np.divide(
            true_positive, gt_count, out=np.full_like(true_positive, np.nan), where=gt_count > 0
        )
        total = float(matrix.sum())
        summary = {
            "mean_iou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0,
            "pixel_accuracy": float(true_positive.sum() / total) if total else 0.0,
            "mean_class_accuracy": float(np.nanmean(accuracy)) if np.any(~np.isnan(accuracy)) else 0.0,
        }
        rows = [
            {
                "class_id": class_id,
                "class_name": name,
                "iou": float(iou[class_id]) if not np.isnan(iou[class_id]) else float("nan"),
                "class_accuracy": float(accuracy[class_id]) if not np.isnan(accuracy[class_id]) else float("nan"),
                "gt_pixels": int(gt_count[class_id]),
                "pred_pixels": int(pred_count[class_id]),
            }
            for class_id, name in enumerate(CITYSCAPES_CLASSES)
        ]
        return summary, rows
