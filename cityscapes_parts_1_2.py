"""Cityscapes robustness project - Parts 1 and 2 only.

The file intentionally follows the order and naming used in the course slides:

Part 1
    1. Prepare the dataset and plot labels.
    2. Run ORB feature detection.
    3. Run Canny edge detection.
    4. Run pretrained YOLO object detection.
    5. Run pretrained SegFormer semantic segmentation.
    6. Compute clean-image metrics.

Part 2
    1. Introduce Gaussian noise, JPEG compression, low light, and motion blur.
    2. Run all four methods on distorted images.
    3. Measure ORB matching, segmentation IoU, and detection AP.
    4. Compute SNR and plot performance per SNR.

There is deliberately no restoration, enhancement, or fine-tuning code here.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


LOGGER = logging.getLogger("cityscapes_parts_1_2")


# Cityscapes train IDs and colors used by its official evaluation protocol.
CITYSCAPES_CLASSES: tuple[str, ...] = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

CITYSCAPES_PALETTE = np.asarray(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)

# Mapping from the raw IDs shipped in gtFine_trainvaltest.zip to the 19
# contiguous train IDs used by Cityscapes models and metrics. All ignored/void
# raw labels remain 255. Some third-party preparations already provide
# *_labelTrainIds.png; the loader accepts both forms.
CITYSCAPES_RAW_ID_TO_TRAIN_ID = np.full(256, 255, dtype=np.uint8)
CITYSCAPES_RAW_ID_TO_TRAIN_ID[
    np.asarray(
        [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    )
] = np.arange(19, dtype=np.uint8)

# Instance IDs in *_gtFine_instanceIds.png use: semantic_label_id * 1000 + instance_number.
CITYSCAPES_INSTANCE_LABELS: Mapping[int, str] = {
    24: "person",
    25: "rider",
    26: "car",
    27: "truck",
    28: "bus",
    31: "train",
    32: "motorcycle",
    33: "bicycle",
}

# COCO has no direct "rider" class, so rider is intentionally excluded.
COCO_ID_TO_SHARED_CLASS: Mapping[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
}
SHARED_DETECTION_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
)

DEFAULT_DISTORTION_LEVELS: Mapping[str, tuple[float, ...]] = {
    # Values are pixel-domain standard deviations on an 8-bit image.
    "GaussNoise": (5.0, 10.0, 20.0, 35.0, 50.0),
    # Values are JPEG quality settings. Lower means more severe compression.
    "SevereJPEG": (80.0, 60.0, 40.0, 20.0, 5.0),
    # Values multiply RGB intensity. Lower means darker.
    "LowLight": (0.80, 0.60, 0.40, 0.25, 0.10),
    # Values are odd motion-kernel lengths in pixels. The direction is fixed so
    # kernel length, rather than direction, is the controlled variable.
    "MotionBlur": (3.0, 5.0, 9.0, 15.0, 25.0),
}


@dataclass(frozen=True)
class CityscapesSample:
    sample_id: str
    image_path: Path
    label_path: Path
    instance_path: Path


@dataclass
class Detection:
    image_id: str
    class_name: str
    bbox: tuple[float, float, float, float]
    score: float = 1.0


@dataclass
class ExperimentConfig:
    dataset_root: Path
    output_dir: Path = Path("outputs")
    split: str = "val"
    max_samples: int = 0
    seed: int = 7
    device: str = "auto"
    nfeatures: int = 800
    orb_ratio_threshold: float = 0.75
    orb_spatial_threshold: float = 3.0
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    canny_blur_kernel: int = 5
    canny_tolerance_radius: int = 2
    yolo_model: str = "yolov8n.pt"
    yolo_eval_confidence: float = 0.001
    yolo_visual_confidence: float = 0.25
    segformer_model: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    distortion_levels: Mapping[str, tuple[float, ...]] | None = None
    gallery_samples: int = 4

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root)
        self.output_dir = Path(self.output_dir)
        if self.distortion_levels is None:
            self.distortion_levels = dict(DEFAULT_DISTORTION_LEVELS)


# ---------------------------------------------------------------------------
# 1. Prepare Dataset and a sample (slides 5-7)
# ---------------------------------------------------------------------------


def discover_cityscapes_samples(
    root: Path | str,
    split: str = "val",
    max_samples: int = 0,
    seed: int = 7,
) -> list[CityscapesSample]:
    """Discover official Cityscapes fine-annotation files.

    Expected layout:
        root/leftImg8bit/{split}/{city}/*_leftImg8bit.png
        root/gtFine/{split}/{city}/*_gtFine_labelIds.png
        root/gtFine/{split}/{city}/*_gtFine_instanceIds.png

    Pre-generated *_labelTrainIds.png files are also accepted and preferred.
    """

    root = Path(root)
    image_root = root / "leftImg8bit" / split
    gt_root = root / "gtFine" / split
    if not image_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError(
            "Cityscapes was not found in the expected official layout.\n"
            f"Expected: {image_root}\n"
            f"      and: {gt_root}\n"
            "Download leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip "
            "from cityscapes-dataset.com and extract both under the same root."
        )

    samples: list[CityscapesSample] = []
    missing: list[Path] = []
    for image_path in sorted(image_root.glob("*/*_leftImg8bit.png")):
        city = image_path.parent.name
        base = image_path.name.removesuffix("_leftImg8bit.png")
        label_path = gt_root / city / f"{base}_gtFine_labelTrainIds.png"
        if not label_path.is_file():
            label_path = gt_root / city / f"{base}_gtFine_labelIds.png"
        instance_path = gt_root / city / f"{base}_gtFine_instanceIds.png"
        if not label_path.is_file():
            missing.append(label_path)
            continue
        if not instance_path.is_file():
            missing.append(instance_path)
            continue
        samples.append(
            CityscapesSample(
                sample_id=base,
                image_path=image_path,
                label_path=label_path,
                instance_path=instance_path,
            )
        )

    if missing:
        preview = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"Found images but {len(missing)} required annotation files are missing.\n{preview}"
        )
    if not samples:
        raise FileNotFoundError(f"No Cityscapes {split!r} images found under {image_root}")

    if max_samples > 0 and max_samples < len(samples):
        rng = random.Random(seed)
        samples = sorted(rng.sample(samples, max_samples), key=lambda sample: sample.sample_id)
    return samples


def load_sample(sample: CityscapesSample) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    image = Image.open(sample.image_path).convert("RGB")
    label = np.asarray(Image.open(sample.label_path), dtype=np.uint8)
    if sample.label_path.name.endswith("_gtFine_labelIds.png"):
        label = raw_label_ids_to_train_ids(label)
    instance = np.asarray(Image.open(sample.instance_path), dtype=np.int32)
    if label.shape != (image.height, image.width):
        raise ValueError(f"Label shape {label.shape} does not match {image.size} for {sample.sample_id}")
    if instance.shape != label.shape:
        raise ValueError(f"Instance shape {instance.shape} does not match labels for {sample.sample_id}")
    return image, label, instance


def load_image_and_label(sample: CityscapesSample) -> tuple[Image.Image, np.ndarray]:
    """Load only what Part 2 needs; instance-derived boxes are cached from Part 1."""

    image = Image.open(sample.image_path).convert("RGB")
    label = np.asarray(Image.open(sample.label_path), dtype=np.uint8)
    if sample.label_path.name.endswith("_gtFine_labelIds.png"):
        label = raw_label_ids_to_train_ids(label)
    if label.shape != (image.height, image.width):
        raise ValueError(f"Label shape {label.shape} does not match {image.size} for {sample.sample_id}")
    return image, label


def raw_label_ids_to_train_ids(raw_label: np.ndarray) -> np.ndarray:
    """Convert official Cityscapes raw label IDs into 0..18 train IDs."""

    raw = np.asarray(raw_label)
    output = np.full(raw.shape, 255, dtype=np.uint8)
    valid = (raw >= 0) & (raw < len(CITYSCAPES_RAW_ID_TO_TRAIN_ID))
    output[valid] = CITYSCAPES_RAW_ID_TO_TRAIN_ID[raw[valid].astype(np.int64)]
    return output


def colorize(mask_idx: np.ndarray) -> np.ndarray:
    """Colorize a Cityscapes train-ID mask; void pixels are black."""

    mask = np.asarray(mask_idx)
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(CITYSCAPES_CLASSES))
    output[valid] = CITYSCAPES_PALETTE[mask[valid].astype(np.int64)]
    return output


def overlay_mask(
    img_pil: Image.Image,
    mask: Image.Image | np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a Cityscapes semantic mask, mirroring the slide helper."""

    img = np.asarray(img_pil.convert("RGB"), dtype=np.float32)
    mask_array = np.asarray(mask, dtype=np.int32)
    color = colorize(mask_array).astype(np.float32)
    valid = ((mask_array >= 0) & (mask_array < len(CITYSCAPES_CLASSES)))[..., None]
    blended = img * (1.0 - alpha) + color * alpha
    output = np.where(valid, blended, img).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output)


def instance_mask_to_boxes(instance_mask: np.ndarray, image_id: str) -> list[Detection]:
    """Derive 2D boxes from Cityscapes instance IDs for COCO-shared classes."""

    detections: list[Detection] = []
    shared_names = set(SHARED_DETECTION_CLASSES)
    for instance_id in np.unique(instance_mask):
        instance_id_int = int(instance_id)
        if instance_id_int < 1000:
            continue
        label_id = instance_id_int // 1000
        class_name = CITYSCAPES_INSTANCE_LABELS.get(label_id)
        if class_name not in shared_names:
            continue
        ys, xs = np.where(instance_mask == instance_id_int)
        if xs.size == 0:
            continue
        # xyxy with an exclusive maximum, matching common detector conventions.
        bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        detections.append(Detection(image_id=image_id, class_name=class_name, bbox=bbox))
    return detections


# ---------------------------------------------------------------------------
# Run ORB feature detector (slides 8-9)
# ---------------------------------------------------------------------------


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("OpenCV is required. Install requirements.txt first.") from exc
    return cv2


def orb_detect(
    img_pil: Image.Image,
    nfeatures: int = 800,
) -> tuple[list[Any], np.ndarray | None]:
    cv2 = _cv2()
    img = np.asarray(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    return orb.detectAndCompute(gray, None)


def orb_overlay(
    img_pil: Image.Image,
    nfeatures: int = 800,
) -> tuple[np.ndarray, list[Any], np.ndarray | None]:
    """Draw ORB keypoints in the same style as the slide example."""

    cv2 = _cv2()
    img = np.asarray(img_pil.convert("RGB"))
    keypoints, descriptors = orb_detect(img_pil, nfeatures=nfeatures)
    output = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return output, keypoints, descriptors


def measure_orb_matching(
    clean_image: Image.Image,
    test_image: Image.Image,
    nfeatures: int = 800,
    ratio_threshold: float = 0.75,
    spatial_threshold: float = 3.0,
) -> dict[str, float]:
    """Measure ORB retention between geometry-aligned clean and test images.

    The distortions do not move pixels, so descriptor matches are also checked for
    spatial agreement. This prevents an accidental descriptor match elsewhere in
    the image from being counted as a retained feature.
    """

    cv2 = _cv2()
    clean_kp, clean_desc = orb_detect(clean_image, nfeatures=nfeatures)
    test_kp, test_desc = orb_detect(test_image, nfeatures=nfeatures)
    clean_count = len(clean_kp)
    test_count = len(test_kp)
    if clean_desc is None or test_desc is None or clean_count == 0:
        return {
            "clean_keypoints": float(clean_count),
            "test_keypoints": float(test_count),
            "keypoint_retention": 0.0,
            "ratio_matches": 0.0,
            "spatial_inliers": 0.0,
            "match_retention": 0.0,
            "inlier_ratio": 0.0,
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(clean_desc, test_desc, k=2)
    good_matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio_threshold * second.distance
    ]
    inliers = []
    for match in good_matches:
        clean_xy = np.asarray(clean_kp[match.queryIdx].pt)
        test_xy = np.asarray(test_kp[match.trainIdx].pt)
        if float(np.linalg.norm(clean_xy - test_xy)) <= spatial_threshold:
            inliers.append(match)

    return {
        "clean_keypoints": float(clean_count),
        "test_keypoints": float(test_count),
        "keypoint_retention": float(test_count / clean_count),
        "ratio_matches": float(len(good_matches)),
        "spatial_inliers": float(len(inliers)),
        "match_retention": float(len(inliers) / clean_count),
        "inlier_ratio": float(len(inliers) / len(good_matches)) if good_matches else 0.0,
    }


# ---------------------------------------------------------------------------
# Run Canny edge detector (edge-detection lectures).
# ---------------------------------------------------------------------------


def canny_detect(
    img_pil: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
    blur_kernel: int = 5,
) -> np.ndarray:
    """Return a binary Canny edge map using fixed thresholds for every condition.

    A small Gaussian pre-filter follows the lecture pipeline and prevents noise
    from dominating the gradient calculation. Fixed parameters are essential:
    retuning Canny for every distortion would conceal robustness degradation.
    """

    if low_threshold < 0 or high_threshold <= low_threshold:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high")
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError("Canny blur_kernel must be a positive odd integer")
    cv2 = _cv2()
    rgb = np.asarray(img_pil.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    return cv2.Canny(gray, low_threshold, high_threshold, L2gradient=True)


def canny_overlay(img_pil: Image.Image, edges: np.ndarray) -> np.ndarray:
    """Draw Canny edges in green over a dimmed RGB image."""

    rgb = np.asarray(img_pil.convert("RGB"), dtype=np.uint8)
    edge_mask = np.asarray(edges) > 0
    output = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
    output[edge_mask] = np.asarray((0, 255, 0), dtype=np.uint8)
    return output


def evaluate_canny_edges(
    reference_edges: np.ndarray,
    test_edges: np.ndarray,
    tolerance_radius: int = 2,
) -> dict[str, float]:
    """Measure edge consistency with spatially tolerant precision/recall/F1.

    Cityscapes has no edge annotations, so clean-image Canny output is the
    reference for later distorted-image comparisons. Dilation permits small
    localization shifts without allowing unrelated edges to match.
    """

    if tolerance_radius < 0:
        raise ValueError("tolerance_radius must be non-negative")
    reference = (np.asarray(reference_edges) > 0).astype(np.uint8)
    test = (np.asarray(test_edges) > 0).astype(np.uint8)
    if reference.shape != test.shape:
        raise ValueError("Reference and test edge maps must have the same shape")

    cv2 = _cv2()
    if tolerance_radius:
        size = 2 * tolerance_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        reference_neighborhood = cv2.dilate(reference, kernel)
        test_neighborhood = cv2.dilate(test, kernel)
    else:
        reference_neighborhood = reference
        test_neighborhood = test

    reference_count = int(reference.sum())
    test_count = int(test.sum())
    matched_test = int(((test > 0) & (reference_neighborhood > 0)).sum())
    matched_reference = int(((reference > 0) & (test_neighborhood > 0)).sum())
    precision = matched_test / test_count if test_count else (1.0 if reference_count == 0 else 0.0)
    recall = (
        matched_reference / reference_count
        if reference_count
        else (1.0 if test_count == 0 else 0.0)
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_edge_pixels": float(reference_count),
        "test_edge_pixels": float(test_count),
        "edge_pixel_retention": (
            float(test_count / reference_count)
            if reference_count
            else (1.0 if test_count == 0 else 0.0)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ---------------------------------------------------------------------------
# Run pretrained object detection (slides 10-11)
# ---------------------------------------------------------------------------


def yolo_overlay(
    img_pil: Image.Image,
    model: Any,
    conf: float = 0.25,
) -> tuple[np.ndarray, Any]:
    """Run and draw YOLO predictions, following the slide function."""

    cv2 = _cv2()
    result = model.predict(img_pil, conf=conf, verbose=False)[0]
    plotted_bgr = result.plot()
    return cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB), result


def yolo_detections(
    img_pil: Image.Image,
    model: Any,
    image_id: str,
    conf: float = 0.001,
) -> list[Detection]:
    result = model.predict(img_pil, conf=conf, max_det=300, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    predictions: list[Detection] = []
    for bbox, score, class_id in zip(boxes, scores, class_ids):
        class_name = COCO_ID_TO_SHARED_CLASS.get(int(class_id))
        if class_name is None:
            continue
        predictions.append(
            Detection(
                image_id=image_id,
                class_name=class_name,
                bbox=tuple(float(value) for value in bbox),
                score=float(score),
            )
        )
    return predictions


# ---------------------------------------------------------------------------
# Run pretrained segmentation and compute IoU (slides 12-16)
# ---------------------------------------------------------------------------


def predict_segmentation(
    img_pil: Image.Image,
    processor: Any,
    model: Any,
    device: str,
) -> np.ndarray:
    """Predict a 0..18 Cityscapes train-ID mask with SegFormer."""

    import torch
    import torch.nn.functional as functional

    inputs = processor(images=img_pil, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    upsampled = functional.interpolate(
        logits,
        size=(img_pil.height, img_pil.width),
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.int32)


def seg_overlay(img_rgb: np.ndarray, mask_idx: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    color = colorize(mask_idx).astype(np.float32)
    image = np.asarray(img_rgb, dtype=np.float32)
    return (image * (1.0 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)


def compute_ious(pred_0idx: np.ndarray, gt_train_ids: np.ndarray) -> dict[int, float]:
    """Compute per-class IoU for the 0..18 Cityscapes training IDs."""

    prediction = np.asarray(pred_0idx)
    ground_truth = np.asarray(gt_train_ids)
    valid = (ground_truth >= 0) & (ground_truth < len(CITYSCAPES_CLASSES))
    ious: dict[int, float] = {}
    for class_id in range(len(CITYSCAPES_CLASSES)):
        predicted = (prediction == class_id) & valid
        actual = (ground_truth == class_id) & valid
        union = int((predicted | actual).sum())
        if union:
            ious[class_id] = float((predicted & actual).sum() / union)
    return ious


class SegmentationAccumulator:
    def __init__(self, num_classes: int = len(CITYSCAPES_CLASSES)) -> None:
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction: np.ndarray, ground_truth: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.int64)
        ground_truth = np.asarray(ground_truth, dtype=np.int64)
        valid = (
            (ground_truth >= 0)
            & (ground_truth < self.num_classes)
            & (prediction >= 0)
            & (prediction < self.num_classes)
        )
        encoded = self.num_classes * ground_truth[valid] + prediction[valid]
        counts = np.bincount(encoded, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def results(self) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
        matrix = self.confusion.astype(np.float64)
        true_positive = np.diag(matrix)
        gt_count = matrix.sum(axis=1)
        pred_count = matrix.sum(axis=0)
        union = gt_count + pred_count - true_positive
        iou = np.divide(
            true_positive,
            union,
            out=np.full_like(true_positive, np.nan),
            where=union > 0,
        )
        class_accuracy = np.divide(
            true_positive,
            gt_count,
            out=np.full_like(true_positive, np.nan),
            where=gt_count > 0,
        )
        total = float(matrix.sum())
        summary = {
            "mean_iou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0,
            "pixel_accuracy": float(true_positive.sum() / total) if total else 0.0,
            "mean_class_accuracy": (
                float(np.nanmean(class_accuracy)) if np.any(~np.isnan(class_accuracy)) else 0.0
            ),
        }
        per_class: list[dict[str, float | int | str]] = []
        for class_id, class_name in enumerate(CITYSCAPES_CLASSES):
            per_class.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "iou": float(iou[class_id]) if not np.isnan(iou[class_id]) else float("nan"),
                    "class_accuracy": (
                        float(class_accuracy[class_id])
                        if not np.isnan(class_accuracy[class_id])
                        else float("nan")
                    ),
                    "gt_pixels": int(gt_count[class_id]),
                    "pred_pixels": int(pred_count[class_id]),
                }
            )
        return summary, per_class


# ---------------------------------------------------------------------------
# Detection metrics: real Cityscapes instance-derived ground truth.
# ---------------------------------------------------------------------------


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP for one class/IoU threshold."""

    if recall.size == 0:
        return 0.0
    levels = np.linspace(0.0, 1.0, 101)
    values = []
    for level in levels:
        candidates = precision[recall >= level]
        values.append(float(candidates.max()) if candidates.size else 0.0)
    return float(np.mean(values))


def _evaluate_detection_class(
    predictions: Sequence[Detection],
    ground_truth: Sequence[Detection],
    iou_threshold: float,
) -> dict[str, float]:
    gt_by_image: dict[str, list[Detection]] = defaultdict(list)
    for item in ground_truth:
        gt_by_image[item.image_id].append(item)
    used = {image_id: np.zeros(len(items), dtype=bool) for image_id, items in gt_by_image.items()}
    sorted_predictions = sorted(predictions, key=lambda item: item.score, reverse=True)
    true_positive = np.zeros(len(sorted_predictions), dtype=np.float64)
    false_positive = np.zeros(len(sorted_predictions), dtype=np.float64)
    matched_ious: list[float] = []

    for index, prediction in enumerate(sorted_predictions):
        candidates = gt_by_image.get(prediction.image_id, [])
        if not candidates:
            false_positive[index] = 1.0
            continue
        overlaps = np.asarray([bbox_iou(prediction.bbox, item.bbox) for item in candidates])
        best_index = int(overlaps.argmax()) if overlaps.size else -1
        best_iou = float(overlaps[best_index]) if best_index >= 0 else 0.0
        if best_iou >= iou_threshold and not used[prediction.image_id][best_index]:
            true_positive[index] = 1.0
            used[prediction.image_id][best_index] = True
            matched_ious.append(best_iou)
        else:
            false_positive[index] = 1.0

    gt_count = len(ground_truth)
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / gt_count if gt_count else np.zeros_like(cumulative_tp)
    precision = np.divide(
        cumulative_tp,
        cumulative_tp + cumulative_fp,
        out=np.zeros_like(cumulative_tp),
        where=(cumulative_tp + cumulative_fp) > 0,
    )
    return {
        "ap": _interpolated_ap(recall, precision) if gt_count else float("nan"),
        "precision": float(precision[-1]) if precision.size else 0.0,
        "recall": float(recall[-1]) if recall.size else 0.0,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "gt_count": float(gt_count),
        "prediction_count": float(len(predictions)),
    }


def evaluate_detections(
    predictions: Sequence[Detection],
    ground_truth: Sequence[Detection],
    classes: Sequence[str] = SHARED_DETECTION_CLASSES,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    thresholds = np.arange(0.50, 0.96, 0.05)
    per_class: list[dict[str, float | str]] = []
    for class_name in classes:
        class_predictions = [item for item in predictions if item.class_name == class_name]
        class_gt = [item for item in ground_truth if item.class_name == class_name]
        evaluations = [
            _evaluate_detection_class(class_predictions, class_gt, float(threshold))
            for threshold in thresholds
        ]
        ap_values = np.asarray([item["ap"] for item in evaluations], dtype=np.float64)
        ap50 = evaluations[0]
        per_class.append(
            {
                "class_name": class_name,
                "map_50_95": (
                    float(np.nanmean(ap_values)) if np.any(~np.isnan(ap_values)) else float("nan")
                ),
                "ap_50": float(ap50["ap"]),
                "precision_50": float(ap50["precision"]),
                "recall_50": float(ap50["recall"]),
                "mean_matched_iou_50": float(ap50["mean_matched_iou"]),
                "gt_count": float(ap50["gt_count"]),
                "prediction_count": float(ap50["prediction_count"]),
            }
        )

    valid_rows = [row for row in per_class if not math.isnan(float(row["map_50_95"]))]
    summary = {
        "map_50_95": (
            float(np.mean([float(row["map_50_95"]) for row in valid_rows])) if valid_rows else 0.0
        ),
        "map_50": float(np.mean([float(row["ap_50"]) for row in valid_rows])) if valid_rows else 0.0,
        "mean_precision_50": (
            float(np.mean([float(row["precision_50"]) for row in valid_rows])) if valid_rows else 0.0
        ),
        "mean_recall_50": (
            float(np.mean([float(row["recall_50"]) for row in valid_rows])) if valid_rows else 0.0
        ),
        "mean_matched_iou_50": (
            float(np.mean([float(row["mean_matched_iou_50"]) for row in valid_rows]))
            if valid_rows
            else 0.0
        ),
        "ground_truth_objects": float(len(ground_truth)),
        "predicted_objects": float(len(predictions)),
        "evaluated_classes": float(len(valid_rows)),
    }
    return summary, per_class


# ---------------------------------------------------------------------------
# Part 2: Introducing Distortions (slides 17-25)
# ---------------------------------------------------------------------------


def gaussian_noise(image_rgb: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=image_rgb.shape)
    return (image_rgb.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


def jpeg_compression(image_rgb: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=int(quality), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return np.asarray(compressed.convert("RGB"), dtype=np.uint8)


def low_light(image_rgb: np.ndarray, brightness: float) -> np.ndarray:
    return (image_rgb.astype(np.float32) * float(brightness)).clip(0, 255).astype(np.uint8)


def motion_blur(
    image_rgb: np.ndarray,
    kernel_size: int,
    angle_degrees: float = 15.0,
) -> np.ndarray:
    """Apply normalized linear motion blur with reflection at image borders.

    The angle is fixed across severity levels so the experiment varies only the
    blur length. Odd kernels have a well-defined central pixel and symmetric PSF.
    """

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Motion-blur kernel_size must be a positive odd integer")
    if kernel_size == 1:
        return np.asarray(image_rgb, dtype=np.uint8).copy()
    cv2 = _cv2()
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0
    radius = center
    radians = math.radians(float(angle_degrees))
    dx = radius * math.cos(radians)
    dy = radius * math.sin(radians)
    start = (int(round(center - dx)), int(round(center - dy)))
    end = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, start, end, color=1.0, thickness=1, lineType=cv2.LINE_8)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:  # Defensive guard for unusual OpenCV builds.
        kernel[int(center), int(center)] = 1.0
        kernel_sum = 1.0
    kernel /= kernel_sum
    blurred = cv2.filter2D(
        np.asarray(image_rgb, dtype=np.uint8),
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT101,
    )
    return np.asarray(blurred, dtype=np.uint8)


def apply_aug(
    img_pil: Image.Image,
    distortion_name: str | Callable[[np.ndarray], np.ndarray],
    level: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Apply one distortion, retaining the slide's ``apply_aug`` naming."""

    image = np.asarray(img_pil.convert("RGB"), dtype=np.uint8)
    if callable(distortion_name):
        return np.asarray(distortion_name(image), dtype=np.uint8)
    if level is None:
        raise ValueError("A numeric level is required for a named distortion")
    if distortion_name == "GaussNoise":
        return gaussian_noise(image, sigma=float(level), seed=seed)
    if distortion_name == "SevereJPEG":
        return jpeg_compression(image, quality=int(level))
    if distortion_name == "LowLight":
        return low_light(image, brightness=float(level))
    if distortion_name == "MotionBlur":
        kernel_size = int(level)
        if float(kernel_size) != float(level):
            raise ValueError("MotionBlur level must be an integer kernel size")
        return motion_blur(image, kernel_size=kernel_size)
    raise KeyError(f"Unknown distortion: {distortion_name}")


def compute_snr(clean_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    """SNR(dB) = 10 log10(signal power / distortion power), as in slide 23."""

    clean = np.asarray(clean_rgb, dtype=np.float64)
    noise = clean - np.asarray(test_rgb, dtype=np.float64)
    signal_power = float(np.mean(clean**2))
    noise_power = float(np.mean(noise**2))
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return float("-inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def _stable_distortion_seed(base_seed: int, sample_id: str, name: str, level_index: int) -> int:
    # Python's hash is intentionally randomized between processes; use deterministic bytes instead.
    text = f"{base_seed}|{sample_id}|{name}|{level_index}".encode("utf-8")
    value = 2166136261
    for byte in text:
        value = (value ^ byte) * 16777619
        value &= 0xFFFFFFFF
    return value


# ---------------------------------------------------------------------------
# Output helpers and plots.
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in fieldnames})


def _matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Matplotlib is required. Install requirements.txt first.") from exc
    return plt


def save_part1_gallery(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    if not records:
        return
    plt = _matplotlib()
    figure, axes = plt.subplots(len(records), 6, figsize=(24, 4.6 * len(records)), squeeze=False)
    titles = ("Clean", "Ground truth", "ORB", "Canny", "YOLO", "SegFormer")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=13)
    for row_index, record in enumerate(records):
        for column, key in enumerate(
            ("clean", "ground_truth", "orb", "canny", "yolo", "segmentation")
        ):
            axes[row_index, column].imshow(record[key])
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(str(record["sample_id"]), fontsize=8)
    figure.suptitle("Part 1 - clean-image baselines", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_distortion_grid(
    image: Image.Image,
    levels: Mapping[str, Sequence[float]],
    output_path: Path,
    seed: int,
) -> None:
    plt = _matplotlib()
    columns = 1 + max(len(values) for values in levels.values())
    figure, axes = plt.subplots(len(levels), columns, figsize=(3.5 * columns, 3.7 * len(levels)))
    axes = np.atleast_2d(axes)
    for row, (name, values) in enumerate(levels.items()):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{name}\nClean")
        axes[row, 0].axis("off")
        for column, level in enumerate(values, 1):
            distorted = apply_aug(image, name, float(level), seed=seed + row * 100 + column)
            axes[row, column].imshow(distorted)
            axes[row, column].set_title(f"level={level:g}\nSNR={compute_snr(np.asarray(image), distorted):.2f} dB")
            axes[row, column].axis("off")
        for column in range(len(values) + 1, columns):
            axes[row, column].axis("off")
    figure.suptitle("Part 2 - distortion intensity ranges", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_performance_snr_plot(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    plt = _matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(14, 10.0))
    axes = axes.ravel()
    metrics = (
        ("orb_match_retention", "ORB spatial match retention"),
        ("canny_f1", "Canny tolerant edge F1"),
        ("seg_mean_iou", "SegFormer mean IoU"),
        ("det_map_50_95", "YOLO mAP@0.50:0.95"),
    )
    distortions = sorted({str(row["distortion"]) for row in rows})
    for axis, (metric, label) in zip(axes, metrics):
        for distortion in distortions:
            selected = [row for row in rows if row["distortion"] == distortion]
            selected.sort(key=lambda row: float(row["mean_snr_db"]), reverse=True)
            axis.plot(
                [float(row["mean_snr_db"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                label=distortion,
            )
        axis.set_xlabel("Mean SNR (dB) - cleaner to the right")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Part 2 - performance per SNR", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_part2_gallery(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        return
    plt = _matplotlib()
    figure, axes = plt.subplots(len(rows), 5, figsize=(20, 4.5 * len(rows)), squeeze=False)
    for column, title in enumerate(("Distorted", "ORB", "Canny", "YOLO", "SegFormer")):
        axes[0, column].set_title(title, fontsize=13)
    for row_index, record in enumerate(rows):
        for column, key in enumerate(("distorted", "orb", "canny", "yolo", "segmentation")):
            axes[row_index, column].imshow(record[key])
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(
            f"{record['distortion']}\nlevel={record['level']:g}\nSNR={record['snr_db']:.2f} dB"
        )
    figure.suptitle("Part 2 - model outputs on distorted images", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def draw_ground_truth_boxes(image: Image.Image, boxes: Sequence[Detection]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for item in boxes:
        draw.rectangle(item.bbox, outline=(0, 255, 0), width=3)
        draw.text((item.bbox[0] + 2, item.bbox[1] + 2), item.class_name, fill=(0, 255, 0))
    return output


# ---------------------------------------------------------------------------
# Model loading and complete Part 1 / Part 2 experiment.
# ---------------------------------------------------------------------------


def select_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(config: ExperimentConfig) -> tuple[Any, Any, Any, str]:
    try:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Model dependencies are missing. Install requirements.txt first.") from exc

    device = select_device(config.device)
    LOGGER.info("Loading YOLO model %s", config.yolo_model)
    detector = YOLO(config.yolo_model)
    detector.to(device)
    LOGGER.info("Loading SegFormer model %s", config.segformer_model)
    processor = AutoImageProcessor.from_pretrained(config.segformer_model)
    segmenter = SegformerForSemanticSegmentation.from_pretrained(config.segformer_model)
    segmenter.to(device)
    segmenter.eval()
    return detector, processor, segmenter, device


@dataclass
class CleanReference:
    sample: CityscapesSample
    gt_detections: list[Detection]
    clean_edges: np.ndarray


def run_part1(
    config: ExperimentConfig,
    samples: Sequence[CityscapesSample],
    detector: Any,
    processor: Any,
    segmenter: Any,
    device: str,
) -> tuple[list[CleanReference], dict[str, Any]]:
    """Part 1: run and evaluate all methods on clean Cityscapes images."""

    LOGGER.info("Part 1: evaluating %d clean images", len(samples))
    output_dir = config.output_dir / "part1"
    segmentation_accumulator = SegmentationAccumulator()
    all_predictions: list[Detection] = []
    all_ground_truth: list[Detection] = []
    per_image_rows: list[dict[str, Any]] = []
    references: list[CleanReference] = []
    gallery: list[dict[str, Any]] = []

    for index, sample in enumerate(samples, 1):
        LOGGER.info("Part 1 [%d/%d] %s", index, len(samples), sample.sample_id)
        image, label, instance = load_sample(sample)
        gt_detections = instance_mask_to_boxes(instance, sample.sample_id)
        predictions = yolo_detections(
            image,
            detector,
            sample.sample_id,
            conf=config.yolo_eval_confidence,
        )
        segmentation = predict_segmentation(image, processor, segmenter, device)
        segmentation_accumulator.update(segmentation, label)
        image_ious = compute_ious(segmentation, label)
        clean_orb = measure_orb_matching(
            image,
            image,
            nfeatures=config.nfeatures,
            ratio_threshold=config.orb_ratio_threshold,
            spatial_threshold=config.orb_spatial_threshold,
        )
        clean_edges = canny_detect(
            image,
            low_threshold=config.canny_low_threshold,
            high_threshold=config.canny_high_threshold,
            blur_kernel=config.canny_blur_kernel,
        )
        clean_canny = evaluate_canny_edges(
            clean_edges,
            clean_edges,
            tolerance_radius=config.canny_tolerance_radius,
        )

        all_predictions.extend(predictions)
        all_ground_truth.extend(gt_detections)
        per_image_rows.append(
            {
                "sample_id": sample.sample_id,
                "orb_keypoints": clean_orb["clean_keypoints"],
                "orb_self_match_retention": clean_orb["match_retention"],
                "canny_edge_pixels": clean_canny["reference_edge_pixels"],
                "canny_self_f1": clean_canny["f1"],
                "seg_mean_iou": float(np.mean(list(image_ious.values()))) if image_ious else 0.0,
                "seg_classes_present": len(image_ious),
                "gt_detection_objects": len(gt_detections),
                "yolo_detection_objects": len(predictions),
            }
        )
        references.append(
            CleanReference(
                sample=sample,
                gt_detections=gt_detections,
                clean_edges=clean_edges,
            )
        )

        if len(gallery) < config.gallery_samples:
            orb_image, keypoints, _ = orb_overlay(image, nfeatures=config.nfeatures)
            canny_image = canny_overlay(image, clean_edges)
            yolo_image, result = yolo_overlay(image, detector, conf=config.yolo_visual_confidence)
            gallery.append(
                {
                    "sample_id": sample.sample_id,
                    "clean": image,
                    "ground_truth": overlay_mask(image, label),
                    "orb": orb_image,
                    "canny": canny_image,
                    "yolo": yolo_image,
                    "segmentation": seg_overlay(np.asarray(image), segmentation),
                    "orb_keypoints": len(keypoints),
                    "yolo_results": result,
                }
            )

    segmentation_summary, segmentation_per_class = segmentation_accumulator.results()
    detection_summary, detection_per_class = evaluate_detections(all_predictions, all_ground_truth)
    summary = {
        "scope": "Part 1 - clean images",
        "sample_count": len(samples),
        "split": config.split,
        "segmentation": segmentation_summary,
        "detection": detection_summary,
        "orb": {
            "mean_keypoints": float(np.mean([row["orb_keypoints"] for row in per_image_rows])),
            "mean_self_match_retention": float(
                np.mean([row["orb_self_match_retention"] for row in per_image_rows])
            ),
        },
        "canny": {
            "mean_edge_pixels": float(
                np.mean([row["canny_edge_pixels"] for row in per_image_rows])
            ),
            "mean_self_f1": float(np.mean([row["canny_self_f1"] for row in per_image_rows])),
            "low_threshold": config.canny_low_threshold,
            "high_threshold": config.canny_high_threshold,
            "gaussian_blur_kernel": config.canny_blur_kernel,
            "tolerance_radius": config.canny_tolerance_radius,
        },
    }
    write_json(output_dir / "clean_summary.json", summary)
    write_csv(output_dir / "clean_per_image.csv", per_image_rows)
    write_csv(output_dir / "segmentation_per_class.csv", segmentation_per_class)
    write_csv(output_dir / "detection_per_class.csv", detection_per_class)
    save_part1_gallery(gallery, output_dir / "figures" / "clean_predictions.png")
    return references, summary


def run_part2(
    config: ExperimentConfig,
    references: Sequence[CleanReference],
    detector: Any,
    processor: Any,
    segmenter: Any,
    device: str,
) -> dict[str, Any]:
    """Part 2: evaluate every method at every configured distortion level."""

    LOGGER.info("Part 2: evaluating distortions on %d images", len(references))
    output_dir = config.output_dir / "part2"
    per_image_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    segmentation_class_rows: list[dict[str, Any]] = []
    detection_class_rows: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []
    distortion_levels = config.distortion_levels or DEFAULT_DISTORTION_LEVELS

    if references:
        first_image, _ = load_image_and_label(references[0].sample)
        save_distortion_grid(
            first_image,
            distortion_levels,
            output_dir / "figures" / "distortion_grid.png",
            config.seed,
        )

    total_variants = sum(len(levels) for levels in distortion_levels.values())
    variant_index = 0
    for distortion_name, levels in distortion_levels.items():
        for level_index, level in enumerate(levels):
            variant_index += 1
            LOGGER.info(
                "Part 2 variant [%d/%d]: %s level=%s",
                variant_index,
                total_variants,
                distortion_name,
                level,
            )
            segmentation_accumulator = SegmentationAccumulator()
            predictions: list[Detection] = []
            ground_truth: list[Detection] = []
            variant_rows: list[dict[str, Any]] = []

            for image_index, reference in enumerate(references, 1):
                LOGGER.info(
                    "  image [%d/%d] %s",
                    image_index,
                    len(references),
                    reference.sample.sample_id,
                )
                clean_image, clean_label = load_image_and_label(reference.sample)
                distortion_seed = _stable_distortion_seed(
                    config.seed,
                    reference.sample.sample_id,
                    distortion_name,
                    level_index,
                )
                distorted_rgb = apply_aug(
                    clean_image,
                    distortion_name,
                    float(level),
                    seed=distortion_seed,
                )
                distorted_image = Image.fromarray(distorted_rgb)
                snr_db = compute_snr(np.asarray(clean_image), distorted_rgb)
                orb_metrics = measure_orb_matching(
                    clean_image,
                    distorted_image,
                    nfeatures=config.nfeatures,
                    ratio_threshold=config.orb_ratio_threshold,
                    spatial_threshold=config.orb_spatial_threshold,
                )
                distorted_edges = canny_detect(
                    distorted_image,
                    low_threshold=config.canny_low_threshold,
                    high_threshold=config.canny_high_threshold,
                    blur_kernel=config.canny_blur_kernel,
                )
                canny_metrics = evaluate_canny_edges(
                    reference.clean_edges,
                    distorted_edges,
                    tolerance_radius=config.canny_tolerance_radius,
                )
                segmentation = predict_segmentation(
                    distorted_image,
                    processor,
                    segmenter,
                    device,
                )
                segmentation_accumulator.update(segmentation, clean_label)
                image_ious = compute_ious(segmentation, clean_label)
                image_predictions = yolo_detections(
                    distorted_image,
                    detector,
                    reference.sample.sample_id,
                    conf=config.yolo_eval_confidence,
                )
                predictions.extend(image_predictions)
                ground_truth.extend(reference.gt_detections)

                row = {
                    "sample_id": reference.sample.sample_id,
                    "distortion": distortion_name,
                    "level_index": level_index,
                    "level": float(level),
                    "snr_db": snr_db,
                    "orb_clean_keypoints": orb_metrics["clean_keypoints"],
                    "orb_distorted_keypoints": orb_metrics["test_keypoints"],
                    "orb_keypoint_retention": orb_metrics["keypoint_retention"],
                    "orb_ratio_matches": orb_metrics["ratio_matches"],
                    "orb_spatial_inliers": orb_metrics["spatial_inliers"],
                    "orb_match_retention": orb_metrics["match_retention"],
                    "orb_inlier_ratio": orb_metrics["inlier_ratio"],
                    "canny_clean_edge_pixels": canny_metrics["reference_edge_pixels"],
                    "canny_distorted_edge_pixels": canny_metrics["test_edge_pixels"],
                    "canny_edge_pixel_retention": canny_metrics["edge_pixel_retention"],
                    "canny_precision": canny_metrics["precision"],
                    "canny_recall": canny_metrics["recall"],
                    "canny_f1": canny_metrics["f1"],
                    "seg_mean_iou": float(np.mean(list(image_ious.values()))) if image_ious else 0.0,
                    "seg_classes_present": len(image_ious),
                    "yolo_detection_objects": len(image_predictions),
                    "gt_detection_objects": len(reference.gt_detections),
                }
                per_image_rows.append(row)
                variant_rows.append(row)

                if (
                    config.gallery_samples > 0
                    and image_index == 1
                    and level_index == len(levels) // 2
                ):
                    orb_image, _, _ = orb_overlay(distorted_image, nfeatures=config.nfeatures)
                    canny_image = canny_overlay(distorted_image, distorted_edges)
                    yolo_image, _ = yolo_overlay(
                        distorted_image,
                        detector,
                        conf=config.yolo_visual_confidence,
                    )
                    gallery_rows.append(
                        {
                            "distortion": distortion_name,
                            "level": float(level),
                            "snr_db": snr_db,
                            "distorted": distorted_rgb,
                            "orb": orb_image,
                            "canny": canny_image,
                            "yolo": yolo_image,
                            "segmentation": seg_overlay(distorted_rgb, segmentation),
                        }
                    )

            segmentation_summary, segmentation_per_class = segmentation_accumulator.results()
            detection_summary, detection_per_class = evaluate_detections(predictions, ground_truth)
            finite_snrs = [float(row["snr_db"]) for row in variant_rows if math.isfinite(float(row["snr_db"]))]
            summary_row = {
                "distortion": distortion_name,
                "level_index": level_index,
                "level": float(level),
                "sample_count": len(variant_rows),
                "mean_snr_db": float(np.mean(finite_snrs)) if finite_snrs else float("inf"),
                "orb_keypoint_retention": float(
                    np.mean([float(row["orb_keypoint_retention"]) for row in variant_rows])
                ),
                "orb_match_retention": float(
                    np.mean([float(row["orb_match_retention"]) for row in variant_rows])
                ),
                "orb_inlier_ratio": float(
                    np.mean([float(row["orb_inlier_ratio"]) for row in variant_rows])
                ),
                "canny_edge_pixel_retention": float(
                    np.mean([float(row["canny_edge_pixel_retention"]) for row in variant_rows])
                ),
                "canny_precision": float(
                    np.mean([float(row["canny_precision"]) for row in variant_rows])
                ),
                "canny_recall": float(
                    np.mean([float(row["canny_recall"]) for row in variant_rows])
                ),
                "canny_f1": float(np.mean([float(row["canny_f1"]) for row in variant_rows])),
                "seg_mean_iou": segmentation_summary["mean_iou"],
                "seg_pixel_accuracy": segmentation_summary["pixel_accuracy"],
                "det_map_50_95": detection_summary["map_50_95"],
                "det_map_50": detection_summary["map_50"],
                "det_precision_50": detection_summary["mean_precision_50"],
                "det_recall_50": detection_summary["mean_recall_50"],
                "det_mean_matched_iou_50": detection_summary["mean_matched_iou_50"],
            }
            summary_rows.append(summary_row)
            for row in segmentation_per_class:
                segmentation_class_rows.append(
                    {"distortion": distortion_name, "level": float(level), **row}
                )
            for row in detection_per_class:
                detection_class_rows.append(
                    {"distortion": distortion_name, "level": float(level), **row}
                )

    write_csv(output_dir / "distorted_per_image.csv", per_image_rows)
    write_csv(output_dir / "distorted_summary.csv", summary_rows)
    write_csv(output_dir / "segmentation_per_class.csv", segmentation_class_rows)
    write_csv(output_dir / "detection_per_class.csv", detection_class_rows)
    write_json(
        output_dir / "distorted_summary.json",
        {
            "scope": "Part 2 - distorted images",
            "sample_count": len(references),
            "distortion_levels": distortion_levels,
            "variants": summary_rows,
        },
    )
    save_performance_snr_plot(summary_rows, output_dir / "figures" / "performance_per_snr.png")
    save_part2_gallery(gallery_rows, output_dir / "figures" / "distorted_predictions.png")
    return {"variants": summary_rows, "sample_count": len(references)}


def run_experiment(config: ExperimentConfig, part: str = "both") -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_cityscapes_samples(
        config.dataset_root,
        split=config.split,
        max_samples=config.max_samples,
        seed=config.seed,
    )
    LOGGER.info("Using %d Cityscapes %s samples", len(samples), config.split)
    detector, processor, segmenter, device = load_models(config)
    LOGGER.info("Inference device: %s", device)

    # Part 2 requires the clean images and ground truth. Running it also computes
    # the Part 1 reference, even if only Part 2 was selected.
    references, part1_summary = run_part1(
        config,
        samples,
        detector,
        processor,
        segmenter,
        device,
    )
    result: dict[str, Any] = {"part1": part1_summary}
    if part in {"2", "both"}:
        result["part2"] = run_part2(
            config,
            references,
            detector,
            processor,
            segmenter,
            device,
        )

    manifest = {
        "scope": "Course project Parts 1 and 2 only",
        "part_requested": part,
        "config": asdict(config),
        "result": result,
    }
    write_json(config.output_dir / "run_manifest.json", manifest)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Cityscapes robustness experiments for course-project Parts 1 and 2."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted Cityscapes root")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--part", choices=("1", "2", "both"), default="both")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Deterministic sample limit; 0 uses the complete split",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--nfeatures", type=int, default=800)
    parser.add_argument("--canny-low-threshold", type=int, default=100)
    parser.add_argument("--canny-high-threshold", type=int, default=200)
    parser.add_argument("--canny-blur-kernel", type=int, default=5)
    parser.add_argument("--canny-tolerance-radius", type=int, default=2)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--segformer-model", default=ExperimentConfig.segformer_model)
    parser.add_argument("--yolo-eval-confidence", type=float, default=0.001)
    parser.add_argument("--yolo-visual-confidence", type=float, default=0.25)
    parser.add_argument("--gallery-samples", type=int, default=4)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use at most four images and the first two levels of each distortion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    max_samples = args.max_samples
    levels: Mapping[str, tuple[float, ...]] = dict(DEFAULT_DISTORTION_LEVELS)
    if args.quick:
        max_samples = min(max_samples, 4) if max_samples > 0 else 4
        levels = {name: values[:2] for name, values in DEFAULT_DISTORTION_LEVELS.items()}

    config = ExperimentConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=max_samples,
        seed=args.seed,
        device=args.device,
        nfeatures=args.nfeatures,
        canny_low_threshold=args.canny_low_threshold,
        canny_high_threshold=args.canny_high_threshold,
        canny_blur_kernel=args.canny_blur_kernel,
        canny_tolerance_radius=args.canny_tolerance_radius,
        yolo_model=args.yolo_model,
        segformer_model=args.segformer_model,
        yolo_eval_confidence=args.yolo_eval_confidence,
        yolo_visual_confidence=args.yolo_visual_confidence,
        gallery_samples=args.gallery_samples,
        distortion_levels=levels,
    )
    run_experiment(config, part=args.part)
    LOGGER.info("Finished. Results are under %s", config.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
