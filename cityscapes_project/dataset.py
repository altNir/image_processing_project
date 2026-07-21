"""Cityscapes discovery, loading, label conversion, and box extraction."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from .config import (
    CITYSCAPES_INSTANCE_LABELS,
    CITYSCAPES_RAW_ID_TO_TRAIN_ID,
    SHARED_DETECTION_CLASSES,
)
from .types import CityscapesSample, Detection


def discover_cityscapes_samples(
    root: Path | str,
    split: str = "val",
    max_samples: int = 0,
    seed: int = 7,
) -> list[CityscapesSample]:
    """Discover images and fine annotations in the official layout."""

    root = Path(root)
    image_root = root / "leftImg8bit" / split
    gt_root = root / "gtFine" / split
    if not image_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError(
            "Cityscapes was not found in the expected official layout.\n"
            f"Expected: {image_root}\n      and: {gt_root}\n"
            "Extract leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip under one root."
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
        samples.append(CityscapesSample(base, image_path, label_path, instance_path))

    if missing:
        preview = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"Found images but {len(missing)} annotation files are missing.\n{preview}"
        )
    if not samples:
        raise FileNotFoundError(f"No Cityscapes {split!r} images found under {image_root}")
    if max_samples > 0 and max_samples < len(samples):
        samples = sorted(
            random.Random(seed).sample(samples, max_samples),
            key=lambda sample: sample.sample_id,
        )
    return samples


def raw_label_ids_to_train_ids(raw_label: np.ndarray) -> np.ndarray:
    """Convert official raw label IDs to contiguous train IDs 0..18."""

    raw = np.asarray(raw_label)
    output = np.full(raw.shape, 255, dtype=np.uint8)
    valid = (raw >= 0) & (raw < len(CITYSCAPES_RAW_ID_TO_TRAIN_ID))
    output[valid] = CITYSCAPES_RAW_ID_TO_TRAIN_ID[raw[valid].astype(np.int64)]
    return output


def _load_label(sample: CityscapesSample) -> np.ndarray:
    label = np.asarray(Image.open(sample.label_path), dtype=np.uint8)
    if sample.label_path.name.endswith("_gtFine_labelIds.png"):
        label = raw_label_ids_to_train_ids(label)
    return label


def load_sample(sample: CityscapesSample) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Load RGB, semantic train IDs, and instance IDs for one sample."""

    image = Image.open(sample.image_path).convert("RGB")
    label = _load_label(sample)
    instance = np.asarray(Image.open(sample.instance_path), dtype=np.int32)
    if label.shape != (image.height, image.width):
        raise ValueError(f"Label shape {label.shape} does not match {image.size} for {sample.sample_id}")
    if instance.shape != label.shape:
        raise ValueError(f"Instance shape {instance.shape} does not match labels for {sample.sample_id}")
    return image, label, instance


def load_image_and_label(sample: CityscapesSample) -> tuple[Image.Image, np.ndarray]:
    """Load RGB and semantic labels when instance IDs are already cached."""

    image = Image.open(sample.image_path).convert("RGB")
    label = _load_label(sample)
    if label.shape != (image.height, image.width):
        raise ValueError(f"Label shape {label.shape} does not match {image.size} for {sample.sample_id}")
    return image, label


def instance_mask_to_boxes(instance_mask: np.ndarray, image_id: str) -> list[Detection]:
    """Derive visible-pixel xyxy boxes for the seven shared object classes."""

    detections: list[Detection] = []
    shared_names = set(SHARED_DETECTION_CLASSES)
    for instance_id in np.unique(instance_mask):
        instance_id_int = int(instance_id)
        if instance_id_int < 1000:
            continue
        class_name = CITYSCAPES_INSTANCE_LABELS.get(instance_id_int // 1000)
        if class_name not in shared_names:
            continue
        ys, xs = np.where(instance_mask == instance_id_int)
        if xs.size:
            detections.append(
                Detection(
                    image_id,
                    class_name,
                    (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
                )
            )
    return detections
