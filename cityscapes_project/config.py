"""Experiment configuration and Cityscapes constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


CITYSCAPES_CLASSES: tuple[str, ...] = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
)

CITYSCAPES_PALETTE = np.asarray(
    [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
        (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
        (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
        (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
        (0, 0, 230), (119, 11, 32),
    ],
    dtype=np.uint8,
)

CITYSCAPES_RAW_ID_TO_TRAIN_ID = np.full(256, 255, dtype=np.uint8)
CITYSCAPES_RAW_ID_TO_TRAIN_ID[
    np.asarray([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])
] = np.arange(19, dtype=np.uint8)

CITYSCAPES_INSTANCE_LABELS: Mapping[int, str] = {
    24: "person", 25: "rider", 26: "car", 27: "truck", 28: "bus",
    31: "train", 32: "motorcycle", 33: "bicycle",
}

COCO_ID_TO_SHARED_CLASS: Mapping[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 6: "train", 7: "truck",
}
SHARED_DETECTION_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
)

DEFAULT_DISTORTION_LEVELS: Mapping[str, tuple[float, ...]] = {
    "GaussNoise": (5.0, 10.0, 20.0, 35.0, 50.0),
    "SevereJPEG": (80.0, 60.0, 40.0, 20.0, 5.0),
    "LowLight": (0.80, 0.60, 0.40, 0.25, 0.10),
    "MotionBlur": (3.0, 5.0, 9.0, 15.0, 25.0),
}


@dataclass
class ExperimentConfig:
    """Configuration shared by clean and distortion evaluation."""

    dataset_root: Path
    output_dir: Path = Path("outputs")
    split: str = "val"
    max_samples: int = 0
    seed: int = 7
    device: str = "auto"
    use_half: bool = True
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
        if self.max_samples < 0 or self.nfeatures <= 0 or self.gallery_samples < 0:
            raise ValueError("Sample limits/gallery size must be non-negative and nfeatures positive")
        if not 0 <= self.canny_low_threshold < self.canny_high_threshold:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high")
        if self.canny_blur_kernel < 1 or self.canny_blur_kernel % 2 == 0:
            raise ValueError("canny_blur_kernel must be a positive odd integer")
        if self.canny_tolerance_radius < 0:
            raise ValueError("canny_tolerance_radius must be non-negative")


@dataclass
class Parts34Config:
    """Configuration for restoration and robust detector fine-tuning."""

    dataset_root: Path
    output_dir: Path = Path("outputs_parts_3_4")
    artifacts_dir: Path = Path("artifacts")
    split: str = "val"
    max_samples: int = 0
    seed: int = 7
    device: str = "auto"
    use_half: bool = True
    nfeatures: int = 800
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    canny_blur_kernel: int = 5
    canny_tolerance_radius: int = 2
    yolo_model: str = "yolov8n.pt"
    yolo_eval_confidence: float = 0.001
    segformer_model: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    distortion_levels: Mapping[str, tuple[float, ...]] | None = None
    gallery_samples: int = 1
    part4_train_samples: int = 0
    part4_val_samples: int = 0
    part4_epochs: int = 40
    part4_image_size: int = 640
    part4_batch: int = 8
    part4_workers: int = 4
    part4_clean_fraction: float = 0.20
    part4_train_views: int = 4
    part4_val_views: int = 2
    part4_internal_val_fraction: float = 0.125
    part4_patience: int = 10
    part4_eval_batch: int = 32
    rebuild_training_data: bool = False
    reuse_part2_results: bool = True
    part3_bootstrap_resamples: int = 1000
    part3_confidence_level: float = 0.95
    fine_tuned_weights: Path | None = None

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root)
        self.output_dir = Path(self.output_dir)
        self.artifacts_dir = Path(self.artifacts_dir)
        if self.fine_tuned_weights is not None:
            self.fine_tuned_weights = Path(self.fine_tuned_weights)
        if self.distortion_levels is None:
            self.distortion_levels = dict(DEFAULT_DISTORTION_LEVELS)
        if not 0.0 <= self.part4_clean_fraction <= 1.0:
            raise ValueError("part4_clean_fraction must be between 0 and 1")
        if self.max_samples < 0 or self.part4_train_samples < 0 or self.part4_val_samples < 0:
            raise ValueError("Sample limits must be non-negative")
        if self.part4_epochs <= 0 or self.part4_image_size <= 0:
            raise ValueError("Part 4 epochs and image size must be positive")
        if self.part4_batch != -1 and self.part4_batch <= 0:
            raise ValueError("Part 4 batch must be positive or -1 for automatic sizing")
        if self.part4_workers < 0:
            raise ValueError("part4_workers must be non-negative")
        if self.part4_train_views <= 0 or self.part4_val_views <= 0:
            raise ValueError("Part 4 view counts must be positive")
        if not 0.0 < self.part4_internal_val_fraction < 0.5:
            raise ValueError("part4_internal_val_fraction must be between zero and 0.5")
        if self.part4_patience < 0 or self.part4_eval_batch <= 0:
            raise ValueError("Part 4 patience must be non-negative and eval batch positive")
        if self.part3_bootstrap_resamples <= 0:
            raise ValueError("part3_bootstrap_resamples must be positive")
        if not 0.0 < self.part3_confidence_level < 1.0:
            raise ValueError("part3_confidence_level must be between zero and one")
        if not 0 <= self.canny_low_threshold < self.canny_high_threshold:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high")
        if self.canny_blur_kernel < 1 or self.canny_blur_kernel % 2 == 0:
            raise ValueError("canny_blur_kernel must be a positive odd integer")
        if self.canny_tolerance_radius < 0:
            raise ValueError("canny_tolerance_radius must be non-negative")


def to_base_config(config: Parts34Config) -> ExperimentConfig:
    """Convert a Parts 3/4 configuration to the shared model configuration."""

    return ExperimentConfig(
        dataset_root=config.dataset_root,
        output_dir=config.output_dir,
        split=config.split,
        max_samples=config.max_samples,
        seed=config.seed,
        device=config.device,
        use_half=config.use_half,
        nfeatures=config.nfeatures,
        canny_low_threshold=config.canny_low_threshold,
        canny_high_threshold=config.canny_high_threshold,
        canny_blur_kernel=config.canny_blur_kernel,
        canny_tolerance_radius=config.canny_tolerance_radius,
        yolo_model=config.yolo_model,
        yolo_eval_confidence=config.yolo_eval_confidence,
        segformer_model=config.segformer_model,
        distortion_levels=config.distortion_levels,
        gallery_samples=config.gallery_samples,
    )
