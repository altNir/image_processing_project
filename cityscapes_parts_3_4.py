"""Cityscapes robustness project - Parts 3 and 4.

Part 3 restores each distorted image, reruns ORB, Canny, SegFormer, and YOLO,
and compares distorted versus restored performance at every severity/SNR level.

Part 4 follows the course slides by building a Cityscapes YOLO dataset with
clean and distorted training images, fine-tuning YOLO, and comparing the
pretrained and fine-tuned detector on clean and distorted validation images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from cityscapes_parts_1_2 import (
    COCO_ID_TO_SHARED_CLASS,
    DEFAULT_DISTORTION_LEVELS,
    SHARED_DETECTION_CLASSES,
    Detection,
    ExperimentConfig,
    SegmentationAccumulator,
    _cv2,
    _matplotlib,
    _stable_distortion_seed,
    apply_aug,
    canny_detect,
    compute_ious,
    compute_snr,
    discover_cityscapes_samples,
    evaluate_canny_edges,
    evaluate_detections,
    instance_mask_to_boxes,
    load_models,
    load_sample,
    measure_orb_matching,
    predict_segmentation,
    select_device,
    write_csv,
    write_json,
    yolo_detections,
)


LOGGER = logging.getLogger("cityscapes_parts_3_4")
PROJECT_CLASS_TO_ID = {name: index for index, name in enumerate(SHARED_DETECTION_CLASSES)}
PROJECT_ID_TO_CLASS = {index: name for name, index in PROJECT_CLASS_TO_ID.items()}


@dataclass
class Parts34Config:
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
    part4_epochs: int = 20
    part4_image_size: int = 640
    part4_batch: int = 8
    part4_workers: int = 4
    part4_clean_fraction: float = 0.20
    rebuild_training_data: bool = False
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


def to_base_config(config: Parts34Config) -> ExperimentConfig:
    """Build the shared model/dataset configuration used by Parts 1-3."""

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


# ---------------------------------------------------------------------------
# Part 3 restoration methods (matching the enhancement direction in slides 27-31).
# ---------------------------------------------------------------------------


def restore_gaussian_noise(image_rgb: np.ndarray) -> np.ndarray:
    """Non-local means followed by a light bilateral edge-preserving filter."""

    cv2 = _cv2()
    bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 25, 25, 7, 21)
    denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=45, sigmaSpace=45)
    return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)


def restore_jpeg(image_rgb: np.ndarray) -> np.ndarray:
    """Reduce JPEG blocking/ringing on luminance while retaining chroma."""

    cv2 = _cv2()
    ycrcb = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)
    y_channel = cv2.bilateralFilter(y_channel, d=7, sigmaColor=35, sigmaSpace=35)
    restored = cv2.merge((y_channel, cr, cb))
    return cv2.cvtColor(restored, cv2.COLOR_YCrCb2RGB)


def restore_low_light(image_rgb: np.ndarray) -> np.ndarray:
    """Gamma lifting followed by CLAHE on LAB luminance, as in the slides."""

    cv2 = _cv2()
    image = np.asarray(image_rgb, dtype=np.uint8)
    gamma = 0.45
    lookup = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0).clip(0, 255)
    lifted = cv2.LUT(image, lookup.astype(np.uint8))
    lab = cv2.cvtColor(lifted, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def restore_motion_blur(image_rgb: np.ndarray, kernel_size: int) -> np.ndarray:
    """Fast unsharp deblurring scaled to the known synthetic blur length."""

    cv2 = _cv2()
    image = np.asarray(image_rgb, dtype=np.uint8)
    sigma = max(0.8, float(kernel_size) / 6.0)
    smooth = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    amount = min(1.6, 0.65 + float(kernel_size) / 25.0)
    sharpened = cv2.addWeighted(image, 1.0 + amount, smooth, -amount, 0)
    return np.asarray(sharpened, dtype=np.uint8)


def restore_image(image_rgb: np.ndarray, distortion_name: str, level: float) -> np.ndarray:
    if distortion_name == "GaussNoise":
        return restore_gaussian_noise(image_rgb)
    if distortion_name == "SevereJPEG":
        return restore_jpeg(image_rgb)
    if distortion_name == "LowLight":
        return restore_low_light(image_rgb)
    if distortion_name == "MotionBlur":
        return restore_motion_blur(image_rgb, int(level))
    raise KeyError(f"No restoration is registered for {distortion_name}")


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def save_restoration_gallery(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    plt = _matplotlib()
    figure, axes = plt.subplots(len(rows), 3, figsize=(15, 4.4 * len(rows)), squeeze=False)
    for column, title in enumerate(("Clean", "Distorted", "Restored")):
        axes[0, column].set_title(title)
    for index, row in enumerate(rows):
        for column, key in enumerate(("clean", "distorted", "restored")):
            axes[index, column].imshow(row[key])
            axes[index, column].axis("off")
        axes[index, 0].set_ylabel(
            f"{row['distortion']}\nlevel={row['level']:g}\n"
            f"{row['distorted_snr']:.1f}->{row['restored_snr']:.1f} dB"
        )
    figure.suptitle("Part 3 - clean, distorted, and restored images", fontsize=16)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_restoration_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    plt = _matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = (
        ("orb_distorted", "orb_restored", "ORB match retention"),
        ("canny_distorted", "canny_restored", "Canny tolerant F1"),
        ("seg_distorted", "seg_restored", "SegFormer mean IoU"),
        ("det_distorted", "det_restored", "YOLO mAP@0.50:0.95"),
    )
    distortions = sorted({str(row["distortion"]) for row in rows})
    for axis, (dist_key, rest_key, title) in zip(axes.ravel(), metrics):
        for distortion in distortions:
            selected = [row for row in rows if row["distortion"] == distortion]
            selected.sort(key=lambda row: float(row["distorted_mean_snr_db"]))
            x = [float(row["distorted_mean_snr_db"]) for row in selected]
            axis.plot(x, [float(row[dist_key]) for row in selected], "--o", label=f"{distortion} distorted")
            axis.plot(x, [float(row[rest_key]) for row in selected], "-s", label=f"{distortion} restored")
        axis.set_xlabel("Distorted-image mean SNR (dB)")
        axis.set_ylabel(title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, ncol=2)
    figure.suptitle("Part 3 - distorted versus restored performance", fontsize=16)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_part3(
    config: Parts34Config,
    detector: Any,
    processor: Any,
    segmenter: Any,
    device: str,
) -> dict[str, Any]:
    """Evaluate all four methods before and after restoration."""

    samples = discover_cityscapes_samples(
        config.dataset_root, config.split, config.max_samples, config.seed
    )
    levels_by_name = config.distortion_levels or DEFAULT_DISTORTION_LEVELS
    output = config.output_dir / "part3"
    per_image: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seg_class_rows: list[dict[str, Any]] = []
    det_class_rows: list[dict[str, Any]] = []
    gallery: list[dict[str, Any]] = []
    total = sum(len(levels) for levels in levels_by_name.values())
    variant_number = 0

    for distortion_name, levels in levels_by_name.items():
        for level_index, level in enumerate(levels):
            variant_number += 1
            LOGGER.info(
                "Part 3 variant [%d/%d]: %s level=%s",
                variant_number,
                total,
                distortion_name,
                level,
            )
            seg_dist = SegmentationAccumulator()
            seg_rest = SegmentationAccumulator()
            pred_dist: list[Detection] = []
            pred_rest: list[Detection] = []
            ground_truth: list[Detection] = []
            variant_rows: list[dict[str, Any]] = []

            for image_index, sample in enumerate(samples, 1):
                LOGGER.info("  image [%d/%d] %s", image_index, len(samples), sample.sample_id)
                clean_image, label, instance = load_sample(sample)
                clean_rgb = np.asarray(clean_image)
                gt = instance_mask_to_boxes(instance, sample.sample_id)
                seed = _stable_distortion_seed(
                    config.seed, sample.sample_id, distortion_name, level_index
                )
                distorted_rgb = apply_aug(clean_image, distortion_name, float(level), seed=seed)
                restored_rgb = restore_image(distorted_rgb, distortion_name, float(level))
                distorted = Image.fromarray(distorted_rgb)
                restored = Image.fromarray(restored_rgb)

                clean_edges = canny_detect(
                    clean_image,
                    config.canny_low_threshold,
                    config.canny_high_threshold,
                    config.canny_blur_kernel,
                )
                distorted_edges = canny_detect(
                    distorted,
                    config.canny_low_threshold,
                    config.canny_high_threshold,
                    config.canny_blur_kernel,
                )
                restored_edges = canny_detect(
                    restored,
                    config.canny_low_threshold,
                    config.canny_high_threshold,
                    config.canny_blur_kernel,
                )
                canny_dist = evaluate_canny_edges(
                    clean_edges, distorted_edges, config.canny_tolerance_radius
                )
                canny_rest = evaluate_canny_edges(
                    clean_edges, restored_edges, config.canny_tolerance_radius
                )
                orb_dist = measure_orb_matching(clean_image, distorted, config.nfeatures)
                orb_rest = measure_orb_matching(clean_image, restored, config.nfeatures)

                segmentation_dist = predict_segmentation(
                    distorted, processor, segmenter, device, config.use_half
                )
                segmentation_rest = predict_segmentation(
                    restored, processor, segmenter, device, config.use_half
                )
                seg_dist.update(segmentation_dist, label)
                seg_rest.update(segmentation_rest, label)

                detections_dist = yolo_detections(
                    distorted,
                    detector,
                    sample.sample_id,
                    config.yolo_eval_confidence,
                    device,
                    config.use_half,
                )
                detections_rest = yolo_detections(
                    restored,
                    detector,
                    sample.sample_id,
                    config.yolo_eval_confidence,
                    device,
                    config.use_half,
                )
                pred_dist.extend(detections_dist)
                pred_rest.extend(detections_rest)
                ground_truth.extend(gt)

                dist_ious = compute_ious(segmentation_dist, label)
                rest_ious = compute_ious(segmentation_rest, label)
                row = {
                    "sample_id": sample.sample_id,
                    "distortion": distortion_name,
                    "level_index": level_index,
                    "level": float(level),
                    "distorted_snr_db": compute_snr(clean_rgb, distorted_rgb),
                    "restored_snr_db": compute_snr(clean_rgb, restored_rgb),
                    "orb_distorted": orb_dist["match_retention"],
                    "orb_restored": orb_rest["match_retention"],
                    "canny_distorted": canny_dist["f1"],
                    "canny_restored": canny_rest["f1"],
                    "seg_distorted_image_miou": float(np.mean(list(dist_ious.values()))) if dist_ious else 0.0,
                    "seg_restored_image_miou": float(np.mean(list(rest_ious.values()))) if rest_ious else 0.0,
                    "detections_distorted": len(detections_dist),
                    "detections_restored": len(detections_rest),
                }
                per_image.append(row)
                variant_rows.append(row)

                if (
                    config.gallery_samples > 0
                    and image_index == 1
                    and level_index == len(levels) // 2
                ):
                    gallery.append(
                        {
                            "distortion": distortion_name,
                            "level": float(level),
                            "distorted_snr": row["distorted_snr_db"],
                            "restored_snr": row["restored_snr_db"],
                            "clean": clean_rgb,
                            "distorted": distorted_rgb,
                            "restored": restored_rgb,
                        }
                    )

            seg_dist_summary, seg_dist_classes = seg_dist.results()
            seg_rest_summary, seg_rest_classes = seg_rest.results()
            det_dist_summary, det_dist_classes = evaluate_detections(pred_dist, ground_truth)
            det_rest_summary, det_rest_classes = evaluate_detections(pred_rest, ground_truth)
            summary = {
                "distortion": distortion_name,
                "level_index": level_index,
                "level": float(level),
                "sample_count": len(variant_rows),
                "distorted_mean_snr_db": _mean(variant_rows, "distorted_snr_db"),
                "restored_mean_snr_db": _mean(variant_rows, "restored_snr_db"),
                "snr_gain_db": _mean(variant_rows, "restored_snr_db") - _mean(variant_rows, "distorted_snr_db"),
                "orb_distorted": _mean(variant_rows, "orb_distorted"),
                "orb_restored": _mean(variant_rows, "orb_restored"),
                "canny_distorted": _mean(variant_rows, "canny_distorted"),
                "canny_restored": _mean(variant_rows, "canny_restored"),
                "seg_distorted": seg_dist_summary["mean_iou"],
                "seg_restored": seg_rest_summary["mean_iou"],
                "det_distorted": det_dist_summary["map_50_95"],
                "det_restored": det_rest_summary["map_50_95"],
            }
            summaries.append(summary)
            for condition, rows in (("distorted", seg_dist_classes), ("restored", seg_rest_classes)):
                for row in rows:
                    seg_class_rows.append(
                        {"distortion": distortion_name, "level": float(level), "condition": condition, **row}
                    )
            for condition, rows in (("distorted", det_dist_classes), ("restored", det_rest_classes)):
                for row in rows:
                    det_class_rows.append(
                        {"distortion": distortion_name, "level": float(level), "condition": condition, **row}
                    )

    write_csv(output / "restoration_per_image.csv", per_image)
    write_csv(output / "restoration_summary.csv", summaries)
    write_csv(output / "segmentation_per_class.csv", seg_class_rows)
    write_csv(output / "detection_per_class.csv", det_class_rows)
    write_json(
        output / "restoration_summary.json",
        {
            "scope": "Part 3 - restored images",
            "sample_count": len(samples),
            "distortion_levels": levels_by_name,
            "variants": summaries,
        },
    )
    save_restoration_gallery(gallery, output / "figures" / "restoration_grid.png")
    save_restoration_plot(summaries, output / "figures" / "restored_performance.png")
    return {"sample_count": len(samples), "variants": summaries}


# ---------------------------------------------------------------------------
# Part 4 - create distorted labels/data, fine-tune YOLO, and evaluate it.
# ---------------------------------------------------------------------------


def detection_to_yolo_row(detection: Detection, width: int, height: int) -> str:
    class_id = PROJECT_CLASS_TO_ID[detection.class_name]
    x1, y1, x2, y2 = detection.bbox
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"


def choose_training_condition(
    index: int,
    sample_id: str,
    seed: int,
    levels_by_name: Mapping[str, Sequence[float]],
    clean_fraction: float,
) -> tuple[str, float | None, int]:
    """Deterministically assign clean/distorted conditions across the dataset."""

    rng = random.Random(f"{seed}|{sample_id}|{index}")
    if rng.random() < clean_fraction:
        return "Clean", None, 0
    names = sorted(levels_by_name)
    name = names[rng.randrange(len(names))]
    levels = levels_by_name[name]
    level_index = rng.randrange(len(levels))
    return name, float(levels[level_index]), level_index


def _training_dataset_key(config: Parts34Config) -> str:
    train = config.part4_train_samples or "full"
    val = config.part4_val_samples or "full"
    recipe = json.dumps(
        {
            "train": train,
            "val": val,
            "seed": config.seed,
            "clean_fraction": config.part4_clean_fraction,
            "distortion_levels": config.distortion_levels,
        },
        sort_keys=True,
    )
    recipe_hash = hashlib.sha256(recipe.encode("utf-8")).hexdigest()[:10]
    return f"cityscapes_robust_train-{train}_val-{val}_seed-{config.seed}_{recipe_hash}"


def prepare_yolo_dataset(config: Parts34Config) -> tuple[Path, Path]:
    """Create a mixed clean/distorted YOLO dataset from Cityscapes instances."""

    root = config.artifacts_dir / "part4" / _training_dataset_key(config)
    manifest_path = root / "dataset_manifest.json"
    yaml_path = root / "dataset.yaml"
    if manifest_path.is_file() and yaml_path.is_file() and not config.rebuild_training_data:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete"):
            LOGGER.info("Reusing prepared Part 4 dataset at %s", root)
            return yaml_path, root

    levels_by_name = config.distortion_levels or DEFAULT_DISTORTION_LEVELS
    rows: list[dict[str, Any]] = []
    for split, limit in (("train", config.part4_train_samples), ("val", config.part4_val_samples)):
        samples = discover_cityscapes_samples(
            config.dataset_root, split=split, max_samples=limit, seed=config.seed
        )
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            if index % 100 == 0:
                LOGGER.info("Preparing Part 4 %s data [%d/%d]", split, index, len(samples))
            image, _, instance = load_sample(sample)
            condition, level, level_index = choose_training_condition(
                index,
                sample.sample_id,
                config.seed + (0 if split == "train" else 100_000),
                levels_by_name,
                config.part4_clean_fraction,
            )
            if condition == "Clean":
                output_rgb = np.asarray(image)
                snr = float("inf")
            else:
                distortion_seed = _stable_distortion_seed(
                    config.seed, sample.sample_id, condition, level_index
                )
                output_rgb = apply_aug(image, condition, float(level), seed=distortion_seed)
                snr = compute_snr(np.asarray(image), output_rgb)
            output_image = image_dir / f"{sample.sample_id}.jpg"
            Image.fromarray(output_rgb).save(output_image, quality=95, subsampling=2)
            boxes = instance_mask_to_boxes(instance, sample.sample_id)
            label_text = "\n".join(
                detection_to_yolo_row(item, image.width, image.height) for item in boxes
            )
            (label_dir / f"{sample.sample_id}.txt").write_text(
                label_text + ("\n" if label_text else ""), encoding="utf-8"
            )
            rows.append(
                {
                    "split": split,
                    "sample_id": sample.sample_id,
                    "condition": condition,
                    "level": level,
                    "snr_db": snr,
                    "objects": len(boxes),
                }
            )

    yaml_lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in PROJECT_ID_TO_CLASS.items())
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    write_csv(root / "samples.csv", rows)
    write_json(
        manifest_path,
        {
            "complete": True,
            "dataset_root": config.dataset_root,
            "seed": config.seed,
            "clean_fraction": config.part4_clean_fraction,
            "distortion_levels": levels_by_name,
            "samples": len(rows),
        },
    )
    return yaml_path, root


def train_yolo(config: Parts34Config, yaml_path: Path, device: str) -> Path:
    """Fine-tune YOLO using the prepared robust Cityscapes dataset."""

    from ultralytics import YOLO

    LOGGER.info("Part 4: fine-tuning %s for %d epochs", config.yolo_model, config.part4_epochs)
    model = YOLO(config.yolo_model)
    train_device: str | int = device
    if device == "cuda":
        train_device = 0
    elif device.startswith("cuda:"):
        train_device = int(device.split(":", 1)[1])
    run_root = config.artifacts_dir / "part4" / "training_runs"
    model.train(
        data=str(yaml_path),
        epochs=config.part4_epochs,
        imgsz=config.part4_image_size,
        batch=config.part4_batch,
        workers=config.part4_workers,
        device=train_device,
        amp=bool(config.use_half and device.startswith("cuda")),
        project=str(run_root),
        name="cityscapes_robust_yolov8n",
        exist_ok=True,
        pretrained=True,
        seed=config.seed,
        deterministic=True,
        plots=True,
        verbose=True,
    )
    best = run_root / "cityscapes_robust_yolov8n" / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"YOLO training completed but best.pt was not found at {best}")
    return best


def model_detections(
    image: Image.Image,
    model: Any,
    image_id: str,
    class_mapping: Mapping[int, str],
    confidence: float,
    device: str,
    use_half: bool,
) -> list[Detection]:
    result = model.predict(
        image,
        conf=confidence,
        max_det=300,
        verbose=False,
        device=device,
        half=bool(use_half and device.startswith("cuda")),
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    detections: list[Detection] = []
    for bbox, score, class_id in zip(boxes, scores, class_ids):
        class_name = class_mapping.get(int(class_id))
        if class_name is None:
            continue
        detections.append(
            Detection(
                image_id=image_id,
                class_name=class_name,
                bbox=tuple(float(value) for value in bbox),
                score=float(score),
            )
        )
    return detections


def save_fine_tuning_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(11, 6))
    distortions = sorted({str(row["distortion"]) for row in rows if row["distortion"] != "Clean"})
    for distortion in distortions:
        selected = [row for row in rows if row["distortion"] == distortion]
        selected.sort(key=lambda row: float(row["mean_snr_db"]))
        x = [float(row["mean_snr_db"]) for row in selected]
        axis.plot(x, [float(row["pretrained_map_50_95"]) for row in selected], "--o", label=f"{distortion} pretrained")
        axis.plot(x, [float(row["finetuned_map_50_95"]) for row in selected], "-s", label=f"{distortion} fine-tuned")
    axis.set_xlabel("Mean SNR (dB)")
    axis.set_ylabel("YOLO mAP@0.50:0.95")
    axis.set_title("Part 4 - pretrained versus distortion-fine-tuned YOLO")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def evaluate_fine_tuned_yolo(
    config: Parts34Config,
    pretrained: Any,
    fine_tuned: Any,
    device: str,
) -> dict[str, Any]:
    samples = discover_cityscapes_samples(
        config.dataset_root, "val", config.max_samples, config.seed
    )
    levels_by_name = config.distortion_levels or DEFAULT_DISTORTION_LEVELS
    conditions: list[tuple[str, int, float | None]] = [("Clean", 0, None)]
    for name, levels in levels_by_name.items():
        conditions.extend((name, index, float(level)) for index, level in enumerate(levels))
    summaries: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for condition, level_index, level in conditions:
        LOGGER.info("Part 4 evaluation: %s level=%s", condition, level)
        pretrained_predictions: list[Detection] = []
        finetuned_predictions: list[Detection] = []
        ground_truth: list[Detection] = []
        snrs: list[float] = []
        for sample in samples:
            clean, _, instance = load_sample(sample)
            if condition == "Clean":
                evaluation_image = clean
            else:
                seed = _stable_distortion_seed(
                    config.seed, sample.sample_id, condition, level_index
                )
                distorted = apply_aug(clean, condition, float(level), seed=seed)
                snrs.append(compute_snr(np.asarray(clean), distorted))
                evaluation_image = Image.fromarray(distorted)
            ground_truth.extend(instance_mask_to_boxes(instance, sample.sample_id))
            pretrained_predictions.extend(
                model_detections(
                    evaluation_image,
                    pretrained,
                    sample.sample_id,
                    COCO_ID_TO_SHARED_CLASS,
                    config.yolo_eval_confidence,
                    device,
                    config.use_half,
                )
            )
            finetuned_predictions.extend(
                model_detections(
                    evaluation_image,
                    fine_tuned,
                    sample.sample_id,
                    PROJECT_ID_TO_CLASS,
                    config.yolo_eval_confidence,
                    device,
                    config.use_half,
                )
            )

        pretrained_summary, pretrained_classes = evaluate_detections(
            pretrained_predictions, ground_truth
        )
        finetuned_summary, finetuned_classes = evaluate_detections(
            finetuned_predictions, ground_truth
        )
        summary = {
            "distortion": condition,
            "level_index": level_index,
            "level": level,
            "sample_count": len(samples),
            "mean_snr_db": float(np.mean(snrs)) if snrs else float("inf"),
            "pretrained_map_50_95": pretrained_summary["map_50_95"],
            "finetuned_map_50_95": finetuned_summary["map_50_95"],
            "map_50_95_gain": finetuned_summary["map_50_95"] - pretrained_summary["map_50_95"],
            "pretrained_map_50": pretrained_summary["map_50"],
            "finetuned_map_50": finetuned_summary["map_50"],
            "pretrained_recall_50": pretrained_summary["mean_recall_50"],
            "finetuned_recall_50": finetuned_summary["mean_recall_50"],
        }
        summaries.append(summary)
        for model_name, rows in (("pretrained", pretrained_classes), ("fine_tuned", finetuned_classes)):
            for row in rows:
                class_rows.append(
                    {"distortion": condition, "level": level, "model": model_name, **row}
                )

    output = config.output_dir / "part4"
    write_csv(output / "fine_tuning_summary.csv", summaries)
    write_csv(output / "detection_per_class.csv", class_rows)
    write_json(
        output / "fine_tuning_summary.json",
        {
            "scope": "Part 4 - YOLO fine-tuning on distorted images",
            "sample_count": len(samples),
            "variants": summaries,
        },
    )
    save_fine_tuning_plot(summaries, output / "figures" / "fine_tuning_per_snr.png")
    return {"sample_count": len(samples), "variants": summaries}


def run_part4(config: Parts34Config, device: str) -> dict[str, Any]:
    from ultralytics import YOLO

    weights = config.fine_tuned_weights
    dataset_root: Path | None = None
    if weights is None:
        yaml_path, dataset_root = prepare_yolo_dataset(config)
        weights = train_yolo(config, yaml_path, device)
    if not weights.is_file():
        raise FileNotFoundError(f"Fine-tuned checkpoint does not exist: {weights}")
    LOGGER.info("Evaluating fine-tuned checkpoint %s", weights)
    pretrained = YOLO(config.yolo_model)
    fine_tuned = YOLO(str(weights))
    pretrained.to(device)
    fine_tuned.to(device)
    result = evaluate_fine_tuned_yolo(config, pretrained, fine_tuned, device)
    result["weights"] = str(weights)
    result["training_dataset"] = str(dataset_root) if dataset_root is not None else None
    write_json(config.output_dir / "part4" / "run_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Cityscapes course-project Parts 3 and 4 with CUDA support."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_parts_3_4"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--part", choices=("3", "4", "both"), default="both")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--segformer-model", default=Parts34Config.segformer_model)
    parser.add_argument("--part4-train-samples", type=int, default=0)
    parser.add_argument("--part4-val-samples", type=int, default=0)
    parser.add_argument("--part4-epochs", type=int, default=20)
    parser.add_argument("--part4-image-size", type=int, default=640)
    parser.add_argument("--part4-batch", type=int, default=8)
    parser.add_argument("--part4-workers", type=int, default=4)
    parser.add_argument("--part4-clean-fraction", type=float, default=0.20)
    parser.add_argument("--rebuild-training-data", action="store_true")
    parser.add_argument("--fine-tuned-weights", type=Path)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Part 3: 4 images/2 levels. Part 4: 32 train, 8 val, 1 epoch.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    levels: Mapping[str, tuple[float, ...]] = dict(DEFAULT_DISTORTION_LEVELS)
    max_samples = args.max_samples
    train_samples = args.part4_train_samples
    val_samples = args.part4_val_samples
    epochs = args.part4_epochs
    batch = args.part4_batch
    if args.quick:
        levels = {name: values[:2] for name, values in DEFAULT_DISTORTION_LEVELS.items()}
        max_samples = min(max_samples, 4) if max_samples else 4
        train_samples = min(train_samples, 32) if train_samples else 32
        val_samples = min(val_samples, 8) if val_samples else 8
        epochs = 1
        batch = min(batch, 4)

    config = Parts34Config(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        max_samples=max_samples,
        seed=args.seed,
        device=args.device,
        use_half=not args.no_half,
        yolo_model=args.yolo_model,
        segformer_model=args.segformer_model,
        distortion_levels=levels,
        part4_train_samples=train_samples,
        part4_val_samples=val_samples,
        part4_epochs=epochs,
        part4_image_size=args.part4_image_size,
        part4_batch=batch,
        part4_workers=args.part4_workers,
        part4_clean_fraction=args.part4_clean_fraction,
        rebuild_training_data=args.rebuild_training_data,
        fine_tuned_weights=args.fine_tuned_weights,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    result: dict[str, Any] = {}
    if args.part in {"3", "both"}:
        detector, processor, segmenter, device = load_models(to_base_config(config))
        result["part3"] = run_part3(config, detector, processor, segmenter, device)
        # Part 4 training needs the full GPU. Drop the Part 3 models first when
        # both parts are run in one command, especially on an 8 GB RTX 4060.
        del detector, processor, segmenter
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
    if args.part in {"4", "both"}:
        result["part4"] = run_part4(config, device)
    write_json(
        config.output_dir / "run_manifest_parts_3_4.json",
        {"config": asdict(config), "result": result},
    )
    LOGGER.info("Parts 3/4 finished. Results are under %s", config.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
