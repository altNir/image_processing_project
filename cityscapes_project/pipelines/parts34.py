"""Part 3 restoration and Part 4 robust-YOLO fine-tuning pipelines."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from cityscapes_project.config import (
    COCO_ID_TO_SHARED_CLASS,
    DEFAULT_DISTORTION_LEVELS,
    SHARED_DETECTION_CLASSES,
    Parts34Config,
)
from cityscapes_project.dataset import (
    discover_cityscapes_samples,
    instance_mask_to_boxes,
    load_sample,
)
from cityscapes_project.methods.classical import (
    canny_detect,
    evaluate_canny_edges,
    measure_orb_matching,
)
from cityscapes_project.methods.detection import (
    DETECTION_EVALUATOR_VERSION,
    evaluate_detections,
    model_detections,
    yolo_detections,
)
from cityscapes_project.methods.distortions import apply_aug, compute_snr, stable_distortion_seed
from cityscapes_project.methods.quality import compute_quality_metrics
from cityscapes_project.methods.restoration import (
    RESTORATION_METHODS,
    RESTORATION_RECIPE_VERSION,
    restoration_parameters,
    restore_image_with_metadata,
)
from cityscapes_project.methods.segmentation import (
    SegmentationAccumulator,
    compute_ious,
    predict_segmentation,
)
from cityscapes_project.types import Detection
from cityscapes_project.utils.io import write_csv, write_json
from cityscapes_project.utils.statistics import paired_bootstrap
from cityscapes_project.utils.visualization import (
    save_fine_tuning_plot,
    save_restoration_gallery,
    save_restoration_plot,
    save_restoration_quality_plot,
)

LOGGER = logging.getLogger(__name__)
PROJECT_CLASS_TO_ID = {name: index for index, name in enumerate(SHARED_DETECTION_CLASSES)}
PROJECT_ID_TO_CLASS = {index: name for name, index in PROJECT_CLASS_TO_ID.items()}


def _software_versions() -> dict[str, str]:
    """Capture the runtime versions needed to reproduce numerical outputs."""

    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("numpy", "Pillow", "opencv-python", "torch", "transformers", "ultralytics"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_matching_part2_results(
    config: Parts34Config,
    sample_ids: set[str],
    levels_by_name: Mapping[str, Sequence[float]],
) -> dict[str, Any] | None:
    """Load expensive Part 2 baselines only after exact sample/variant validation.

    Early project outputs predated completion metadata, so structural completeness
    is accepted when every expected row and key is present. Detection is deliberately
    not reused because Part 3 uses the corrected evaluator version.
    """

    if not config.reuse_part2_results:
        return None
    part2 = config.output_dir / "part2"
    required = {
        "json": part2 / "distorted_summary.json",
        "per_image": part2 / "distorted_per_image.csv",
        "summary": part2 / "distorted_summary.csv",
        "segmentation": part2 / "segmentation_per_class.csv",
    }
    if not all(path.is_file() for path in required.values()):
        return None
    metadata = json.loads(required["json"].read_text(encoding="utf-8"))
    expected_variants = sum(len(levels) for levels in levels_by_name.values())
    expected_levels = {name: [float(level) for level in levels] for name, levels in levels_by_name.items()}
    recorded_levels = metadata.get("distortion_levels")
    if (
        metadata.get("complete") is False
        or int(metadata.get("sample_count", -1)) != len(sample_ids)
        or (
            "completed_variants" in metadata
            and int(metadata["completed_variants"]) != expected_variants
        )
        or (
            recorded_levels is not None
            and {name: [float(level) for level in levels] for name, levels in recorded_levels.items()}
            != expected_levels
        )
    ):
        LOGGER.warning("Part 2 outputs exist but do not match this Part 3 configuration")
        return None

    per_image = _read_csv_rows(required["per_image"])
    if len(per_image) != len(sample_ids) * expected_variants:
        return None
    if {row["sample_id"] for row in per_image} != sample_ids:
        return None
    summary_rows = _read_csv_rows(required["summary"])
    expected_keys = {
        (name, index) for name, levels in levels_by_name.items() for index in range(len(levels))
    }
    summary_by_key = {
        (row["distortion"], int(row["level_index"])): row for row in summary_rows
    }
    if len(summary_rows) != expected_variants or set(summary_by_key) != expected_keys:
        return None
    for name, index in expected_keys:
        if float(summary_by_key[(name, index)]["level"]) != float(levels_by_name[name][index]):
            return None
    segmentation_rows = _read_csv_rows(required["segmentation"])
    segmentation_keys = {
        (row["distortion"], float(row["level"])) for row in segmentation_rows
    }
    expected_level_keys = {
        (name, float(level)) for name, levels in levels_by_name.items() for level in levels
    }
    if not expected_level_keys.issubset(segmentation_keys):
        return None
    return {
        "per_image": {
            (row["sample_id"], row["distortion"], int(row["level_index"])): row
            for row in per_image
        },
        "summary": summary_by_key,
        "segmentation": segmentation_rows,
        "provenance": {
            "source": str(part2),
            "legacy_completion_inferred": "complete" not in metadata,
            "reused_metrics": ["orb_match_retention", "canny_f1", "segmentation"],
            "recomputed_metrics": [
                "snr", "psnr", "ssim", "mae", "detection"
            ],
        },
    }


def run_part3(
    config: Parts34Config,
    detector: Any,
    processor: Any,
    segmenter: Any,
    device: str,
) -> dict[str, Any]:
    """Run a paired, severity-aware, auditable restoration evaluation."""

    samples = discover_cityscapes_samples(
        config.dataset_root, config.split, config.max_samples, config.seed
    )
    levels_by_name = config.distortion_levels or DEFAULT_DISTORTION_LEVELS
    output = config.output_dir / "part3"
    per_image: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seg_class_rows: list[dict[str, Any]] = []
    det_class_rows: list[dict[str, Any]] = []
    statistical_rows: list[dict[str, Any]] = []
    gallery: list[dict[str, Any]] = []
    total = sum(len(levels) for levels in levels_by_name.values())
    part2_cache = _load_matching_part2_results(
        config, {sample.sample_id for sample in samples}, levels_by_name
    )
    if part2_cache is not None:
        LOGGER.info(
            "Part 3 will reuse matching Part 2 distorted baselines and evaluate restored images only"
        )
    variant_number = 0

    for distortion_name, levels in levels_by_name.items():
        for level_index, level in enumerate(levels):
            variant_number += 1
            LOGGER.info(
                "Part 3 variant [%d/%d]: %s level=%s",
                variant_number, total, distortion_name, level,
            )
            seg_dist = SegmentationAccumulator() if part2_cache is None else None
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
                seed = stable_distortion_seed(
                    config.seed, sample.sample_id, distortion_name, level_index
                )
                distorted_rgb = apply_aug(clean_image, distortion_name, float(level), seed=seed)
                restoration_started = time.perf_counter()
                restored_rgb, restoration_metadata = restore_image_with_metadata(
                    distorted_rgb, distortion_name, float(level)
                )
                restoration_ms = 1000.0 * (time.perf_counter() - restoration_started)
                distorted = Image.fromarray(distorted_rgb)
                restored = Image.fromarray(restored_rgb)

                distorted_quality = compute_quality_metrics(clean_rgb, distorted_rgb)
                restored_quality = compute_quality_metrics(clean_rgb, restored_rgb)

                clean_edges = canny_detect(
                    clean_image, config.canny_low_threshold,
                    config.canny_high_threshold, config.canny_blur_kernel,
                )
                restored_edges = canny_detect(
                    restored, config.canny_low_threshold,
                    config.canny_high_threshold, config.canny_blur_kernel,
                )
                canny_rest = evaluate_canny_edges(
                    clean_edges, restored_edges, config.canny_tolerance_radius
                )
                orb_rest = measure_orb_matching(clean_image, restored, config.nfeatures)

                segmentation_rest = predict_segmentation(
                    restored, processor, segmenter, device, config.use_half
                )
                seg_rest.update(segmentation_rest, label)

                detections_rest = yolo_detections(
                    restored, detector, sample.sample_id,
                    config.yolo_eval_confidence, device, config.use_half,
                )
                pred_rest.extend(detections_rest)
                ground_truth.extend(gt)

                # Detection is always recomputed so both conditions use evaluator
                # version 2, even when costly Part 2 segmentation/classical rows are reused.
                detections_dist = yolo_detections(
                    distorted, detector, sample.sample_id,
                    config.yolo_eval_confidence, device, config.use_half,
                )
                pred_dist.extend(detections_dist)

                rest_ious = compute_ious(segmentation_rest, label)
                if part2_cache is None:
                    distorted_edges = canny_detect(
                        distorted, config.canny_low_threshold,
                        config.canny_high_threshold, config.canny_blur_kernel,
                    )
                    canny_dist = evaluate_canny_edges(
                        clean_edges, distorted_edges, config.canny_tolerance_radius
                    )
                    orb_dist = measure_orb_matching(clean_image, distorted, config.nfeatures)
                    segmentation_dist = predict_segmentation(
                        distorted, processor, segmenter, device, config.use_half
                    )
                    assert seg_dist is not None
                    seg_dist.update(segmentation_dist, label)
                    dist_ious = compute_ious(segmentation_dist, label)
                    distorted_values = {
                        "orb_distorted": orb_dist["match_retention"],
                        "canny_distorted": canny_dist["f1"],
                        "seg_distorted_image_miou": (
                            float(np.mean(list(dist_ious.values()))) if dist_ious else 0.0
                        ),
                    }
                else:
                    cached = part2_cache["per_image"][(
                        sample.sample_id, distortion_name, level_index
                    )]
                    distorted_values = {
                        "orb_distorted": float(cached["orb_match_retention"]),
                        "canny_distorted": float(cached["canny_f1"]),
                        "seg_distorted_image_miou": float(cached["seg_mean_iou"]),
                    }
                row = {
                    "sample_id": sample.sample_id,
                    "distortion": distortion_name,
                    "level_index": level_index,
                    "level": float(level),
                    "restoration_recipe_version": RESTORATION_RECIPE_VERSION,
                    "restoration_method": restoration_metadata["method"],
                    "restoration_parameters": json.dumps(
                        restoration_metadata["parameters"], sort_keys=True
                    ),
                    "restoration_runtime_ms": restoration_ms,
                    "distorted_snr_db": distorted_quality["snr_db"],
                    "restored_snr_db": restored_quality["snr_db"],
                    "snr_gain_db": restored_quality["snr_db"] - distorted_quality["snr_db"],
                    "distorted_psnr_db": distorted_quality["psnr_db"],
                    "restored_psnr_db": restored_quality["psnr_db"],
                    "psnr_gain_db": restored_quality["psnr_db"] - distorted_quality["psnr_db"],
                    "distorted_ssim": distorted_quality["ssim"],
                    "restored_ssim": restored_quality["ssim"],
                    "ssim_gain": restored_quality["ssim"] - distorted_quality["ssim"],
                    "distorted_mae": distorted_quality["mae"],
                    "restored_mae": restored_quality["mae"],
                    "mae_reduction": distorted_quality["mae"] - restored_quality["mae"],
                    "orb_restored": orb_rest["match_retention"],
                    "canny_restored": canny_rest["f1"],
                    "seg_restored_image_miou": float(np.mean(list(rest_ious.values()))) if rest_ious else 0.0,
                    "detections_restored": len(detections_rest),
                    "detections_distorted": len(detections_dist),
                    **distorted_values,
                }
                per_image.append(row)
                variant_rows.append(row)

                if config.gallery_samples > 0 and image_index == 1 and level_index == len(levels) // 2:
                    gallery.append({
                        "distortion": distortion_name,
                        "level": float(level),
                        "distorted_snr": row["distorted_snr_db"],
                        "restored_snr": row["restored_snr_db"],
                        "distorted_psnr": row["distorted_psnr_db"],
                        "restored_psnr": row["restored_psnr_db"],
                        "distorted_ssim": row["distorted_ssim"],
                        "restored_ssim": row["restored_ssim"],
                        "clean": clean_rgb,
                        "distorted": distorted_rgb,
                        "restored": restored_rgb,
                    })

            seg_rest_summary, seg_rest_classes = seg_rest.results()
            det_rest_summary, det_rest_classes = evaluate_detections(pred_rest, ground_truth)
            det_dist_summary, det_dist_classes = evaluate_detections(pred_dist, ground_truth)
            if part2_cache is None:
                assert seg_dist is not None
                seg_dist_summary, seg_dist_classes = seg_dist.results()
            else:
                cached_summary = part2_cache["summary"][(distortion_name, level_index)]
                seg_dist_summary = {"mean_iou": float(cached_summary["seg_mean_iou"])}
                seg_dist_classes = [
                    {key: value for key, value in row.items() if key not in {"distortion", "level"}}
                    for row in part2_cache["segmentation"]
                    if row["distortion"] == distortion_name
                    and float(row["level"]) == float(level)
                ]
            summary = {
                "distortion": distortion_name,
                "level_index": level_index,
                "level": float(level),
                "sample_count": len(variant_rows),
                "restoration_recipe_version": RESTORATION_RECIPE_VERSION,
                "restoration_method": restoration_metadata["method"],
                "restoration_parameters": json.dumps(
                    restoration_metadata["parameters"], sort_keys=True
                ),
                "mean_restoration_runtime_ms": _mean(variant_rows, "restoration_runtime_ms"),
                "distorted_mean_snr_db": _mean(variant_rows, "distorted_snr_db"),
                "restored_mean_snr_db": _mean(variant_rows, "restored_snr_db"),
                "snr_gain_db": _mean(variant_rows, "restored_snr_db")
                - _mean(variant_rows, "distorted_snr_db"),
                "distorted_mean_psnr_db": _mean(variant_rows, "distorted_psnr_db"),
                "restored_mean_psnr_db": _mean(variant_rows, "restored_psnr_db"),
                "psnr_gain_db": _mean(variant_rows, "restored_psnr_db")
                - _mean(variant_rows, "distorted_psnr_db"),
                "distorted_mean_ssim": _mean(variant_rows, "distorted_ssim"),
                "restored_mean_ssim": _mean(variant_rows, "restored_ssim"),
                "ssim_gain": _mean(variant_rows, "restored_ssim") - _mean(variant_rows, "distorted_ssim"),
                "distorted_mean_mae": _mean(variant_rows, "distorted_mae"),
                "restored_mean_mae": _mean(variant_rows, "restored_mae"),
                "mae_reduction": _mean(variant_rows, "distorted_mae") - _mean(variant_rows, "restored_mae"),
                "orb_distorted": _mean(variant_rows, "orb_distorted"),
                "orb_restored": _mean(variant_rows, "orb_restored"),
                "orb_gain": _mean(variant_rows, "orb_restored") - _mean(variant_rows, "orb_distorted"),
                "canny_distorted": _mean(variant_rows, "canny_distorted"),
                "canny_restored": _mean(variant_rows, "canny_restored"),
                "canny_gain": _mean(variant_rows, "canny_restored") - _mean(variant_rows, "canny_distorted"),
                "seg_distorted": seg_dist_summary["mean_iou"],
                "seg_restored": seg_rest_summary["mean_iou"],
                "seg_gain": seg_rest_summary["mean_iou"] - seg_dist_summary["mean_iou"],
                "det_distorted": det_dist_summary["map_50_95"],
                "det_restored": det_rest_summary["map_50_95"],
                "det_gain": det_rest_summary["map_50_95"] - det_dist_summary["map_50_95"],
            }
            summaries.append(summary)

            statistical_metrics = (
                ("snr_db", "distorted_snr_db", "restored_snr_db", True),
                ("psnr_db", "distorted_psnr_db", "restored_psnr_db", True),
                ("ssim", "distorted_ssim", "restored_ssim", True),
                ("mae", "distorted_mae", "restored_mae", False),
                ("orb_match_retention", "orb_distorted", "orb_restored", True),
                ("canny_f1", "canny_distorted", "canny_restored", True),
                (
                    "segmentation_image_miou",
                    "seg_distorted_image_miou",
                    "seg_restored_image_miou",
                    True,
                ),
            )
            for metric_index, (
                metric, before_key, after_key, higher_is_better
            ) in enumerate(statistical_metrics):
                statistics = paired_bootstrap(
                    [float(row[before_key]) for row in variant_rows],
                    [float(row[after_key]) for row in variant_rows],
                    higher_is_better=higher_is_better,
                    resamples=config.part3_bootstrap_resamples,
                    confidence_level=config.part3_confidence_level,
                    seed=config.seed + variant_number * 100 + metric_index,
                )
                statistical_rows.append({
                    "distortion": distortion_name,
                    "level_index": level_index,
                    "level": float(level),
                    "metric": metric,
                    "confidence_level": config.part3_confidence_level,
                    "bootstrap_resamples": config.part3_bootstrap_resamples,
                    **statistics,
                })
            for condition, rows in (("distorted", seg_dist_classes), ("restored", seg_rest_classes)):
                seg_class_rows.extend(
                    {"distortion": distortion_name, "level": float(level), "condition": condition, **row}
                    for row in rows
                )
            for condition, rows in (("distorted", det_dist_classes), ("restored", det_rest_classes)):
                det_class_rows.extend(
                    {"distortion": distortion_name, "level": float(level), "condition": condition, **row}
                    for row in rows
                )

            # Checkpoint completed variants because Part 3 is the longest CPU-heavy stage.
            write_csv(output / "restoration_per_image.csv", per_image)
            write_csv(output / "restoration_summary.csv", summaries)
            write_csv(output / "segmentation_per_class.csv", seg_class_rows)
            write_csv(output / "detection_per_class.csv", det_class_rows)
            write_csv(output / "paired_statistics.csv", statistical_rows)
            write_json(output / "restoration_summary.json", {
                "scope": "Part 3 - restored images",
                "complete": False,
                "completed_variants": len(summaries),
                "total_variants": total,
                "sample_count": len(samples),
                "reused_part2_results": part2_cache is not None,
                "part2_cache_provenance": part2_cache["provenance"] if part2_cache else None,
                "restoration_recipe_version": RESTORATION_RECIPE_VERSION,
                "detection_evaluator_version": DETECTION_EVALUATOR_VERSION,
                "statistical_protocol": {
                    "design": "paired by Cityscapes sample",
                    "bootstrap_resamples": config.part3_bootstrap_resamples,
                    "confidence_level": config.part3_confidence_level,
                },
                "distortion_levels": levels_by_name,
                "variants": summaries,
            })

    write_csv(output / "restoration_per_image.csv", per_image)
    write_csv(output / "restoration_summary.csv", summaries)
    write_csv(output / "segmentation_per_class.csv", seg_class_rows)
    write_csv(output / "detection_per_class.csv", det_class_rows)
    write_csv(output / "paired_statistics.csv", statistical_rows)
    write_json(output / "restoration_summary.json", {
        "scope": "Part 3 - restored images",
        "complete": True,
        "completed_variants": len(summaries),
        "total_variants": total,
        "sample_count": len(samples),
        "reused_part2_results": part2_cache is not None,
        "part2_cache_provenance": part2_cache["provenance"] if part2_cache else None,
        "restoration_recipe_version": RESTORATION_RECIPE_VERSION,
        "detection_evaluator_version": DETECTION_EVALUATOR_VERSION,
        "statistical_protocol": {
            "design": "paired by Cityscapes sample",
            "bootstrap_resamples": config.part3_bootstrap_resamples,
            "confidence_level": config.part3_confidence_level,
        },
        "distortion_levels": levels_by_name,
        "variants": summaries,
    })
    write_json(output / "restoration_manifest.json", {
        "scope": "Part 3 restoration methodology and reproducibility manifest",
        "recipe_version": RESTORATION_RECIPE_VERSION,
        "detection_evaluator_version": DETECTION_EVALUATOR_VERSION,
        "dataset_root": str(config.dataset_root.resolve()),
        "seed": config.seed,
        "split": config.split,
        "sample_count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "distortion_levels": levels_by_name,
        "distortion_seed_rule": "FNV-1a seed from run seed, sample ID, distortion, and level index",
        "models": {
            "detection": config.yolo_model,
            "segmentation": config.segformer_model,
            "yolo_evaluation_confidence": config.yolo_eval_confidence,
        },
        "execution": {
            "device": device,
            "half_precision_requested": config.use_half,
            "software_versions": _software_versions(),
        },
        "methods": RESTORATION_METHODS,
        "parameters_by_variant": [
            {
                "distortion": name,
                "level_index": index,
                "level": float(level),
                "parameters": restoration_parameters(name, float(level)),
            }
            for name, levels in levels_by_name.items()
            for index, level in enumerate(levels)
        ],
        "part2_cache_provenance": part2_cache["provenance"] if part2_cache else None,
        "statistical_protocol": {
            "paired_unit": "Cityscapes image",
            "interval": "percentile bootstrap of paired mean improvement",
            "bootstrap_resamples": config.part3_bootstrap_resamples,
            "confidence_level": config.part3_confidence_level,
            "non_finite_pairs": "excluded per metric",
            "positive_delta": "improvement (MAE sign is inverted)",
        },
        "quality_metric_protocol": {
            "snr": "RGB signal power divided by RGB error power, in dB",
            "psnr": "RGB MSE with 8-bit peak value, in dB",
            "ssim": "luminance SSIM, 11x11 Gaussian window, sigma 1.5",
            "mae": "mean absolute RGB error in 0-255 intensity units",
        },
    })
    save_restoration_gallery(gallery, output / "figures" / "restoration_grid.png")
    save_restoration_plot(summaries, output / "figures" / "restored_performance.png")
    save_restoration_quality_plot(summaries, output / "figures" / "restoration_quality.png")
    return {
        "sample_count": len(samples),
        "reused_part2_results": part2_cache is not None,
        "restoration_recipe_version": RESTORATION_RECIPE_VERSION,
        "detection_evaluator_version": DETECTION_EVALUATOR_VERSION,
        "variants": summaries,
    }


def detection_to_yolo_row(detection: Detection, width: int, height: int) -> str:
    """Convert an xyxy detection to one normalized YOLO label row."""

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
    recipe = json.dumps({
        "recipe_version": 2,
        "train": train,
        "val": val,
        "seed": config.seed,
        "clean_fraction": config.part4_clean_fraction,
        "distortion_levels": config.distortion_levels,
    }, sort_keys=True)
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
    class_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
    }
    condition_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
    }
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
                index, sample.sample_id,
                config.seed + (0 if split == "train" else 100_000),
                levels_by_name, config.part4_clean_fraction,
            )
            if condition == "Clean":
                output_rgb = np.asarray(image)
                snr = float("inf")
            else:
                distortion_seed = stable_distortion_seed(
                    config.seed, sample.sample_id, condition, level_index
                )
                output_rgb = apply_aug(image, condition, float(level), seed=distortion_seed)
                snr = compute_snr(np.asarray(image), output_rgb)
            # PNG avoids silently adding JPEG artifacts to clean and non-JPEG
            # training conditions. The prepared dataset is cached between runs.
            output_image = image_dir / f"{sample.sample_id}.png"
            Image.fromarray(output_rgb).save(output_image, compress_level=3)
            boxes = instance_mask_to_boxes(instance, sample.sample_id)
            condition_counts[split][condition] += 1
            class_counts[split].update(item.class_name for item in boxes)
            label_text = "\n".join(
                detection_to_yolo_row(item, image.width, image.height) for item in boxes
            )
            (label_dir / f"{sample.sample_id}.txt").write_text(
                label_text + ("\n" if label_text else ""), encoding="utf-8"
            )
            rows.append({
                "split": split,
                "sample_id": sample.sample_id,
                "condition": condition,
                "level": level,
                "snr_db": snr,
                "objects": len(boxes),
            })

    missing_train_classes = [
        name for name in SHARED_DETECTION_CLASSES if class_counts["train"][name] == 0
    ]
    if missing_train_classes:
        LOGGER.warning(
            "Part 4 training subset has no instances for: %s. "
            "Use the full training split for final results.",
            ", ".join(missing_train_classes),
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
    write_json(manifest_path, {
        "complete": True,
        "recipe_version": 2,
        "dataset_root": config.dataset_root,
        "seed": config.seed,
        "clean_fraction": config.part4_clean_fraction,
        "distortion_levels": levels_by_name,
        "samples": len(rows),
        "condition_counts": {
            split: dict(sorted(counts.items())) for split, counts in condition_counts.items()
        },
        "class_instance_counts": {
            split: {name: int(counts[name]) for name in SHARED_DETECTION_CLASSES}
            for split, counts in class_counts.items()
        },
        "missing_train_classes": missing_train_classes,
    })
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
    run_root = (config.artifacts_dir / "part4" / "training_runs").resolve()
    run_name = _training_dataset_key(config)
    model.train(
        data=str(yaml_path), epochs=config.part4_epochs, imgsz=config.part4_image_size,
        batch=config.part4_batch, workers=config.part4_workers, device=train_device,
        amp=bool(config.use_half and device.startswith("cuda")),
        project=str(run_root), name=run_name, exist_ok=True, pretrained=True,
        seed=config.seed, deterministic=True, plots=True, verbose=True,
        warmup_epochs=min(3.0, max(0.5, 0.10 * config.part4_epochs)),
        close_mosaic=min(5, max(1, config.part4_epochs // 5)),
    )
    trainer = getattr(model, "trainer", None)
    trainer_best = getattr(trainer, "best", None)
    best = Path(trainer_best) if trainer_best is not None else run_root / run_name / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"YOLO training completed but best.pt was not found at {best}")
    return best.resolve()


def evaluate_fine_tuned_yolo(
    config: Parts34Config,
    pretrained: Any,
    fine_tuned: Any,
    device: str,
) -> dict[str, Any]:
    """Compare pretrained and fine-tuned YOLO under every condition."""

    samples = discover_cityscapes_samples(config.dataset_root, "val", config.max_samples, config.seed)
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
                seed = stable_distortion_seed(
                    config.seed, sample.sample_id, condition, level_index
                )
                distorted = apply_aug(clean, condition, float(level), seed=seed)
                snrs.append(compute_snr(np.asarray(clean), distorted))
                evaluation_image = Image.fromarray(distorted)
            ground_truth.extend(instance_mask_to_boxes(instance, sample.sample_id))
            pretrained_predictions.extend(model_detections(
                evaluation_image, pretrained, sample.sample_id, COCO_ID_TO_SHARED_CLASS,
                config.yolo_eval_confidence, device, config.use_half,
            ))
            finetuned_predictions.extend(model_detections(
                evaluation_image, fine_tuned, sample.sample_id, PROJECT_ID_TO_CLASS,
                config.yolo_eval_confidence, device, config.use_half,
            ))

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
            "operating_confidence": pretrained_summary["operating_confidence"],
            "pretrained_precision_50_at_operating_confidence": pretrained_summary[
                "mean_precision_50_at_operating_confidence"
            ],
            "finetuned_precision_50_at_operating_confidence": finetuned_summary[
                "mean_precision_50_at_operating_confidence"
            ],
            "pretrained_recall_50_at_operating_confidence": pretrained_summary[
                "mean_recall_50_at_operating_confidence"
            ],
            "finetuned_recall_50_at_operating_confidence": finetuned_summary[
                "mean_recall_50_at_operating_confidence"
            ],
            "pretrained_f1_50_at_operating_confidence": pretrained_summary[
                "mean_f1_50_at_operating_confidence"
            ],
            "finetuned_f1_50_at_operating_confidence": finetuned_summary[
                "mean_f1_50_at_operating_confidence"
            ],
        }
        summaries.append(summary)
        for model_name, rows in (("pretrained", pretrained_classes), ("fine_tuned", finetuned_classes)):
            class_rows.extend(
                {"distortion": condition, "level": level, "model": model_name, **row}
                for row in rows
            )

        write_csv(config.output_dir / "part4" / "fine_tuning_summary.csv", summaries)
        write_csv(config.output_dir / "part4" / "detection_per_class.csv", class_rows)
        write_json(config.output_dir / "part4" / "fine_tuning_summary.json", {
            "scope": "Part 4 - YOLO fine-tuning on distorted images",
            "detection_evaluator_version": DETECTION_EVALUATOR_VERSION,
            "complete": False,
            "completed_conditions": len(summaries),
            "total_conditions": len(conditions),
            "sample_count": len(samples),
            "variants": summaries,
        })

    output = config.output_dir / "part4"
    write_csv(output / "fine_tuning_summary.csv", summaries)
    write_csv(output / "detection_per_class.csv", class_rows)
    write_json(output / "fine_tuning_summary.json", {
        "scope": "Part 4 - YOLO fine-tuning on distorted images",
        "detection_evaluator_version": DETECTION_EVALUATOR_VERSION,
        "complete": True,
        "completed_conditions": len(summaries),
        "total_conditions": len(conditions),
        "sample_count": len(samples),
        "variants": summaries,
    })
    save_fine_tuning_plot(summaries, output / "figures" / "fine_tuning_per_snr.png")
    return {
        "sample_count": len(samples),
        "detection_evaluator_version": DETECTION_EVALUATOR_VERSION,
        "variants": summaries,
    }


def run_part4(config: Parts34Config, device: str) -> dict[str, Any]:
    """Prepare/train when needed, then evaluate the robust detector."""

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


_stable_distortion_seed = stable_distortion_seed
