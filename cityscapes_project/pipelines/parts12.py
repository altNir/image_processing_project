"""Part 1 clean baselines and Part 2 distortion robustness pipeline."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image

from cityscapes_project.config import DEFAULT_DISTORTION_LEVELS, ExperimentConfig
from cityscapes_project.dataset import (
    discover_cityscapes_samples,
    instance_mask_to_boxes,
    load_image_and_label,
    load_sample,
)
from cityscapes_project.methods.classical import (
    canny_detect,
    canny_overlay,
    evaluate_canny_edges,
    measure_orb_matching,
    orb_overlay,
)
from cityscapes_project.methods.detection import (
    DETECTION_EVALUATOR_VERSION,
    evaluate_detections,
    yolo_detections,
    yolo_overlay,
)
from cityscapes_project.methods.distortions import apply_aug, compute_snr, stable_distortion_seed
from cityscapes_project.methods.segmentation import (
    SegmentationAccumulator,
    compute_ious,
    predict_segmentation,
)
from cityscapes_project.types import CityscapesSample, Detection
from cityscapes_project.utils.device import load_models
from cityscapes_project.utils.io import write_csv, write_json
from cityscapes_project.utils.visualization import (
    overlay_mask,
    save_distortion_grid,
    save_part1_gallery,
    save_part2_gallery,
    save_performance_snr_plot,
    seg_overlay,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class CleanReference:
    """Compact clean-image state reused by every Part 2 variant."""

    sample: CityscapesSample
    gt_detections: list[Detection]
    clean_edges_packed: np.ndarray
    clean_edges_shape: tuple[int, int]


def pack_binary_map(binary_map: np.ndarray) -> np.ndarray:
    """Pack a binary image to one bit per pixel for low-memory references."""

    return np.packbits((np.asarray(binary_map) > 0).reshape(-1))


def unpack_binary_map(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Restore a packed binary image as an OpenCV-style 0/255 uint8 map."""

    count = int(shape[0] * shape[1])
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), count=count)
    return (bits.reshape(shape) * 255).astype(np.uint8)


def run_part1(
    config: ExperimentConfig,
    samples: Sequence[CityscapesSample],
    detector: Any,
    processor: Any,
    segmenter: Any,
    device: str,
) -> tuple[list[CleanReference], dict[str, Any]]:
    """Run and evaluate all methods on clean Cityscapes images."""

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
            image, detector, sample.sample_id, conf=config.yolo_eval_confidence,
            device=device, use_half=config.use_half,
        )
        segmentation = predict_segmentation(
            image, processor, segmenter, device, use_half=config.use_half
        )
        segmentation_accumulator.update(segmentation, label)
        image_ious = compute_ious(segmentation, label)
        clean_orb = measure_orb_matching(
            image, image, nfeatures=config.nfeatures,
            ratio_threshold=config.orb_ratio_threshold,
            spatial_threshold=config.orb_spatial_threshold,
        )
        clean_edges = canny_detect(
            image, low_threshold=config.canny_low_threshold,
            high_threshold=config.canny_high_threshold,
            blur_kernel=config.canny_blur_kernel,
        )
        clean_canny = evaluate_canny_edges(
            clean_edges, clean_edges, tolerance_radius=config.canny_tolerance_radius
        )

        all_predictions.extend(predictions)
        all_ground_truth.extend(gt_detections)
        per_image_rows.append({
            "sample_id": sample.sample_id,
            "orb_keypoints": clean_orb["clean_keypoints"],
            "orb_self_match_retention": clean_orb["match_retention"],
            "canny_edge_pixels": clean_canny["reference_edge_pixels"],
            "canny_self_f1": clean_canny["f1"],
            "seg_mean_iou": float(np.mean(list(image_ious.values()))) if image_ious else 0.0,
            "seg_classes_present": len(image_ious),
            "gt_detection_objects": len(gt_detections),
            "yolo_detection_objects": len(predictions),
        })
        references.append(CleanReference(
            sample=sample,
            gt_detections=gt_detections,
            clean_edges_packed=pack_binary_map(clean_edges),
            clean_edges_shape=clean_edges.shape,
        ))

        if len(gallery) < config.gallery_samples:
            orb_image, keypoints, _ = orb_overlay(image, nfeatures=config.nfeatures)
            canny_image = canny_overlay(image, clean_edges)
            yolo_image, result = yolo_overlay(
                image, detector, conf=config.yolo_visual_confidence,
                device=device, use_half=config.use_half,
            )
            gallery.append({
                "sample_id": sample.sample_id,
                "clean": image,
                "ground_truth": overlay_mask(image, label),
                "orb": orb_image,
                "canny": canny_image,
                "yolo": yolo_image,
                "segmentation": seg_overlay(np.asarray(image), segmentation),
                "orb_keypoints": len(keypoints),
                "yolo_results": result,
            })

    segmentation_summary, segmentation_per_class = segmentation_accumulator.results()
    detection_summary, detection_per_class = evaluate_detections(all_predictions, all_ground_truth)
    summary = {
        "scope": "Part 1 - clean images",
        "detection_evaluator_version": DETECTION_EVALUATOR_VERSION,
        "sample_count": len(samples),
        "split": config.split,
        "segmentation": segmentation_summary,
        "detection": detection_summary,
        "orb": {
            "mean_keypoints": float(np.mean([row["orb_keypoints"] for row in per_image_rows])),
            "mean_self_match_retention": float(np.mean([
                row["orb_self_match_retention"] for row in per_image_rows
            ])),
        },
        "canny": {
            "mean_edge_pixels": float(np.mean([row["canny_edge_pixels"] for row in per_image_rows])),
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
    """Evaluate every method at every configured distortion level."""

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
            first_image, distortion_levels,
            output_dir / "figures" / "distortion_grid.png", config.seed,
        )

    total_variants = sum(len(levels) for levels in distortion_levels.values())
    variant_index = 0
    for distortion_name, levels in distortion_levels.items():
        for level_index, level in enumerate(levels):
            variant_index += 1
            LOGGER.info(
                "Part 2 variant [%d/%d]: %s level=%s",
                variant_index, total_variants, distortion_name, level,
            )
            segmentation_accumulator = SegmentationAccumulator()
            predictions: list[Detection] = []
            ground_truth: list[Detection] = []
            variant_rows: list[dict[str, Any]] = []

            for image_index, reference in enumerate(references, 1):
                LOGGER.info(
                    "  image [%d/%d] %s", image_index, len(references), reference.sample.sample_id
                )
                clean_image, clean_label = load_image_and_label(reference.sample)
                distortion_seed = stable_distortion_seed(
                    config.seed, reference.sample.sample_id, distortion_name, level_index
                )
                distorted_rgb = apply_aug(
                    clean_image, distortion_name, float(level), seed=distortion_seed
                )
                distorted_image = Image.fromarray(distorted_rgb)
                snr_db = compute_snr(np.asarray(clean_image), distorted_rgb)
                orb_metrics = measure_orb_matching(
                    clean_image, distorted_image, nfeatures=config.nfeatures,
                    ratio_threshold=config.orb_ratio_threshold,
                    spatial_threshold=config.orb_spatial_threshold,
                )
                distorted_edges = canny_detect(
                    distorted_image, low_threshold=config.canny_low_threshold,
                    high_threshold=config.canny_high_threshold,
                    blur_kernel=config.canny_blur_kernel,
                )
                clean_edges = unpack_binary_map(
                    reference.clean_edges_packed, reference.clean_edges_shape
                )
                canny_metrics = evaluate_canny_edges(
                    clean_edges, distorted_edges, tolerance_radius=config.canny_tolerance_radius
                )
                segmentation = predict_segmentation(
                    distorted_image, processor, segmenter, device, use_half=config.use_half
                )
                segmentation_accumulator.update(segmentation, clean_label)
                image_ious = compute_ious(segmentation, clean_label)
                image_predictions = yolo_detections(
                    distorted_image, detector, reference.sample.sample_id,
                    conf=config.yolo_eval_confidence, device=device, use_half=config.use_half,
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

                if config.gallery_samples > 0 and image_index == 1 and level_index == len(levels) // 2:
                    orb_image, _, _ = orb_overlay(distorted_image, nfeatures=config.nfeatures)
                    canny_image = canny_overlay(distorted_image, distorted_edges)
                    yolo_image, _ = yolo_overlay(
                        distorted_image, detector, conf=config.yolo_visual_confidence,
                        device=device, use_half=config.use_half,
                    )
                    gallery_rows.append({
                        "distortion": distortion_name,
                        "level": float(level),
                        "snr_db": snr_db,
                        "distorted": distorted_rgb,
                        "orb": orb_image,
                        "canny": canny_image,
                        "yolo": yolo_image,
                        "segmentation": seg_overlay(distorted_rgb, segmentation),
                    })

            segmentation_summary, segmentation_per_class = segmentation_accumulator.results()
            detection_summary, detection_per_class = evaluate_detections(predictions, ground_truth)
            finite_snrs = [
                float(row["snr_db"]) for row in variant_rows if math.isfinite(float(row["snr_db"]))
            ]
            summary_row = {
                "distortion": distortion_name,
                "level_index": level_index,
                "level": float(level),
                "sample_count": len(variant_rows),
                "mean_snr_db": float(np.mean(finite_snrs)) if finite_snrs else float("inf"),
                "orb_keypoint_retention": float(np.mean([float(row["orb_keypoint_retention"]) for row in variant_rows])),
                "orb_match_retention": float(np.mean([float(row["orb_match_retention"]) for row in variant_rows])),
                "orb_inlier_ratio": float(np.mean([float(row["orb_inlier_ratio"]) for row in variant_rows])),
                "canny_edge_pixel_retention": float(np.mean([float(row["canny_edge_pixel_retention"]) for row in variant_rows])),
                "canny_precision": float(np.mean([float(row["canny_precision"]) for row in variant_rows])),
                "canny_recall": float(np.mean([float(row["canny_recall"]) for row in variant_rows])),
                "canny_f1": float(np.mean([float(row["canny_f1"]) for row in variant_rows])),
                "seg_mean_iou": segmentation_summary["mean_iou"],
                "seg_pixel_accuracy": segmentation_summary["pixel_accuracy"],
                "det_map_50_95": detection_summary["map_50_95"],
                "det_map_50": detection_summary["map_50"],
                "det_precision_50": detection_summary["mean_precision_50"],
                "det_recall_50": detection_summary["mean_recall_50"],
                "det_operating_confidence": detection_summary["operating_confidence"],
                "det_precision_50_at_operating_confidence": detection_summary[
                    "mean_precision_50_at_operating_confidence"
                ],
                "det_recall_50_at_operating_confidence": detection_summary[
                    "mean_recall_50_at_operating_confidence"
                ],
                "det_f1_50_at_operating_confidence": detection_summary[
                    "mean_f1_50_at_operating_confidence"
                ],
                "det_mean_matched_iou_50": detection_summary["mean_matched_iou_50"],
            }
            summary_rows.append(summary_row)
            segmentation_class_rows.extend(
                {"distortion": distortion_name, "level": float(level), **row}
                for row in segmentation_per_class
            )
            detection_class_rows.extend(
                {"distortion": distortion_name, "level": float(level), **row}
                for row in detection_per_class
            )

            # Persist every completed severity variant. A multi-hour run can be
            # inspected or recovered even if a later model call is interrupted.
            write_csv(output_dir / "distorted_per_image.csv", per_image_rows)
            write_csv(output_dir / "distorted_summary.csv", summary_rows)
            write_csv(output_dir / "segmentation_per_class.csv", segmentation_class_rows)
            write_csv(output_dir / "detection_per_class.csv", detection_class_rows)
            write_json(output_dir / "distorted_summary.json", {
                "scope": "Part 2 - distorted images",
                "complete": False,
                "completed_variants": len(summary_rows),
                "total_variants": total_variants,
                "sample_count": len(references),
                "distortion_levels": distortion_levels,
                "detection_evaluator_version": DETECTION_EVALUATOR_VERSION,
                "variants": summary_rows,
            })

    write_csv(output_dir / "distorted_per_image.csv", per_image_rows)
    write_csv(output_dir / "distorted_summary.csv", summary_rows)
    write_csv(output_dir / "segmentation_per_class.csv", segmentation_class_rows)
    write_csv(output_dir / "detection_per_class.csv", detection_class_rows)
    write_json(output_dir / "distorted_summary.json", {
        "scope": "Part 2 - distorted images",
        "complete": True,
        "completed_variants": len(summary_rows),
        "total_variants": total_variants,
        "sample_count": len(references),
        "distortion_levels": distortion_levels,
        "detection_evaluator_version": DETECTION_EVALUATOR_VERSION,
        "variants": summary_rows,
    })
    save_performance_snr_plot(summary_rows, output_dir / "figures" / "performance_per_snr.png")
    save_part2_gallery(gallery_rows, output_dir / "figures" / "distorted_predictions.png")
    return {"variants": summary_rows, "sample_count": len(references)}


def run_experiment(config: ExperimentConfig, part: str = "both") -> dict[str, Any]:
    """Run Part 1 and, when selected, Part 2 with shared loaded models."""

    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_cityscapes_samples(
        config.dataset_root, split=config.split, max_samples=config.max_samples, seed=config.seed
    )
    LOGGER.info("Using %d Cityscapes %s samples", len(samples), config.split)
    detector, processor, segmenter, device = load_models(config)
    LOGGER.info("Inference device: %s", device)
    elapsed_by_part: dict[str, float] = {}
    part_started = time.perf_counter()
    references, part1_summary = run_part1(
        config, samples, detector, processor, segmenter, device
    )
    elapsed_by_part["part1"] = time.perf_counter() - part_started
    result: dict[str, Any] = {"part1": part1_summary}
    if part in {"2", "both"}:
        part_started = time.perf_counter()
        result["part2"] = run_part2(
            config, references, detector, processor, segmenter, device
        )
        elapsed_by_part["part2"] = time.perf_counter() - part_started
    write_json(config.output_dir / "run_manifest.json", {
        "scope": "Course project Parts 1 and 2",
        "part_requested": part,
        "elapsed_seconds": time.perf_counter() - started,
        "elapsed_seconds_by_part": elapsed_by_part,
        "config": asdict(config),
        "result": result,
    })
    return result


# Compatibility with the original private helper name.
_stable_distortion_seed = stable_distortion_seed
