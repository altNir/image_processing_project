"""Confirm train-only Part 3 tuning on unseen training cities and DL tasks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cityscapes_project.config import (
    COCO_ID_TO_SHARED_CLASS,
    DEFAULT_DISTORTION_LEVELS,
    ExperimentConfig,
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
from cityscapes_project.methods.detection import batched_model_detections, evaluate_detections
from cityscapes_project.methods.distortions import apply_aug, stable_distortion_seed
from cityscapes_project.methods.quality import compute_quality_metrics
from cityscapes_project.methods.restoration import restore_image_at_strength
from cityscapes_project.methods.segmentation import SegmentationAccumulator, predict_segmentation
from cityscapes_project.utils.dependencies import cv2_module
from cityscapes_project.utils.device import load_models
from cityscapes_project.utils.io import write_csv, write_json


def _select(samples: list[Any], cities: tuple[str, ...], count: int) -> list[Any]:
    available = [sample for sample in samples if sample.image_path.parent.name in cities]
    if len(available) < count:
        raise ValueError(f"Only {len(available)} samples available in {cities}")
    step = max(1, len(available) // count)
    return sorted(available[::step][:count], key=lambda sample: sample.sample_id)


def confirm(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    tuning = json.loads(args.tuning_manifest.read_text(encoding="utf-8"))
    strengths = {name: float(value) for name, value in tuning["recommendations"].items()}
    train = discover_cityscapes_samples(args.dataset_root, "train")
    samples = _select(train, tuple(args.cities), args.samples)
    config = ExperimentConfig(dataset_root=args.dataset_root, device=args.device, use_half=True)
    detector, processor, segmenter, device = load_models(config)
    rows: list[dict[str, Any]] = []

    for distortion, levels in DEFAULT_DISTORTION_LEVELS.items():
        strength = strengths[distortion]
        for level_index, level in enumerate(levels):
            base_images: list[Image.Image] = []
            selected_images: list[Image.Image] = []
            image_ids: list[str] = []
            ground_truth = []
            base_seg, selected_seg = SegmentationAccumulator(), SegmentationAccumulator()
            classical: list[dict[str, float]] = []
            for sample in samples:
                clean, label, instance = load_sample(sample)
                clean_rgb = np.asarray(clean)
                seed = stable_distortion_seed(args.seed, sample.sample_id, distortion, level_index)
                distorted = apply_aug(clean, distortion, float(level), seed=seed)
                base = restore_image_at_strength(distorted, distortion, float(level), 1.0)
                selected = base if strength == 1.0 else cv2_module().addWeighted(
                    base, strength, distorted, 1.0 - strength, 0.0
                )
                base_image, selected_image = Image.fromarray(base), Image.fromarray(selected)
                base_prediction = predict_segmentation(
                    base_image, processor, segmenter, device, True
                )
                selected_prediction = base_prediction if strength == 1.0 else predict_segmentation(
                    selected_image, processor, segmenter, device, True
                )
                base_seg.update(base_prediction, label)
                selected_seg.update(selected_prediction, label)
                clean_edges = canny_detect(clean, 100, 200, 5)
                distorted_image = Image.fromarray(distorted)
                distorted_quality = compute_quality_metrics(clean_rgb, distorted)
                selected_quality = compute_quality_metrics(clean_rgb, selected)
                classical.append({
                    "psnr_gain_db": selected_quality["psnr_db"] - distorted_quality["psnr_db"],
                    "ssim_gain": selected_quality["ssim"] - distorted_quality["ssim"],
                    "mae_reduction": distorted_quality["mae"] - selected_quality["mae"],
                    "orb_gain": measure_orb_matching(clean, selected_image, 800)["match_retention"]
                    - measure_orb_matching(clean, distorted_image, 800)["match_retention"],
                    "canny_gain": evaluate_canny_edges(
                        clean_edges, canny_detect(selected_image, 100, 200, 5), 2
                    )["f1"] - evaluate_canny_edges(
                        clean_edges, canny_detect(distorted_image, 100, 200, 5), 2
                    )["f1"],
                })
                base_images.append(base_image)
                selected_images.append(selected_image)
                image_ids.append(sample.sample_id)
                ground_truth.extend(instance_mask_to_boxes(instance, sample.sample_id))
                distorted_image.close()
                clean.close()

            base_detections = batched_model_detections(
                base_images, detector, image_ids, COCO_ID_TO_SHARED_CLASS, 0.001,
                device, True, batch=args.batch, image_size=640,
            )
            selected_detections = base_detections if strength == 1.0 else batched_model_detections(
                selected_images, detector, image_ids, COCO_ID_TO_SHARED_CLASS, 0.001,
                device, True, batch=args.batch, image_size=640,
            )
            base_detection, _ = evaluate_detections(base_detections, ground_truth)
            selected_detection, _ = evaluate_detections(selected_detections, ground_truth)
            base_segmentation, _ = base_seg.results()
            selected_segmentation, _ = selected_seg.results()
            rows.append({
                "distortion": distortion, "level_index": level_index, "level": float(level),
                "output_strength": strength, "sample_count": len(samples),
                **{name: float(np.mean([row[name] for row in classical])) for name in classical[0]},
                "base_seg_miou": base_segmentation["mean_iou"],
                "selected_seg_miou": selected_segmentation["mean_iou"],
                "selected_minus_base_seg_miou": selected_segmentation["mean_iou"] - base_segmentation["mean_iou"],
                "base_detection_map_50_95": base_detection["map_50_95"],
                "selected_detection_map_50_95": selected_detection["map_50_95"],
                "selected_minus_base_detection_map_50_95": selected_detection["map_50_95"] - base_detection["map_50_95"],
            })
            for image in base_images + selected_images:
                image.close()

    family_summary = []
    for distortion in DEFAULT_DISTORTION_LEVELS:
        family = [row for row in rows if row["distortion"] == distortion]
        family_summary.append({
            "distortion": distortion,
            "output_strength": strengths[distortion],
            "mean_seg_change_vs_v3": float(np.mean([
                row["selected_minus_base_seg_miou"] for row in family
            ])),
            "mean_detection_change_vs_v3": float(np.mean([
                row["selected_minus_base_detection_map_50_95"] for row in family
            ])),
            "mean_psnr_gain_vs_distorted": float(np.mean([row["psnr_gain_db"] for row in family])),
            "mean_ssim_gain_vs_distorted": float(np.mean([row["ssim_gain"] for row in family])),
            "mean_orb_gain_vs_distorted": float(np.mean([row["orb_gain"] for row in family])),
            "mean_canny_gain_vs_distorted": float(np.mean([row["canny_gain"] for row in family])),
        })
    accepted = all(
        row["mean_seg_change_vs_v3"] >= -args.guardrail
        and row["mean_detection_change_vs_v3"] >= -args.guardrail
        and row["mean_psnr_gain_vs_distorted"] > 0.0
        and row["mean_ssim_gain_vs_distorted"] > 0.0
        for row in family_summary
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "confirmation_by_condition.csv", rows)
    write_csv(args.output_dir / "confirmation_by_family.csv", family_summary)
    result = {
        "complete": True,
        "accepted": accepted,
        "scope": "unseen-train-city downstream confirmation",
        "development_cities": tuning["cities"],
        "confirmation_cities": args.cities,
        "sample_count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "guardrail": f"selected-minus-v3 mean segmentation and detection >= -{args.guardrail}",
        "recommendations": strengths,
        "family_summary": family_summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "confirmation_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tuning-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_part3_v4_confirmation"))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--cities", nargs="+", default=["aachen", "bochum"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--guardrail", type=float, default=0.005)
    return parser


if __name__ == "__main__":
    result = confirm(build_parser().parse_args())
    print(json.dumps({"accepted": result["accepted"], "families": result["family_summary"]}, indent=2))
