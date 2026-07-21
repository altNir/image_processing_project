"""Unified command-line interface for all four project parts."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cityscapes_project.config import (
    DEFAULT_DISTORTION_LEVELS,
    ExperimentConfig,
    Parts34Config,
    to_base_config,
)
from cityscapes_project.pipelines.parts12 import run_experiment
from cityscapes_project.pipelines.parts34 import run_part3, run_part4
from cityscapes_project.utils.device import load_models, select_device
from cityscapes_project.utils.io import write_json

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the project-level command interface used by ``main.py``."""

    parser = argparse.ArgumentParser(description="Run any Cityscapes course-project part.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--part", choices=("1", "2", "3", "4", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--no-half", action="store_true", help="Disable CUDA half precision")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--nfeatures", type=int, default=800)
    parser.add_argument("--orb-ratio-threshold", type=float, default=0.75)
    parser.add_argument("--orb-spatial-threshold", type=float, default=3.0)
    parser.add_argument("--canny-low-threshold", type=int, default=100)
    parser.add_argument("--canny-high-threshold", type=int, default=200)
    parser.add_argument("--canny-blur-kernel", type=int, default=5)
    parser.add_argument("--canny-tolerance-radius", type=int, default=2)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--segformer-model", default=ExperimentConfig.segformer_model)
    parser.add_argument("--yolo-eval-confidence", type=float, default=0.001)
    parser.add_argument("--yolo-visual-confidence", type=float, default=0.25)
    parser.add_argument("--gallery-samples", type=int, default=4)
    parser.add_argument("--part4-train-samples", type=int, default=0)
    parser.add_argument("--part4-val-samples", type=int, default=0)
    parser.add_argument("--part4-epochs", type=int, default=20)
    parser.add_argument("--part4-image-size", type=int, default=640)
    parser.add_argument("--part4-batch", type=int, default=8)
    parser.add_argument("--part4-workers", type=int, default=4)
    parser.add_argument("--part4-clean-fraction", type=float, default=0.20)
    parser.add_argument("--rebuild-training-data", action="store_true")
    parser.add_argument("--part3-bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--part3-confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--no-reuse-part2",
        action="store_true",
        help="Recompute distorted Part 3 baselines instead of reusing a complete matching Part 2 run",
    )
    parser.add_argument("--fine-tuned-weights", type=Path)
    return parser


def _quick_limits(args: argparse.Namespace) -> tuple[int, Mapping[str, tuple[float, ...]]]:
    levels: Mapping[str, tuple[float, ...]] = dict(DEFAULT_DISTORTION_LEVELS)
    max_samples = args.max_samples
    if args.quick:
        levels = {name: values[:2] for name, values in DEFAULT_DISTORTION_LEVELS.items()}
        max_samples = min(max_samples, 4) if max_samples else 4
        args.part4_train_samples = min(args.part4_train_samples, 32) if args.part4_train_samples else 32
        args.part4_val_samples = min(args.part4_val_samples, 8) if args.part4_val_samples else 8
        args.part4_epochs = 1
        args.part4_batch = min(args.part4_batch, 4)
    return max_samples, levels


def _parts34_config(
    args: argparse.Namespace,
    max_samples: int,
    levels: Mapping[str, tuple[float, ...]],
) -> Parts34Config:
    return Parts34Config(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        split=args.split,
        max_samples=max_samples,
        seed=args.seed,
        device=args.device,
        use_half=not args.no_half,
        nfeatures=args.nfeatures,
        canny_low_threshold=args.canny_low_threshold,
        canny_high_threshold=args.canny_high_threshold,
        canny_blur_kernel=args.canny_blur_kernel,
        canny_tolerance_radius=args.canny_tolerance_radius,
        yolo_model=args.yolo_model,
        yolo_eval_confidence=args.yolo_eval_confidence,
        segformer_model=args.segformer_model,
        distortion_levels=levels,
        gallery_samples=args.gallery_samples,
        part4_train_samples=args.part4_train_samples,
        part4_val_samples=args.part4_val_samples,
        part4_epochs=args.part4_epochs,
        part4_image_size=args.part4_image_size,
        part4_batch=args.part4_batch,
        part4_workers=args.part4_workers,
        part4_clean_fraction=args.part4_clean_fraction,
        rebuild_training_data=args.rebuild_training_data,
        reuse_part2_results=not args.no_reuse_part2,
        part3_bootstrap_resamples=args.part3_bootstrap_resamples,
        part3_confidence_level=args.part3_confidence_level,
        fine_tuned_weights=args.fine_tuned_weights,
    )


def _run_parts34(config: Parts34Config, part: str) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    result: dict[str, Any] = {}
    elapsed: dict[str, float] = {}
    overall_started = time.perf_counter()
    if part in {"3", "all"}:
        started = time.perf_counter()
        detector, processor, segmenter, device = load_models(to_base_config(config))
        result["part3"] = run_part3(config, detector, processor, segmenter, device)
        elapsed["part3"] = time.perf_counter() - started
        del detector, processor, segmenter
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
    if part in {"4", "all"}:
        started = time.perf_counter()
        result["part4"] = run_part4(config, device)
        elapsed["part4"] = time.perf_counter() - started
    write_json(
        config.output_dir / "run_manifest_parts_3_4.json",
        {
            "elapsed_seconds": time.perf_counter() - overall_started,
            "elapsed_seconds_by_part": elapsed,
            "config": asdict(config),
            "result": result,
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run one part or the complete pipeline from the repository root."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    max_samples, levels = _quick_limits(args)

    if args.part in {"1", "2", "all"}:
        config12 = ExperimentConfig(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            split=args.split,
            max_samples=max_samples,
            seed=args.seed,
            device=args.device,
            use_half=not args.no_half,
            nfeatures=args.nfeatures,
            orb_ratio_threshold=args.orb_ratio_threshold,
            orb_spatial_threshold=args.orb_spatial_threshold,
            canny_low_threshold=args.canny_low_threshold,
            canny_high_threshold=args.canny_high_threshold,
            canny_blur_kernel=args.canny_blur_kernel,
            canny_tolerance_radius=args.canny_tolerance_radius,
            yolo_model=args.yolo_model,
            yolo_eval_confidence=args.yolo_eval_confidence,
            yolo_visual_confidence=args.yolo_visual_confidence,
            segformer_model=args.segformer_model,
            distortion_levels=levels,
            gallery_samples=args.gallery_samples,
        )
        run_experiment(config12, "both" if args.part in {"2", "all"} else "1")

    if args.part in {"3", "4", "all"}:
        _run_parts34(_parts34_config(args, max_samples, levels), args.part)

    LOGGER.info("Finished selected project part(s).")
    return 0
