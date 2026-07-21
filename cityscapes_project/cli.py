"""Command-line interfaces for the unified and legacy entry points."""

from __future__ import annotations

import argparse
import logging
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


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def build_parts12_parser() -> argparse.ArgumentParser:
    """Build the backward-compatible Parts 1/2 argument parser."""

    parser = argparse.ArgumentParser(
        description="Run Cityscapes robustness experiments for Parts 1 and 2."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted Cityscapes root")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--part", choices=("1", "2", "both"), default="both")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-samples", type=int, default=0, help="0 uses the complete split")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--no-half", action="store_true", help="Disable CUDA half precision")
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
    parser.add_argument("--quick", action="store_true", help="Four images and two severity levels")
    return parser


def parts12_main(argv: Sequence[str] | None = None) -> int:
    """Run the original Parts 1/2 command-line interface."""

    _configure_logging()
    args = build_parts12_parser().parse_args(argv)
    max_samples = args.max_samples
    levels: Mapping[str, tuple[float, ...]] = dict(DEFAULT_DISTORTION_LEVELS)
    if args.quick:
        max_samples = min(max_samples, 4) if max_samples > 0 else 4
        levels = {name: values[:2] for name, values in DEFAULT_DISTORTION_LEVELS.items()}
    config = ExperimentConfig(
        dataset_root=args.dataset_root, output_dir=args.output_dir, split=args.split,
        max_samples=max_samples, seed=args.seed, device=args.device,
        use_half=not args.no_half, nfeatures=args.nfeatures,
        canny_low_threshold=args.canny_low_threshold,
        canny_high_threshold=args.canny_high_threshold,
        canny_blur_kernel=args.canny_blur_kernel,
        canny_tolerance_radius=args.canny_tolerance_radius,
        yolo_model=args.yolo_model, segformer_model=args.segformer_model,
        yolo_eval_confidence=args.yolo_eval_confidence,
        yolo_visual_confidence=args.yolo_visual_confidence,
        gallery_samples=args.gallery_samples, distortion_levels=levels,
    )
    run_experiment(config, part=args.part)
    LOGGER.info("Finished. Results are under %s", config.output_dir.resolve())
    return 0


def build_parts34_parser() -> argparse.ArgumentParser:
    """Build the backward-compatible Parts 3/4 argument parser."""

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
        "--quick", action="store_true",
        help="Part 3: 4 images/2 levels. Part 4: 32 train, 8 val, 1 epoch.",
    )
    return parser


def _parts34_config(args: argparse.Namespace) -> Parts34Config:
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
    return Parts34Config(
        dataset_root=args.dataset_root, output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir, max_samples=max_samples, seed=args.seed,
        device=args.device, use_half=not args.no_half, yolo_model=args.yolo_model,
        segformer_model=args.segformer_model, distortion_levels=levels,
        part4_train_samples=train_samples, part4_val_samples=val_samples,
        part4_epochs=epochs, part4_image_size=args.part4_image_size,
        part4_batch=batch, part4_workers=args.part4_workers,
        part4_clean_fraction=args.part4_clean_fraction,
        rebuild_training_data=args.rebuild_training_data,
        fine_tuned_weights=args.fine_tuned_weights,
    )


def _run_parts34(config: Parts34Config, part: str) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    result: dict[str, Any] = {}
    if part in {"3", "both"}:
        detector, processor, segmenter, device = load_models(to_base_config(config))
        result["part3"] = run_part3(config, detector, processor, segmenter, device)
        del detector, processor, segmenter
        if device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()
    if part in {"4", "both"}:
        result["part4"] = run_part4(config, device)
    write_json(
        config.output_dir / "run_manifest_parts_3_4.json",
        {"config": asdict(config), "result": result},
    )
    return result


def parts34_main(argv: Sequence[str] | None = None) -> int:
    """Run the original Parts 3/4 command-line interface."""

    _configure_logging()
    args = build_parts34_parser().parse_args(argv)
    config = _parts34_config(args)
    _run_parts34(config, args.part)
    LOGGER.info("Parts 3/4 finished. Results are under %s", config.output_dir.resolve())
    return 0


def build_unified_parser() -> argparse.ArgumentParser:
    """Build the small, project-level command interface used by ``main.py``."""

    parser = argparse.ArgumentParser(description="Run any Cityscapes course-project part.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--part", choices=("1", "2", "3", "4", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--segformer-model", default=ExperimentConfig.segformer_model)
    parser.add_argument("--part4-train-samples", type=int, default=0)
    parser.add_argument("--part4-val-samples", type=int, default=0)
    parser.add_argument("--part4-epochs", type=int, default=20)
    parser.add_argument("--part4-image-size", type=int, default=640)
    parser.add_argument("--part4-batch", type=int, default=8)
    parser.add_argument("--part4-workers", type=int, default=4)
    parser.add_argument("--part4-clean-fraction", type=float, default=0.20)
    parser.add_argument("--rebuild-training-data", action="store_true")
    parser.add_argument("--fine-tuned-weights", type=Path)
    return parser


def unified_main(argv: Sequence[str] | None = None) -> int:
    """Run one part or the complete pipeline from the repository root."""

    _configure_logging()
    args = build_unified_parser().parse_args(argv)
    levels: Mapping[str, tuple[float, ...]] = dict(DEFAULT_DISTORTION_LEVELS)
    max_samples = args.max_samples
    if args.quick:
        levels = {name: values[:2] for name, values in DEFAULT_DISTORTION_LEVELS.items()}
        max_samples = min(max_samples, 4) if max_samples else 4
    if args.part in {"1", "2", "all"}:
        config12 = ExperimentConfig(
            dataset_root=args.dataset_root, output_dir=args.output_dir,
            max_samples=max_samples, seed=args.seed, device=args.device,
            use_half=not args.no_half, yolo_model=args.yolo_model,
            segformer_model=args.segformer_model, distortion_levels=levels,
        )
        run_experiment(config12, "both" if args.part in {"2", "all"} else "1")
    if args.part in {"3", "4", "all"}:
        # Reuse the established Parts 3/4 configuration conversion and runner.
        args.max_samples = max_samples
        args.part4_train_samples = (
            min(args.part4_train_samples, 32) if args.quick and args.part4_train_samples else
            (32 if args.quick else args.part4_train_samples)
        )
        args.part4_val_samples = (
            min(args.part4_val_samples, 8) if args.quick and args.part4_val_samples else
            (8 if args.quick else args.part4_val_samples)
        )
        config34 = _parts34_config(args)
        config34.distortion_levels = levels
        _run_parts34(config34, "both" if args.part == "all" else args.part)
    LOGGER.info("Finished selected project part(s).")
    return 0
