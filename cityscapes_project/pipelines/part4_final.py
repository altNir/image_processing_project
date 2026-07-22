"""Leakage-free multi-seed Part 4 protocol used for the final report."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, replace
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
from cityscapes_project.dataset import discover_cityscapes_samples, instance_mask_to_boxes, load_sample
from cityscapes_project.methods.detection import (
    DETECTION_EVALUATOR_VERSION,
    batched_model_detections,
    evaluate_detections,
)
from cityscapes_project.methods.distortions import apply_aug, compute_snr, stable_distortion_seed
from cityscapes_project.pipelines.parts34 import (
    PROJECT_ID_TO_CLASS,
    _software_versions,
    prepare_yolo_dataset,
    train_yolo,
)
from cityscapes_project.utils.io import write_csv, write_json
from cityscapes_project.utils.statistics import paired_bootstrap

LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _conditions(
    levels: Mapping[str, Sequence[float]],
) -> list[tuple[str, int, float | None]]:
    rows: list[tuple[str, int, float | None]] = [("Clean", 0, None)]
    for name, values in levels.items():
        rows.extend((name, index, float(value)) for index, value in enumerate(values))
    return rows


def train_final_models(
    config: Parts34Config,
    device: str,
    seeds: Sequence[int],
) -> tuple[Path, dict[int, Path], dict[int, float]]:
    """Prepare one common dataset and train independent initialization/order seeds."""

    started = time.perf_counter()
    yaml_path, prepared_root = prepare_yolo_dataset(config)
    LOGGER.info("Prepared final Part 4 data in %.1f minutes", (time.perf_counter() - started) / 60)
    checkpoints: dict[int, Path] = {}
    elapsed: dict[int, float] = {}
    for seed in seeds:
        run_name = f"final_recipe-v3_trainseed-{seed}"
        expected = config.artifacts_dir / "part4" / "training_runs" / run_name / "weights" / "best.pt"
        results_csv = expected.parents[1] / "results.csv"
        if expected.is_file() and results_csv.is_file():
            LOGGER.info("Reusing complete seed-%d run at %s", seed, expected)
            checkpoints[seed], elapsed[seed] = expected.resolve(), 0.0
            continue
        seed_config = replace(config, seed=seed)
        seed_started = time.perf_counter()
        checkpoints[seed] = train_yolo(
            seed_config, yaml_path, device, run_name=run_name
        )
        elapsed[seed] = time.perf_counter() - seed_started
    return prepared_root, checkpoints, elapsed


def evaluate_final_models(
    config: Parts34Config,
    device: str,
    checkpoints: Mapping[int, Path],
) -> dict[str, Any]:
    """Evaluate a pretrained baseline and every robust seed on untouched official val."""

    from ultralytics import YOLO

    output = config.output_dir / "part4"
    output.mkdir(parents=True, exist_ok=True)
    samples = discover_cityscapes_samples(
        config.dataset_root, split="val", max_samples=config.max_samples, seed=7
    )
    image_ids = [sample.sample_id for sample in samples]
    ground_truth = []
    for index, sample in enumerate(samples):
        if index % 100 == 0:
            LOGGER.info("Caching final ground truth [%d/%d]", index, len(samples))
        clean, _, instance = load_sample(sample)
        clean.close()
        ground_truth.extend(instance_mask_to_boxes(instance, sample.sample_id))

    models: dict[str, tuple[Any, Mapping[int, str]]] = {
        "pretrained": (YOLO(config.yolo_model), COCO_ID_TO_SHARED_CLASS),
    }
    for seed, checkpoint in checkpoints.items():
        models[f"robust_seed_{seed}"] = (YOLO(str(checkpoint)), PROJECT_ID_TO_CLASS)
    for model, _ in models.values():
        model.to(device)

    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    levels = config.distortion_levels or DEFAULT_DISTORTION_LEVELS
    conditions = _conditions(levels)
    for condition_number, (condition, level_index, level) in enumerate(conditions, 1):
        condition_started = time.perf_counter()
        LOGGER.info(
            "Final evaluation [%d/%d]: %s level=%s",
            condition_number, len(conditions), condition, level,
        )
        images: list[Image.Image] = []
        snrs: list[float] = []
        for sample in samples:
            with Image.open(sample.image_path) as source:
                clean = source.convert("RGB")
            if condition == "Clean":
                evaluated = clean
            else:
                distortion_seed = stable_distortion_seed(7, sample.sample_id, condition, level_index)
                array = apply_aug(clean, condition, float(level), seed=distortion_seed)
                snrs.append(compute_snr(np.asarray(clean), array))
                evaluated = Image.fromarray(array)
                clean.close()
            images.append(evaluated)

        current_metrics: dict[str, dict[str, Any]] = {}
        for model_name, (model, mapping) in models.items():
            predictions = batched_model_detections(
                images, model, image_ids, mapping, config.yolo_eval_confidence,
                device, config.use_half, batch=config.part4_eval_batch,
                image_size=config.part4_image_size,
            )
            summary, per_class = evaluate_detections(predictions, ground_truth)
            current_metrics[model_name] = summary
            metric_rows.append({
                "model": model_name, "distortion": condition,
                "severity_index": level_index + 1 if condition != "Clean" else 0,
                "level": level, "sample_count": len(samples),
                "mean_snr_db": float(np.mean(snrs)) if snrs else float("inf"),
                **summary,
            })
            class_rows.extend({
                "model": model_name, "distortion": condition,
                "severity_index": level_index + 1 if condition != "Clean" else 0,
                "level": level, **row,
            } for row in per_class)
        for image in images:
            image.close()

        robust_values = [
            float(metrics["map_50_95"])
            for name, metrics in current_metrics.items() if name.startswith("robust_seed_")
        ]
        baseline = float(current_metrics["pretrained"]["map_50_95"])
        condition_rows.append({
            "distortion": condition,
            "severity_index": level_index + 1 if condition != "Clean" else 0,
            "level": level, "sample_count": len(samples),
            "mean_snr_db": float(np.mean(snrs)) if snrs else float("inf"),
            "pretrained_map_50_95": baseline,
            "robust_mean_map_50_95": float(np.mean(robust_values)),
            "robust_std_map_50_95": float(np.std(robust_values, ddof=1)) if len(robust_values) > 1 else 0.0,
            "mean_gain": float(np.mean(robust_values)) - baseline,
            "condition_elapsed_seconds": time.perf_counter() - condition_started,
        })
        write_csv(output / "model_condition_metrics.csv", metric_rows)
        write_csv(output / "detection_per_class.csv", class_rows)
        write_csv(output / "fine_tuning_summary.csv", condition_rows)
        write_json(output / "evaluation_progress.json", {
            "complete": False, "completed_conditions": condition_number,
            "total_conditions": len(conditions), "sample_count": len(samples),
        })

    write_json(output / "evaluation_progress.json", {
        "complete": True, "completed_conditions": len(conditions),
        "total_conditions": len(conditions), "sample_count": len(samples),
    })
    return {
        "sample_count": len(samples), "ground_truth_objects": len(ground_truth),
        "condition_rows": condition_rows, "metric_rows": metric_rows,
        "class_rows": class_rows,
    }


def analyze_final_results(
    config: Parts34Config,
    evaluation: Mapping[str, Any],
    checkpoints: Mapping[int, Path],
    training_seconds: Mapping[int, float],
    prepared_root: Path,
) -> dict[str, Any]:
    """Create uncertainty summaries, acceptance checks, plots, and provenance."""

    output = config.output_dir / "part4"
    metric_rows = list(evaluation["metric_rows"])
    condition_rows = list(evaluation["condition_rows"])
    class_rows = list(evaluation["class_rows"])
    model_names = [f"robust_seed_{seed}" for seed in checkpoints]
    distorted = [row for row in condition_rows if row["distortion"] != "Clean"]
    bootstrap = paired_bootstrap(
        [float(row["pretrained_map_50_95"]) for row in distorted],
        [float(row["robust_mean_map_50_95"]) for row in distorted],
        resamples=20_000, confidence_level=0.95, seed=20260722,
    )
    clean = next(row for row in condition_rows if row["distortion"] == "Clean")
    positive_conditions = sum(float(row["mean_gain"]) >= 0.0 for row in distorted)

    per_seed: list[dict[str, Any]] = []
    for model_name in model_names:
        rows = {str(row["distortion"]) + "|" + str(row["level"]): row for row in metric_rows if row["model"] == model_name}
        baseline_rows = {
            str(row["distortion"]) + "|" + str(row["level"]): row
            for row in metric_rows if row["model"] == "pretrained"
        }
        corrupt_keys = [key for key in rows if not key.startswith("Clean|")]
        clean_key = next(key for key in rows if key.startswith("Clean|"))
        gains = [float(rows[key]["map_50_95"]) - float(baseline_rows[key]["map_50_95"]) for key in corrupt_keys]
        per_seed.append({
            "model": model_name,
            "clean_map_50_95": float(rows[clean_key]["map_50_95"]),
            "clean_gain": float(rows[clean_key]["map_50_95"]) - float(baseline_rows[clean_key]["map_50_95"]),
            "mean_corrupted_map_50_95": float(np.mean([float(rows[key]["map_50_95"]) for key in corrupt_keys])),
            "mean_corrupted_gain": float(np.mean(gains)),
            "nonnegative_conditions": int(sum(gain >= 0 for gain in gains)),
        })
    write_csv(output / "seed_summary.csv", per_seed)

    class_gain_rows: list[dict[str, Any]] = []
    for class_name in SHARED_DETECTION_CLASSES:
        baseline_values, robust_values = [], []
        for distortion_row in distorted:
            distortion, level = distortion_row["distortion"], distortion_row["level"]
            base = [row for row in class_rows if row["model"] == "pretrained" and row["class_name"] == class_name and row["distortion"] == distortion and row["level"] == level]
            robust = [row for row in class_rows if row["model"] in model_names and row["class_name"] == class_name and row["distortion"] == distortion and row["level"] == level]
            if base and robust:
                baseline_values.append(float(base[0]["map_50_95"]))
                robust_values.append(float(np.mean([float(row["map_50_95"]) for row in robust])))
        class_gain_rows.append({
            "class_name": class_name,
            "official_val_gt_instances": int(float(next(
                row["gt_count"] for row in class_rows
                if row["model"] == "pretrained" and row["class_name"] == class_name
                and row["distortion"] == "Clean"
            ))),
            "pretrained_mean_corrupted_map_50_95": float(np.mean(baseline_values)),
            "robust_mean_corrupted_map_50_95": float(np.mean(robust_values)),
            "mean_corrupted_gain": float(np.mean(np.asarray(robust_values) - baseline_values)),
        })
    write_csv(output / "per_class_robustness_summary.csv", class_gain_rows)

    criteria = {
        "clean_map_loss_no_worse_than_0.005": float(clean["mean_gain"]) >= -0.005,
        "corrupted_gain_95ci_strictly_positive": float(bootstrap["ci_low"]) > 0.0,
        "at_least_16_of_20_corruptions_nonnegative": positive_conditions >= 16,
        "no_supported_class_mean_corrupted_regression_below_0.01": min(
            float(row["mean_corrupted_gain"]) for row in class_gain_rows
            if int(row["official_val_gt_instances"]) >= 20
        ) >= -0.01,
    }
    summary = {
        "protocol": "Part 4 final recipe v3",
        "detection_evaluator_version": DETECTION_EVALUATOR_VERSION,
        "official_final_sample_count": evaluation["sample_count"],
        "training_seeds": list(checkpoints),
        "clean": clean,
        "mean_corrupted_pretrained_map_50_95": float(np.mean([float(row["pretrained_map_50_95"]) for row in distorted])),
        "mean_corrupted_robust_map_50_95": float(np.mean([float(row["robust_mean_map_50_95"]) for row in distorted])),
        "mean_corrupted_gain": float(np.mean([float(row["mean_gain"]) for row in distorted])),
        "nonnegative_corrupted_conditions": positive_conditions,
        "condition_paired_bootstrap": bootstrap,
        "acceptance_criteria": criteria,
        "all_acceptance_criteria_pass": all(criteria.values()),
        "per_seed": per_seed,
        "per_class": class_gain_rows,
        "per_class_inference_minimum_gt_instances": 20,
        "rare_class_policy": "classes below 20 official-val instances are reported descriptively but excluded from the regression gate",
    }
    write_json(output / "final_analysis.json", summary)
    _save_plots(output, condition_rows, class_gain_rows, checkpoints)

    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata: list[dict[str, Any]] = []
    for seed, checkpoint in checkpoints.items():
        destination = checkpoint_dir / f"robust_seed_{seed}_best.pt"
        shutil.copy2(checkpoint, destination)
        run_dir = checkpoint.parents[1]
        training_dir = output / "training" / f"seed_{seed}"
        training_dir.mkdir(parents=True, exist_ok=True)
        for name in ("results.csv", "args.yaml", "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
            source = run_dir / name
            if source.is_file():
                shutil.copy2(source, training_dir / name)
        checkpoint_metadata.append({
            "seed": seed, "path": destination, "sha256": _sha256(destination),
            "training_seconds_this_invocation": training_seconds.get(seed, 0.0),
        })
    dataset_metadata_dir = output / "training_dataset"
    dataset_metadata_dir.mkdir(parents=True, exist_ok=True)
    for name in ("dataset_manifest.json", "samples.csv", "dataset.yaml"):
        source = prepared_root / name
        if source.is_file():
            shutil.copy2(source, dataset_metadata_dir / name)
    write_json(output / "experiment_manifest.json", {
        "complete": True, "recipe_version": 3,
        "software_versions": _software_versions(),
        "config": asdict(config), "prepared_dataset": prepared_root,
        "tracked_prepared_dataset_metadata": dataset_metadata_dir,
        "prepared_dataset_manifest_sha256": _sha256(prepared_root / "dataset_manifest.json"),
        "checkpoints": checkpoint_metadata,
        "methodological_guards": [
            "official Cityscapes val is untouched until final evaluation",
            "internal validation is city-disjoint and drawn only from official train",
            "every source has an exact clean PNG anchor",
            "corrupted views are balanced over all 20 distortion/severity pairs",
            "three independent deterministic training seeds are reported without cherry-picking",
            "all models use the same 500 images and deterministic corruptions",
        ],
    })
    return summary


def _save_plots(
    output: Path,
    condition_rows: Sequence[Mapping[str, Any]],
    class_rows: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[int, Path],
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"GaussNoise": "#5B5FDE", "SevereJPEG": "#E76F51", "LowLight": "#2A9D8F", "MotionBlur": "#E9A23B"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, distortion in zip(axes.flat, colors):
        rows = sorted(
            (row for row in condition_rows if row["distortion"] == distortion),
            key=lambda row: int(row["severity_index"]),
        )
        x = np.arange(1, len(rows) + 1)
        baseline = np.asarray([float(row["pretrained_map_50_95"]) for row in rows])
        robust = np.asarray([float(row["robust_mean_map_50_95"]) for row in rows])
        spread = np.asarray([float(row["robust_std_map_50_95"]) for row in rows])
        axis.plot(x, baseline, "--o", color="#4B5563", label="COCO pretrained")
        axis.plot(x, robust, "-o", color=colors[distortion], label="Robust mean (3 seeds)")
        axis.fill_between(x, robust - spread, robust + spread, color=colors[distortion], alpha=0.18, label="±1 seed SD")
        axis.set_title(distortion)
        axis.set_xticks(x)
        axis.set_xlabel("Severity (1=mild, 5=severe)")
        axis.set_ylabel("mAP@0.50:0.95")
        axis.legend(fontsize=8)
    figure.suptitle("Part 4 — Robust YOLO on untouched Cityscapes validation", fontsize=15, weight="bold")
    figure.tight_layout()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "robustness_curves_three_seeds.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    distortions = list(colors)
    matrix = np.asarray([
        [float(next(row for row in condition_rows if row["distortion"] == distortion and int(row["severity_index"]) == severity)["mean_gain"]) for severity in range(1, 6)]
        for distortion in distortions
    ])
    limit = max(abs(float(matrix.min())), abs(float(matrix.max())), 0.001)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            text_color = "white" if abs(float(matrix[row, column])) > 0.55 * limit else "#111827"
            axis.text(
                column, row, f"{matrix[row, column]:+.3f}", ha="center",
                va="center", fontsize=9, color=text_color, weight="bold",
            )
    axis.set_xticks(range(5), [f"Severity {index}" for index in range(1, 6)])
    axis.set_yticks(range(4), distortions)
    axis.set_title("Mean robust mAP gain over pretrained baseline (3 seeds)", weight="bold")
    figure.colorbar(image, ax=axis, label="Δ mAP@0.50:0.95")
    figure.tight_layout()
    figure.savefig(figures / "robustness_gain_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    names = [str(row["class_name"]) for row in class_rows]
    gains = [float(row["mean_corrupted_gain"]) for row in class_rows]
    axis.bar(names, gains, color=["#2A9D8F" if gain >= 0 else "#E76F51" for gain in gains])
    axis.axhline(0, color="#1F2937", linewidth=1)
    axis.set_ylabel("Mean Δ mAP@0.50:0.95")
    axis.set_title("Per-class robustness change across 20 corruptions", weight="bold")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(figures / "per_class_robustness_gain.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for seed, checkpoint in checkpoints.items():
        rows = _read_csv(checkpoint.parents[1] / "results.csv")
        epochs = [int(float(row["epoch"])) for row in rows]
        axes[0].plot(epochs, [float(row["train/box_loss"]) for row in rows], label=f"seed {seed}")
        axes[1].plot(epochs, [float(row["metrics/mAP50-95(B)"]) for row in rows], label=f"seed {seed}")
    axes[0].set(title="Training box loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Internal city-disjoint validation", xlabel="Epoch", ylabel="mAP@0.50:0.95")
    for axis in axes:
        axis.legend()
    figure.suptitle("Part 4 training convergence", fontsize=15, weight="bold")
    figure.tight_layout()
    figure.savefig(figures / "training_convergence_three_seeds.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_final_part4(
    config: Parts34Config,
    device: str,
    seeds: Sequence[int] = (7, 17, 29),
) -> dict[str, Any]:
    """Execute the complete final protocol and persist recoverable progress."""

    overall_started = time.perf_counter()
    prepared, checkpoints, training_seconds = train_final_models(config, device, seeds)
    evaluation_started = time.perf_counter()
    evaluation = evaluate_final_models(config, device, checkpoints)
    summary = analyze_final_results(
        config, evaluation, checkpoints, training_seconds, prepared
    )
    write_json(config.output_dir / "part4" / "run_summary.json", {
        "complete": True, "elapsed_seconds": time.perf_counter() - overall_started,
        "evaluation_seconds": time.perf_counter() - evaluation_started,
        "analysis": summary,
    })
    return summary
