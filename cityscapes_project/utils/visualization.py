"""Rendering and plotting helpers for all four project parts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from cityscapes_project.config import CITYSCAPES_PALETTE
from cityscapes_project.methods.distortions import apply_aug, compute_snr
from cityscapes_project.types import Detection
from cityscapes_project.utils.dependencies import matplotlib_pyplot


def colorize(mask_idx: np.ndarray) -> np.ndarray:
    """Map Cityscapes train IDs to their official RGB colors."""

    mask = np.asarray(mask_idx)
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(CITYSCAPES_PALETTE))
    output[valid] = CITYSCAPES_PALETTE[mask[valid].astype(np.int64)]
    return output


def overlay_mask(
    img_pil: Image.Image,
    mask: Image.Image | np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a semantic mask while preserving the original slide-style interface."""

    image = np.asarray(img_pil.convert("RGB"), dtype=np.float32)
    mask_array = np.asarray(mask, dtype=np.int32)
    colors = colorize(mask_array).astype(np.float32)
    valid = ((mask_array >= 0) & (mask_array < len(CITYSCAPES_PALETTE)))[..., None]
    blended = image * (1.0 - alpha) + colors * alpha
    output = np.where(valid, blended, image).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output)


def seg_overlay(image_rgb: np.ndarray, mask_idx: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a segmentation prediction over an RGB array."""

    image = np.asarray(image_rgb, dtype=np.float32)
    colors = colorize(mask_idx).astype(np.float32)
    return (image * (1.0 - alpha) + colors * alpha).clip(0, 255).astype(np.uint8)


def draw_ground_truth_boxes(image: Image.Image, boxes: Sequence[Detection]) -> Image.Image:
    """Draw Cityscapes ground-truth boxes and class labels."""

    output = image.copy()
    draw = ImageDraw.Draw(output)
    for item in boxes:
        draw.rectangle(item.bbox, outline=(0, 255, 0), width=3)
        draw.text((item.bbox[0] + 2, item.bbox[1] + 2), item.class_name, fill=(0, 255, 0))
    return output


def save_part1_gallery(records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Save the clean baseline gallery."""

    if not records:
        return
    plt = matplotlib_pyplot()
    figure, axes = plt.subplots(len(records), 6, figsize=(24, 4.6 * len(records)), squeeze=False)
    for column, title in enumerate(("Clean", "Ground truth", "ORB", "Canny", "YOLO", "SegFormer")):
        axes[0, column].set_title(title, fontsize=13)
    for row_index, record in enumerate(records):
        for column, key in enumerate(("clean", "ground_truth", "orb", "canny", "yolo", "segmentation")):
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
    """Save one row per synthetic distortion across all severity levels."""

    plt = matplotlib_pyplot()
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
            snr = compute_snr(np.asarray(image), distorted)
            axes[row, column].set_title(f"level={level:g}\nSNR={snr:.2f} dB")
            axes[row, column].axis("off")
        for column in range(len(values) + 1, columns):
            axes[row, column].axis("off")
    figure.suptitle("Part 2 - distortion intensity ranges", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_performance_snr_plot(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Plot every Part 2 metric against mean SNR."""

    if not rows:
        return
    plt = matplotlib_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(14, 10.0))
    metrics = (
        ("orb_match_retention", "ORB spatial match retention"),
        ("canny_f1", "Canny tolerant edge F1"),
        ("seg_mean_iou", "SegFormer mean IoU"),
        ("det_map_50_95", "YOLO mAP@0.50:0.95"),
    )
    distortions = sorted({str(row["distortion"]) for row in rows})
    for axis, (metric, label) in zip(axes.ravel(), metrics):
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


def save_part2_gallery(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Save representative distorted outputs from all four methods."""

    if not rows:
        return
    plt = matplotlib_pyplot()
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


def save_restoration_gallery(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Save clean/distorted/restored triplets for Part 3."""

    if not rows:
        return
    plt = matplotlib_pyplot()
    figure, axes = plt.subplots(len(rows), 3, figsize=(15, 4.4 * len(rows)), squeeze=False)
    for column, title in enumerate(("Clean", "Distorted", "Restored")):
        axes[0, column].set_title(title)
    for index, row in enumerate(rows):
        for column, key in enumerate(("clean", "distorted", "restored")):
            axes[index, column].imshow(row[key])
            axes[index, column].axis("off")
        axes[index, 0].set_ylabel(
            f"{row['distortion']}\nlevel={row['level']:g}\n"
            f"PSNR {row['distorted_psnr']:.1f}->{row['restored_psnr']:.1f} dB\n"
            f"SSIM {row['distorted_ssim']:.3f}->{row['restored_ssim']:.3f}"
        )
    figure.suptitle("Part 3 - clean, distorted, and restored images", fontsize=16)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_restoration_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Plot distorted versus restored metrics for Part 3."""

    if not rows:
        return
    plt = matplotlib_pyplot()
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


def save_restoration_quality_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Plot full-reference fidelity and restoration runtime by severity."""

    if not rows:
        return
    plt = matplotlib_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = (
        ("psnr_gain_db", "PSNR gain (dB)"),
        ("ssim_gain", "SSIM gain"),
        ("mae_reduction", "MAE reduction (intensity levels)"),
        ("mean_restoration_runtime_ms", "Restoration time per image (ms)"),
    )
    distortions = sorted({str(row["distortion"]) for row in rows})
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for distortion in distortions:
            selected = [row for row in rows if row["distortion"] == distortion]
            selected.sort(key=lambda row: int(row["level_index"]))
            axis.plot(
                [int(row["level_index"]) + 1 for row in selected],
                [float(row[metric]) for row in selected],
                "-o",
                label=distortion,
            )
        if metric != "mean_restoration_runtime_ms":
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xlabel("Severity index (1=mild, 5=severe)")
        axis.set_ylabel(title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Part 3 - restoration fidelity gains and computational cost", fontsize=16)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_fine_tuning_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Plot pretrained versus fine-tuned YOLO robustness for Part 4."""

    if not rows:
        return
    plt = matplotlib_pyplot()
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


# Backward-compatible dependency helper name.
_matplotlib = matplotlib_pyplot
