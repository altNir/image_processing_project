"""Create a paired recipe-v3 versus recipe-v4 Part 3 comparison artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "psnr_gain_db",
    "ssim_gain",
    "mae_reduction",
    "orb_gain",
    "canny_gain",
    "seg_gain",
    "det_gain",
)
CHANGED_FAMILIES = ("GaussNoise", "SevereJPEG")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare(old_dir: Path, new_dir: Path, output_dir: Path) -> None:
    old_manifest = json.loads((old_dir / "restoration_manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((new_dir / "restoration_manifest.json").read_text(encoding="utf-8"))
    if old_manifest["sample_ids"] != new_manifest["sample_ids"]:
        raise ValueError("Recipe comparison requires identical validation image IDs")
    old_rows = {
        (row["distortion"], int(row["level_index"])): row
        for row in _read_csv(old_dir / "restoration_summary.csv")
    }
    new_rows = {
        (row["distortion"], int(row["level_index"])): row
        for row in _read_csv(new_dir / "restoration_summary.csv")
    }
    if old_rows.keys() != new_rows.keys():
        raise ValueError("Recipe comparison requires identical distortion conditions")

    comparison = []
    for key in sorted(new_rows):
        new, old = new_rows[key], old_rows[key]
        comparison.append({
            "distortion": key[0],
            "level_index": key[1],
            "level": float(new["level"]),
            **{
                f"v4_minus_v3_{metric}": float(new[metric]) - float(old[metric])
                for metric in METRICS
            },
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "recipe_v4_vs_v3.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = (
        ("psnr_gain_db", "PSNR gain change (dB)"),
        ("ssim_gain", "SSIM gain change"),
        ("orb_gain", "ORB-retention gain change"),
        ("seg_gain", "Segmentation mIoU-gain change"),
    )
    colors = {"GaussNoise": "#2563EB", "SevereJPEG": "#DC2626"}
    for axis, (metric, title) in zip(axes.flat, panels):
        for family in CHANGED_FAMILIES:
            rows = sorted(
                (row for row in comparison if row["distortion"] == family),
                key=lambda row: int(row["level_index"]),
            )
            axis.plot(
                np.arange(1, 6),
                [row[f"v4_minus_v3_{metric}"] for row in rows],
                marker="o",
                linewidth=2,
                color=colors[family],
                label=family,
            )
        axis.axhline(0.0, color="#6B7280", linewidth=1)
        axis.set_title(title, weight="bold")
        axis.set_xlabel("Severity index (1=mild, 5=severe)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=True)
    figure.suptitle(
        "Part 3 recipe v4 minus v3 on identical validation images",
        fontsize=16,
        weight="bold",
    )
    figure.savefig(output_dir / "recipe_v4_vs_v3.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    compare(arguments.old_dir, arguments.new_dir, arguments.output_dir)
