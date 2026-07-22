"""Tune one conservative output-strength scalar per Part 3 restoration family.

The utility uses only official Cityscapes ``train`` images.  It does not alter
the restoration algorithms, choose parameters per image, or inspect reported
validation results.  Candidate selection balances fidelity and classical
structure preservation with equal group weight.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cityscapes_project.config import DEFAULT_DISTORTION_LEVELS
from cityscapes_project.dataset import discover_cityscapes_samples, load_sample
from cityscapes_project.methods.classical import (
    canny_detect,
    evaluate_canny_edges,
    measure_orb_matching,
)
from cityscapes_project.methods.distortions import apply_aug, stable_distortion_seed
from cityscapes_project.methods.quality import compute_quality_metrics
from cityscapes_project.methods.restoration import restore_image_at_strength
from cityscapes_project.utils.dependencies import cv2_module
from cityscapes_project.utils.io import write_csv, write_json


def _select_city_balanced(
    samples: list[Any], cities: tuple[str, ...], count: int, seed: int
) -> list[Any]:
    pools = {
        city: [sample for sample in samples if sample.image_path.parent.name == city]
        for city in cities
    }
    if any(not pool for pool in pools.values()):
        missing = [city for city, pool in pools.items() if not pool]
        raise ValueError(f"No training samples found for cities: {missing}")
    base, remainder = divmod(count, len(cities))
    selected: list[Any] = []
    for city_index, city in enumerate(cities):
        city_count = base + int(city_index < remainder)
        pool = pools[city]
        if len(pool) < city_count:
            raise ValueError(
                f"City {city!r} has {len(pool)} samples but {city_count} were requested"
            )
        selected.extend(random.Random(f"{seed}|{city}").sample(pool, city_count))
    if len({sample.sample_id for sample in selected}) != count:
        raise AssertionError("City-balanced selection must contain exactly the requested IDs")
    return sorted(selected, key=lambda sample: sample.sample_id)


def _rank_scores(values: dict[float, float]) -> dict[float, float]:
    ordered = sorted(values, key=lambda strength: (values[strength], strength))
    denominator = max(1, len(ordered) - 1)
    return {strength: index / denominator for index, strength in enumerate(ordered)}


def tune(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    all_train = discover_cityscapes_samples(args.dataset_root, split="train")
    samples = _select_city_balanced(all_train, tuple(args.cities), args.samples, args.seed)
    strengths = tuple(sorted(set(float(value) for value in args.strengths)))
    rows: list[dict[str, Any]] = []

    for distortion, levels in DEFAULT_DISTORTION_LEVELS.items():
        condition_values: dict[tuple[float, int], list[dict[str, float]]] = defaultdict(list)
        for level_index, level in enumerate(levels):
            for sample in samples:
                clean_image, _, _ = load_sample(sample)
                clean_rgb = np.asarray(clean_image)
                seed = stable_distortion_seed(args.seed, sample.sample_id, distortion, level_index)
                distorted_rgb = apply_aug(clean_image, distortion, float(level), seed=seed)
                distorted_image = Image.fromarray(distorted_rgb)
                distorted_quality = compute_quality_metrics(clean_rgb, distorted_rgb)
                clean_edges = canny_detect(clean_image, 100, 200, 5)
                distorted_edges = canny_detect(distorted_image, 100, 200, 5)
                distorted_canny = evaluate_canny_edges(clean_edges, distorted_edges, 2)["f1"]
                distorted_orb = measure_orb_matching(clean_image, distorted_image, 800)[
                    "match_retention"
                ]
                fully_restored = restore_image_at_strength(
                    distorted_rgb, distortion, float(level), 1.0
                )
                for strength in strengths:
                    restored_rgb = fully_restored if strength == 1.0 else cv2_module().addWeighted(
                        fully_restored, strength, distorted_rgb, 1.0 - strength, 0.0
                    )
                    restored_image = Image.fromarray(restored_rgb)
                    quality = compute_quality_metrics(clean_rgb, restored_rgb)
                    restored_edges = canny_detect(restored_image, 100, 200, 5)
                    canny = evaluate_canny_edges(clean_edges, restored_edges, 2)["f1"]
                    orb = measure_orb_matching(clean_image, restored_image, 800)[
                        "match_retention"
                    ]
                    condition_values[(strength, level_index)].append({
                        "psnr_gain_db": quality["psnr_db"] - distorted_quality["psnr_db"],
                        "ssim_gain": quality["ssim"] - distorted_quality["ssim"],
                        "mae_reduction": distorted_quality["mae"] - quality["mae"],
                        "orb_gain": orb - distorted_orb,
                        "canny_gain": canny - distorted_canny,
                    })
                    restored_image.close()
                distorted_image.close()
                clean_image.close()

        aggregate: dict[float, dict[str, float]] = {}
        metrics = ("psnr_gain_db", "ssim_gain", "mae_reduction", "orb_gain", "canny_gain")
        for strength in strengths:
            records = [
                record for level_index in range(len(levels))
                for record in condition_values[(strength, level_index)]
            ]
            aggregate[strength] = {
                metric: float(np.mean([record[metric] for record in records]))
                for metric in metrics
            }
            aggregate[strength].update({
                f"positive_{metric}_conditions": int(sum(
                    np.mean([record[metric] for record in condition_values[(strength, level_index)]]) > 0
                    for level_index in range(len(levels))
                )) for metric in metrics
            })

        ranks = {
            metric: _rank_scores({strength: aggregate[strength][metric] for strength in strengths})
            for metric in metrics
        }
        for strength in strengths:
            fidelity_rank = float(np.mean([
                ranks[metric][strength] for metric in ("psnr_gain_db", "ssim_gain", "mae_reduction")
            ]))
            structure_rank = float(np.mean([
                ranks[metric][strength] for metric in ("orb_gain", "canny_gain")
            ]))
            aggregate[strength]["fidelity_rank"] = fidelity_rank
            aggregate[strength]["structure_rank"] = structure_rank
            aggregate[strength]["balanced_rank_score"] = 0.5 * fidelity_rank + 0.5 * structure_rank

        eligible = [
            strength for strength in strengths
            if aggregate[strength]["positive_psnr_gain_db_conditions"] >= 4
            and aggregate[strength]["positive_ssim_gain_conditions"] == 5
        ]
        recommended = max(
            eligible or list(strengths),
            key=lambda strength: (
                aggregate[strength]["balanced_rank_score"],
                aggregate[strength]["structure_rank"],
                strength,
            ),
        )
        for strength in strengths:
            rows.append({
                "distortion": distortion,
                "output_strength": strength,
                "recommended": strength == recommended,
                **aggregate[strength],
            })

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    recommendations = {
        row["distortion"]: row["output_strength"] for row in rows if row["recommended"]
    }
    write_csv(output / "candidate_summary.csv", rows)
    result = {
        "complete": True,
        "scope": "train-only family-level Part 3 output-strength tuning",
        "dataset_split": "official train",
        "cities": args.cities,
        "sample_count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "candidate_strengths": strengths,
        "selection_rule": {
            "fidelity_group": ["PSNR gain", "SSIM gain", "MAE reduction"],
            "structure_group": ["ORB retention gain", "Canny F1 gain"],
            "weighting": "50% mean fidelity rank, 50% mean structure rank",
            "guardrail": "PSNR improves at >=4/5 levels and SSIM at 5/5 levels",
            "granularity": "one fixed scalar per distortion family; never per image or task",
        },
        "recommendations": recommendations,
        "elapsed_seconds": time.perf_counter() - started,
        "candidates": rows,
    }
    write_json(output / "tuning_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_part3_tuning"))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--cities", nargs="+", default=["darmstadt", "krefeld"])
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.70, 0.85, 1.0])
    parser.add_argument("--seed", type=int, default=7)
    return parser


if __name__ == "__main__":
    result = tune(build_parser().parse_args())
    print(json.dumps(result["recommendations"], indent=2))
