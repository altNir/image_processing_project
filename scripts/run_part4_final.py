"""Run the final leakage-free, three-seed Part 4 experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from cityscapes_project.config import Parts34Config
from cityscapes_project.pipelines.part4_final import run_final_part4
from cityscapes_project.utils.device import select_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_part4_v3_official"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts_final"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--train-views", type=int, default=4)
    parser.add_argument("--val-views", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = Parts34Config(
        dataset_root=args.dataset_root, output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir, max_samples=args.max_eval_samples, seed=7,
        device="cuda:0", use_half=True, part4_train_samples=args.train_samples,
        part4_val_samples=args.val_samples, part4_epochs=args.epochs,
        part4_image_size=640, part4_batch=args.batch,
        part4_workers=args.workers, part4_train_views=args.train_views,
        part4_val_views=args.val_views, part4_internal_val_fraction=0.125,
        part4_patience=10, part4_eval_batch=args.eval_batch,
    )
    run_final_part4(config, select_device(config.device), seeds=tuple(args.seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
