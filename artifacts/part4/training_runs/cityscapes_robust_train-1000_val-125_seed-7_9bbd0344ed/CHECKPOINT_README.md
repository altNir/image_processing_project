# Part 4 checkpoint

This folder contains the successful Cityscapes distortion-aware YOLO run:

- 1,000 training images
- 125 validation images
- 20 epochs
- seed 7
- seven shared Cityscapes/COCO detection classes

## Checkpoints

- `weights/best.pt`: checkpoint with the best validation fitness; use this to continue fine-tuning from the strongest learned weights.
- `weights/last.pt`: final epoch checkpoint; use this when an exact Ultralytics resume workflow needs the latest optimizer/training state.

The generated 601.93 MB image dataset is not stored in Git. The project rebuilds it deterministically from Cityscapes with the same seed and configuration.

## Continue training through this project

From the repository root in PowerShell:

```powershell
python .\main.py `
  --dataset-root .\data\cityscapes `
  --output-dir .\outputs_continued_part4 `
  --artifacts-dir .\artifacts `
  --part 4 `
  --max-samples 125 `
  --part4-train-samples 1000 `
  --part4-val-samples 125 `
  --part4-epochs 20 `
  --yolo-model .\artifacts\part4\training_runs\cityscapes_robust_train-1000_val-125_seed-7_9bbd0344ed\weights\best.pt `
  --device cuda
```

This starts a new training session initialized from `best.pt`, so it preserves the learned weights but creates a fresh optimizer schedule. Use a different `--output-dir` to keep the original results unchanged.

`--fine-tuned-weights` evaluates a checkpoint without training it; it is not the option for continuing training.
