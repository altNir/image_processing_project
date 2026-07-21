# Cityscapes Vision Robustness Project - Parts 1-4

This project evaluates computer-vision methods on clean and distorted Cityscapes images, restores the distorted images, and fine-tunes a detector for improved distortion robustness. The implementation follows the course-slide pipeline and helper style (`overlay_mask`, `orb_overlay`, `yolo_overlay`, `predict_segmentation`, `compute_ious`, `apply_aug`, and `compute_snr`) while adapting it to Cityscapes annotations.

## Project design

| Part | Work performed |
|---|---|
| Part 1 | Evaluate ORB, Canny, pretrained YOLOv8n, and Cityscapes SegFormer-B0 on clean images |
| Part 2 | Apply Gaussian noise, JPEG compression, low light, and motion blur at five levels; rerun and evaluate all methods |
| Part 3 | Restore every distorted image with a distortion-specific enhancement method; compare distorted versus restored results |
| Part 4 | Build a mixed clean/distorted Cityscapes detection set, fine-tune YOLO, and compare pretrained versus fine-tuned detection |

The additional Canny method and motion-blur distortion are Arik's changes and are part of the final project direction.

### Tasks and metrics

| Task | Method | Main metrics |
|---|---|---|
| Local feature detection/matching | ORB | keypoint retention, spatial match retention, inlier ratio |
| Edge detection | Canny | edge-pixel retention, tolerant precision, recall, F1 |
| Semantic segmentation | SegFormer-B0 | per-class IoU, mean IoU, pixel accuracy, mean class accuracy |
| Object detection | YOLOv8n | per-class AP@0.50, mAP@0.50:0.95, precision, recall, matched-box IoU |
| Image quality | SNR | SNR in dB before and after restoration |

Cityscapes instance masks are converted to real object-detection ground-truth boxes. The evaluated classes shared directly by Cityscapes and COCO are `person`, `bicycle`, `car`, `motorcycle`, `bus`, `train`, and `truck`. Cityscapes `rider` is excluded because COCO has no equivalent class.

## Files

- `cityscapes_parts_1_2.py` - clean and distorted evaluation.
- `cityscapes_parts_3_4.py` - restoration, robust YOLO fine-tuning, and evaluation.
- `setup_cuda.ps1` - installs a CUDA-enabled PyTorch wheel and performs a real GPU smoke test.
- `tests/` - unit and lightweight orchestration tests that do not download model weights.

## Dataset setup

Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from the [Cityscapes website](https://www.cityscapes-dataset.com/), then extract both under the same directory. This repository expects the existing local layout:

```text
data/cityscapes/
|-- leftImg8bit/
|   |-- train/<city>/*_leftImg8bit.png
|   `-- val/<city>/*_leftImg8bit.png
`-- gtFine/
    |-- train/<city>/*_gtFine_labelIds.png and *_instanceIds.png
    `-- val/<city>/*_gtFine_labelIds.png and *_instanceIds.png
```

Official raw `labelIds` are converted in memory to the 19 contiguous Cityscapes train IDs. Prepared `labelTrainIds` masks are also accepted. Use the validation split for reported metrics because Cityscapes test labels are withheld.

## Installation and CUDA

Python 3.10 or newer and an NVIDIA GPU/driver are recommended. In PowerShell, from this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\setup_cuda.ps1
```

`setup_cuda.ps1` replaces a CPU-only PyTorch installation with the official CUDA 13.0 wheel and verifies `torch.cuda.is_available()`, the GPU name, GPU memory, and an FP16 matrix operation. CUDA 13.0 is appropriate for the RTX 4060 and the installed recent NVIDIA driver. To select another supported wheel:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_cuda.ps1 -CudaWheel cu126
```

Confirm the exact interpreter before a long run:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The project uses CUDA through PyTorch, Transformers, and Ultralytics. CuPy is not required: the expensive SegFormer, YOLO inference, and YOLO training already run on the GPU, while the small OpenCV preprocessing/restoration operations remain on the CPU. FP16 is enabled by default on CUDA; pass `--no-half` only if a model or GPU has a precision issue.

Pretrained weights are downloaded automatically on first use:

- `yolov8n.pt`
- `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`

## Quick CUDA test of all four parts

Run these before a complete experiment:

```powershell
python cityscapes_parts_1_2.py `
  --dataset-root .\data\cityscapes `
  --output-dir .\outputs `
  --part both `
  --quick `
  --device cuda

python cityscapes_parts_3_4.py `
  --dataset-root .\data\cityscapes `
  --output-dir .\outputs_parts_3_4 `
  --artifacts-dir .\artifacts `
  --part both `
  --quick `
  --device cuda
```

Quick mode uses four evaluation images and two levels per distortion. Part 4 prepares 32 training images and 8 validation images and trains for one epoch. It verifies the complete path but is not intended to produce final report scores.

## Complete experiment

### Parts 1 and 2

```powershell
python cityscapes_parts_1_2.py `
  --dataset-root .\data\cityscapes `
  --output-dir .\outputs_full `
  --part both `
  --split val `
  --max-samples 0 `
  --device cuda
```

This evaluates 500 clean images plus 10,000 distorted images: four distortions, five levels, and 500 images.

### Parts 3 and 4

```powershell
python cityscapes_parts_3_4.py `
  --dataset-root .\data\cityscapes `
  --output-dir .\outputs_parts_3_4_full `
  --artifacts-dir .\artifacts `
  --part both `
  --max-samples 0 `
  --device cuda `
  --part4-epochs 20 `
  --part4-image-size 640 `
  --part4-batch 8
```

An 8 GB RTX 4060 should normally handle YOLOv8n at batch 8. If CUDA reports out-of-memory, rerun with `--part4-batch 4`. When both parts run in one command, Part 3 models are released and the CUDA cache is cleared before Part 4 training.

Part 4 training artifacts can be large and are intentionally excluded by `.gitignore`. The prepared dataset is deterministic and is reused on later runs with the same recipe. Use `--rebuild-training-data` to regenerate it.

To evaluate an existing fine-tuned checkpoint without retraining:

```powershell
python cityscapes_parts_3_4.py `
  --dataset-root .\data\cityscapes `
  --part 4 `
  --device cuda `
  --fine-tuned-weights .\artifacts\part4\training_runs\cityscapes_robust_yolov8n\weights\best.pt
```

## Part 1 - clean evaluation

For every clean validation image, the script loads RGB, semantic, and instance annotations; creates the slide-style overlays; runs ORB, Canny, YOLO, and SegFormer; and evaluates predictions against ground truth. Clean Canny maps are packed to one bit per pixel before Part 2, avoiding roughly one gigabyte of unnecessary in-memory edge maps on the full split.

## Part 2 - controlled distortions

| Name in code | Default levels | Interpretation |
|---|---|---|
| `GaussNoise` | sigma 5, 10, 20, 35, 50 | additive RGB Gaussian noise |
| `SevereJPEG` | quality 80, 60, 40, 20, 5 | lower quality is more severe |
| `LowLight` | factor 0.80, 0.60, 0.40, 0.25, 0.10 | lower brightness is more severe |
| `MotionBlur` | kernel 3, 5, 9, 15, 25 | longer fixed-angle motion kernel is more severe |

The script reruns all four methods at every level and measures actual image SNR:

```text
SNR(dB) = 10 * log10(mean(clean^2) / mean((clean - test)^2))
```

## Part 3 - restoration and reevaluation

Each distorted image is restored by a method suited to its corruption, then ORB, Canny, SegFormer, and YOLO are rerun on both distorted and restored versions.

| Distortion | Restoration method |
|---|---|
| Gaussian noise | colored non-local means plus bilateral filtering |
| JPEG artifacts | bilateral filtering on the luminance channel |
| Low light | gamma lifting plus CLAHE local contrast enhancement |
| Motion blur | severity-scaled unsharp deblurring |

The result tables include distorted/restored SNR, ORB retention, Canny F1, SegFormer mIoU, YOLO mAP, per-class metrics, and the gain or loss caused by restoration. A restoration is not assumed to help; the measured comparison is the result.

## Part 4 - distortion-robust YOLO

Part 4 follows the slide-style supervised fine-tuning sequence:

1. Read Cityscapes train and validation images and instance masks.
2. Convert the seven shared instance classes to normalized YOLO labels.
3. Deterministically assign clean or distorted training conditions; the default clean fraction is 20%.
4. Fine-tune pretrained YOLOv8n with CUDA automatic mixed precision.
5. Evaluate the original pretrained detector and fine-tuned detector on clean images and every distortion level.
6. Report clean accuracy, robustness by SNR, per-class AP, and fine-tuning gains.

The full train split is used when `--part4-train-samples 0`; the full validation split is used when `--part4-val-samples 0`. `--max-samples` separately controls the final Part 3/4 evaluation set.

## Outputs

```text
outputs_full/
|-- run_manifest.json
|-- part1/
|   |-- clean_summary.json
|   |-- clean_per_image.csv
|   |-- segmentation_per_class.csv
|   `-- detection_per_class.csv
`-- part2/
    |-- distorted_summary.json and .csv
    |-- distorted_per_image.csv
    |-- segmentation_per_class.csv
    |-- detection_per_class.csv
    `-- figures/

outputs_parts_3_4_full/
|-- run_manifest_parts_3_4.json
|-- part3/
|   |-- restoration_summary.json and .csv
|   |-- restoration_per_image.csv
|   |-- segmentation_per_class.csv
|   |-- detection_per_class.csv
|   `-- figures/restoration_grid.png and restored_performance.png
`-- part4/
    |-- fine_tuning_summary.json and .csv
    |-- detection_per_class.csv
    |-- run_summary.json
    `-- figures/fine_tuning_per_snr.png
```

## Reproducibility

- Default random seed: `7`, matching the slide example.
- Sample selection, distortions, training-condition assignment, and Ultralytics training seed are deterministic.
- Run manifests record the full configuration and checkpoint path.
- Low confidence (`0.001`) is used for YOLO AP curves so they are not truncated.
- Semantic void label `255` is ignored.
- AP uses 101-point interpolation at IoU thresholds 0.50 to 0.95.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover dataset discovery, Cityscapes ID mapping, all distortions, all restorations, packed Canny references, edge consistency, SNR, segmentation, detection AP, YOLO label conversion, deterministic training assignment, and pipeline output creation.

## References

- [Cityscapes dataset](https://www.cityscapes-dataset.com/)
- [SegFormer-B0 Cityscapes checkpoint](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024)
- [Ultralytics training API](https://docs.ultralytics.com/modes/train/)
- [Official PyTorch CUDA installation](https://pytorch.org/get-started/locally/)
