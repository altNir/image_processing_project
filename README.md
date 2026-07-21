# Cityscapes Vision Robustness Project

This class project measures how common vision methods behave on clean and degraded street scenes, whether classical restoration recovers performance, and whether distortion-aware detector training improves robustness. It follows the function style used in the course slides while using Cityscapes semantic and instance annotations for quantitative evaluation.

## Project stages and numbered parts

The work has three main stages containing four numbered implementation parts:

| Stage | Course part | What it does |
|---|---:|---|
| Clean baselines | Part 1 | Runs ORB, Canny, pretrained YOLOv8n, and Cityscapes SegFormer-B0 on clean validation images |
| Degradation and recovery | Parts 2–3 | Applies four controlled distortions, measures robustness, restores each image, and measures recovery |
| Robust adaptation | Part 4 | Builds a mixed clean/distorted Cityscapes detection set, fine-tunes YOLO, and compares it with the pretrained detector |

Canny and motion blur are the additional methods chosen for the three-person project direction.

## Tasks and evaluation metrics

| Vision task | Method | Metrics |
|---|---|---|
| Local features | ORB | keypoint retention, spatial match retention, inlier ratio |
| Edge detection | Canny | edge-pixel retention, tolerant precision, recall, F1 |
| Semantic segmentation | SegFormer-B0 | per-class IoU, mean IoU, pixel accuracy, mean class accuracy |
| Object detection | YOLOv8n | AP@0.50, mAP@0.50:0.95, precision, recall, matched-box IoU |
| Image quality | SNR | signal-to-noise ratio in dB before and after restoration |

Cityscapes instance masks are converted to object boxes. Evaluation uses the seven direct Cityscapes/COCO matches: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `train`, and `truck`. `rider` is excluded because COCO has no direct equivalent.

## Repository organization

```text
images_project/
├── main.py                         # Small unified entry point
├── cityscapes_project/
│   ├── config.py                   # Constants and dataclass configurations
│   ├── dataset.py                  # Discovery, loading, label/box conversion
│   ├── types.py                    # Shared sample and detection records
│   ├── cli.py                      # Unified command parser
│   ├── methods/
│   │   ├── classical.py            # ORB and Canny
│   │   ├── distortions.py          # Noise, JPEG, low light, motion blur, SNR
│   │   ├── restoration.py          # Part 3 restoration algorithms
│   │   ├── segmentation.py         # SegFormer inference and metrics
│   │   └── detection.py            # YOLO conversion and AP metrics
│   ├── pipelines/
│   │   ├── parts12.py              # Part 1/2 orchestration
│   │   └── parts34.py              # Part 3/4 orchestration
│   └── utils/
│       ├── dependencies.py         # Optional dependency error messages
│       ├── device.py               # CUDA/CPU selection and model loading
│       ├── io.py                   # JSON and CSV writing
│       ├── timing.py               # Runtime extrapolation
│       └── visualization.py        # Overlays, galleries, and plots
├── tests/
│   ├── test_core_methods.py
│   ├── test_restoration_and_training.py
│   └── test_timing.py
├── requirements.txt
└── setup_cuda.ps1
```

## Dataset

Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from the [Cityscapes website](https://www.cityscapes-dataset.com/) and extract both beneath one root:

```text
data/cityscapes/
├── leftImg8bit/
│   ├── train/<city>/*_leftImg8bit.png
│   └── val/<city>/*_leftImg8bit.png
└── gtFine/
    ├── train/<city>/*_gtFine_labelIds.png and *_instanceIds.png
    └── val/<city>/*_gtFine_labelIds.png and *_instanceIds.png
```

Raw `labelIds` are converted in memory to the 19 Cityscapes train IDs. Existing `labelTrainIds` files also work. Reported scores should use `val`, because test labels are withheld.

## Installation

Python 3.10 or newer is recommended. From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install and verify CUDA-enabled PyTorch with:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_cuda.ps1
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The first model run downloads `yolov8n.pt` and `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`.

## GPU and CuPy policy

Use `--device cuda` to enable GPU model inference/training, `--device cuda:0` to select a GPU, or `--device cpu` to disable GPU use. CUDA half precision is enabled by default; add `--no-half` if it causes a precision or compatibility problem.

CuPy is intentionally **not** a project dependency. Gaussian noise generation is a small NumPy operation, while Gaussian/non-local-means restoration is implemented by OpenCV on the CPU. Moving each full-resolution image to and from a CuPy array would add transfer and installation overhead without accelerating the expensive YOLO and SegFormer stages. Those stages already remain on the GPU through PyTorch, so arrays are not repeatedly transferred between NumPy and CuPy.

## Running the project

The unified entry point accepts `--part 1`, `2`, `3`, `4`, or `all`.

Quick smoke test of the complete pipeline:

```powershell
python .\main.py `
  --dataset-root .\data\cityscapes `
  --output-dir .\outputs_quick `
  --artifacts-dir .\artifacts `
  --part all `
  --quick `
  --device cuda
```

Run individual parts:

```powershell
python .\main.py --dataset-root .\data\cityscapes --output-dir .\outputs --part 1 --device cuda
python .\main.py --dataset-root .\data\cityscapes --output-dir .\outputs --part 2 --device cuda
python .\main.py --dataset-root .\data\cityscapes --output-dir .\outputs --part 3 --device cuda
python .\main.py --dataset-root .\data\cityscapes --output-dir .\outputs --part 4 --device cuda
```

Part 2 automatically computes the clean Part 1 references it needs. Use a single command line if PowerShell backticks are inconvenient.

Evaluate an existing fine-tuned checkpoint without training again:

```powershell
python .\main.py `
  --dataset-root .\data\cityscapes `
  --part 4 `
  --device cuda `
  --fine-tuned-weights .\artifacts\part4\training_runs\<run-name>\weights\best.pt
```

## Main configuration options

| Option | Default | Meaning |
|---|---:|---|
| `--part` | `all` in `main.py` | One numbered part or the complete pipeline |
| `--split` | `val` | Cityscapes split used for evaluation |
| `--max-samples` | `0` | Deterministic evaluation limit; `0` uses all 500 validation images |
| `--seed` | `7` | Sampling, distortion, assignment, and training seed |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, or `mps` |
| `--no-half` | off | Disable CUDA half precision |
| `--quick` | off | Four evaluation images, two levels, and tiny Part 4 training |
| `--nfeatures` | `800` | Maximum number of ORB features |
| `--orb-ratio-threshold` | `0.75` | ORB descriptor ratio-test threshold |
| `--orb-spatial-threshold` | `3.0` | Maximum aligned-keypoint distance in pixels |
| `--canny-low-threshold` | `100` | Lower Canny hysteresis threshold |
| `--canny-high-threshold` | `200` | Upper Canny hysteresis threshold |
| `--canny-blur-kernel` | `5` | Positive odd Gaussian pre-blur size |
| `--canny-tolerance-radius` | `2` | Edge-matching tolerance in pixels |
| `--yolo-eval-confidence` | `0.001` | Low confidence floor used for AP evaluation |
| `--yolo-visual-confidence` | `0.25` | Confidence floor used in gallery figures |
| `--gallery-samples` | `4` | Number of representative gallery samples |
| `--part4-train-samples` | `0` | Training-image limit; `0` uses all 2,975 train images |
| `--part4-val-samples` | `0` | Training-validation limit; `0` uses all 500 validation images |
| `--part4-epochs` | `20` | YOLO fine-tuning epochs |
| `--part4-image-size` | `640` | YOLO training resolution |
| `--part4-batch` | `8` | Training batch size; try `4` after CUDA out-of-memory |
| `--part4-clean-fraction` | `0.20` | Fraction of clean images in the robust training mixture |
| `--rebuild-training-data` | off | Ignore and replace the reusable prepared Part 4 dataset |

Run `python .\main.py --help` for the complete option list.

## Distortions and restoration

| Distortion | Five default levels | Part 3 restoration |
|---|---|---|
| Gaussian noise | sigma 5, 10, 20, 35, 50 | colored non-local means plus bilateral filtering |
| JPEG | quality 80, 60, 40, 20, 5 | luminance-channel bilateral filtering |
| Low light | factor 0.80, 0.60, 0.40, 0.25, 0.10 | gamma lifting plus CLAHE |
| Motion blur | kernel 3, 5, 9, 15, 25 | severity-scaled unsharp deblurring |

SNR is calculated as `10 * log10(mean(clean²) / mean((clean - test)²))`. Restoration is evaluated rather than assumed to help; a negative gain remains in the result.

## Outputs and example results

```text
outputs/
├── run_manifest.json
├── run_manifest_parts_3_4.json
├── part1/clean_summary.json, per-image/per-class CSVs, figures/
├── part2/distorted_summary.json, per-image/per-class CSVs, figures/
├── part3/restoration_summary.json, per-image/per-class CSVs, figures/
└── part4/fine_tuning_summary.json, per-class CSV, run_summary.json, figures/
```

The completed 20-image validation run produced these **illustrative**, non-final results:

- Part 1 SegFormer mIoU: `0.5586`; pixel accuracy: `0.9120`.
- Part 1 YOLO mAP@0.50:0.95: `0.2252`; mAP@0.50: `0.4418`.
- Gaussian sigma 5 retained ORB matches at `0.9143`; sigma 50 reduced retention to `0.4545`.
- Low-light factor 0.10 improved from `0.875 dB` distorted SNR to `9.930 dB` after restoration.
- Motion-blur restoration was mixed: some levels helped downstream metrics and others did not.

The 20-image Part 4 checkpoint used only 20 training images, so its extremely low fine-tuned score is a smoke-test result, not evidence about the full training recipe.

## Runtime estimate

The measured 20-image, all-level run took approximately **21 minutes 1 second** from the start of Part 1 through the end of Part 4. A direct 500/20 scale-up is **8.76 hours**. The full Part 4 recipe also expands from 20 training images and 5 epochs to 2,975 images and 20 epochs, although GPU training scales differently from evaluation.

A realistic planning range on the same machine is therefore **about 9.5–11 hours for Parts 1–4**, not seven hours. Approximate contributions from the measured timestamps are:

- Parts 1–2: about 1 hour 40 minutes when scaled to 500 images.
- Part 3: about 5 hours 40 minutes, dominated by CPU non-local-means denoising.
- Part 4 preparation, 20-epoch training, and full evaluation: roughly 2–3.5 hours.

Machine load, model download/cache state, disk speed, CUDA version, and thermal limits can change the result. Run Parts 1–2, Part 3, and Part 4 separately if a single long session is risky; completed outputs and prepared Part 4 data are reusable.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The suite covers dataset discovery and ID mapping, distortions, restorations, Canny/ORB support, packed clean references, SNR, segmentation, detection AP, YOLO label conversion, deterministic training assignment, runtime extrapolation, and lightweight output creation. Tests do not download model weights.

## Reproducibility, assumptions, and limitations

- The default seed is `7`; sample selection and synthetic conditions are deterministic.
- Semantic void label `255` is ignored.
- Detection AP uses 101-point interpolation at IoU thresholds 0.50–0.95.
- The COCO/Cityscapes label spaces are not identical; only seven direct class matches are scored.
- Box ground truth is derived from visible instance-mask pixels, not Cityscapes amodal boxes.
- Part 3 restoration is CPU-heavy and is the largest evaluation bottleneck.
- The fine-tuned detector needs the full training recipe before its results are suitable for the report.
- Large datasets, generated outputs, checkpoints, and training artifacts are excluded by `.gitignore`.
