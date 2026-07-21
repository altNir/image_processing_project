# Cityscapes Vision Robustness Project - Parts 1 and 2

This repository implements **only Part 1 and Part 2** of the course project:

1. Measure ORB, Canny, YOLO, and SegFormer on clean Cityscapes images.
2. Apply controlled distortions and measure how performance changes by distortion intensity and SNR.

It intentionally does **not** implement image restoration/enhancement (Part 3) or model fine-tuning (Part 4).

The implementation follows the code flow shown in the course slides as closely as practical. The main script retains the slide-style helpers `overlay_mask`, `orb_overlay`, `yolo_overlay`, `predict_segmentation`, `compute_ious`, `apply_aug`, and `compute_snr`, adapted from ADE20K to Cityscapes labels and annotations. Canny edge detection and motion blur extend the slide example to the four-method/four-distortion scope required for a three-person team.

## Project choices

| Item | Choice |
|---|---|
| Dataset | Cityscapes fine annotations, normally the 500-image validation split |
| Low-level task 1 | ORB feature detection and clean-to-distorted feature matching |
| Low-level task 2 | Canny edge detection and clean-to-distorted edge consistency |
| High-level task 1 | Semantic segmentation with SegFormer-B0 fine-tuned on Cityscapes |
| High-level task 2 | Object detection with pretrained YOLOv8n |
| Distortions | Gaussian noise, JPEG compression, low light, and motion blur |
| ORB metrics | Keypoint retention, spatially verified match retention, and inlier ratio |
| Canny metrics | Edge-pixel retention and tolerance-aware precision, recall, and F1 |
| Segmentation metrics | Per-class IoU, mean IoU, pixel accuracy, and mean class accuracy |
| Detection metrics | Per-class AP@0.50, mAP@0.50:0.95, precision, recall, and matched-box IoU |

Cityscapes provides real semantic and instance annotations. Object-detection ground-truth boxes are derived from the instance masks rather than from clean YOLO predictions. This makes the detection results genuine ground-truth evaluation.

## Dataset setup

Cityscapes requires a free account and acceptance of its academic/non-commercial terms. The script does not download or redistribute the dataset.

1. Register at [cityscapes-dataset.com](https://www.cityscapes-dataset.com/).
2. Download:
   - `leftImg8bit_trainvaltest.zip`
   - `gtFine_trainvaltest.zip`
3. Extract both archives under one directory.

The expected structure is:

```text
CITYSCAPES_ROOT/
├── leftImg8bit/
│   ├── train/<city>/*_leftImg8bit.png
│   └── val/<city>/*_leftImg8bit.png
└── gtFine/
    ├── train/<city>/
    │   ├── *_gtFine_labelIds.png
    │   └── *_gtFine_instanceIds.png
    └── val/<city>/
        ├── *_gtFine_labelIds.png
        └── *_gtFine_instanceIds.png
```

The official archive provides raw `labelIds` masks. The loader converts them in memory to the 19 contiguous train IDs used by SegFormer and the metrics. If a prepared dataset already contains `labelTrainIds` masks, those are detected and preferred automatically.

Use the validation split for reported metrics. Cityscapes test labels are withheld for its evaluation server.

## Environment installation

Python 3.10 or newer is recommended. From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation can depend on the available GPU. If you want CUDA acceleration, install the appropriate PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) before installing the remaining requirements.

On their first use, Ultralytics and Transformers download the pretrained `yolov8n.pt` and `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` weights.

## Running the project

### Fast pipeline check

This runs four deterministic validation images and two levels per distortion:

```powershell
python cityscapes_parts_1_2.py `
  --dataset-root ".\data\cityscapes" `
  --output-dir outputs `
  --quick
```

### Complete Parts 1 and 2 experiment

This runs all 500 validation images and all five levels of each distortion:

```powershell
python cityscapes_parts_1_2.py `
  --dataset-root ".\data\cityscapes" `
  --output-dir outputs `
  --part both `
  --split val `
  --max-samples 0
```

The complete experiment evaluates 500 clean images plus 10,000 distorted images. A CUDA-capable GPU is strongly recommended. Run `--quick` first to verify the installation and dataset paths.

### Part 1 only

```powershell
python cityscapes_parts_1_2.py `
  --dataset-root ".\data\cityscapes" `
  --part 1
```

Selecting `--part 2` still computes the clean Part 1 reference first because ORB, Canny, and degradation comparisons require it.

Useful options:

```text
--max-samples N          Deterministic sample limit; 0 means the complete split
--device auto            Automatically use CUDA, MPS, or CPU
--device cuda:0          Select a particular CUDA device
--nfeatures 800          Maximum ORB features, matching the slides
--canny-low-threshold 100
--canny-high-threshold 200
--canny-blur-kernel 5
--canny-tolerance-radius 2
--yolo-model yolov8n.pt  Ultralytics detection checkpoint
--gallery-samples 4      Number of clean qualitative examples
```

Run `python cityscapes_parts_1_2.py --help` for every option.

## Part 1: clean-image evaluation

For every selected clean image, the script:

1. Loads the RGB image, converts the official `labelIds` semantic mask to train IDs, and loads the instance-ID mask.
2. Produces a Cityscapes-color semantic overlay.
3. Detects and draws up to 800 ORB keypoints.
4. Detects Canny edges after fixed Gaussian pre-smoothing.
5. Runs pretrained YOLOv8n.
6. Runs the Cityscapes SegFormer-B0 checkpoint.
7. Computes semantic metrics against the 19-class ground truth.
8. Derives ground-truth boxes from the instance mask and evaluates YOLO.

The seven object classes shared directly by Cityscapes and COCO are evaluated:

- `person`
- `bicycle`
- `car`
- `motorcycle`
- `bus`
- `train`
- `truck`

Cityscapes `rider` is excluded because COCO has no direct rider class. It is not incorrectly merged into `person`.

## Part 2: distorted-image evaluation

Each clean image is transformed at five intensity levels:

| Distortion name in code | Levels | Meaning |
|---|---:|---|
| `GaussNoise` | sigma = 5, 10, 20, 35, 50 | Additive RGB Gaussian noise in the 0-255 pixel domain |
| `SevereJPEG` | quality = 80, 60, 40, 20, 5 | JPEG encoding quality; lower is more severe |
| `LowLight` | factor = 0.80, 0.60, 0.40, 0.25, 0.10 | Multiplicative brightness; lower is darker |
| `MotionBlur` | kernel = 3, 5, 9, 15, 25 | Linear motion point-spread function at a fixed 15-degree direction; longer is more severe |

For each distorted image, the script reruns all four methods and computes:

- Actual SNR relative to the clean image.
- ORB keypoint and spatial match retention.
- Canny edge-pixel retention and tolerant precision, recall, and F1.
- Segmentation IoU against the unchanged semantic ground truth.
- Detection AP against the unchanged instance-derived boxes.
- Per-class and aggregate results for every distortion level.

SNR is calculated exactly in the form used in the slides:

```text
SNR(dB) = 10 * log10(mean(clean²) / mean((clean - distorted)²))
```

## Output files

```text
outputs/
├── run_manifest.json
├── part1/
│   ├── clean_summary.json
│   ├── clean_per_image.csv
│   ├── segmentation_per_class.csv
│   ├── detection_per_class.csv
│   └── figures/
│       └── clean_predictions.png
└── part2/
    ├── distorted_summary.json
    ├── distorted_summary.csv
    ├── distorted_per_image.csv
    ├── segmentation_per_class.csv
    ├── detection_per_class.csv
    └── figures/
        ├── distortion_grid.png
        ├── distorted_predictions.png
        └── performance_per_snr.png
```

The run manifest records the complete configuration used for reproducibility. CSV files are suitable for additional plots or report tables, while the generated figures can be embedded directly in the course README or presentation.

## Metric details

### ORB

Descriptors are matched with a Hamming-distance brute-force matcher and Lowe's ratio test. Because all four distortions preserve image geometry, a match is retained only when the matched keypoints are within three pixels of one another.

```text
match retention = spatially verified matches / clean keypoints
inlier ratio    = spatially verified matches / ratio-test matches
```

### Canny edge detection

Canny uses fixed thresholds (`100`, `200`) and a fixed 5x5 Gaussian pre-filter for clean and distorted images. Parameters are not retuned by distortion because doing so would hide robustness loss. Cityscapes does not provide edge annotations, so the clean Canny map is the reference for distorted-image evaluation. A two-pixel dilation tolerance permits small localization shifts. Precision, recall, F1, and edge-pixel retention are reported.

Motion blur uses normalized odd-length kernels, reflection padding, and a fixed 15-degree direction. Holding direction constant isolates kernel length as the experimental severity variable and avoids artificial dark image borders.

### Semantic segmentation

Predicted SegFormer logits are bilinearly resized to the original Cityscapes resolution. Void ground-truth pixels (`255`) are ignored. IoU is computed per class and then averaged only over classes present in the evaluated split/subset.

### Object detection

For each Cityscapes instance, its visible-pixel extent becomes an `xyxy` ground-truth box. Predictions are matched greedily by class and confidence. AP uses 101-point interpolation at IoU thresholds 0.50 through 0.95 in increments of 0.05.

YOLO inference uses a low confidence threshold (`0.001`) for AP calculation so the precision-recall curve is not truncated. Qualitative plots use `0.25`, matching the course slides.

## Reproducibility and scope notes

- The default seed is `7`, matching the slide example.
- Sample selection and all distortions are deterministic.
- No image restoration or enhancement is performed.
- No model is trained or fine-tuned.
- The code measures robustness only on clean and synthetically distorted Cityscapes images.
- ORB and Canny are classical low-level methods; SegFormer and YOLO satisfy the deep-learning requirement.

## Verification

The unit tests exercise dataset discovery, mask-to-box conversion, all distortions, Canny consistency, SNR, semantic metrics, box IoU, and detection AP without downloading model weights:

```powershell
python -m unittest discover -s tests -v
```

## References

- [Cityscapes dataset](https://www.cityscapes-dataset.com/)
- [Cityscapes benchmark metrics](https://www.cityscapes-dataset.com/benchmarks/)
- [SegFormer-B0 fine-tuned on Cityscapes](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024)
- [Ultralytics Python API](https://docs.ultralytics.com/usage/python/)
