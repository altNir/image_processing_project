"""Compute-device selection and pretrained model loading."""

from __future__ import annotations

import logging
from typing import Any

from cityscapes_project.config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


def select_device(requested: str) -> str:
    """Resolve ``auto`` or validate an explicitly requested PyTorch device."""

    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. Install the CUDA-enabled "
            "PyTorch wheel with setup_cuda.ps1, then retry."
        )
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(config: ExperimentConfig) -> tuple[Any, Any, Any, str]:
    """Load the pretrained YOLO and SegFormer models once for a pipeline run."""

    try:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Model dependencies are missing. Install requirements.txt first.") from exc

    device = select_device(config.device)
    LOGGER.info("Loading YOLO model %s", config.yolo_model)
    detector = YOLO(config.yolo_model)
    detector.to(device)
    LOGGER.info("Loading SegFormer model %s", config.segformer_model)
    processor = AutoImageProcessor.from_pretrained(config.segformer_model)
    segmenter = SegformerForSemanticSegmentation.from_pretrained(config.segformer_model)
    segmenter.to(device)
    segmenter.eval()
    LOGGER.info(
        "Inference precision: %s",
        "FP16 autocast" if config.use_half and device.startswith("cuda") else "FP32",
    )
    return detector, processor, segmenter, device
