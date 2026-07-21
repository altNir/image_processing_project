"""Deterministic synthetic distortions and image-quality measurement."""

from __future__ import annotations

import io
import math
from typing import Callable

import numpy as np
from PIL import Image

from ..utils.dependencies import cv2_module


def gaussian_noise(image_rgb: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Add seeded Gaussian noise on the CPU; transfer overhead makes GPU use unhelpful here."""

    noise = np.random.default_rng(seed).normal(0.0, sigma, size=image_rgb.shape)
    return (image_rgb.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


def jpeg_compression(image_rgb: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip an RGB image through JPEG at the requested quality."""

    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=int(quality), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return np.asarray(compressed.convert("RGB"), dtype=np.uint8)


def low_light(image_rgb: np.ndarray, brightness: float) -> np.ndarray:
    """Apply multiplicative brightness reduction."""

    return (image_rgb.astype(np.float32) * float(brightness)).clip(0, 255).astype(np.uint8)


def motion_blur(
    image_rgb: np.ndarray, kernel_size: int, angle_degrees: float = 15.0
) -> np.ndarray:
    """Apply a normalized fixed-angle motion point-spread function."""

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Motion-blur kernel_size must be a positive odd integer")
    if kernel_size == 1:
        return np.asarray(image_rgb, dtype=np.uint8).copy()
    cv2 = cv2_module()
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0
    radians = math.radians(float(angle_degrees))
    dx, dy = center * math.cos(radians), center * math.sin(radians)
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        color=1.0,
        thickness=1,
        lineType=cv2.LINE_8,
    )
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        kernel[int(center), int(center)], kernel_sum = 1.0, 1.0
    kernel /= kernel_sum
    return np.asarray(
        cv2.filter2D(
            np.asarray(image_rgb, dtype=np.uint8), -1, kernel, borderType=cv2.BORDER_REFLECT101
        ),
        dtype=np.uint8,
    )


def apply_aug(
    img_pil: Image.Image,
    distortion_name: str | Callable[[np.ndarray], np.ndarray],
    level: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Apply one named or callable distortion."""

    image = np.asarray(img_pil.convert("RGB"), dtype=np.uint8)
    if callable(distortion_name):
        return np.asarray(distortion_name(image), dtype=np.uint8)
    if level is None:
        raise ValueError("A numeric level is required for a named distortion")
    if distortion_name == "GaussNoise":
        return gaussian_noise(image, float(level), seed)
    if distortion_name == "SevereJPEG":
        return jpeg_compression(image, int(level))
    if distortion_name == "LowLight":
        return low_light(image, float(level))
    if distortion_name == "MotionBlur":
        kernel_size = int(level)
        if float(kernel_size) != float(level):
            raise ValueError("MotionBlur level must be an integer kernel size")
        return motion_blur(image, kernel_size)
    raise KeyError(f"Unknown distortion: {distortion_name}")


def compute_snr(clean_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    """Compute signal-to-noise ratio in decibels."""

    clean = np.asarray(clean_rgb, dtype=np.float64)
    noise = clean - np.asarray(test_rgb, dtype=np.float64)
    signal_power, noise_power = float(np.mean(clean**2)), float(np.mean(noise**2))
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return float("-inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def stable_distortion_seed(base_seed: int, sample_id: str, name: str, level_index: int) -> int:
    """Return a process-independent FNV-1a seed for one distortion."""

    value = 2166136261
    for byte in f"{base_seed}|{sample_id}|{name}|{level_index}".encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value
