"""Full-reference image-quality metrics used by Part 3."""

from __future__ import annotations

import math

import numpy as np

from cityscapes_project.methods.distortions import compute_snr
from cityscapes_project.utils.dependencies import cv2_module


def compute_mae(reference_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    """Mean absolute RGB error in the native 0--255 intensity range."""

    reference = np.asarray(reference_rgb, dtype=np.float32)
    test = np.asarray(test_rgb, dtype=np.float32)
    if reference.shape != test.shape:
        raise ValueError("Reference and test images must have identical shapes")
    return float(np.mean(np.abs(reference - test)))


def compute_psnr(reference_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    """Peak signal-to-noise ratio with an 8-bit peak value."""

    reference = np.asarray(reference_rgb, dtype=np.float32)
    test = np.asarray(test_rgb, dtype=np.float32)
    if reference.shape != test.shape:
        raise ValueError("Reference and test images must have identical shapes")
    mse = float(np.mean((reference - test) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * math.log10((255.0**2) / mse))


def compute_ssim(reference_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    """Luminance SSIM using the standard 11x11 Gaussian window.

    Computing the structural term on luminance follows common image-quality
    practice and keeps memory bounded for native 2048x1024 Cityscapes images.
    """

    cv2 = cv2_module()
    reference = np.asarray(reference_rgb)
    test = np.asarray(test_rgb)
    if reference.shape != test.shape:
        raise ValueError("Reference and test images must have identical shapes")
    if reference.ndim not in (2, 3):
        raise ValueError("SSIM expects a grayscale or color image")
    if reference.ndim == 3:
        if reference.shape[2] != 3:
            raise ValueError("Color SSIM expects exactly three RGB channels")
        reference = cv2.cvtColor(reference.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        test = cv2.cvtColor(test.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    reference = reference.astype(np.float32)
    test = test.astype(np.float32)
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    mu_ref = cv2.GaussianBlur(reference, (11, 11), 1.5)
    mu_test = cv2.GaussianBlur(test, (11, 11), 1.5)
    ref_sq, test_sq, ref_test = mu_ref**2, mu_test**2, mu_ref * mu_test
    sigma_ref = cv2.GaussianBlur(reference**2, (11, 11), 1.5) - ref_sq
    sigma_test = cv2.GaussianBlur(test**2, (11, 11), 1.5) - test_sq
    covariance = cv2.GaussianBlur(reference * test, (11, 11), 1.5) - ref_test
    numerator = (2.0 * ref_test + c1) * (2.0 * covariance + c2)
    denominator = (ref_sq + test_sq + c1) * (sigma_ref + sigma_test + c2)
    score = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator != 0)
    return float(np.clip(np.mean(score), -1.0, 1.0))


def compute_quality_metrics(
    reference_rgb: np.ndarray, test_rgb: np.ndarray
) -> dict[str, float]:
    """Return complementary pixel fidelity and structural quality measurements."""

    return {
        "snr_db": compute_snr(reference_rgb, test_rgb),
        "psnr_db": compute_psnr(reference_rgb, test_rgb),
        "ssim": compute_ssim(reference_rgb, test_rgb),
        "mae": compute_mae(reference_rgb, test_rgb),
    }
