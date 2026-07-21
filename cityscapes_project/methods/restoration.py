"""Deterministic, severity-aware classical restoration methods for Part 3."""

from __future__ import annotations

from typing import Any

import numpy as np

from cityscapes_project.methods.distortions import motion_blur_kernel
from cityscapes_project.utils.dependencies import cv2_module


RESTORATION_RECIPE_VERSION = 3

RESTORATION_METHODS: dict[str, dict[str, Any]] = {
    "GaussNoise": {
        "method": "severity-aware non-local means with edge-preserving residual cleanup",
        "reference": "Buades, Coll and Morel, CVPR 2005",
        "doi": "https://doi.org/10.1109/CVPR.2005.38",
    },
    "SevereJPEG": {
        "method": "8x8 boundary-aware luminance deblocking",
        "reference": "signal-adaptive JPEG deblocking principles",
        "doi": "https://doi.org/10.1109/83.661000",
    },
    "LowLight": {
        "method": "exposure-compensating gamma lift followed by luminance CLAHE",
        "reference": "contrast-limited adaptive histogram equalization",
        "doi": "https://doi.org/10.1016/0734-189X(87)90186-X",
    },
    "MotionBlur": {
        "method": "known-PSF Tikhonov deconvolution with Laplacian regularization",
        "reference": "regularized linear inverse restoration",
        "doi": "https://doi.org/10.2307/2006224",
    },
}


def restoration_parameters(distortion_name: str, level: float) -> dict[str, float | int]:
    """Return the fixed parameter recipe for a distortion and severity.

    Parameters depend only on the declared synthetic distortion level. This avoids
    tuning on the evaluation images and makes every result exactly reproducible.
    """

    if distortion_name == "GaussNoise":
        sigma = float(level)
        severity = float(np.clip((sigma - 5.0) / 45.0, 0.0, 1.0))
        return {
            "h_luminance": max(2.0, 0.46 * sigma),
            "h_color": max(2.0, 0.40 * sigma),
            "template_window": 7,
            "search_window": 21,
            "blend": 0.30 + 0.65 * severity,
            "bilateral_sigma": 0.0 if sigma < 35.0 else 8.0 + 0.30 * sigma,
        }
    if distortion_name == "SevereJPEG":
        quality = float(level)
        severity = float(np.clip((80.0 - quality) / 75.0, 0.0, 1.0))
        return {
            "block_size": 8,
            "boundary_radius": 1 + int(round(2.0 * severity)),
            "diameter": 5 + 2 * int(round(2.0 * severity)),
            "sigma_color": 8.0 + 36.0 * severity,
            "sigma_space": 4.0 + 12.0 * severity,
            # A steep schedule avoids softening moderate JPEG images while still
            # treating conspicuous boundaries at quality 5 aggressively.
            "boundary_blend": 0.15 + 0.65 * severity**4,
            "global_blend": 0.01 + 0.06 * severity**6,
        }
    if distortion_name == "LowLight":
        factor = float(level)
        severity = float(np.clip((0.8 - factor) / 0.7, 0.0, 1.0))
        return {
            "gamma": 0.88 - 0.36 * severity,
            "clahe_clip": 1.15 + 1.55 * severity,
            "clahe_grid": 8,
            "blend": 0.42 + 0.50 * severity,
        }
    if distortion_name == "MotionBlur":
        kernel_size = int(level)
        if kernel_size <= 15:
            blend = 0.45 + 0.25 * (kernel_size - 3.0) / 12.0
        else:
            blend = 0.70 - 0.30 * (kernel_size - 15.0) / 10.0
        return {
            "kernel_size": kernel_size,
            # Larger kernels have deeper spectral nulls and therefore need a
            # strongly nonlinear regularization increase to suppress ringing.
            "regularization": 0.0015 + 0.00060 * (kernel_size - 3) ** 2,
            "blend": float(np.clip(blend, 0.40, 0.70)),
            "padding": max(12, 2 * kernel_size),
        }
    raise KeyError(f"No restoration is registered for {distortion_name}")


def restore_gaussian_noise(image_rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Denoise with NLM and retain more original detail at mild severities."""

    cv2 = cv2_module()
    image = np.asarray(image_rgb, dtype=np.uint8)
    parameters = restoration_parameters("GaussNoise", sigma)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        float(parameters["h_luminance"]),
        float(parameters["h_color"]),
        int(parameters["template_window"]),
        int(parameters["search_window"]),
    )
    bilateral_sigma = float(parameters["bilateral_sigma"])
    if bilateral_sigma > 0.0:
        denoised = cv2.bilateralFilter(
            denoised, d=5, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma
        )
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    blend = float(parameters["blend"])
    return cv2.addWeighted(denoised_rgb, blend, image, 1.0 - blend, 0)


def _jpeg_boundary_weights(height: int, width: int, block_size: int, radius: int) -> np.ndarray:
    """Create a soft mask around internal JPEG block boundaries."""

    weights = np.zeros((height, width), dtype=np.float32)
    for boundary in range(block_size, width, block_size):
        for offset in range(-radius, radius + 1):
            column = boundary + offset
            if 0 <= column < width:
                weights[:, column] = np.maximum(
                    weights[:, column], 1.0 - abs(offset) / (radius + 1.0)
                )
    for boundary in range(block_size, height, block_size):
        for offset in range(-radius, radius + 1):
            row = boundary + offset
            if 0 <= row < height:
                weights[row, :] = np.maximum(
                    weights[row, :], 1.0 - abs(offset) / (radius + 1.0)
                )
    return weights


def restore_jpeg(image_rgb: np.ndarray, quality: float) -> np.ndarray:
    """Deblock known 8x8 JPEG boundaries without globally blurring textures."""

    cv2 = cv2_module()
    image = np.asarray(image_rgb, dtype=np.uint8)
    parameters = restoration_parameters("SevereJPEG", quality)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    luminance = ycrcb[..., 0]
    filtered = cv2.bilateralFilter(
        luminance,
        d=int(parameters["diameter"]),
        sigmaColor=float(parameters["sigma_color"]),
        sigmaSpace=float(parameters["sigma_space"]),
    )
    weights = _jpeg_boundary_weights(
        luminance.shape[0], luminance.shape[1],
        int(parameters["block_size"]), int(parameters["boundary_radius"]),
    )
    alpha = (
        float(parameters["global_blend"])
        + float(parameters["boundary_blend"]) * weights
    ).clip(0.0, 1.0)
    blended = (
        luminance.astype(np.float32) * (1.0 - alpha)
        + filtered.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)
    restored_ycrcb = ycrcb.copy()
    restored_ycrcb[..., 0] = blended
    return cv2.cvtColor(restored_ycrcb, cv2.COLOR_YCrCb2RGB)


def restore_low_light(image_rgb: np.ndarray, factor: float) -> np.ndarray:
    """Lift exposure and locally normalize luminance with bounded enhancement."""

    cv2 = cv2_module()
    image = np.asarray(image_rgb, dtype=np.uint8)
    parameters = restoration_parameters("LowLight", factor)
    gamma = float(parameters["gamma"])
    lookup = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0).clip(0, 255)
    lifted = cv2.LUT(image, lookup.astype(np.uint8))
    lab = cv2.cvtColor(lifted, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(parameters["clahe_clip"]),
        tileGridSize=(int(parameters["clahe_grid"]),) * 2,
    )
    enhanced = cv2.cvtColor(
        cv2.merge((clahe.apply(lightness), a_channel, b_channel)),
        cv2.COLOR_LAB2RGB,
    )
    blend = float(parameters["blend"])
    return cv2.addWeighted(enhanced, blend, image, 1.0 - blend, 0)


def _tikhonov_deconvolution(
    image_rgb: np.ndarray,
    kernel_size: int,
    regularization: float,
    padding: int,
) -> np.ndarray:
    """Known-PSF inverse with a Laplacian smoothness prior and reflected borders."""

    image = np.asarray(image_rgb, dtype=np.float32) / 255.0
    padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="reflect")
    psf = motion_blur_kernel(kernel_size)
    transfer = np.zeros(padded.shape[:2], dtype=np.float32)
    transfer[:kernel_size, :kernel_size] = psf
    transfer = np.roll(transfer, -(kernel_size // 2), axis=(0, 1))
    h_frequency = np.fft.fft2(transfer)

    laplacian = np.zeros(padded.shape[:2], dtype=np.float32)
    laplacian[0, 0] = 4.0
    laplacian[0, 1] = laplacian[1, 0] = -1.0
    laplacian[0, -1] = laplacian[-1, 0] = -1.0
    l_frequency = np.fft.fft2(laplacian)
    inverse = np.conj(h_frequency) / (
        np.abs(h_frequency) ** 2
        + float(regularization) * np.abs(l_frequency) ** 2
        + 1e-8
    )
    restored_channels = [
        np.fft.ifft2(np.fft.fft2(padded[..., channel]) * inverse).real
        for channel in range(3)
    ]
    restored = np.stack(restored_channels, axis=2)[
        padding:-padding, padding:-padding
    ]
    return np.rint(restored.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def restore_motion_blur(image_rgb: np.ndarray, kernel_size: int) -> np.ndarray:
    """Deblur using the exact synthetic PSF and regularized inversion."""

    cv2 = cv2_module()
    image = np.asarray(image_rgb, dtype=np.uint8)
    parameters = restoration_parameters("MotionBlur", float(kernel_size))
    deconvolved = _tikhonov_deconvolution(
        image,
        kernel_size,
        float(parameters["regularization"]),
        int(parameters["padding"]),
    )
    blend = float(parameters["blend"])
    return cv2.addWeighted(deconvolved, blend, image, 1.0 - blend, 0)


def restore_image_with_metadata(
    image_rgb: np.ndarray, distortion_name: str, level: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Restore an image and return the complete auditable recipe metadata."""

    parameters = restoration_parameters(distortion_name, level)
    if distortion_name == "GaussNoise":
        restored = restore_gaussian_noise(image_rgb, float(level))
    elif distortion_name == "SevereJPEG":
        restored = restore_jpeg(image_rgb, float(level))
    elif distortion_name == "LowLight":
        restored = restore_low_light(image_rgb, float(level))
    elif distortion_name == "MotionBlur":
        restored = restore_motion_blur(image_rgb, int(level))
    else:
        raise KeyError(f"No restoration is registered for {distortion_name}")
    metadata = {
        "recipe_version": RESTORATION_RECIPE_VERSION,
        **RESTORATION_METHODS[distortion_name],
        "parameters": parameters,
    }
    return restored, metadata


def restore_image(image_rgb: np.ndarray, distortion_name: str, level: float) -> np.ndarray:
    """Backward-compatible restoration dispatcher."""

    return restore_image_with_metadata(image_rgb, distortion_name, level)[0]
