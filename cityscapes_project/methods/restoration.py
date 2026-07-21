"""Part 3 image-restoration methods."""

from __future__ import annotations

import numpy as np

from cityscapes_project.methods.distortions import motion_blur_kernel
from cityscapes_project.utils.dependencies import cv2_module


def restoration_parameters(distortion_name: str, level: float) -> dict[str, float | int]:
    """Return the deterministic, severity-aware parameters used for restoration."""

    if distortion_name == "GaussNoise":
        sigma = float(level)
        return {
            "h_luminance": max(2.0, 0.45 * sigma),
            "h_color": max(2.0, 0.40 * sigma),
            "bilateral_sigma": 0.0 if sigma < 35.0 else 10.0 + 0.35 * sigma,
        }
    if distortion_name == "SevereJPEG":
        quality = float(level)
        severity = float(np.clip((100.0 - quality) / 95.0, 0.0, 1.0))
        diameter = 3 + 2 * int(round(3.0 * severity))
        return {
            "diameter": diameter,
            "sigma_color": 6.0 + 30.0 * severity,
            "sigma_space": 6.0 + 30.0 * severity,
            "blend": severity**4,
        }
    if distortion_name == "LowLight":
        factor = float(level)
        severity = float(np.clip((1.0 - factor) / 0.9, 0.0, 1.0))
        return {
            "gamma": 1.0 - 0.62 * severity,
            "clahe_clip": 1.0 + 1.8 * severity,
            "blend": 0.15 + 0.85 * severity,
        }
    if distortion_name == "MotionBlur":
        kernel_size = int(level)
        return {
            "kernel_size": kernel_size,
            "balance": 0.006 + 0.00045 * kernel_size,
            "blend": min(0.90, 0.15 + 0.030 * kernel_size),
        }
    raise KeyError(f"No restoration is registered for {distortion_name}")


def restore_gaussian_noise(image_rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Denoise using strength scaled to the known synthetic noise sigma."""

    cv2 = cv2_module()
    parameters = restoration_parameters("GaussNoise", sigma)
    bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        float(parameters["h_luminance"]),
        float(parameters["h_color"]),
        7,
        21,
    )
    bilateral_sigma = float(parameters["bilateral_sigma"])
    if bilateral_sigma > 0.0:
        denoised = cv2.bilateralFilter(
            denoised, d=5, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma
        )
    return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)


def restore_jpeg(image_rgb: np.ndarray, quality: float) -> np.ndarray:
    """Reduce blocking and ringing on luminance while retaining chroma."""

    cv2 = cv2_module()
    parameters = restoration_parameters("SevereJPEG", quality)
    ycrcb = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)
    y_channel = cv2.bilateralFilter(
        y_channel,
        d=int(parameters["diameter"]),
        sigmaColor=float(parameters["sigma_color"]),
        sigmaSpace=float(parameters["sigma_space"]),
    )
    restored = cv2.merge((y_channel, cr, cb))
    filtered = cv2.cvtColor(restored, cv2.COLOR_YCrCb2RGB)
    blend = float(parameters["blend"])
    return cv2.addWeighted(filtered, blend, np.asarray(image_rgb), 1.0 - blend, 0)


def restore_low_light(image_rgb: np.ndarray, factor: float) -> np.ndarray:
    """Apply severity-scaled gamma lifting and CLAHE without overprocessing mild darkness."""

    cv2 = cv2_module()
    image = np.asarray(image_rgb, dtype=np.uint8)
    parameters = restoration_parameters("LowLight", factor)
    gamma = float(parameters["gamma"])
    lookup = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0).clip(0, 255)
    lifted = cv2.LUT(image, lookup.astype(np.uint8))
    lab = cv2.cvtColor(lifted, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(parameters["clahe_clip"]), tileGridSize=(8, 8)
    )
    lightness = clahe.apply(lightness)
    restored = cv2.merge((lightness, a_channel, b_channel))
    enhanced = cv2.cvtColor(restored, cv2.COLOR_LAB2RGB)
    blend = float(parameters["blend"])
    return cv2.addWeighted(enhanced, blend, image, 1.0 - blend, 0)


def _wiener_deconvolution(
    image_rgb: np.ndarray,
    kernel_size: int,
    balance: float,
) -> np.ndarray:
    """Deblur RGB channels with a regularized frequency-domain Wiener inverse."""

    image = np.asarray(image_rgb, dtype=np.float32) / 255.0
    pad = kernel_size
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    psf = motion_blur_kernel(kernel_size)
    transfer = np.zeros(padded.shape[:2], dtype=np.float32)
    transfer[:kernel_size, :kernel_size] = psf
    transfer = np.roll(transfer, -(kernel_size // 2), axis=(0, 1))
    frequency_response = np.fft.fft2(transfer)
    inverse = np.conj(frequency_response) / (
        np.abs(frequency_response) ** 2 + float(balance)
    )
    restored_channels = []
    for channel in range(3):
        observed = np.fft.fft2(padded[..., channel])
        restored_channels.append(np.fft.ifft2(observed * inverse).real)
    restored = np.stack(restored_channels, axis=2)[pad:-pad, pad:-pad]
    return (restored.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def restore_motion_blur(image_rgb: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply severity-scaled Wiener deconvolution for the known motion PSF."""

    cv2 = cv2_module()
    image = np.asarray(image_rgb, dtype=np.uint8)
    parameters = restoration_parameters("MotionBlur", float(kernel_size))
    deconvolved = _wiener_deconvolution(
        image, kernel_size, float(parameters["balance"])
    )
    blend = float(parameters["blend"])
    return cv2.addWeighted(deconvolved, blend, image, 1.0 - blend, 0)


def restore_image(image_rgb: np.ndarray, distortion_name: str, level: float) -> np.ndarray:
    """Dispatch a distorted image to its matching restoration method."""

    if distortion_name == "GaussNoise":
        return restore_gaussian_noise(image_rgb, float(level))
    if distortion_name == "SevereJPEG":
        return restore_jpeg(image_rgb, float(level))
    if distortion_name == "LowLight":
        return restore_low_light(image_rgb, float(level))
    if distortion_name == "MotionBlur":
        return restore_motion_blur(image_rgb, int(level))
    raise KeyError(f"No restoration is registered for {distortion_name}")
