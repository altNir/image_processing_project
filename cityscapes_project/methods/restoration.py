"""Part 3 image-restoration methods."""

from __future__ import annotations

import numpy as np

from cityscapes_project.utils.dependencies import cv2_module


def restore_gaussian_noise(image_rgb: np.ndarray) -> np.ndarray:
    """Denoise with non-local means and a light edge-preserving filter."""

    cv2 = cv2_module()
    bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 25, 25, 7, 21)
    denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=45, sigmaSpace=45)
    return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)


def restore_jpeg(image_rgb: np.ndarray) -> np.ndarray:
    """Reduce blocking and ringing on luminance while retaining chroma."""

    cv2 = cv2_module()
    ycrcb = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)
    y_channel = cv2.bilateralFilter(y_channel, d=7, sigmaColor=35, sigmaSpace=35)
    restored = cv2.merge((y_channel, cr, cb))
    return cv2.cvtColor(restored, cv2.COLOR_YCrCb2RGB)


def restore_low_light(image_rgb: np.ndarray) -> np.ndarray:
    """Lift gamma, then apply CLAHE to LAB luminance."""

    cv2 = cv2_module()
    image = np.asarray(image_rgb, dtype=np.uint8)
    gamma = 0.45
    lookup = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0).clip(0, 255)
    lifted = cv2.LUT(image, lookup.astype(np.uint8))
    lab = cv2.cvtColor(lifted, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    restored = cv2.merge((lightness, a_channel, b_channel))
    return cv2.cvtColor(restored, cv2.COLOR_LAB2RGB)


def restore_motion_blur(image_rgb: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply fast unsharp deblurring scaled to the synthetic blur length."""

    cv2 = cv2_module()
    image = np.asarray(image_rgb, dtype=np.uint8)
    sigma = max(0.8, float(kernel_size) / 6.0)
    smooth = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    amount = min(1.6, 0.65 + float(kernel_size) / 25.0)
    sharpened = cv2.addWeighted(image, 1.0 + amount, smooth, -amount, 0)
    return np.asarray(sharpened, dtype=np.uint8)


def restore_image(image_rgb: np.ndarray, distortion_name: str, level: float) -> np.ndarray:
    """Dispatch a distorted image to its matching restoration method."""

    if distortion_name == "GaussNoise":
        return restore_gaussian_noise(image_rgb)
    if distortion_name == "SevereJPEG":
        return restore_jpeg(image_rgb)
    if distortion_name == "LowLight":
        return restore_low_light(image_rgb)
    if distortion_name == "MotionBlur":
        return restore_motion_blur(image_rgb, int(level))
    raise KeyError(f"No restoration is registered for {distortion_name}")
