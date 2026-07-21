"""Classical ORB feature and Canny edge methods."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..utils.dependencies import cv2_module


def orb_detect(img_pil: Image.Image, nfeatures: int = 800) -> tuple[list[Any], np.ndarray | None]:
    """Detect ORB keypoints and descriptors."""

    cv2 = cv2_module()
    gray = cv2.cvtColor(np.asarray(img_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return cv2.ORB_create(nfeatures=nfeatures).detectAndCompute(gray, None)


def orb_overlay(
    img_pil: Image.Image, nfeatures: int = 800
) -> tuple[np.ndarray, list[Any], np.ndarray | None]:
    """Draw rich ORB keypoints over an RGB image."""

    cv2 = cv2_module()
    image = np.asarray(img_pil.convert("RGB"))
    keypoints, descriptors = orb_detect(img_pil, nfeatures)
    output = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return output, keypoints, descriptors


def measure_orb_matching(
    clean_image: Image.Image,
    test_image: Image.Image,
    nfeatures: int = 800,
    ratio_threshold: float = 0.75,
    spatial_threshold: float = 3.0,
) -> dict[str, float]:
    """Measure descriptor and spatial retention for aligned image pairs."""

    cv2 = cv2_module()
    clean_kp, clean_desc = orb_detect(clean_image, nfeatures)
    test_kp, test_desc = orb_detect(test_image, nfeatures)
    clean_count, test_count = len(clean_kp), len(test_kp)
    if clean_desc is None or test_desc is None or clean_count == 0:
        return {
            "clean_keypoints": float(clean_count), "test_keypoints": float(test_count),
            "keypoint_retention": 0.0, "ratio_matches": 0.0, "spatial_inliers": 0.0,
            "match_retention": 0.0, "inlier_ratio": 0.0,
        }
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(
        clean_desc, test_desc, k=2
    )
    good = [
        first for pair in pairs if len(pair) == 2 for first, second in [pair]
        if first.distance < ratio_threshold * second.distance
    ]
    inliers = [
        match for match in good
        if float(np.linalg.norm(
            np.asarray(clean_kp[match.queryIdx].pt) - np.asarray(test_kp[match.trainIdx].pt)
        )) <= spatial_threshold
    ]
    return {
        "clean_keypoints": float(clean_count),
        "test_keypoints": float(test_count),
        "keypoint_retention": float(test_count / clean_count),
        "ratio_matches": float(len(good)),
        "spatial_inliers": float(len(inliers)),
        "match_retention": float(len(inliers) / clean_count),
        "inlier_ratio": float(len(inliers) / len(good)) if good else 0.0,
    }


def canny_detect(
    img_pil: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
    blur_kernel: int = 5,
) -> np.ndarray:
    """Return a fixed-parameter binary Canny edge map."""

    if low_threshold < 0 or high_threshold <= low_threshold:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high")
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError("Canny blur_kernel must be a positive odd integer")
    cv2 = cv2_module()
    gray = cv2.cvtColor(np.asarray(img_pil.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    return cv2.Canny(gray, low_threshold, high_threshold, L2gradient=True)


def canny_overlay(img_pil: Image.Image, edges: np.ndarray) -> np.ndarray:
    """Draw edges in green over a dimmed image."""

    output = (np.asarray(img_pil.convert("RGB"), dtype=np.float32) * 0.45).astype(np.uint8)
    output[np.asarray(edges) > 0] = np.asarray((0, 255, 0), dtype=np.uint8)
    return output


def evaluate_canny_edges(
    reference_edges: np.ndarray, test_edges: np.ndarray, tolerance_radius: int = 2
) -> dict[str, float]:
    """Compute spatially tolerant edge precision, recall, F1, and retention."""

    if tolerance_radius < 0:
        raise ValueError("tolerance_radius must be non-negative")
    reference = (np.asarray(reference_edges) > 0).astype(np.uint8)
    test = (np.asarray(test_edges) > 0).astype(np.uint8)
    if reference.shape != test.shape:
        raise ValueError("Reference and test edge maps must have the same shape")
    if tolerance_radius:
        cv2 = cv2_module()
        size = 2 * tolerance_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        reference_neighborhood = cv2.dilate(reference, kernel)
        test_neighborhood = cv2.dilate(test, kernel)
    else:
        reference_neighborhood, test_neighborhood = reference, test
    reference_count, test_count = int(reference.sum()), int(test.sum())
    matched_test = int(((test > 0) & (reference_neighborhood > 0)).sum())
    matched_reference = int(((reference > 0) & (test_neighborhood > 0)).sum())
    precision = matched_test / test_count if test_count else (1.0 if not reference_count else 0.0)
    recall = matched_reference / reference_count if reference_count else (1.0 if not test_count else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_edge_pixels": float(reference_count), "test_edge_pixels": float(test_count),
        "edge_pixel_retention": float(test_count / reference_count) if reference_count else (1.0 if not test_count else 0.0),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }
