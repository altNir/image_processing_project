"""Lazy optional-dependency imports with actionable errors."""

from typing import Any


def cv2_module() -> Any:
    """Import OpenCV only when an OpenCV-backed method is used."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required. Install requirements.txt first.") from exc
    return cv2


def matplotlib_pyplot() -> Any:
    """Import matplotlib with a non-interactive backend."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Matplotlib is required. Install requirements.txt first.") from exc
    return plt
