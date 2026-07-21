"""Small data objects shared across the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CityscapesSample:
    """Paths belonging to one Cityscapes image and its annotations."""

    sample_id: str
    image_path: Path
    label_path: Path
    instance_path: Path


@dataclass
class Detection:
    """One object-detection prediction or ground-truth box."""

    image_id: str
    class_name: str
    bbox: tuple[float, float, float, float]
    score: float = 1.0
