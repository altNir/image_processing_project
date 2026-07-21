"""Backward-compatible launcher for Parts 3 and 4.

The implementation now lives in the focused :mod:`cityscapes_project`
package. Existing imports and commands remain supported by this module.
"""

from cityscapes_project.cli import build_parts34_parser as build_parser
from cityscapes_project.cli import parts34_main as main
from cityscapes_project.config import *  # noqa: F401,F403
from cityscapes_project.methods.detection import model_detections
from cityscapes_project.methods.restoration import *  # noqa: F401,F403
from cityscapes_project.pipelines.parts34 import *  # noqa: F401,F403
from cityscapes_project.types import Detection
from cityscapes_project.utils.dependencies import cv2_module as _cv2
from cityscapes_project.utils.dependencies import matplotlib_pyplot as _matplotlib
from cityscapes_project.utils.device import load_models, select_device
from cityscapes_project.utils.visualization import (
    save_fine_tuning_plot,
    save_restoration_gallery,
    save_restoration_plot,
)


if __name__ == "__main__":
    raise SystemExit(main())
