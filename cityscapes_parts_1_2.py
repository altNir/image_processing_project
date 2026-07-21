"""Backward-compatible launcher for Parts 1 and 2.

The implementation now lives in the focused :mod:`cityscapes_project`
package. Existing imports and commands remain supported by this module.
"""

from cityscapes_project.cli import build_parts12_parser as build_parser
from cityscapes_project.cli import parts12_main as main
from cityscapes_project.config import *  # noqa: F401,F403
from cityscapes_project.dataset import *  # noqa: F401,F403
from cityscapes_project.methods.classical import *  # noqa: F401,F403
from cityscapes_project.methods.detection import *  # noqa: F401,F403
from cityscapes_project.methods.distortions import *  # noqa: F401,F403
from cityscapes_project.methods.segmentation import *  # noqa: F401,F403
from cityscapes_project.pipelines.parts12 import *  # noqa: F401,F403
from cityscapes_project.types import *  # noqa: F401,F403
from cityscapes_project.utils.dependencies import cv2_module as _cv2
from cityscapes_project.utils.dependencies import matplotlib_pyplot as _matplotlib
from cityscapes_project.utils.device import load_models, select_device
from cityscapes_project.utils.io import json_safe as _json_safe
from cityscapes_project.utils.io import write_csv, write_json
from cityscapes_project.utils.visualization import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
