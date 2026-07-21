import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import cityscapes_parts_3_4 as project
from cityscapes_parts_1_2 import Detection
from cityscapes_parts_3_4 import (
    PROJECT_CLASS_TO_ID,
    Parts34Config,
    choose_training_condition,
    detection_to_yolo_row,
    restore_image,
)


class RestorationTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.image = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)

    def test_every_part3_restoration_preserves_image_contract(self) -> None:
        variants = (
            ("GaussNoise", 20.0),
            ("SevereJPEG", 20.0),
            ("LowLight", 0.4),
            ("MotionBlur", 9.0),
        )
        for name, level in variants:
            with self.subTest(name=name):
                restored = restore_image(self.image, name, level)
                self.assertEqual(restored.shape, self.image.shape)
                self.assertEqual(restored.dtype, np.uint8)

    def test_unknown_restoration_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            restore_image(self.image, "Unknown", 1.0)


class YoloDatasetTests(unittest.TestCase):
    def test_detection_is_converted_to_normalized_yolo_row(self) -> None:
        detection = Detection("image", "car", (20.0, 10.0, 60.0, 30.0))
        values = detection_to_yolo_row(detection, width=100, height=50).split()
        self.assertEqual(int(values[0]), PROJECT_CLASS_TO_ID["car"])
        self.assertAlmostEqual(float(values[1]), 0.4)
        self.assertAlmostEqual(float(values[2]), 0.4)
        self.assertAlmostEqual(float(values[3]), 0.4)
        self.assertAlmostEqual(float(values[4]), 0.4)

    def test_training_condition_is_deterministic(self) -> None:
        levels = {"GaussNoise": (5.0, 10.0), "LowLight": (0.8, 0.4)}
        first = choose_training_condition(3, "sample", 7, levels, 0.2)
        second = choose_training_condition(3, "sample", 7, levels, 0.2)
        self.assertEqual(first, second)

    def test_clean_fraction_extremes_are_respected(self) -> None:
        levels = {"GaussNoise": (5.0,)}
        self.assertEqual(
            choose_training_condition(0, "sample", 7, levels, 1.0),
            ("Clean", None, 0),
        )
        self.assertEqual(
            choose_training_condition(0, "sample", 7, levels, 0.0)[0],
            "GaussNoise",
        )

    def test_train_yolo_returns_the_checkpoint_reported_by_ultralytics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual_best = root / "ultralytics-selected-run" / "weights" / "best.pt"
            actual_best.parent.mkdir(parents=True)
            actual_best.write_bytes(b"checkpoint")
            captured: dict[str, object] = {}

            class FakeYOLO:
                def __init__(self, _checkpoint: str) -> None:
                    self.trainer = None

                def train(self, **kwargs: object) -> None:
                    captured.update(kwargs)
                    self.trainer = types.SimpleNamespace(best=actual_best)

            fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
            config = Parts34Config(
                dataset_root=root,
                artifacts_dir=root / "relative-artifacts",
                part4_epochs=1,
            )
            yaml_path = root / "dataset.yaml"
            yaml_path.write_text("names: {}\n", encoding="utf-8")
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                returned = project.train_yolo(config, yaml_path, "cpu")

            self.assertEqual(returned, actual_best.resolve())
            self.assertTrue(Path(str(captured["project"])).is_absolute())
            self.assertIn("train-", str(captured["name"]))
            self.assertIn("val-", str(captured["name"]))


class Part3OrchestrationTests(unittest.TestCase):
    def test_part3_writes_distorted_and_restored_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cityscapes"
            output = Path(directory) / "outputs"
            image_dir = root / "leftImg8bit" / "val" / "sample_city"
            gt_dir = root / "gtFine" / "val" / "sample_city"
            image_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            sample_id = "sample_city_000000_000001"
            image = np.full((24, 32, 3), 150, dtype=np.uint8)
            image[:, 16:] = 80
            label = np.zeros((24, 32), dtype=np.uint8)
            instance = np.zeros((24, 32), dtype=np.uint16)
            instance[5:18, 8:25] = 26001
            Image.fromarray(image).save(image_dir / f"{sample_id}_leftImg8bit.png")
            Image.fromarray(label).save(gt_dir / f"{sample_id}_gtFine_labelTrainIds.png")
            Image.fromarray(instance).save(gt_dir / f"{sample_id}_gtFine_instanceIds.png")

            config = Parts34Config(
                dataset_root=root,
                output_dir=output,
                distortion_levels={"LowLight": (0.5,)},
                gallery_samples=0,
            )
            fake_orb = {
                "clean_keypoints": 10.0,
                "test_keypoints": 8.0,
                "keypoint_retention": 0.8,
                "ratio_matches": 8.0,
                "spatial_inliers": 7.0,
                "match_retention": 0.7,
                "inlier_ratio": 0.875,
            }

            def fake_yolo(_image, _model, image_id, *_args, **_kwargs):
                return [Detection(image_id, "car", (8.0, 5.0, 25.0, 18.0), score=0.9)]

            with (
                patch.object(project, "measure_orb_matching", return_value=fake_orb),
                patch.object(project, "predict_segmentation", return_value=label.astype(np.int32)),
                patch.object(project, "yolo_detections", side_effect=fake_yolo),
                patch.object(project, "save_restoration_gallery"),
                patch.object(project, "save_restoration_plot"),
            ):
                result = project.run_part3(config, None, None, None, "cpu")

            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(len(result["variants"]), 1)
            self.assertIn("seg_restored", result["variants"][0])
            self.assertIn("det_restored", result["variants"][0])
            self.assertTrue((output / "part3" / "restoration_summary.json").is_file())
            self.assertTrue((output / "part3" / "restoration_per_image.csv").is_file())
            self.assertTrue((output / "part3" / "segmentation_per_class.csv").is_file())
            self.assertTrue((output / "part3" / "detection_per_class.csv").is_file())


if __name__ == "__main__":
    unittest.main()
