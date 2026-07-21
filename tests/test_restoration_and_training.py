import sys
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import cityscapes_project.pipelines.parts34 as project
from cityscapes_project.config import Parts34Config
from cityscapes_project.methods.distortions import apply_aug, compute_snr
from cityscapes_project.methods.restoration import restoration_parameters, restore_image
from cityscapes_project.methods.quality import compute_mae, compute_psnr, compute_ssim
from cityscapes_project.utils.statistics import paired_bootstrap
from cityscapes_project.pipelines.parts34 import (
    PROJECT_CLASS_TO_ID,
    choose_training_condition,
    detection_to_yolo_row,
)
from cityscapes_project.types import Detection


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

    def test_restoration_strength_tracks_distortion_severity(self) -> None:
        mild_noise = restoration_parameters("GaussNoise", 5.0)
        severe_noise = restoration_parameters("GaussNoise", 50.0)
        self.assertLess(mild_noise["h_luminance"], severe_noise["h_luminance"])
        mild_darkness = restoration_parameters("LowLight", 0.8)
        severe_darkness = restoration_parameters("LowLight", 0.1)
        self.assertGreater(mild_darkness["gamma"], severe_darkness["gamma"])
        self.assertLess(mild_darkness["blend"], severe_darkness["blend"])

    def test_low_light_restoration_improves_snr_at_mild_and_severe_levels(self) -> None:
        horizontal = np.linspace(20, 235, 128, dtype=np.uint8)
        clean = np.repeat(horizontal[None, :, None], 64, axis=0)
        clean = np.repeat(clean, 3, axis=2)
        image = Image.fromarray(clean)
        for level in (0.8, 0.1):
            with self.subTest(level=level):
                distorted = apply_aug(image, "LowLight", level)
                restored = restore_image(distorted, "LowLight", level)
                self.assertGreater(
                    compute_snr(clean, restored), compute_snr(clean, distorted)
                )

    def test_gaussian_restoration_is_bounded_and_severity_aware(self) -> None:
        mild = restore_image(self.image, "GaussNoise", 5.0)
        severe = restore_image(self.image, "GaussNoise", 50.0)
        mild_change = float(np.mean(np.abs(mild.astype(float) - self.image.astype(float))))
        severe_change = float(np.mean(np.abs(severe.astype(float) - self.image.astype(float))))
        self.assertGreater(severe_change, mild_change)

    def test_jpeg_deblocking_targets_block_boundaries(self) -> None:
        restored = restore_image(self.image, "SevereJPEG", 5.0)
        change = np.mean(np.abs(restored.astype(float) - self.image.astype(float)), axis=2)
        boundary = np.zeros(change.shape, dtype=bool)
        boundary[:, 7:10] = True
        boundary[:, 15:18] = True
        boundary[:, 23:26] = True
        boundary[:, 31:34] = True
        boundary[:, 39:42] = True
        boundary[:, 47:50] = True
        boundary[:, 55:58] = True
        boundary[7:10, :] = True
        boundary[15:18, :] = True
        boundary[23:26, :] = True
        boundary[31:34, :] = True
        boundary[39:42, :] = True
        self.assertGreater(float(np.mean(change[boundary])), float(np.mean(change[~boundary])))

    def test_regularized_motion_restoration_improves_structured_blur(self) -> None:
        x = np.linspace(0, 255, 128, dtype=np.uint8)
        clean = np.repeat(x[None, :, None], 64, axis=0)
        clean = np.repeat(clean, 3, axis=2)
        clean[12:52, 28:34] = 255
        clean[18:45, 75:104] = 25
        distorted = apply_aug(Image.fromarray(clean), "MotionBlur", 15.0)
        restored = restore_image(distorted, "MotionBlur", 15.0)
        self.assertGreater(compute_psnr(clean, restored), compute_psnr(clean, distorted))


class QualityAndStatisticsTests(unittest.TestCase):
    def test_reference_metrics_are_exact_for_identical_images(self) -> None:
        image = np.full((20, 24, 3), 73, dtype=np.uint8)
        self.assertTrue(np.isinf(compute_psnr(image, image)))
        self.assertAlmostEqual(compute_ssim(image, image), 1.0)
        self.assertEqual(compute_mae(image, image), 0.0)

    def test_quality_metrics_order_degradation(self) -> None:
        reference = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        reference = np.repeat(reference[..., None], 3, axis=2)
        mild = np.clip(reference.astype(int) + 3, 0, 255).astype(np.uint8)
        severe = np.clip(reference.astype(int) + 20, 0, 255).astype(np.uint8)
        self.assertGreater(compute_psnr(reference, mild), compute_psnr(reference, severe))
        self.assertLess(compute_mae(reference, mild), compute_mae(reference, severe))

    def test_paired_bootstrap_is_deterministic_and_orients_mae(self) -> None:
        before = [10.0, 8.0, 12.0, 9.0]
        after = [8.0, 7.0, 9.0, 9.0]
        first = paired_bootstrap(
            before, after, higher_is_better=False, resamples=200, seed=11
        )
        second = paired_bootstrap(
            before, after, higher_is_better=False, resamples=200, seed=11
        )
        self.assertEqual(first, second)
        self.assertGreater(first["mean_improvement"], 0.0)
        self.assertEqual(first["win_rate"], 0.75)


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
            self.assertEqual(captured["warmup_epochs"], 0.5)

    def test_prepared_training_data_uses_png_and_records_class_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cityscapes"
            for split in ("train", "val"):
                image_dir = root / "leftImg8bit" / split / "sample_city"
                gt_dir = root / "gtFine" / split / "sample_city"
                image_dir.mkdir(parents=True)
                gt_dir.mkdir(parents=True)
                sample_id = f"sample_city_{split}_000001"
                image = np.full((16, 24, 3), 120, dtype=np.uint8)
                labels = np.zeros((16, 24), dtype=np.uint8)
                instances = np.zeros((16, 24), dtype=np.uint16)
                instances[2:12, 4:20] = 26001
                Image.fromarray(image).save(image_dir / f"{sample_id}_leftImg8bit.png")
                Image.fromarray(labels).save(gt_dir / f"{sample_id}_gtFine_labelTrainIds.png")
                Image.fromarray(instances).save(gt_dir / f"{sample_id}_gtFine_instanceIds.png")

            config = Parts34Config(
                dataset_root=root,
                artifacts_dir=Path(directory) / "artifacts",
                part4_clean_fraction=1.0,
            )
            _, prepared = project.prepare_yolo_dataset(config)
            self.assertEqual(len(list((prepared / "images" / "train").glob("*.png"))), 1)
            self.assertEqual(len(list((prepared / "images" / "train").glob("*.jpg"))), 0)
            manifest = json.loads((prepared / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["recipe_version"], 2)
            self.assertEqual(manifest["class_instance_counts"]["train"]["car"], 1)


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
                patch.object(project, "save_restoration_quality_plot"),
            ):
                result = project.run_part3(config, None, None, None, "cpu")

            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(len(result["variants"]), 1)
            self.assertIn("seg_restored", result["variants"][0])
            self.assertIn("det_restored", result["variants"][0])
            self.assertIn("psnr_gain_db", result["variants"][0])
            self.assertTrue((output / "part3" / "restoration_summary.json").is_file())
            self.assertTrue((output / "part3" / "restoration_per_image.csv").is_file())
            self.assertTrue((output / "part3" / "segmentation_per_class.csv").is_file())
            self.assertTrue((output / "part3" / "detection_per_class.csv").is_file())
            self.assertTrue((output / "part3" / "paired_statistics.csv").is_file())
            self.assertTrue((output / "part3" / "restoration_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
