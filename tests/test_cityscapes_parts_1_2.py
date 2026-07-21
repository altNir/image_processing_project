import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import cityscapes_parts_1_2 as project
from cityscapes_parts_1_2 import (
    Detection,
    ExperimentConfig,
    SegmentationAccumulator,
    apply_aug,
    bbox_iou,
    canny_detect,
    compute_ious,
    compute_snr,
    discover_cityscapes_samples,
    evaluate_detections,
    evaluate_canny_edges,
    instance_mask_to_boxes,
    load_sample,
    raw_label_ids_to_train_ids,
)


class DatasetTests(unittest.TestCase):
    def test_discover_official_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "leftImg8bit" / "val" / "sample_city"
            gt_dir = root / "gtFine" / "val" / "sample_city"
            image_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            base = "sample_city_000000_000001"
            Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(
                image_dir / f"{base}_leftImg8bit.png"
            )
            raw_label = np.asarray(
                [[7, 8, 11, 255], [24, 25, 26, 33], [0, 1, 2, 3]], dtype=np.uint8
            )
            Image.fromarray(raw_label).save(
                gt_dir / f"{base}_gtFine_labelIds.png"
            )
            Image.fromarray(np.zeros((3, 4), dtype=np.uint16)).save(
                gt_dir / f"{base}_gtFine_instanceIds.png"
            )

            samples = discover_cityscapes_samples(root, split="val")
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, base)
            _, converted, _ = load_sample(samples[0])
            self.assertTrue(np.array_equal(converted[0], np.asarray([0, 1, 2, 255])))

    def test_raw_id_mapping(self) -> None:
        raw = np.asarray([7, 8, 11, 24, 25, 26, 33, 0, 255], dtype=np.uint8)
        expected = np.asarray([0, 1, 2, 11, 12, 13, 18, 255, 255], dtype=np.uint8)
        self.assertTrue(np.array_equal(raw_label_ids_to_train_ids(raw), expected))


class DistortionTests(unittest.TestCase):
    def setUp(self) -> None:
        gradient = np.arange(16, dtype=np.uint8).reshape(4, 4) * 16
        self.image_array = np.repeat(gradient[..., None], 3, axis=2)
        self.image = Image.fromarray(self.image_array)

    def test_gaussian_noise_is_seeded(self) -> None:
        first = apply_aug(self.image, "GaussNoise", 10.0, seed=123)
        second = apply_aug(self.image, "GaussNoise", 10.0, seed=123)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, self.image_array))

    def test_jpeg_and_low_light_keep_shape(self) -> None:
        jpeg = apply_aug(self.image, "SevereJPEG", 20.0)
        dark = apply_aug(self.image, "LowLight", 0.5)
        self.assertEqual(jpeg.shape, self.image_array.shape)
        self.assertEqual(dark.shape, self.image_array.shape)
        self.assertLess(float(dark.mean()), float(self.image_array.mean()))

    def test_motion_blur_keeps_shape_and_changes_image(self) -> None:
        blurred = apply_aug(self.image, "MotionBlur", 5.0)
        self.assertEqual(blurred.shape, self.image_array.shape)
        self.assertEqual(blurred.dtype, np.uint8)
        self.assertFalse(np.array_equal(blurred, self.image_array))

    def test_motion_blur_rejects_even_kernel(self) -> None:
        with self.assertRaises(ValueError):
            apply_aug(self.image, "MotionBlur", 4.0)

    def test_motion_blur_preserves_constant_brightness(self) -> None:
        constant = np.full((32, 32, 3), 137, dtype=np.uint8)
        blurred = apply_aug(Image.fromarray(constant), "MotionBlur", 15.0)
        self.assertTrue(np.array_equal(blurred, constant))

    def test_stronger_motion_blur_reduces_snr_on_texture(self) -> None:
        rng = np.random.default_rng(19)
        textured = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        image = Image.fromarray(textured)
        mild = apply_aug(image, "MotionBlur", 3.0)
        strong = apply_aug(image, "MotionBlur", 15.0)
        self.assertGreater(compute_snr(textured, mild), compute_snr(textured, strong))

    def test_snr_identical_is_infinite(self) -> None:
        self.assertTrue(math.isinf(compute_snr(self.image_array, self.image_array)))

    def test_snr_decreases_for_stronger_low_light(self) -> None:
        mild = apply_aug(self.image, "LowLight", 0.8)
        strong = apply_aug(self.image, "LowLight", 0.2)
        self.assertGreater(compute_snr(self.image_array, mild), compute_snr(self.image_array, strong))


class SegmentationTests(unittest.TestCase):
    def test_compute_ious(self) -> None:
        ground_truth = np.asarray([[0, 0], [1, 1]], dtype=np.uint8)
        prediction = np.asarray([[0, 1], [1, 1]], dtype=np.uint8)
        ious = compute_ious(prediction, ground_truth)
        self.assertAlmostEqual(ious[0], 0.5)
        self.assertAlmostEqual(ious[1], 2.0 / 3.0)

    def test_accumulator_perfect_prediction(self) -> None:
        ground_truth = np.asarray([[0, 1], [1, 255]], dtype=np.uint8)
        prediction = np.asarray([[0, 1], [1, 4]], dtype=np.uint8)
        accumulator = SegmentationAccumulator()
        accumulator.update(prediction, ground_truth)
        summary, rows = accumulator.results()
        self.assertAlmostEqual(summary["mean_iou"], 1.0)
        self.assertAlmostEqual(summary["pixel_accuracy"], 1.0)
        self.assertEqual(int(rows[0]["gt_pixels"]), 1)


class CannyTests(unittest.TestCase):
    def test_tolerant_f1_accepts_one_pixel_shift(self) -> None:
        reference = np.zeros((12, 12), dtype=np.uint8)
        test = np.zeros_like(reference)
        reference[:, 5] = 255
        test[:, 6] = 255
        metrics = evaluate_canny_edges(reference, test, tolerance_radius=1)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_empty_reference_and_test_are_perfectly_consistent(self) -> None:
        empty = np.zeros((8, 8), dtype=np.uint8)
        metrics = evaluate_canny_edges(empty, empty, tolerance_radius=2)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["edge_pixel_retention"], 1.0)

    def test_canny_detect_returns_binary_map(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, 16:] = 255
        edges = canny_detect(Image.fromarray(image))
        self.assertEqual(edges.shape, image.shape[:2])
        self.assertTrue(set(np.unique(edges)).issubset({0, 255}))
        self.assertGreater(int((edges > 0).sum()), 0)


class DetectionTests(unittest.TestCase):
    def test_instance_mask_to_boxes_uses_shared_classes(self) -> None:
        mask = np.zeros((6, 7), dtype=np.int32)
        mask[1:3, 2:5] = 26001  # car
        mask[3:5, 1:3] = 25001  # rider - excluded because COCO has no rider class
        boxes = instance_mask_to_boxes(mask, "image")
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].class_name, "car")
        self.assertEqual(boxes[0].bbox, (2.0, 1.0, 5.0, 3.0))

    def test_bbox_iou(self) -> None:
        self.assertAlmostEqual(bbox_iou((0, 0, 2, 2), (1, 1, 3, 3)), 1.0 / 7.0)
        self.assertEqual(bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)

    def test_perfect_detection_has_perfect_ap(self) -> None:
        gt = [Detection("image", "car", (0, 0, 10, 10))]
        predictions = [Detection("image", "car", (0, 0, 10, 10), score=0.9)]
        summary, rows = evaluate_detections(predictions, gt, classes=("car",))
        self.assertAlmostEqual(summary["map_50_95"], 1.0)
        self.assertAlmostEqual(summary["map_50"], 1.0)
        self.assertAlmostEqual(float(rows[0]["recall_50"]), 1.0)

    def test_duplicate_prediction_reduces_precision(self) -> None:
        gt = [Detection("image", "car", (0, 0, 10, 10))]
        predictions = [
            Detection("image", "car", (0, 0, 10, 10), score=0.9),
            Detection("image", "car", (0, 0, 10, 10), score=0.8),
        ]
        summary, rows = evaluate_detections(predictions, gt, classes=("car",))
        self.assertAlmostEqual(summary["map_50"], 1.0)
        self.assertAlmostEqual(float(rows[0]["precision_50"]), 0.5)


class PipelineOrchestrationTests(unittest.TestCase):
    def test_parts_1_and_2_write_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cityscapes"
            output = Path(directory) / "outputs"
            image_dir = root / "leftImg8bit" / "val" / "sample_city"
            gt_dir = root / "gtFine" / "val" / "sample_city"
            image_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            base = "sample_city_000000_000001"
            image_array = np.full((8, 10, 3), 120, dtype=np.uint8)
            label = np.zeros((8, 10), dtype=np.uint8)
            instance = np.zeros((8, 10), dtype=np.uint16)
            instance[2:6, 3:8] = 26001  # car
            Image.fromarray(image_array).save(image_dir / f"{base}_leftImg8bit.png")
            Image.fromarray(label).save(gt_dir / f"{base}_gtFine_labelTrainIds.png")
            Image.fromarray(instance).save(gt_dir / f"{base}_gtFine_instanceIds.png")
            samples = discover_cityscapes_samples(root, split="val")
            config = ExperimentConfig(
                dataset_root=root,
                output_dir=output,
                gallery_samples=0,
                distortion_levels={"LowLight": (0.5,)},
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

            def fake_yolo(_image, _model, image_id, conf=0.001):
                return [Detection(image_id, "car", (3.0, 2.0, 8.0, 6.0), score=0.9)]

            with (
                patch.object(project, "measure_orb_matching", return_value=fake_orb),
                patch.object(project, "predict_segmentation", return_value=label.astype(np.int32)),
                patch.object(project, "yolo_detections", side_effect=fake_yolo),
                patch.object(project, "save_distortion_grid"),
                patch.object(project, "save_performance_snr_plot"),
                patch.object(project, "save_part2_gallery"),
            ):
                references, part1 = project.run_part1(config, samples, None, None, None, "cpu")
                part2 = project.run_part2(config, references, None, None, None, "cpu")

            self.assertEqual(part1["sample_count"], 1)
            self.assertEqual(part2["sample_count"], 1)
            self.assertIn("canny", part1)
            self.assertIn("canny_f1", part2["variants"][0])
            self.assertTrue((output / "part1" / "clean_summary.json").is_file())
            self.assertTrue((output / "part1" / "segmentation_per_class.csv").is_file())
            self.assertTrue((output / "part2" / "distorted_summary.csv").is_file())
            self.assertTrue((output / "part2" / "detection_per_class.csv").is_file())


if __name__ == "__main__":
    unittest.main()
