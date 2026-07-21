"""Tests for transparent runtime extrapolation."""

import unittest

from cityscapes_project.utils.timing import estimate_runtime


class RuntimeEstimateTests(unittest.TestCase):
    def test_twenty_image_measurement_scales_to_full_validation_split(self) -> None:
        estimate = estimate_runtime(sample_count=20, elapsed_seconds=21 * 60 + 1, target_count=500)
        self.assertAlmostEqual(estimate.seconds_per_sample, 63.05)
        self.assertAlmostEqual(estimate.projected_hours, 8.7569, places=3)

    def test_invalid_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_runtime(sample_count=0, elapsed_seconds=10, target_count=500)


if __name__ == "__main__":
    unittest.main()
