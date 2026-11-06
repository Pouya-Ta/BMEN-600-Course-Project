"""Focused safeguards for discontinuities, amplitude rules and anti-aliasing."""
import unittest

import numpy as np

from eeg_common import HERE, original_indices
from preprocess_eeg import config_from, quality, resample_run, window_starts


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.config = config_from(HERE / "preprocessing_config.json")

    def test_mask_handles_leading_trailing_and_internal_removal(self):
        mask = np.ones(13000, dtype=np.uint8)
        mask[:500] = mask[6000:6500] = mask[-300:] = 0
        indices = np.flatnonzero(mask)
        eeg = {"etc": {"clean_sample_mask": mask}, "event": {"type": "boundary", "latency": 5500.5}}
        recovered, edges = original_indices(eeg, len(indices), len(mask))
        np.testing.assert_array_equal(recovered, indices)
        np.testing.assert_array_equal(edges, [0, 5500, len(indices)])
        for left, right in zip(edges[:-1], edges[1:]):
            for start in window_starts(left, right, 500, self.config):
                self.assertTrue(np.all(np.diff(recovered[start:start + 4000]) == 1))
                self.assertGreaterEqual(start - left, 500)
                self.assertGreaterEqual(right - start - 4000, 500)

    def test_bad_boundary_or_mask_fails(self):
        eeg = {"etc": {"clean_sample_mask": [1, 1, 0, 1]}, "event": []}
        with self.assertRaises(ValueError):
            original_indices(eeg, 3, 4)
        with self.assertRaises(ValueError):
            original_indices(eeg, 4, 4)

    def test_artifacts_are_rejected_without_diagnosis(self):
        t = np.arange(4000) / 500
        good = np.tile(10 * np.sin(2 * np.pi * 10 * t), (19, 1))
        self.assertEqual(quality(good, self.config)[0], [])
        high = good.copy()
        high[0] *= 30
        self.assertIn("high_peak_to_peak", quality(high, self.config)[0])
        flat = good.copy()
        flat[4] = 2
        self.assertIn("flat_channel", quality(flat, self.config)[0])
        good[0, 20] = np.nan
        self.assertEqual(quality(good, self.config)[0], ["nonfinite"])

    def test_resampling_preserves_band_amplitude_and_rejects_alias(self):
        t = np.arange(5000) / 500
        output = resample_run((10 * np.sin(2 * np.pi * 10 * t))[None], 500, 256)[0]
        target_t = np.arange(2560) / 256
        core = slice(256, -256)
        np.testing.assert_allclose(output[core], 10 * np.sin(2 * np.pi * 10 * target_t[core]), atol=0.02)
        aliased = resample_run(np.sin(2 * np.pi * 200 * t)[None], 500, 256)[0]
        self.assertLess(float(np.std(aliased[core])), 0.01)


if __name__ == "__main__":
    unittest.main()
