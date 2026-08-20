import unittest

import numpy as np

from srk_model.analysis import aggregate_wiener_increments, compute_log_centroid


class AnalysisTests(unittest.TestCase):
    def test_wiener_aggregation_preserves_path_endpoint(self):
        fine = np.arange(12, dtype=float)
        coarse = aggregate_wiener_increments(fine, 3)
        np.testing.assert_allclose(coarse, [3.0, 12.0, 21.0, 30.0])
        self.assertAlmostEqual(float(np.sum(coarse)), float(np.sum(fine)))

    def test_log_centroid_ignores_nonpositive_and_invalid_values(self):
        sigmas = np.array([1e-4, 1e-3, 1e-2, 1e-1])
        values = np.array([np.nan, 0.0, 1.0, 1.05])
        centre, lower, upper = compute_log_centroid(sigmas, values, delta=0.1)
        self.assertAlmostEqual(centre, np.sqrt(1e-3))
        self.assertAlmostEqual(lower, 1e-2)
        self.assertAlmostEqual(upper, 1e-1)


if __name__ == "__main__":
    unittest.main()
