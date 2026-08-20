import unittest

import numpy as np

from srk_model.statistics import analyze_rhythm, detect_burst_times, synchronization_index


class StatisticsTests(unittest.TestCase):
    def test_burn_in_excludes_early_events(self):
        voltage = -np.ones(50)
        voltage[[2, 10, 20, 30, 40]] = 1.0
        times = detect_burst_times(
            voltage,
            dt=1.0,
            burn_in=5.0,
            threshold=0.0,
            lockout=2.0,
        )
        np.testing.assert_allclose(times, [10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(
            analyze_rhythm(voltage, 1.0, burn_in=5.0, threshold=0.0, lockout=2.0),
            0.0,
        )

    def test_synchronization_index_is_binary_occupancy_correlation(self):
        first = np.array([-1.0, 1.0, -1.0, 1.0])
        second = first.copy()
        inverse = -first
        self.assertAlmostEqual(
            synchronization_index(first, second, 1.0, burn_in=0.0, threshold=0.0),
            1.0,
        )
        self.assertAlmostEqual(
            synchronization_index(first, inverse, 1.0, burn_in=0.0, threshold=0.0),
            -1.0,
        )


if __name__ == "__main__":
    unittest.main()
