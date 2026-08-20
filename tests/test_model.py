import unittest

import numpy as np

from srk_model.model import simulate_coupled_srk, simulate_srk


class ModelTests(unittest.TestCase):
    def test_single_solver_is_finite_and_conductance_sensitive(self):
        noise = np.zeros(200)
        low_conductance = simulate_srk(noise, 0.1, 0.0, g_s=3.5)
        high_conductance = simulate_srk(noise, 0.1, 0.0, g_s=4.5)
        self.assertTrue(np.all(np.isfinite(low_conductance)))
        self.assertFalse(np.allclose(low_conductance, high_conductance))

    def test_identical_coupled_cells_match_under_identical_noise(self):
        noise = np.zeros((2, 200))
        first, second = simulate_coupled_srk(noise, 0.1, 1e-4)
        np.testing.assert_allclose(first, second)

    def test_noise_shape_is_validated(self):
        with self.assertRaises(ValueError):
            simulate_coupled_srk(np.zeros(20), 0.1, 1e-4)


if __name__ == "__main__":
    unittest.main()
