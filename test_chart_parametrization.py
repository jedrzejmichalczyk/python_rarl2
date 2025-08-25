#!/usr/bin/env python3
"""
Tests for ChartParametrization (no chart switching)
"""
import numpy as np
import unittest

from chart_parametrization import ChartParametrization
from bop import create_unitary_chart_center
from lossless_embedding import verify_lossless


class TestChartParametrization(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_roundtrip_and_lossless(self):
        n, m = 3, 2
        W, X, Y, Z = create_unitary_chart_center(n, m)

        cp = ChartParametrization(n=n, p=m, m=m)
        cp.set_chart_center((W, X, Y, Z))

        # Small V inside the chart domain
        V = 0.05 * (np.random.randn(m, n) + 1j * np.random.randn(m, n))
        Do = np.eye(m, dtype=np.complex128)

        A, B, C, D = cp.deparametrize(V, Do)
        self.assertEqual(A.shape, (n, n))
        self.assertEqual(B.shape, (n, m))
        self.assertEqual(C.shape, (m, n))
        self.assertEqual(D.shape, (m, m))

        # Lossless on the unit circle
        max_err = verify_lossless(A, B, C, D, n_points=50)
        self.assertLess(max_err, 1e-9)

        # Parametrize back (should return cached V, Do exactly)
        V2, Do2 = cp.parametrize((A, B, C, D))
        self.assertAlmostEqual(np.linalg.norm(V2 - V), 0.0, places=12)
        self.assertAlmostEqual(np.linalg.norm(Do2 - Do), 0.0, places=12)

        # Boundary check
        self.assertTrue(cp.check_chart_boundary())

    def test_center_V_zero(self):
        n, m = 2, 2
        W, X, Y, Z = create_unitary_chart_center(n, m)

        cp = ChartParametrization(n=n, p=m, m=m)
        cp.set_chart_center((W, X, Y, Z))

        V = np.zeros((m, n), dtype=np.complex128)
        Do = np.eye(m, dtype=np.complex128)

        A, B, C, D = cp.deparametrize(V, Do)
        max_err = verify_lossless(A, B, C, D, n_points=50)
        self.assertLess(max_err, 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)

