#!/usr/bin/env python3
import unittest
import numpy as np
import torch

from bop import create_unitary_chart_center
from torch_chart_optimizer import ChartRARL2Torch


class TestTorchChartOptimizer(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        torch.manual_seed(123)

    def test_forward_backward_and_optimization(self):
        n, m = 2, 2  # small system
        # Chart center
        W, X, Y, Z = create_unitary_chart_center(n, m)
        # Create a stable random target with same I/O dims
        A_F = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        A_F = A_F * (0.6 / np.max(np.abs(np.linalg.eigvals(A_F))))
        B_F = (np.random.randn(n, m) + 1j * np.random.randn(n, m)) * 0.3
        C_F = (np.random.randn(m, n) + 1j * np.random.randn(m, n)) * 0.3
        D_F = np.zeros((m, m), dtype=np.complex128)

        model = ChartRARL2Torch(n, m, (W, X, Y, Z), (A_F, B_F, C_F, D_F))
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.2, max_iter=5)

        # Initial loss
        with torch.no_grad():
            loss0 = model().item()

        # One backward pass sanity check
        loss = model()
        loss.backward()
        # Ensure gradients exist
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        # Zero for optimizer
        optimizer.zero_grad(set_to_none=True)

        def closure():
            optimizer.zero_grad()
            l = model()
            l.backward()
            return l

        # A few steps
        for _ in range(5):
            optimizer.step(closure)

        with torch.no_grad():
            loss1 = model().item()

        # Loss should not increase and typically decreases
        self.assertLessEqual(loss1, loss0 + 1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)

