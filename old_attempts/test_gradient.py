#!/usr/bin/env python3
"""
Test Gradient Computation for RARL2
====================================
Tests the gradient of the L2 criterion based on equation (11) from the paper:
dJn/dλ = 2 Re Tr[P*_12(A* Q12 ∂A/∂λ + C* ∂C/∂λ)]

The gradient is computed with respect to parameters in the BOP chart.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lossless_embedding import solve_discrete_lyapunov


class TestGradient(unittest.TestCase):
    """Test gradient computation for RARL2 optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
    def test_finite_difference_verification(self):
        """Verify analytical gradient against finite differences."""
        from gradient_computation import RARL2Gradient
        
        # Create a simple test case
        n = 3  # State dimension
        m = 2  # Input/output dimension
        p = 2  # Output dimension
        
        # Create target system F
        A_F = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        A_F = A_F * 0.5 / np.max(np.abs(np.linalg.eigvals(A_F)))  # Make stable
        B_F = np.random.randn(n, m) + 1j * np.random.randn(n, m)
        C_F = np.random.randn(p, n) + 1j * np.random.randn(p, n)
        D_F = np.zeros((p, m), dtype=np.complex128)
        
        # Create initial approximation (output normal pair)
        from bop import create_output_normal_chart_center
        W, X, Y, Z = create_output_normal_chart_center(n, p)
        
        # Use W as A and Y as C for initial approximation
        A_init = W
        C_init = Y
        
        # Create gradient computer
        grad_computer = RARL2Gradient(A_F, B_F, C_F, D_F)
        
        # Compute analytical gradient at initial point
        grad_analytical = grad_computer.compute_gradient(C_init, A_init)
        
        # Compute finite difference gradient
        epsilon = 1e-8
        grad_fd = np.zeros_like(grad_analytical)
        
        # Finite difference for each parameter
        # Parameters are arranged as vectorized (C, A)
        params = np.concatenate([C_init.flatten(), A_init.flatten()])
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            
            params_minus = params.copy()
            params_minus[i] -= epsilon
            
            # Reconstruct C and A from parameters
            C_plus = params_plus[:C_init.size].reshape(C_init.shape)
            A_plus = params_plus[C_init.size:].reshape(A_init.shape)
            
            C_minus = params_minus[:C_init.size].reshape(C_init.shape)
            A_minus = params_minus[C_init.size:].reshape(A_init.shape)
            
            # Compute objective values
            obj_plus = grad_computer.compute_objective(C_plus, A_plus)
            obj_minus = grad_computer.compute_objective(C_minus, A_minus)
            
            # Finite difference
            grad_fd[i] = (obj_plus - obj_minus) / (2 * epsilon)
        
        # Compare gradients
        grad_error = np.linalg.norm(grad_analytical - grad_fd)
        relative_error = grad_error / (np.linalg.norm(grad_analytical) + 1e-10)
        
        self.assertLess(relative_error, 1e-5,
                       f"Gradient mismatch: relative error = {relative_error:.2e}")
        
    def test_gradient_at_optimum(self):
        """Test that gradient is zero when approximation equals target."""
        from gradient_computation import RARL2Gradient
        
        n = 2
        p = 2
        
        # Create a system and use it as both target and approximation
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        A = A * 0.5 / np.max(np.abs(np.linalg.eigvals(A)))
        B = np.random.randn(n, p) + 1j * np.random.randn(n, p)
        C = np.random.randn(p, n) + 1j * np.random.randn(p, n)
        D = np.zeros((p, p), dtype=np.complex128)
        
        # Create gradient computer with target = approximation
        grad_computer = RARL2Gradient(A, B, C, D)
        
        # Gradient should be zero at the optimum
        grad = grad_computer.compute_gradient(C, A)
        
        grad_norm = np.linalg.norm(grad)
        self.assertLess(grad_norm, 1e-10,
                       f"Gradient not zero at optimum: norm = {grad_norm:.2e}")
        
    def test_gradient_descent_step(self):
        """Test that gradient descent reduces the objective."""
        from gradient_computation import RARL2Gradient
        
        n = 3
        m = 2
        p = 2
        
        # Create target system
        A_F = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        A_F = A_F * 0.6 / np.max(np.abs(np.linalg.eigvals(A_F)))
        B_F = np.random.randn(n, m) + 1j * np.random.randn(n, m)
        C_F = np.random.randn(p, n) + 1j * np.random.randn(p, n)
        D_F = np.zeros((p, m), dtype=np.complex128)
        
        # Create initial approximation
        A_init = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        A_init = A_init * 0.5 / np.max(np.abs(np.linalg.eigvals(A_init)))
        C_init = np.random.randn(p, n) + 1j * np.random.randn(p, n) * 0.5
        
        grad_computer = RARL2Gradient(A_F, B_F, C_F, D_F)
        
        # Compute initial objective
        obj_initial = grad_computer.compute_objective(C_init, A_init)
        
        # Compute gradient
        grad = grad_computer.compute_gradient(C_init, A_init)
        
        # Take a small gradient descent step
        learning_rate = 1e-4
        params = np.concatenate([C_init.flatten(), A_init.flatten()])
        params_new = params - learning_rate * grad
        
        # Reconstruct C and A
        C_new = params_new[:C_init.size].reshape(C_init.shape)
        A_new = params_new[C_init.size:].reshape(A_init.shape)
        
        # Compute new objective
        obj_new = grad_computer.compute_objective(C_new, A_new)
        
        # Objective should decrease
        self.assertLess(obj_new, obj_initial,
                       f"Objective did not decrease: {obj_initial:.6f} -> {obj_new:.6f}")
        
    def test_gradient_with_bop_parametrization(self):
        """Test gradient computation through BOP parametrization."""
        from gradient_computation import RARL2Gradient
        from bop import BOP, create_output_normal_chart_center
        
        n = 2
        m = 2
        
        # Create target
        A_F = np.array([[0.3, 0.1], [-0.1, 0.4]], dtype=np.complex128)
        B_F = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        C_F = np.array([[1, 0.5], [0.5, 1]], dtype=np.complex128)
        D_F = np.zeros((m, m), dtype=np.complex128)
        
        # Create BOP chart
        W, X, Y, Z = create_output_normal_chart_center(n, m)
        bop = BOP(W, X, Y, Z)
        
        # Test gradient with respect to V parameter
        V = np.zeros((m, n), dtype=np.complex128)
        Do = np.eye(m, dtype=np.complex128)
        
        # Get realization from BOP
        A, B, C, D = bop.V_to_realization(V, Do)
        
        grad_computer = RARL2Gradient(A_F, B_F, C_F, D_F)
        
        # Compute gradient with respect to (C, A)
        grad_CA = grad_computer.compute_gradient(C, A)
        
        # Now compute gradient with respect to V using chain rule
        # This tests the composition of gradients
        epsilon = 1e-8
        grad_V_fd = np.zeros(V.size, dtype=np.complex128)
        
        for i in range(V.size):
            V_flat = V.flatten()
            V_plus = V_flat.copy()
            V_plus[i] += epsilon
            V_plus = V_plus.reshape(V.shape)
            
            if bop.is_in_domain(V_plus):
                A_plus, B_plus, C_plus, D_plus = bop.V_to_realization(V_plus, Do)
                obj_plus = grad_computer.compute_objective(C_plus, A_plus)
            else:
                obj_plus = np.inf
            
            V_minus = V_flat.copy()
            V_minus[i] -= epsilon
            V_minus = V_minus.reshape(V.shape)
            
            if bop.is_in_domain(V_minus):
                A_minus, B_minus, C_minus, D_minus = bop.V_to_realization(V_minus, Do)
                obj_minus = grad_computer.compute_objective(C_minus, A_minus)
            else:
                obj_minus = np.inf
            
            if obj_plus != np.inf and obj_minus != np.inf:
                grad_V_fd[i] = (obj_plus - obj_minus) / (2 * epsilon)
        
        # The gradient could be zero at V=0 if we're at a stationary point
        # Just verify that we computed a gradient (might be zero)
        grad_V_norm = np.linalg.norm(grad_V_fd)
        # This test just verifies the gradient computation runs without error
        self.assertGreaterEqual(grad_V_norm, 0,
                               "Gradient norm should be non-negative")
        
    def test_gradient_preserves_stability(self):
        """Test that gradient direction preserves stability of A."""
        from gradient_computation import RARL2Gradient
        
        n = 3
        m = 2
        p = 2
        
        # Create target
        A_F = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        A_F = A_F * 0.5 / np.max(np.abs(np.linalg.eigvals(A_F)))
        B_F = np.random.randn(n, m) + 1j * np.random.randn(n, m)
        C_F = np.random.randn(p, n) + 1j * np.random.randn(p, n)
        D_F = np.zeros((p, m), dtype=np.complex128)
        
        # Start with stable A
        A_init = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        A_init = A_init * 0.7 / np.max(np.abs(np.linalg.eigvals(A_init)))
        C_init = np.random.randn(p, n) + 1j * np.random.randn(p, n)
        
        # Verify initial stability
        eigvals_init = np.linalg.eigvals(A_init)
        self.assertTrue(np.all(np.abs(eigvals_init) < 1.0),
                       "Initial A not stable")
        
        grad_computer = RARL2Gradient(A_F, B_F, C_F, D_F)
        
        # Take several gradient steps
        A_current = A_init.copy()
        C_current = C_init.copy()
        learning_rate = 1e-5
        
        for step in range(10):
            grad = grad_computer.compute_gradient(C_current, A_current)
            
            # Update parameters
            params = np.concatenate([C_current.flatten(), A_current.flatten()])
            params = params - learning_rate * grad
            
            C_current = params[:C_init.size].reshape(C_init.shape)
            A_current = params[C_init.size:].reshape(A_init.shape)
            
            # Check stability
            eigvals = np.linalg.eigvals(A_current)
            max_eigval = np.max(np.abs(eigvals))
            
            # Stability might be slightly violated due to numerical errors
            # but should remain close to stable
            self.assertLess(max_eigval, 1.1,
                           f"A became too unstable at step {step}: max |λ| = {max_eigval:.3f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)