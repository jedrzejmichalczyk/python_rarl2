#!/usr/bin/env python3
"""
Test Suite for Integrated RARL2 Algorithm
==========================================
Tests the complete pipeline: lossless parametrization, H2 projection,
Douglas-Shapiro factorization, and gradient computation.

Following TDD approach - tests written before implementation.
"""

import numpy as np
import unittest
from scipy import linalg


class TestRARL2Integrated(unittest.TestCase):
    """Test the integrated RARL2 algorithm components."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create a simple stable system
        self.n = 3
        self.p = 2
        self.m = 2
        
        # Create stable A matrix
        A = np.random.randn(self.n, self.n) + 1j * np.random.randn(self.n, self.n)
        eigvals = np.linalg.eigvals(A)
        A = A * 0.8 / np.max(np.abs(eigvals))  # Ensure stability
        self.A = A
        
        # Create observable (C, A) pair
        self.C = np.random.randn(self.p, self.n) + 1j * np.random.randn(self.p, self.n)
        
        # Create target system F for approximation
        self.A_F = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        eigvals_F = np.linalg.eigvals(self.A_F)
        self.A_F = self.A_F * 0.7 / np.max(np.abs(eigvals_F))
        self.B_F = np.random.randn(4, self.m) + 1j * np.random.randn(4, self.m)
        self.C_F = np.random.randn(self.p, 4) + 1j * np.random.randn(self.p, 4)
        self.D_F = np.random.randn(self.p, self.m) + 1j * np.random.randn(self.p, self.m)
    
    def test_douglas_shapiro_factorization(self):
        """
        Test Douglas-Shapiro factorization: H = C*G where G is lossless.
        
        Key property: Given any stable H, we can factor it as H = C*G
        where G is lossless (all-pass) and C is the outer factor.
        """
        from rarl2_integrated import douglas_shapiro_factorization
        
        # Create a stable system H
        A_H = self.A
        B_H = np.random.randn(self.n, self.m) + 1j * np.random.randn(self.n, self.m)
        C_H = self.C
        D_H = np.eye(self.p, self.m)
        
        # Perform factorization
        C_factor, G_lossless = douglas_shapiro_factorization(
            A_H, B_H, C_H, D_H
        )
        
        # G_lossless should be a lossless system (4-tuple)
        A_G, B_G, C_G, D_G = G_lossless
        
        # Test 1: G should be lossless (unitary on unit circle)
        # Check at several points on unit circle
        for theta in np.linspace(0, 2*np.pi, 10):
            z = np.exp(1j * theta)
            G_z = D_G + C_G @ np.linalg.inv(z * np.eye(self.n) - A_G) @ B_G
            
            # G(z) * G(z)^H should be identity
            should_be_I = G_z @ G_z.conj().T
            np.testing.assert_allclose(should_be_I, np.eye(self.m), atol=1e-10)
        
        # Test 2: The factorization should approximately recover H
        # Note: This is NOT exact unless H was constructed from lossless G
        # The factorization finds the "best" lossless G and outer C
        
    def test_h2_projection_with_lossless(self):
        """
        Test H2 projection: compute optimal C = π+(F*G†).
        
        Given target F and lossless G, find optimal C such that
        ||F - C*G||₂ is minimized.
        """
        from rarl2_integrated import h2_projection_lossless
        from lossless_embedding import lossless_embedding
        
        # Create lossless G from (C, A)
        B_G, D_G = lossless_embedding(self.C, self.A)
        
        # Compute optimal C via H2 projection
        C_opt = h2_projection_lossless(
            self.A_F, self.B_F, self.C_F, self.D_F,  # Target F
            self.A, B_G, self.C, D_G  # Lossless G
        )
        
        # Test: C_opt should minimize ||F - C_opt*G||₂
        # We can't directly test the optimality, but we can verify:
        # 1. C_opt is stable (has proper state-space realization)
        # 2. The error ||F - C_opt*G||₂ is finite
        
        self.assertIsNotNone(C_opt)
        A_C, B_C, C_C, D_C = C_opt
        
        # Check stability of C
        eigvals_C = np.linalg.eigvals(A_C)
        self.assertTrue(np.all(np.abs(eigvals_C) < 1.0))
    
    def test_gradient_via_lossless_parametrization(self):
        """
        Test gradient computation using the concentrated criterion.
        
        The gradient of ψₙ(G) = ||F - π+(F*G†)*G||₂² with respect to
        the lossless parametrization of G.
        """
        from rarl2_integrated import compute_gradient_lossless
        from lossless_embedding import lossless_embedding
        
        # Create lossless G from (C, A)
        B_G, D_G = lossless_embedding(self.C, self.A)
        
        # Compute gradient with respect to (C, A) parameters
        grad = compute_gradient_lossless(
            self.C, self.A, B_G, D_G,
            self.A_F, self.B_F, self.C_F, self.D_F
        )
        
        # Test gradient properties
        # 1. Gradient should be real-valued for real objective
        # 2. Gradient should be zero at optimum
        # 3. Gradient should have correct shape
        
        expected_size = self.C.size + self.A.size
        self.assertEqual(grad.shape[0], expected_size)
        
        # Verify gradient via finite differences
        eps = 1e-7
        from rarl2_integrated import compute_objective_lossless
        
        # Compute base objective
        obj_base = compute_objective_lossless(
            self.C, self.A,
            self.A_F, self.B_F, self.C_F, self.D_F
        )
        
        # Perturb first parameter and check gradient
        C_pert = self.C.copy()
        C_pert.flat[0] += eps
        obj_pert = compute_objective_lossless(
            C_pert, self.A,
            self.A_F, self.B_F, self.C_F, self.D_F
        )
        
        grad_fd = (obj_pert - obj_base) / eps
        grad_analytical = grad[0]
        
        # Should match within numerical precision
        # Note: finite differences have limited accuracy, especially for complex functions
        self.assertAlmostEqual(grad_fd, grad_analytical, places=4)
    
    def test_lossless_to_lossy_conversion(self):
        """
        Test converting a lossless system to lossy via H2 projection.
        
        Given lossless G, compute H = C*G where C = π+(F*G†) to get
        the best approximation to target F.
        """
        from rarl2_integrated import lossless_to_lossy
        from lossless_embedding import lossless_embedding
        
        # Create lossless G
        B_G, D_G = lossless_embedding(self.C, self.A)
        
        # Convert to lossy system that approximates F
        A_H, B_H, C_H, D_H = lossless_to_lossy(
            self.A, B_G, self.C, D_G,  # Lossless G
            self.A_F, self.B_F, self.C_F, self.D_F  # Target F
        )
        
        # Test properties:
        # 1. H should be stable
        eigvals_H = np.linalg.eigvals(A_H)
        self.assertTrue(np.all(np.abs(eigvals_H) < 1.0))
        
        # 2. H should have the same I/O dimensions as F
        self.assertEqual(C_H.shape[0], self.C_F.shape[0])
        self.assertEqual(B_H.shape[1], self.B_F.shape[1])
        
        # 3. ||H - F||₂ should be finite
        # (We compute a simple bound using nuclear norm)
        error_bound = np.linalg.norm(D_H - self.D_F, 'nuc')
        self.assertLess(error_bound, 100)  # Reasonable bound
    
    def test_rarl2_convergence(self):
        """
        Test that RARL2 optimization converges and improves the approximation.
        
        Starting from an initial approximation, RARL2 should iteratively
        improve the L2 error ||F - H||₂.
        """
        from rarl2_integrated import RARL2Optimizer
        
        # Initialize optimizer
        optimizer = RARL2Optimizer(
            target=(self.A_F, self.B_F, self.C_F, self.D_F),
            order=self.n  # Approximation order
        )
        
        # Run optimization for a few iterations
        initial_error = optimizer.get_error()
        
        for i in range(10):
            optimizer.step(learning_rate=0.01)
            
            # Error should generally decrease (allow small increases for stability)
            new_error = optimizer.get_error()
            if i > 5:  # After initial phase
                self.assertLessEqual(new_error, initial_error * 1.1)
        
        final_error = optimizer.get_error()
        
        # Final error should be less than initial
        self.assertLess(final_error, initial_error)
        
        # Get the final approximation
        A_final, B_final, C_final, D_final = optimizer.get_approximation()
        
        # Should be stable
        eigvals_final = np.linalg.eigvals(A_final)
        self.assertTrue(np.all(np.abs(eigvals_final) < 1.0))
    
    def test_concentrated_criterion(self):
        """
        Test the concentrated criterion ψₙ(G) = ||F - π+(F*G†)*G||₂².
        
        This is the key objective function that RARL2 minimizes.
        """
        from rarl2_integrated import concentrated_criterion
        from lossless_embedding import lossless_embedding
        
        # Create lossless G
        B_G, D_G = lossless_embedding(self.C, self.A)
        
        # Compute concentrated criterion
        psi = concentrated_criterion(
            self.A, B_G, self.C, D_G,  # Lossless G
            self.A_F, self.B_F, self.C_F, self.D_F  # Target F
        )
        
        # Test properties:
        # 1. Should be non-negative
        self.assertGreaterEqual(psi, 0)
        
        # 2. Should be zero only if F = C*G for some C
        # (This is hard to test directly)
        
        # 3. Should be finite for stable systems
        self.assertLess(psi, np.inf)
    
    def test_h2_norm_computation(self):
        """Test H2 norm computation for error system."""
        from rarl2_integrated import compute_h2_norm
        
        # Create two systems and compute ||H1 - H2||₂
        A1 = self.A
        B1 = np.random.randn(self.n, self.m) + 1j * np.random.randn(self.n, self.m)
        C1 = self.C
        D1 = np.zeros((self.p, self.m))
        
        A2 = self.A_F
        B2 = self.B_F
        C2 = self.C_F  
        D2 = self.D_F
        
        h2_norm = compute_h2_norm(
            A1, B1, C1, D1,
            A2, B2, C2, D2
        )
        
        # Should be non-negative
        self.assertGreaterEqual(h2_norm, 0)
        
        # Should be zero for identical systems
        h2_norm_zero = compute_h2_norm(
            A1, B1, C1, D1,
            A1, B1, C1, D1
        )
        self.assertAlmostEqual(h2_norm_zero, 0, places=10)
    
    def test_output_normal_parametrization(self):
        """
        Test output normal parametrization where A^H*A + C^H*C = I.
        
        This is the specific parametrization used in RARL2.
        """
        from rarl2_integrated import create_output_normal_pair
        
        C_on, A_on = create_output_normal_pair(self.n, self.p)
        
        # Test output normal property
        M = A_on.conj().T @ A_on + C_on.conj().T @ C_on
        np.testing.assert_allclose(M, np.eye(self.n), atol=1e-10)
        
        # A should be stable
        eigvals = np.linalg.eigvals(A_on)
        self.assertTrue(np.all(np.abs(eigvals) < 1.0))
    
    def test_gradient_descent_step(self):
        """Test a single gradient descent step improves the objective."""
        from rarl2_integrated import gradient_descent_step
        from lossless_embedding import lossless_embedding
        
        # Current parameters
        C_current = self.C
        A_current = self.A
        
        # Compute gradient and take a step
        C_new, A_new, obj_old, obj_new = gradient_descent_step(
            C_current, A_current,
            self.A_F, self.B_F, self.C_F, self.D_F,
            learning_rate=0.01
        )
        
        # Objective should generally decrease (allow small increase for numerical issues)
        self.assertLessEqual(obj_new, obj_old * 1.01)
        
        # New A should remain stable
        eigvals_new = np.linalg.eigvals(A_new)
        self.assertTrue(np.all(np.abs(eigvals_new) < 1.0))


if __name__ == '__main__':
    unittest.main()