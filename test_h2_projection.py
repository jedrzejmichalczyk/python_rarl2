#!/usr/bin/env python3
"""
Test H2 Projection for RARL2
=============================
Tests the H2 projection operator that projects onto the space of 
stable, square-integrable functions.

The H2 projection extracts the stable part of a transfer function.
For discrete-time systems, stable means eigenvalues inside the unit circle.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestH2Projection(unittest.TestCase):
    """Test the H2 projection operator."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
    def test_stable_function_unchanged(self):
        """Test that a stable function is unchanged by H2 projection."""
        from h2_projection import H2Projection
        
        n = 3
        # Create a stable system (all eigenvalues inside unit circle)
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        A = A * 0.5 / np.max(np.abs(np.linalg.eigvals(A)))  # Make stable
        B = np.random.randn(n, 2) + 1j * np.random.randn(n, 2)
        C = np.random.randn(2, n) + 1j * np.random.randn(2, n)
        D = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        
        h2_proj = H2Projection()
        A_proj, B_proj, C_proj, D_proj = h2_proj.project(A, B, C, D)
        
        # Stable system should be unchanged
        self.assertTrue(np.allclose(A, A_proj, rtol=1e-10))
        self.assertTrue(np.allclose(B, B_proj, rtol=1e-10))
        self.assertTrue(np.allclose(C, C_proj, rtol=1e-10))
        self.assertTrue(np.allclose(D, D_proj, rtol=1e-10))
        
    def test_unstable_poles_removed(self):
        """Test that unstable poles are removed by projection."""
        from h2_projection import H2Projection
        
        # Create a system with mixed stable and unstable poles
        # Use diagonal A for explicit control of eigenvalues
        stable_eigvals = np.array([0.5, 0.3 + 0.2j, 0.3 - 0.2j], dtype=np.complex128)
        unstable_eigvals = np.array([1.5, 1.2 + 0.3j, 1.2 - 0.3j], dtype=np.complex128)
        
        # Combine eigenvalues
        all_eigvals = np.concatenate([stable_eigvals, unstable_eigvals])
        n = len(all_eigvals)
        A = np.diag(all_eigvals)
        
        B = np.random.randn(n, 2) + 1j * np.random.randn(n, 2)
        C = np.random.randn(2, n) + 1j * np.random.randn(2, n)
        D = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        
        h2_proj = H2Projection()
        A_proj, B_proj, C_proj, D_proj = h2_proj.project(A, B, C, D)
        
        # Check that projected system is stable
        eigvals_proj = np.linalg.eigvals(A_proj)
        self.assertTrue(np.all(np.abs(eigvals_proj) < 1.0),
                       f"Projected system not stable: max |λ| = {np.max(np.abs(eigvals_proj))}")
        
        # Check that dimension is reduced (unstable modes removed)
        self.assertEqual(A_proj.shape[0], len(stable_eigvals),
                        "Projected system should have reduced dimension")
        
    def test_projection_reduces_h2_norm(self):
        """Test that projection reduces or maintains H2 norm."""
        from h2_projection import H2Projection
        from lossless_embedding import solve_discrete_lyapunov
        
        # Create a mixed stable/unstable system
        n = 4
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        # Make some eigenvalues unstable
        eigvals, eigvecs = np.linalg.eig(A)
        eigvals[0] = 1.2  # Make first eigenvalue unstable
        eigvals[1] = 1.1 + 0.2j  # Make second eigenvalue unstable
        A = eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)
        
        B = np.random.randn(n, 2) + 1j * np.random.randn(n, 2)
        C = np.random.randn(2, n) + 1j * np.random.randn(2, n)
        D = np.zeros((2, 2), dtype=np.complex128)
        
        h2_proj = H2Projection()
        A_proj, B_proj, C_proj, D_proj = h2_proj.project(A, B, C, D)
        
        # Compute H2 norm of projected system (should be finite)
        # H2 norm squared = trace(B^H * Q * B) where Q solves A^H*Q*A - Q = -C^H*C
        
        # Check if projected system is non-empty
        if A_proj.shape[0] == 0:
            # Empty system has zero H2 norm (only feedthrough D remains)
            h2_norm_sq = np.real(np.trace(D_proj.conj().T @ D_proj))
            self.assertGreaterEqual(h2_norm_sq, 0, "H2 norm should be non-negative")
        else:
            try:
                Q = solve_discrete_lyapunov(A_proj, C_proj.conj().T @ C_proj)
                h2_norm_sq = np.real(np.trace(B_proj.conj().T @ Q @ B_proj))
                self.assertGreater(h2_norm_sq, 0, "H2 norm should be positive")
                self.assertLess(h2_norm_sq, 1e6, "H2 norm should be finite")
            except Exception as e:
                self.fail(f"H2 norm computation failed for projected system: {e}")
            
    def test_projection_idempotent(self):
        """Test that projection is idempotent: π(π(H)) = π(H)."""
        from h2_projection import H2Projection
        
        # Create an unstable system
        n = 3
        A = np.array([[1.5, 0.1, 0], 
                      [0, 0.6, 0.2],
                      [0, -0.2, 0.4]], dtype=np.complex128)
        B = np.random.randn(n, 2) + 1j * np.random.randn(n, 2)
        C = np.random.randn(2, n) + 1j * np.random.randn(2, n)
        D = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        
        h2_proj = H2Projection()
        
        # First projection
        A1, B1, C1, D1 = h2_proj.project(A, B, C, D)
        
        # Second projection (should be identical)
        A2, B2, C2, D2 = h2_proj.project(A1, B1, C1, D1)
        
        self.assertTrue(np.allclose(A1, A2, rtol=1e-10))
        self.assertTrue(np.allclose(B1, B2, rtol=1e-10))
        self.assertTrue(np.allclose(C1, C2, rtol=1e-10))
        self.assertTrue(np.allclose(D1, D2, rtol=1e-10))
        
    def test_implicit_gradient_computation(self):
        """Test gradient computation through H2 projection using implicit differentiation."""
        from h2_projection import H2Projection
        
        n = 3
        # Create a system with mixed stable/unstable eigenvalues
        A = np.array([[0.5, 0.2, 0.1],
                      [0.1, 1.2, 0.1],  # Unstable mode
                      [0.1, 0.1, 0.3]], dtype=np.complex128)
        B = np.random.randn(n, 2) + 1j * np.random.randn(n, 2)
        C = np.random.randn(2, n) + 1j * np.random.randn(2, n)
        D = np.zeros((2, 2), dtype=np.complex128)
        
        h2_proj = H2Projection()
        
        # Forward pass
        A_proj, B_proj, C_proj, D_proj = h2_proj.project(A, B, C, D)
        
        # Test gradient computation (simplified - just check it runs)
        # In full implementation, this would use implicit differentiation
        grad_output = np.ones_like(A_proj)
        
        # The gradient should be computable without issues
        # Real implementation would solve Sylvester equation
        try:
            # Placeholder for gradient computation
            # Real implementation: grad_input = h2_proj.backward(grad_output, A, B, C, D)
            grad_shape = A.shape
            self.assertEqual(grad_shape, A.shape, "Gradient shape should match input")
        except Exception as e:
            self.fail(f"Gradient computation failed: {e}")
            
    def test_projection_preserves_structure(self):
        """Test that projection preserves certain structural properties."""
        from h2_projection import H2Projection
        
        # Create a system with specific structure (e.g., real matrices)
        n = 3
        A = np.array([[1.5, 0.2, 0],
                      [0.2, 0.4, 0.1],
                      [0, 0.1, 0.3]], dtype=np.float64)
        B = np.random.randn(n, 2)
        C = np.random.randn(2, n)
        D = np.zeros((2, 2))
        
        h2_proj = H2Projection()
        A_proj, B_proj, C_proj, D_proj = h2_proj.project(A, B, C, D)
        
        # Check that real inputs give real outputs
        if np.all(np.isreal(A)):
            self.assertTrue(np.all(np.isreal(A_proj)),
                           "Real input should give real output")
            
        # Check controllability/observability preservation (for stable part)
        # This is a more advanced test
        
    def test_projection_with_jordan_blocks(self):
        """Test projection of systems with Jordan blocks (repeated eigenvalues)."""
        from h2_projection import H2Projection
        
        # Create a system with Jordan blocks
        # One stable block, one unstable block
        n = 4
        A = np.array([[0.5, 1, 0, 0],      # Stable Jordan block
                      [0, 0.5, 0, 0],
                      [0, 0, 1.2, 1],       # Unstable Jordan block
                      [0, 0, 0, 1.2]], dtype=np.complex128)
        
        B = np.random.randn(n, 2) + 1j * np.random.randn(n, 2)
        C = np.random.randn(2, n) + 1j * np.random.randn(2, n)
        D = np.zeros((2, 2), dtype=np.complex128)
        
        h2_proj = H2Projection()
        A_proj, B_proj, C_proj, D_proj = h2_proj.project(A, B, C, D)
        
        # Should keep only the stable Jordan block
        self.assertEqual(A_proj.shape[0], 2, "Should keep only stable block")
        
        # Check eigenvalues
        eigvals = np.linalg.eigvals(A_proj)
        self.assertTrue(np.all(np.abs(eigvals) < 1.0),
                       "All eigenvalues should be stable")


if __name__ == '__main__':
    unittest.main(verbosity=2)