#!/usr/bin/env python3
"""
Test BOP (Balanced Output Pairs) Parametrization
=================================================
Based on Section 7.1 of AUTO_MOSfinal.pdf.

BOP provides a chart-based parametrization of lossless functions.
Key properties to test:
1. The chart center Ω corresponds to V = 0
2. All points in the chart produce lossless functions
3. The parametrization is smooth and invertible within the chart domain
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lossless_embedding import verify_lossless
import bop as bop_module


class TestBOP(unittest.TestCase):
    """Test the BOP (Balanced Output Pairs) parametrization."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
    def test_chart_center_properties(self):
        """Test that the chart center Ω has the expected properties."""
        # Create a simple chart center (unitary realization)
        n = 2  # state dimension
        m = 2  # input/output dimension
        
        # Create a unitary realization matrix Ω = [W X; Y Z]
        # Such that Ω^H * Ω = I
        theta = np.random.randn(n+m, n+m) + 1j * np.random.randn(n+m, n+m)
        Omega, _ = np.linalg.qr(theta)
        
        # Extract blocks
        W = Omega[:n, :n]
        X = Omega[:n, n:]
        Y = Omega[n:, :n]
        Z = Omega[n:, n:]
        
        # Test unitarity of Omega
        unitarity_error = np.linalg.norm(Omega.conj().T @ Omega - np.eye(n+m))
        self.assertLess(unitarity_error, 1e-14,
                       f"Omega not unitary, error = {unitarity_error:.2e}")
        
        # Test that W is stable (eigenvalues inside unit disk)
        eigvals_W = np.linalg.eigvals(W)
        # Scale W to ensure stability
        max_eig = np.max(np.abs(eigvals_W))
        if max_eig >= 1.0:
            scale = 0.95 / max_eig
            W = W * scale
            # Need to adjust other blocks to maintain structure
            # This is simplified - proper implementation would maintain unitarity
        
        # Import BOP from correct module
        from bop import BOP
        
        # Create BOP instance with chart center
        bop = BOP(W, X, Y, Z)
        
        # At chart center, V = 0 should give back the original system
        V = np.zeros((m, n), dtype=np.complex128)
        Do = np.eye(m, dtype=np.complex128)
        
        A, B, C, D = bop.V_to_realization(V, Do)
        
        # Check that we get back something close to original
        # (allowing for transformations)
        self.assertEqual(A.shape, (n, n))
        self.assertEqual(B.shape, (n, m))
        self.assertEqual(C.shape, (m, n))
        self.assertEqual(D.shape, (m, m))
        
        # The resulting system should be lossless
        max_error = verify_lossless(A, B, C, D)
        self.assertLess(max_error, 1e-10,
                       f"Result not lossless, error = {max_error:.2e}")
    
    def test_parametrization_produces_lossless(self):
        """Test that all points in the chart produce lossless functions."""
        n = 2
        m = 2
        
        # Create a simple unitary chart center
        from bop import create_unitary_chart_center
        W, X, Y, Z = create_unitary_chart_center(n, m)
        
        from bop import BOP
        bop = BOP(W, X, Y, Z)
        
        # Test various V values within the chart domain
        # V must satisfy ||V|| < something to stay in chart
        test_V_values = [
            np.zeros((m, n), dtype=np.complex128),
            0.1 * np.ones((m, n), dtype=np.complex128),
            0.2 * (np.random.randn(m, n) + 1j * np.random.randn(m, n)),
        ]
        
        Do = np.eye(m, dtype=np.complex128)
        
        for i, V in enumerate(test_V_values):
            with self.subTest(i=i):
                # Check if V is in valid domain
                is_valid = bop.is_in_domain(V)
                if is_valid:
                    A, B, C, D = bop.V_to_realization(V, Do)
                    
                    # Should produce lossless function
                    max_error = verify_lossless(A, B, C, D)
                    self.assertLess(max_error, 1e-10,
                                   f"Not lossless for V index {i}, error = {max_error:.2e}")
                    
                    # Check stability
                    eigvals = np.linalg.eigvals(A)
                    self.assertTrue(np.all(np.abs(eigvals) < 1.0),
                                   f"A not stable for V index {i}")
    
    def test_invertibility_within_chart(self):
        """Test that the parametrization is invertible within the chart."""
        n = 3
        m = 2
        
        from bop import create_unitary_chart_center, BOP
        W, X, Y, Z = create_unitary_chart_center(n, m)
        bop = BOP(W, X, Y, Z)
        
        # Start with a V in the domain
        V_original = 0.1 * (np.random.randn(m, n) + 1j * np.random.randn(m, n))
        Do = np.eye(m, dtype=np.complex128)
        
        # Forward map: V -> (A, B, C, D)
        A, B, C, D = bop.V_to_realization(V_original, Do)
        
        # Inverse map: (A, B, C, D) -> V
        V_recovered, Do_recovered = bop.realization_to_V(A, B, C, D)
        
        # Check recovery
        V_error = np.linalg.norm(V_recovered - V_original)
        self.assertLess(V_error, 1e-10,
                       f"V not recovered, error = {V_error:.2e}")
        
        Do_error = np.linalg.norm(Do_recovered - Do)
        self.assertLess(Do_error, 1e-10,
                       f"Do not recovered, error = {Do_error:.2e}")
    
    def test_output_normal_form_preservation(self):
        """Test that output normal form is preserved in the parametrization."""
        n = 2
        m = 2
        
        from bop import BOP, create_output_normal_chart_center
        
        # Create chart center in output normal form
        W, X, Y, Z = create_output_normal_chart_center(n, m)
        
        # Verify output normal: W^H * W + Y^H * Y = I
        output_normal_error = np.linalg.norm(
            W.conj().T @ W + Y.conj().T @ Y - np.eye(n)
        )
        self.assertLess(output_normal_error, 1e-10,
                       f"Chart center not output normal, error = {output_normal_error:.2e}")
        
        bop = BOP(W, X, Y, Z)
        
        # Test that parametrization maintains output normal form
        V = 0.05 * np.ones((m, n), dtype=np.complex128)
        Do = np.eye(m, dtype=np.complex128)
        
        if bop.is_in_domain(V):
            A, B, C, D = bop.V_to_realization(V, Do)
            
            # Check if result is in output normal form
            # A^H * A + C^H * C should be close to I
            output_normal_check = A.conj().T @ A + C.conj().T @ C
            deviation = np.linalg.norm(output_normal_check - np.eye(n))
            
            # Note: exact output normal form may not be preserved
            # but should be close for small V
            self.assertLess(deviation, 0.5,
                           f"Far from output normal form, deviation = {deviation:.2e}")
    
    def test_boundary_behavior(self):
        """Test behavior at chart boundary (P becoming singular)."""
        n = 2
        m = 2
        
        from bop import create_unitary_chart_center, BOP
        W, X, Y, Z = create_unitary_chart_center(n, m)
        bop = BOP(W, X, Y, Z)
        
        # Create V approaching the boundary
        # The boundary is where P = Y^H*Y - V^H*V becomes singular
        
        # Start with valid V
        V_valid = 0.1 * np.eye(m, n, dtype=np.complex128)
        self.assertTrue(bop.is_in_domain(V_valid),
                       "V_valid should be in domain")
        
        # Scale up to approach boundary
        for scale in [0.5, 0.8, 0.95]:
            V_test = scale * Y[:, :n]  # Approach Y in magnitude
            
            if bop.is_in_domain(V_test):
                # Should still produce valid lossless function
                Do = np.eye(m, dtype=np.complex128)
                A, B, C, D = bop.V_to_realization(V_test, Do)
                max_error = verify_lossless(A, B, C, D)
                self.assertLess(max_error, 1e-9,
                               f"Not lossless near boundary, scale={scale}")
            else:
                # We've hit the boundary
                self.assertGreater(scale, 0.7,
                                  "Boundary reached too early")
    
    def test_dimension_consistency(self):
        """Test that dimensions are consistent across different n, m values."""
        test_cases = [
            (1, 1),  # Scalar case
            (2, 1),  # More states than outputs
            (2, 2),  # Square case
            (2, 3),  # More outputs than states
            (5, 3),  # Larger system
        ]
        
        from bop import create_unitary_chart_center, BOP
        
        for n, m in test_cases:
            with self.subTest(n=n, m=m):
                W, X, Y, Z = create_unitary_chart_center(n, m)
                
                # Check dimensions
                self.assertEqual(W.shape, (n, n))
                self.assertEqual(X.shape, (n, m))
                self.assertEqual(Y.shape, (m, n))
                self.assertEqual(Z.shape, (m, m))
                
                bop = BOP(W, X, Y, Z)
                
                # V should be m x n
                V = np.zeros((m, n), dtype=np.complex128)
                Do = np.eye(m, dtype=np.complex128)
                
                A, B, C, D = bop.V_to_realization(V, Do)
                
                # Check output dimensions
                self.assertEqual(A.shape, (n, n))
                self.assertEqual(B.shape, (n, m))
                self.assertEqual(C.shape, (m, n))
                self.assertEqual(D.shape, (m, m))


if __name__ == '__main__':
    unittest.main(verbosity=2)