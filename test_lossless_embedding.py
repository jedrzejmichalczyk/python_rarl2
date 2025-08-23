#!/usr/bin/env python3
"""
Test Lossless Embedding - First unit test for RARL2
=====================================================
According to Proposition 1 in AUTO_MOSfinal.pdf:
Given an observable pair (C, A) with A asymptotically stable, 
the lossless embedding creates a lossless matrix G(z) = D + C(zI - A)^(-1)B

Key property to test: G(z) should be lossless (unitary on unit circle)
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lossless_embedding import (
    lossless_embedding, 
    verify_lossless,
    solve_discrete_lyapunov,
    create_output_normal_pair
)


class TestLosslessEmbedding(unittest.TestCase):
    """Test the lossless embedding from observable pairs."""
    
    def setUp(self):
        """Set up test cases with different observable pairs."""
        np.random.seed(42)  # For reproducibility
        
    def test_simple_2x2_stable_system(self):
        """Test lossless embedding for a simple 2x2 stable system."""
        # Create a simple stable system
        # A should have all eigenvalues inside unit disk
        A = np.array([[0.5, 0.2],
                      [-0.2, 0.3]], dtype=np.complex128)
        
        # C should make (C, A) observable
        C = np.array([[1.0, 0.0],
                      [0.0, 1.0]], dtype=np.complex128)
        
        # Verify stability (all eigenvalues |λ| < 1)
        eigvals = np.linalg.eigvals(A)
        self.assertTrue(np.all(np.abs(eigvals) < 1.0), 
                       f"A is not stable, eigenvalues: {eigvals}")
        
        # Verify observability: rank([C; CA]) should be n
        n = A.shape[0]
        obs_matrix = np.vstack([C, C @ A])
        rank = np.linalg.matrix_rank(obs_matrix)
        self.assertEqual(rank, n, 
                        f"(C, A) is not observable, rank={rank}, n={n}")
        
        # Compute lossless embedding with ν = 1
        nu = 1.0
        B, D = lossless_embedding(C, A, nu)
        
        # Verify dimensions
        self.assertEqual(B.shape, (n, C.shape[0]))
        self.assertEqual(D.shape, (C.shape[0], C.shape[0]))
        
        # Key test: Verify losslessness on unit circle
        max_error = verify_lossless(A, B, C, D, n_points=100)
        self.assertLess(max_error, 1e-10,
                       f"G is not lossless on unit circle, max error = {max_error:.2e}")
        
        # Verify G(ν) = I
        G_nu = D + C @ np.linalg.inv(nu * np.eye(n) - A) @ B
        identity_error = np.linalg.norm(G_nu - np.eye(G_nu.shape[0]))
        self.assertLess(identity_error, 1e-10, 
                       f"G(ν) != I, error={identity_error:.2e}")
        
    def test_scalar_case(self):
        """Test lossless embedding for scalar (1x1) case."""
        # Scalar stable system
        A = np.array([[0.5]], dtype=np.complex128)
        C = np.array([[1.0]], dtype=np.complex128)
        
        # Lossless embedding with ν = 1
        nu = 1.0
        B, D = lossless_embedding(C, A, nu)
        
        # Test on unit circle
        theta_values = np.linspace(0, 2*np.pi, 50)
        for theta in theta_values:
            z = np.exp(1j * theta)
            G_z = D + C * (1.0 / (z - A[0,0])) * B
            
            # In scalar case, |G(z)| should be 1 on unit circle
            magnitude = np.abs(G_z[0,0])
            self.assertAlmostEqual(magnitude, 1.0, places=10,
                                 msg=f"|G(e^(i*{theta:.2f}))| = {magnitude:.6f} != 1")

    def test_output_normal_form(self):
        """Test lossless embedding with output normal form."""
        # Create an output normal pair: A^H * A + C^H * C = I
        n = 3
        p = 2
        
        C, A = create_output_normal_pair(n, p)
        
        # Verify output normal form
        norm_error = np.linalg.norm(A.conj().T @ A + C.conj().T @ C - np.eye(n))
        self.assertLess(norm_error, 1e-9, 
                       f"Not in output normal form, error={norm_error:.2e}")
        
        # Compute lossless embedding
        nu = -1.0  # Different choice of ν
        B, D = lossless_embedding(C, A, nu)
        
        # The realization matrix should be unitary for output normal form
        # when Q = I (which it is for output normal form)
        R = np.block([[A, B],
                      [C, D]])
        
        # For output normal form, we get a balanced realization
        # which means the realization matrix is close to unitary
        unitarity_error = np.linalg.norm(R @ R.conj().T - np.eye(n+p))
        
        # Note: The realization matrix is exactly unitary only when
        # the observability Gramian Q = I, which happens when we have
        # perfect output normal form
        self.assertLess(unitarity_error, 1e-8,
                       f"Realization matrix far from unitary, error={unitarity_error:.2e}")
        
        # Verify losslessness
        max_error = verify_lossless(A, B, C, D)
        self.assertLess(max_error, 1e-10,
                       f"Not lossless, max error = {max_error:.2e}")
    
    def test_different_nu_values(self):
        """Test that lossless embedding works for different ν values."""
        # Simple stable system
        A = np.array([[0.3, 0.1],
                      [-0.1, 0.4]], dtype=np.complex128)
        C = np.array([[1.0, 0.5]], dtype=np.complex128)
        
        # Test different values of ν on unit circle
        nu_values = [1.0, -1.0, 1j, -1j, np.exp(1j * np.pi/4)]
        
        for nu in nu_values:
            with self.subTest(nu=nu):
                B, D = lossless_embedding(C, A, nu)
                
                # Verify losslessness
                max_error = verify_lossless(A, B, C, D)
                self.assertLess(max_error, 1e-10,
                               f"Not lossless for nu={nu}, error={max_error:.2e}")
                
                # Verify G(ν) = I
                n = A.shape[0]
                G_nu = D + C @ np.linalg.inv(nu * np.eye(n) - A) @ B
                identity_error = np.linalg.norm(G_nu - np.eye(G_nu.shape[0]))
                self.assertLess(identity_error, 1e-10,
                               f"G({nu}) != I, error={identity_error:.2e}")
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        # Unstable system
        A_unstable = np.array([[1.5, 0.0],
                               [0.0, 0.8]], dtype=np.complex128)
        C = np.array([[1.0, 0.0]], dtype=np.complex128)
        
        with self.assertRaises(ValueError) as context:
            lossless_embedding(C, A_unstable)
        self.assertIn("not stable", str(context.exception))
        
        # Non-observable pair
        A_stable = np.array([[0.5, 0.0],
                            [0.0, 0.3]], dtype=np.complex128)
        C_unobs = np.array([[1.0, 0.0]], dtype=np.complex128)  # Can't observe second state
        
        with self.assertRaises(ValueError) as context:
            lossless_embedding(C_unobs, A_stable)
        self.assertIn("not observable", str(context.exception))
        
        # Invalid nu (not on unit circle)
        A = np.array([[0.5]], dtype=np.complex128)
        C = np.array([[1.0]], dtype=np.complex128)
        
        with self.assertRaises(ValueError) as context:
            lossless_embedding(C, A, nu=0.5)
        self.assertIn("|nu|", str(context.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)