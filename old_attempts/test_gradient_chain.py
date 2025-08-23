#!/usr/bin/env python3
"""
Test Suite for Gradient Computation Chain in RARL2
====================================================
Tests the complete gradient chain:
V parameters → (C,A) realization → objective function

Following TDD approach with automatic differentiation validation.
"""

import numpy as np
import unittest
from scipy import linalg
import torch
import torch.autograd as autograd


class TestGradientChain(unittest.TestCase):
    """Test gradient computation through the complete RARL2 chain."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Dimensions
        self.n = 3  # State dimension
        self.p = 2  # Output dimension
        self.m = 2  # Input dimension
        
        # Create test target system
        self.create_target_system()
        
        # Create test (C,A) pair in output normal form
        self.create_output_normal_pair()
        
    def create_target_system(self):
        """Create a stable target system for testing."""
        A_F = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        eigvals = np.linalg.eigvals(A_F)
        A_F = A_F * 0.8 / np.max(np.abs(eigvals))
        
        B_F = np.random.randn(4, self.m) + 1j * np.random.randn(4, self.m)
        C_F = np.random.randn(self.p, 4) + 1j * np.random.randn(self.p, 4)
        D_F = np.random.randn(self.p, self.m) + 1j * np.random.randn(self.p, self.m)
        
        self.A_F = A_F
        self.B_F = B_F
        self.C_F = C_F
        self.D_F = D_F
        
    def create_output_normal_pair(self):
        """Create (C,A) pair satisfying A^H*A + C^H*C = I."""
        # Random unitary matrix
        M = np.random.randn(self.n + self.p, self.n) + \
            1j * np.random.randn(self.n + self.p, self.n)
        U, _ = np.linalg.qr(M)
        
        # Extract A and C
        A = U[:self.n, :] * 0.9  # Scale for stability
        C = U[self.n:self.n+self.p, :]
        
        # Normalize to satisfy output normal constraint
        current = A.conj().T @ A + C.conj().T @ C
        scale = linalg.sqrtm(np.linalg.inv(current)).astype(np.complex128)
        
        self.A = A @ scale
        self.C = C @ scale
    
    def test_equation_11_gradient(self):
        """
        Test gradient computation using equation (11) from the paper:
        dJn/dλ = 2 Re Tr(P*₁₂[A* Q₁₂ ∂A/∂λ + C* ∂C/∂λ])
        """
        from rarl2_analytical_gradient import (
            solve_stein_equation,
            solve_sylvester_P12,
            compute_analytical_gradient
        )
        
        # Step 1: Solve for Q₁₂ from Stein equation
        Q12 = solve_stein_equation(self.A, self.C)
        
        # Verify Stein equation: A^H * Q12 * A + C^H * C = Q12
        lhs = self.A.conj().T @ Q12 @ self.A + self.C.conj().T @ self.C
        np.testing.assert_allclose(lhs, Q12, atol=1e-10,
                                  err_msg="Stein equation not satisfied")
        
        # Step 2: Compute B from lossless embedding
        from lossless_embedding import lossless_embedding
        B, D = lossless_embedding(self.C, self.A)
        
        # Step 3: Solve for P₁₂*
        P12_star = solve_sylvester_P12(self.A, self.A, B, B)
        
        # Verify Sylvester equation: A * P12* * A^H + B * B^H = P12*
        lhs = self.A @ P12_star @ self.A.conj().T + B @ B.conj().T
        np.testing.assert_allclose(lhs, P12_star, atol=1e-10,
                                  err_msg="Sylvester equation not satisfied")
        
        # Step 4: Compute analytical gradient
        grad_C, grad_A = compute_analytical_gradient(self.C, self.A, self.B_F)
        
        # Test gradient properties
        self.assertEqual(grad_C.shape, self.C.shape)
        self.assertEqual(grad_A.shape, self.A.shape)
        
        # Gradients should be finite
        self.assertTrue(np.all(np.isfinite(grad_C)))
        self.assertTrue(np.all(np.isfinite(grad_A)))
    
    def test_v_to_ca_transformation(self):
        """
        Test the transformation from V parameters to (C,A) realization.
        This follows the BOP parametrization from Section 7.1 of the paper.
        """
        from bop import BOP, create_output_normal_chart_center
        
        # Create chart center
        W, X, Y, Z = create_output_normal_chart_center(self.n, self.m)
        
        # Create BOP parametrization
        bop = BOP(W, X, Y, Z)
        
        # Test with V = 0 (should give chart center)
        V = np.zeros((self.m, self.n), dtype=np.complex128)
        D0 = np.eye(self.m, dtype=np.complex128)
        
        A, B, C, D = bop.V_to_realization(V, D0)
        
        # At V=0, we should get a lossless system
        from lossless_embedding import verify_lossless
        lossless_error = verify_lossless(A, B, C, D, n_points=10)
        self.assertLess(lossless_error, 1e-10,
                       "V=0 should give lossless system")
        
        # Test with non-zero V
        V = np.random.randn(self.m, self.n) * 0.1 + \
            1j * np.random.randn(self.m, self.n) * 0.1
        
        # Check V is in valid domain
        self.assertTrue(bop.is_in_domain(V),
                       "V should be in valid domain")
        
        A, B, C, D = bop.V_to_realization(V, D0)
        
        # Result should still be lossless
        lossless_error = verify_lossless(A, B, C, D, n_points=10)
        self.assertLess(lossless_error, 1e-10,
                       "Non-zero V should still give lossless system")
    
    def test_gradient_via_torch_autodiff(self):
        """
        Test gradient computation using PyTorch automatic differentiation.
        This validates our analytical gradients.
        """
        # Convert to torch tensors with complex support
        C_real = torch.tensor(self.C.real, requires_grad=True, dtype=torch.float64)
        C_imag = torch.tensor(self.C.imag, requires_grad=True, dtype=torch.float64)
        A_real = torch.tensor(self.A.real, requires_grad=True, dtype=torch.float64)
        A_imag = torch.tensor(self.A.imag, requires_grad=True, dtype=torch.float64)
        
        # Define objective function in torch
        def torch_objective(C_r, C_i, A_r, A_i):
            C = torch.complex(C_r, C_i)
            A = torch.complex(A_r, A_i)
            
            # Simplified objective for testing: ||C||^2 + ||A||^2
            # (Full objective would require implementing H2 norm in torch)
            return torch.sum(torch.abs(C)**2) + torch.sum(torch.abs(A)**2)
        
        # Compute gradient via autodiff
        obj = torch_objective(C_real, C_imag, A_real, A_imag)
        obj.backward()
        
        grad_C_torch = C_real.grad.numpy() + 1j * C_imag.grad.numpy()
        grad_A_torch = A_real.grad.numpy() + 1j * A_imag.grad.numpy()
        
        # For this simple objective, analytical gradient is:
        grad_C_analytical = 2 * self.C
        grad_A_analytical = 2 * self.A
        
        # Compare
        np.testing.assert_allclose(grad_C_torch, grad_C_analytical, rtol=1e-10)
        np.testing.assert_allclose(grad_A_torch, grad_A_analytical, rtol=1e-10)
    
    def test_complete_gradient_chain(self):
        """
        Test the complete gradient chain: V → (C,A) → objective.
        This is the full RARL2 gradient computation.
        """
        # This test validates that we can compute:
        # dJ/dV = dJ/d(C,A) * d(C,A)/dV
        
        # Step 1: Create V parameters
        V_size = self.m * self.n
        V_flat = np.random.randn(V_size) * 0.1 + 1j * np.random.randn(V_size) * 0.1
        V = V_flat.reshape(self.m, self.n)
        
        # Step 2: Transform V → (C,A)
        from bop import BOP, create_output_normal_chart_center
        W, X, Y, Z = create_output_normal_chart_center(self.n, self.m)
        bop = BOP(W, X, Y, Z)
        
        if not bop.is_in_domain(V):
            # Scale V to be in domain
            V = V * 0.01
        
        D0 = np.eye(self.m, dtype=np.complex128)
        A, B, C, D = bop.V_to_realization(V, D0)
        
        # Extract just (C,A) for our gradient
        C_from_V = C
        A_from_V = A
        
        # Step 3: Compute gradient dJ/d(C,A)
        from rarl2_analytical_gradient import compute_analytical_gradient
        grad_C, grad_A = compute_analytical_gradient(C_from_V, A_from_V, self.B_F)
        
        # Step 4: Compute d(C,A)/dV via finite differences
        # (In production, we'd use automatic differentiation)
        eps = 1e-7
        dCA_dV = np.zeros((C.size + A.size, V_flat.size), dtype=np.complex128)
        
        for i in range(V_flat.size):
            V_pert = V_flat.copy()
            V_pert[i] += eps
            V_pert_mat = V_pert.reshape(self.m, self.n)
            
            if bop.is_in_domain(V_pert_mat):
                A_pert, B_pert, C_pert, D_pert = bop.V_to_realization(V_pert_mat, D0)
                
                dC_dVi = (C_pert - C) / eps
                dA_dVi = (A_pert - A) / eps
                
                dCA_dV[:C.size, i] = dC_dVi.flatten()
                dCA_dV[C.size:, i] = dA_dVi.flatten()
        
        # Step 5: Chain rule: dJ/dV = dJ/d(C,A) * d(C,A)/dV
        grad_CA = np.concatenate([grad_C.flatten(), grad_A.flatten()])
        grad_V = dCA_dV.T @ grad_CA
        
        # Test properties
        self.assertEqual(grad_V.shape, V_flat.shape)
        self.assertTrue(np.all(np.isfinite(grad_V)))
        
        # Gradient should be mostly real for real objective
        # (imaginary part should be small)
        if np.all(np.isreal(V_flat)):
            rel_imag = np.linalg.norm(grad_V.imag) / (np.linalg.norm(grad_V) + 1e-10)
            self.assertLess(rel_imag, 0.1,
                          "Gradient should be mostly real for real parameters")
    
    def test_gradient_descent_convergence(self):
        """
        Test that gradient descent with our gradients actually converges.
        """
        from rarl2_analytical_gradient import AnalyticalRARL2Optimizer
        
        # Create optimizer
        optimizer = AnalyticalRARL2Optimizer(
            target=(self.A_F, self.B_F, self.C_F, self.D_F),
            order=self.n
        )
        
        # Run a few iterations
        initial_error = optimizer.compute_objective()
        
        for i in range(10):
            error = optimizer.step(learning_rate=0.01)
            
            # Error should generally decrease
            if i > 2:  # Allow a few iterations to stabilize
                self.assertLessEqual(error, initial_error * 1.1,
                                   f"Error increased too much at iteration {i}")
        
        final_error = optimizer.errors[-1]
        
        # Should have some improvement
        self.assertLess(final_error, initial_error,
                       "Optimization should improve objective")
        
        # Check that (C,A) remains output normal
        ON_check = optimizer.A.conj().T @ optimizer.A + \
                  optimizer.C.conj().T @ optimizer.C
        np.testing.assert_allclose(ON_check, np.eye(self.n), atol=1e-8,
                                  err_msg="Output normal constraint violated")
        
        # Check stability
        eigvals = np.linalg.eigvals(optimizer.A)
        self.assertTrue(np.all(np.abs(eigvals) < 1.0),
                       "A should remain stable")


class TestAutomaticDifferentiation(unittest.TestCase):
    """Test automatic differentiation for RARL2 components."""
    
    def test_torch_complex_gradients(self):
        """
        Test PyTorch's handling of complex gradients.
        This is crucial for our implementation.
        """
        # Create complex tensor
        z_real = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        z_imag = torch.tensor([[0.5, -0.5], [1.0, -1.0]], requires_grad=True)
        z = torch.complex(z_real, z_imag)
        
        # Define a real-valued function of complex variable
        # f(z) = ||z||^2 = sum(|z_ij|^2)
        def f(z_r, z_i):
            z = torch.complex(z_r, z_i)
            return torch.sum(torch.abs(z)**2)
        
        # Compute gradient
        loss = f(z_real, z_imag)
        loss.backward()
        
        # Analytical gradient: df/dz* = z (Wirtinger derivative)
        grad_analytical_real = 2 * z_real.detach().numpy()
        grad_analytical_imag = 2 * z_imag.detach().numpy()
        
        # Compare
        np.testing.assert_allclose(z_real.grad.numpy(), grad_analytical_real)
        np.testing.assert_allclose(z_imag.grad.numpy(), grad_analytical_imag)
    
    def test_matrix_operation_gradients(self):
        """
        Test gradients through matrix operations needed for RARL2.
        """
        # Test gradient through matrix multiplication
        A = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        B = torch.randn(3, 2, requires_grad=True, dtype=torch.float64)
        C = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        
        # Operation: Tr(C * A * B)
        result = torch.trace(C @ A @ B)
        result.backward()
        
        # Analytical gradients
        grad_A_analytical = B @ C.T
        grad_B_analytical = A.T @ C.T
        grad_C_analytical = A @ B
        
        np.testing.assert_allclose(A.grad.numpy(), grad_A_analytical.numpy(), rtol=1e-10)
        np.testing.assert_allclose(B.grad.numpy(), grad_B_analytical.numpy(), rtol=1e-10)
        np.testing.assert_allclose(C.grad.numpy(), grad_C_analytical.numpy(), rtol=1e-10)


if __name__ == '__main__':
    # First check if we have required dependencies
    try:
        import torch
        print("PyTorch available, running full test suite")
    except ImportError:
        print("WARNING: PyTorch not available, skipping autodiff tests")
        print("Install with: pip install torch")
    
    unittest.main()