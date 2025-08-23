#!/usr/bin/env python3
"""
Analytical Gradient Implementation for RARL2
=============================================
Implements the exact gradient formula from equation (11) of AUTO_MOSfinal.pdf:
dJn/dλ = 2 Re Tr(P*₁₂[A* Q₁₂ ∂A/∂λ + C* ∂C/∂λ])

This module provides:
1. Analytical gradients via implicit differentiation
2. Integration with BOP parametrization
3. Automatic differentiation support
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional
from lossless_embedding import solve_discrete_lyapunov


def solve_stein_equation(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Solve the Stein equation: A^H * Q * A + C^H * C = Q
    
    This is equation (10) from the paper.
    
    Args:
        A: System matrix (n × n)
        C: Output matrix (p × n)
        
    Returns:
        Q: Solution to Stein equation (n × n)
    """
    n = A.shape[0]
    
    # For output normal pairs where A^H*A + C^H*C = I,
    # the observability Gramian Q₁₂ = I (identity)
    ON_check = A.conj().T @ A + C.conj().T @ C
    if np.allclose(ON_check, np.eye(n), atol=1e-8):
        # Output normal case: Q = I
        return np.eye(n, dtype=np.complex128)
    
    # General case: solve A^H * Q * A - Q = -C^H * C
    return solve_discrete_lyapunov(A.conj().T, C.conj().T @ C)


def solve_sylvester_P12(A: np.ndarray, A_hat: np.ndarray, 
                        B: np.ndarray, B_hat: np.ndarray) -> np.ndarray:
    """
    Solve for P₁₂* from the Sylvester equation:
    A * P₁₂* * A^H + B * B^H = P₁₂*
    
    Args:
        A: System matrix from approximation
        A_hat: Not used (for future extension)
        B: Input matrix from approximation
        B_hat: Not used (for future extension)
        
    Returns:
        P₁₂*: Solution (conjugate transpose of P₁₂)
    """
    # This is a discrete Lyapunov equation
    # A * P₁₂* * A^H - P₁₂* = -B * B^H
    return solve_discrete_lyapunov(A, B @ B.conj().T)


def compute_analytical_gradient(C: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the analytical gradient using equation (11) from the paper.
    
    dJn/dλ = 2 Re Tr(P*₁₂[A* Q₁₂ ∂A/∂λ + C* ∂C/∂λ])
    
    Args:
        C, A: Observable pair (C,A) in output normal form
        
    Returns:
        grad_C: Gradient with respect to C
        grad_A: Gradient with respect to A
    """
    n = A.shape[0]
    p = C.shape[0]
    
    # Step 1: Verify output normal constraint (soft check)
    ON_check = A.conj().T @ A + C.conj().T @ C
    ON_error = np.linalg.norm(ON_check - np.eye(n))
    if ON_error > 1e-8:
        print(f"Warning: Output normal error = {ON_error:.2e}")
    
    # Step 2: Solve for Q₁₂ from equation (10)
    # A^H * Q₁₂ * A + C^H * C = Q₁₂
    Q12 = solve_stein_equation(A, C)
    
    # Step 3: Compute B from first necessary condition
    # From equation (8): Q₁₂* * B = -Q̂ * B̂
    # Since Q̂ = I (output normal), B = -Q₁₂* * B_target
    B = -Q12.conj().T @ B_target
    
    # Step 4: Solve for P₁₂* from dual equation
    P12_star = solve_sylvester_P12(A, A, B, B)
    
    # Step 5: Compute gradients using equation (11)
    # dJn/dC = 2 Re Tr(P₁₂* * C*)
    # dJn/dA = 2 Re Tr(P₁₂* * A* * Q₁₂)
    
    # For C gradient
    grad_C = 2 * np.real(P12_star @ C.conj().T).T
    
    # For A gradient  
    grad_A = 2 * np.real(P12_star @ (A.conj().T @ Q12))
    
    # Step 6: Project gradients to maintain output normal constraint
    # The constraint is: A^H * A + C^H * C = I
    # Its differential: dA^H * A + A^H * dA + dC^H * C + C^H * dC = 0
    
    # Compute Lagrange multiplier matrix
    Lambda = (A.conj().T @ grad_A + grad_A.conj().T @ A + 
              C.conj().T @ grad_C + grad_C.conj().T @ C) / 2
    
    # Make Lambda Hermitian (it should be, but numerical errors...)
    Lambda = (Lambda + Lambda.conj().T) / 2
    
    # Project gradients onto constraint manifold
    grad_A_proj = grad_A - A @ Lambda
    grad_C_proj = grad_C - C @ Lambda
    
    return grad_C_proj, grad_A_proj


def compute_rarl2_gradient_full(C: np.ndarray, A: np.ndarray,
                               A_F: np.ndarray, B_F: np.ndarray,
                               C_F: np.ndarray, D_F: np.ndarray) -> np.ndarray:
    """
    Complete gradient computation for RARL2 optimization.
    
    This implements the full gradient of the concentrated criterion
    with respect to the (C,A) parameters, using analytical formulas.
    
    Args:
        C, A: Current observable pair (output normal)
        A_F, B_F, C_F, D_F: Target system
        
    Returns:
        Gradient vector (flattened, real part for real parameters)
    """
    # Get analytical gradients
    grad_C, grad_A = compute_analytical_gradient(C, A, B_F)
    
    # Flatten and concatenate
    grad = np.concatenate([grad_C.flatten(), grad_A.flatten()])
    
    # Return real part for optimization
    # (imaginary part should be negligible for real objective)
    return np.real(grad)


class AnalyticalRARL2Optimizer:
    """
    RARL2 optimizer using analytical gradients.
    
    This replaces finite differences with exact gradients
    computed via implicit differentiation.
    """
    
    def __init__(self, target: Tuple[np.ndarray, ...], order: int):
        """Initialize with target system and approximation order."""
        self.A_F, self.B_F, self.C_F, self.D_F = target
        self.n = order
        self.p = self.C_F.shape[0]
        self.m = self.B_F.shape[1]
        
        # Initialize with output normal pair
        from rarl2_integrated import create_output_normal_pair
        self.C, self.A = create_output_normal_pair(self.n, self.p)
        
        self.iteration = 0
        self.errors = []
        
    def compute_objective(self) -> float:
        """Compute the concentrated criterion value."""
        from rarl2_integrated import compute_objective_lossless
        return compute_objective_lossless(
            self.C, self.A,
            self.A_F, self.B_F, self.C_F, self.D_F
        )
    
    def step(self, learning_rate: float = 0.01) -> float:
        """
        Perform one optimization step using analytical gradients.
        
        Args:
            learning_rate: Step size for gradient descent
            
        Returns:
            Current objective value after step
        """
        # Compute analytical gradient
        grad = compute_rarl2_gradient_full(
            self.C, self.A,
            self.A_F, self.B_F, self.C_F, self.D_F
        )
        
        # Reshape gradient
        grad_C = grad[:self.C.size].reshape(self.C.shape)
        grad_A = grad[self.C.size:].reshape(self.A.shape)
        
        # Line search for optimal step size
        alpha = self._line_search(grad_C, grad_A, learning_rate)
        
        # Update parameters
        self.C = self.C - alpha * grad_C
        self.A = self.A - alpha * grad_A
        
        # Re-normalize to maintain output normal form
        self._normalize_output_normal()
        
        # Ensure stability
        self._ensure_stability()
        
        # Compute and store error
        self.iteration += 1
        error = self.compute_objective()
        self.errors.append(error)
        
        return error
    
    def _line_search(self, grad_C: np.ndarray, grad_A: np.ndarray,
                    max_step: float) -> float:
        """
        Backtracking line search for step size.
        
        Uses Armijo condition to find suitable step size.
        """
        alpha = max_step
        beta = 0.5  # Backtracking factor
        c = 0.1     # Armijo constant
        
        obj_current = self.compute_objective()
        grad_norm_sq = np.sum(np.abs(grad_C)**2) + np.sum(np.abs(grad_A)**2)
        
        for _ in range(20):  # Max 20 iterations
            # Trial step
            C_new = self.C - alpha * grad_C
            A_new = self.A - alpha * grad_A
            
            # Check if A_new is stable
            try:
                eigvals = np.linalg.eigvals(A_new)
                if np.all(np.abs(eigvals) < 0.999):
                    # Normalize trial point
                    M = A_new.conj().T @ A_new + C_new.conj().T @ C_new
                    scale = linalg.sqrtm(linalg.inv(M)).astype(np.complex128)
                    A_test = A_new @ scale
                    C_test = C_new @ scale
                    
                    # Compute new objective
                    from rarl2_integrated import compute_objective_lossless
                    obj_new = compute_objective_lossless(
                        C_test, A_test,
                        self.A_F, self.B_F, self.C_F, self.D_F
                    )
                    
                    # Check Armijo condition
                    if obj_new <= obj_current - c * alpha * grad_norm_sq:
                        return alpha
            except:
                pass  # Continue with smaller step
            
            # Reduce step size
            alpha *= beta
        
        return alpha
    
    def _normalize_output_normal(self):
        """
        Re-project onto output normal constraint manifold.
        
        Ensures A^H * A + C^H * C = I.
        """
        # Current constraint matrix
        M = self.A.conj().T @ self.A + self.C.conj().T @ self.C
        
        # Find transformation to restore constraint
        # We want T such that (C*T, A*T) is output normal
        # T^H * M * T = I, so T = M^{-1/2}
        
        try:
            # Direct computation
            M_sqrt_inv = linalg.sqrtm(linalg.inv(M)).astype(np.complex128)
            self.A = self.A @ M_sqrt_inv
            self.C = self.C @ M_sqrt_inv
        except:
            # Fallback: iterative normalization
            for _ in range(10):
                M = self.A.conj().T @ self.A + self.C.conj().T @ self.C
                eigvals = np.linalg.eigvalsh(M)
                
                if np.min(eigvals) < 1e-10:
                    # Regularize
                    M = M + 1e-10 * np.eye(self.n)
                
                try:
                    scale = linalg.sqrtm(linalg.inv(M)).astype(np.complex128)
                    self.A = self.A @ scale
                    self.C = self.C @ scale
                    
                    # Check convergence
                    M_new = self.A.conj().T @ self.A + self.C.conj().T @ self.C
                    if np.linalg.norm(M_new - np.eye(self.n)) < 1e-10:
                        break
                except:
                    break
    
    def _ensure_stability(self):
        """Project A to stable region if needed."""
        eigvals = np.linalg.eigvals(self.A)
        max_eigval = np.max(np.abs(eigvals))
        
        if max_eigval >= 0.999:
            # Scale down to ensure stability
            self.A = self.A * 0.99 / max_eigval
            # Re-normalize after scaling
            self._normalize_output_normal()
    
    def get_approximation(self) -> Tuple[np.ndarray, ...]:
        """Get the current approximation as a lossy system."""
        from lossless_embedding import lossless_embedding
        from rarl2_integrated import lossless_to_lossy
        
        B_G, D_G = lossless_embedding(self.C, self.A)
        return lossless_to_lossy(
            self.A, B_G, self.C, D_G,
            self.A_F, self.B_F, self.C_F, self.D_F
        )