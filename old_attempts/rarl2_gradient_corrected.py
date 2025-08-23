#!/usr/bin/env python3
"""
Corrected Gradient Implementation for RARL2
============================================
Properly handles the two different B matrices:
1. B from lossless embedding (for parametrization)
2. B̂ from necessary conditions (for gradient computation)
"""

import numpy as np
from scipy import linalg
from typing import Tuple
from lossless_embedding import solve_discrete_lyapunov, lossless_embedding


def solve_stein_equation_Q12(A: np.ndarray, A_F: np.ndarray, 
                             C: np.ndarray, C_F: np.ndarray) -> np.ndarray:
    """
    Solve for Q₁₂ from the augmented Stein equation.
    
    From the error system with augmented matrices:
    Ã = [A_F  0]    C̃ = [C_F  -C]
        [0    A]
    
    The Stein equation gives us Q₁₂ in the (1,2) block.
    Q₁₂ maps from approximation space to target space: n_F × n
    
    It satisfies: A_F^H Q₁₂ A + C_F^H C = Q₁₂
    """
    n_F = A_F.shape[0]
    n = A.shape[0]
    
    # Q₁₂ is n_F × n (maps from approximation to target)
    # It satisfies: A_F^H Q₁₂ A + C_F^H C = Q₁₂
    
    # Iterative solution
    Q12 = np.zeros((n_F, n), dtype=np.complex128)
    for _ in range(100):
        Q12_new = A_F.conj().T @ Q12 @ A + C_F.conj().T @ C
        if np.linalg.norm(Q12_new - Q12) < 1e-12:
            break
        Q12 = Q12_new
    
    return Q12


def solve_sylvester_P12(A: np.ndarray, A_F: np.ndarray,
                        B_hat: np.ndarray, B_F: np.ndarray) -> np.ndarray:
    """
    Solve for P₁₂ from the augmented Sylvester equation.
    
    P₁₂ satisfies: A P₁₂ A_F^H + B̂ B_F^H = P₁₂
    """
    # Reshape to standard Sylvester form
    # A P₁₂ A_F^H - P₁₂ = -B̂ B_F^H
    # This is solved by scipy.linalg.solve_sylvester
    P12 = linalg.solve_sylvester(A, -A_F.conj().T, B_hat @ B_F.conj().T)
    return P12


def compute_rarl2_gradient_corrected(
    C: np.ndarray, A: np.ndarray,
    A_F: np.ndarray, B_F: np.ndarray, 
    C_F: np.ndarray, D_F: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the analytical gradient for RARL2 with proper handling of B matrices.
    
    Args:
        C, A: Observable pair (C,A) in output normal form (approximation)
        A_F, B_F, C_F, D_F: Target system to approximate
        
    Returns:
        grad_C: Gradient with respect to C
        grad_A: Gradient with respect to A
    """
    n = A.shape[0]
    n_F = A_F.shape[0]
    p = C.shape[0]
    
    # Verify output normal constraint
    ON_check = A.conj().T @ A + C.conj().T @ C
    ON_error = np.linalg.norm(ON_check - np.eye(n))
    if ON_error > 1e-8:
        print(f"Warning: Output normal error = {ON_error:.2e}")
    
    # Step 1: Solve for Q₁₂ from the coupled Stein equation
    Q12 = solve_stein_equation_Q12(A, A_F, C, C_F)
    
    # Step 2: Compute B̂ from first necessary condition
    # From equation (8): Q₁₂* B_F = -Q̂ B̂
    # Since Q̂ = I (output normal), B̂ = -Q₁₂* B_F
    B_hat = -Q12.conj().T @ B_F
    
    # Step 3: Solve for P₁₂ from the coupled Sylvester equation
    P12 = solve_sylvester_P12(A, A_F, B_hat, B_F)
    
    # Step 4: Compute gradients using equation (11)
    # From the paper: dJn/dλ = 2 Re Tr(P*₁₂[A* Q₁₂ ∂A/∂λ + C* ∂C/∂λ])
    # Since P₁₂ is n × n_F and Q₁₂ is n_F × n:
    # dJn/dC = 2 Re Tr(P₁₂^T Q₁₂^T C^T) = 2 Re Tr(Q₁₂^T P₁₂^T C^T)
    # dJn/dA = 2 Re Tr(P₁₂^T Q₁₂^T A^T) = 2 Re Tr(Q₁₂^T P₁₂^T A^T)
    
    # Actually, from the necessary conditions:
    # grad_C comes from: CP₁₂ = ĈP̂ where P̂ = P₁₂^T P₁₂
    # grad_A comes from: Q₁₂^T A P₁₂ = -Q̂ Â P̂
    
    grad_C = 2 * np.real(P12 @ C_F.conj().T).T
    grad_A = 2 * np.real(Q12.conj().T @ A_F @ P12.T)
    
    # Step 5: Project gradients to maintain output normal constraint
    # The constraint is: A^H * A + C^H * C = I
    Lambda = (A.conj().T @ grad_A + grad_A.conj().T @ A + 
              C.conj().T @ grad_C + grad_C.conj().T @ C) / 2
    
    # Make Lambda Hermitian
    Lambda = (Lambda + Lambda.conj().T) / 2
    
    # Project gradients
    grad_A_proj = grad_A - A @ Lambda
    grad_C_proj = grad_C - C @ Lambda
    
    return grad_C_proj, grad_A_proj


def verify_gradient_computation():
    """Verify the gradient computation with a simple example."""
    np.random.seed(42)
    
    # Create test systems
    n = 2  # Approximation order
    n_F = 3  # Target order
    p = 2  # Output dimension
    m = 2  # Input dimension
    
    # Create output normal pair for approximation
    from rarl2_integrated import create_output_normal_pair
    C, A = create_output_normal_pair(n, p)
    
    # Create target system
    A_F = np.random.randn(n_F, n_F) + 1j * np.random.randn(n_F, n_F)
    A_F = A_F * 0.8 / np.max(np.abs(np.linalg.eigvals(A_F)))
    B_F = np.random.randn(n_F, m) + 1j * np.random.randn(n_F, m)
    C_F = np.random.randn(p, n_F) + 1j * np.random.randn(p, n_F)
    D_F = np.zeros((p, m), dtype=np.complex128)
    
    # Compute gradient
    grad_C, grad_A = compute_rarl2_gradient_corrected(C, A, A_F, B_F, C_F, D_F)
    
    print("Gradient computation successful!")
    print(f"grad_C shape: {grad_C.shape}, norm: {np.linalg.norm(grad_C):.4f}")
    print(f"grad_A shape: {grad_A.shape}, norm: {np.linalg.norm(grad_A):.4f}")
    
    # Verify output normal constraint is preserved
    dON = (A.conj().T @ grad_A + grad_A.conj().T @ A + 
           C.conj().T @ grad_C + grad_C.conj().T @ C)
    print(f"Output normal constraint violation: {np.linalg.norm(dON):.2e}")
    
    # Compare with finite differences
    from rarl2_integrated import compute_objective_lossless
    
    eps = 1e-7
    obj_base = compute_objective_lossless(C, A, A_F, B_F, C_F, D_F)
    
    # Test gradient for first element of C
    C_pert = C.copy()
    C_pert.flat[0] += eps
    obj_pert = compute_objective_lossless(C_pert, A, A_F, B_F, C_F, D_F)
    
    grad_fd = (obj_pert - obj_base) / eps
    grad_analytical = grad_C.flat[0]
    
    print(f"\nFinite difference check (first element of C):")
    print(f"  Analytical: {grad_analytical:.6f}")
    print(f"  Finite diff: {np.real(grad_fd):.6f}")
    print(f"  Relative error: {abs(grad_analytical - np.real(grad_fd))/abs(grad_analytical + 1e-10):.2%}")


if __name__ == "__main__":
    verify_gradient_computation()