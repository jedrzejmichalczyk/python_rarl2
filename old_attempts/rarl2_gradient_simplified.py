#!/usr/bin/env python3
"""
Simplified Gradient for Output Normal Pairs
============================================
When (C,A) is in output normal form, the gradient simplifies significantly.
Based on Section 6 of AUTO_MOSfinal.pdf.
"""

import numpy as np
from typing import Tuple


def compute_gradient_output_normal(
    C: np.ndarray, A: np.ndarray,
    A_F: np.ndarray, B_F: np.ndarray, 
    C_F: np.ndarray, D_F: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient for output normal pairs using simplified formulas.
    
    For output normal pairs where A^H*A + C^H*C = I:
    - The observability Gramian Q̂ = I
    - The first necessary condition: B̂ = -Q₁₂* B_F
    - The gradient simplifies significantly
    
    Args:
        C, A: Observable pair in output normal form
        A_F, B_F, C_F, D_F: Target system
        
    Returns:
        grad_C, grad_A: Gradients with respect to C and A
    """
    n = A.shape[0]
    n_F = A_F.shape[0]
    p = C.shape[0]
    m = B_F.shape[1]
    
    # Verify output normal
    ON_check = A.conj().T @ A + C.conj().T @ C
    ON_error = np.linalg.norm(ON_check - np.eye(n))
    if ON_error > 1e-6:
        raise ValueError(f"Not output normal: error = {ON_error:.2e}")
    
    # For output normal pairs, the gradient computation simplifies:
    # We need to compute the interaction terms Q₁₂ and P₁₂
    
    # Method 1: Direct computation from necessary conditions
    # Since we're doing model reduction (n < n_F), we can't directly
    # match the target. The gradient tells us how to improve.
    
    # Simplified approach for testing:
    # The gradient should point in the direction that reduces ||F - H||²
    
    # Compute current approximation error
    from lossless_embedding import lossless_embedding
    B, D = lossless_embedding(C, A, nu=1.0)
    
    # For gradient, we need the sensitivity of the error
    # with respect to changes in (C,A)
    
    # Use finite differences for now (to validate the concept)
    eps = 1e-7
    grad_C = np.zeros_like(C, dtype=np.complex128)
    grad_A = np.zeros_like(A, dtype=np.complex128)
    
    # Base objective
    from rarl2_integrated import compute_objective_lossless
    obj_base = compute_objective_lossless(C, A, A_F, B_F, C_F, D_F)
    
    # Gradient w.r.t C
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C_pert = C.copy()
            C_pert[i,j] += eps
            
            # Re-normalize to maintain output normal
            M = A.conj().T @ A + C_pert.conj().T @ C_pert
            scale = np.linalg.inv(np.linalg.cholesky(M))
            C_pert_norm = C_pert @ scale
            A_pert_norm = A @ scale
            
            obj_pert = compute_objective_lossless(
                C_pert_norm, A_pert_norm, A_F, B_F, C_F, D_F
            )
            grad_C[i,j] = (obj_pert - obj_base) / eps
    
    # Gradient w.r.t A
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_pert = A.copy()
            A_pert[i,j] += eps
            
            # Re-normalize to maintain output normal
            M = A_pert.conj().T @ A_pert + C.conj().T @ C
            scale = np.linalg.inv(np.linalg.cholesky(M))
            C_pert_norm = C @ scale
            A_pert_norm = A_pert @ scale
            
            obj_pert = compute_objective_lossless(
                C_pert_norm, A_pert_norm, A_F, B_F, C_F, D_F
            )
            grad_A[i,j] = (obj_pert - obj_base) / eps
    
    # Make gradients real (since objective is real)
    grad_C = np.real(grad_C)
    grad_A = np.real(grad_A)
    
    # Project to satisfy constraint
    # The constraint is d(A^H*A + C^H*C) = 0
    # which gives: A^H*dA + dA^H*A + C^H*dC + dC^H*C = 0
    
    Lambda = (A.conj().T @ grad_A + grad_A.T @ A + 
              C.conj().T @ grad_C + grad_C.T @ C) / 2
    
    grad_A_proj = grad_A - A @ Lambda
    grad_C_proj = grad_C - C @ Lambda
    
    return grad_C_proj, grad_A_proj


def test_simplified_gradient():
    """Test the simplified gradient computation."""
    np.random.seed(42)
    
    # Create test case
    n = 2
    n_F = 3
    p = 2
    m = 2
    
    # Output normal pair
    from rarl2_integrated import create_output_normal_pair
    C, A = create_output_normal_pair(n, p)
    
    # Target system
    A_F = np.random.randn(n_F, n_F) + 1j * np.random.randn(n_F, n_F)
    A_F = A_F * 0.8 / np.max(np.abs(np.linalg.eigvals(A_F)))
    B_F = np.random.randn(n_F, m) + 1j * np.random.randn(n_F, m)
    C_F = np.random.randn(p, n_F) + 1j * np.random.randn(p, n_F)
    D_F = np.zeros((p, m), dtype=np.complex128)
    
    print("Testing simplified gradient for output normal pairs...")
    
    # Compute gradient
    grad_C, grad_A = compute_gradient_output_normal(
        C, A, A_F, B_F, C_F, D_F
    )
    
    print(f"grad_C norm: {np.linalg.norm(grad_C):.4f}")
    print(f"grad_A norm: {np.linalg.norm(grad_A):.4f}")
    
    # Check constraint preservation
    dON = (A.conj().T @ grad_A + grad_A.T @ A + 
           C.conj().T @ grad_C + grad_C.T @ C)
    print(f"Constraint violation: {np.linalg.norm(dON):.2e}")
    
    # Test gradient descent step
    from rarl2_integrated import compute_objective_lossless
    
    obj_before = compute_objective_lossless(C, A, A_F, B_F, C_F, D_F)
    
    # Take a small step
    alpha = 0.001
    C_new = C - alpha * grad_C
    A_new = A - alpha * grad_A
    
    # Re-normalize
    M = A_new.conj().T @ A_new + C_new.conj().T @ C_new
    scale = np.linalg.inv(np.linalg.cholesky(M))
    C_new = C_new @ scale
    A_new = A_new @ scale
    
    obj_after = compute_objective_lossless(C_new, A_new, A_F, B_F, C_F, D_F)
    
    print(f"\nGradient descent test:")
    print(f"  Objective before: {obj_before:.6f}")
    print(f"  Objective after:  {obj_after:.6f}")
    print(f"  Improvement:      {obj_before - obj_after:.6f}")
    
    if obj_after < obj_before:
        print("✓ Gradient descent reduces objective!")
    else:
        print("✗ Gradient descent increased objective (may need smaller step)")


if __name__ == "__main__":
    test_simplified_gradient()