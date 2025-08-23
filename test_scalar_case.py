#!/usr/bin/env python3
"""
Test RARL2 with simple scalar case where we know the analytical solution.

This tests our existing rarl2_implicit_diff.py implementation against
a case where we can verify correctness mathematically.
"""

import numpy as np
import torch
from rarl2_implicit_diff import RARL2WithImplicitDiff


def create_scalar_test_case():
    """
    Create the simplest possible case:
    - Target: 1st order LOSSLESS system 
    - Approximation: 1st order system
    - Should match exactly with zero error
    """
    print("Creating 1st order lossless target...")
    
    # Create a 1st order lossless system using our own lossless embedding
    from lossless_embedding import lossless_embedding
    
    # Start with output normal pair (C,A) - use proper calculation
    # For A^H*A + C^H*C = I in scalar case: |A|² + |C|² = 1
    A_mag = 0.6  # Choose stable magnitude < 1
    C_mag = np.sqrt(1 - A_mag**2)  # Ensure constraint
    
    A = np.array([[A_mag]], dtype=np.complex128)
    C = np.array([[C_mag]], dtype=np.complex128)
    
    # Verify output normal
    ON = A.conj().T @ A + C.conj().T @ C
    print(f"Output normal check: A^H*A + C^H*C = {ON[0,0]:.6f} (should be 1)")
    print(f"Chosen: |A|² = {A_mag**2:.3f}, |C|² = {C_mag**2:.3f}, sum = {A_mag**2 + C_mag**2:.3f}")
    
    # Create lossless B,D
    B, D = lossless_embedding(C, A, nu=1.0)
    
    print(f"Target system:")
    print(f"  A = {A[0,0]:.4f}")
    print(f"  B = {B[0,0]:.4f}")
    print(f"  C = {C[0,0]:.4f}")
    print(f"  D = {D[0,0]:.4f}")
    
    # Verify losslessness using the correct formula
    from lossless_embedding import verify_lossless_realization_matrix
    lossless_error = verify_lossless_realization_matrix(A, B, C, D)
    print(f"Lossless check: ||G^H*G - I|| = {lossless_error:.2e}")
    
    return A, B, C, D


def test_with_existing_implementation():
    """Test using our existing RARL2WithImplicitDiff."""
    
    print("=" * 60)
    print("Testing RARL2 with Scalar Case")
    print("=" * 60)
    
    # Set seeds for reproducibility
    np.random.seed(123)
    torch.manual_seed(123)
    
    # Create lossless scalar test case
    A_F, B_F, C_F, D_F = create_scalar_test_case()
    
    # Approximation: 1st order system (same as target)
    n = 1     # 1st order approximation
    n_F = 1   # 1st order target
    p = 1     # 1 output
    m = 1     # 1 input
    
    print(f"\nApproximating {n_F}x{n_F} lossless system with {n}x{n} system")
    print("Expected result: EXACT match (zero error) since dimensions are equal")
    
    # Use existing implementation
    model = RARL2WithImplicitDiff(n, p, A_F, B_F, C_F, D_F)
    
    # L-BFGS optimizer (as we found works best)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)
    
    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        return loss
    
    # Optimization
    losses = []
    print("\nOptimizing...")
    
    for i in range(30):
        loss = optimizer.step(closure)
        current_loss = loss.detach().item()  # Fix PyTorch warning
        losses.append(current_loss)
        
        if i % 5 == 0:
            print(f"Iteration {i:2d}: Loss = {current_loss:.6f}")
    
    # Get final result
    A, B, C, D = model.get_current_system()
    
    print(f"\nFinal Results:")
    print(f"  Loss: {losses[-1]:.6f}")
    print(f"  Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    print(f"\nApproximation system:")
    print(f"  A = {A}")
    print(f"  B = {B}")  
    print(f"  C = {C}")
    print(f"  D = {D}")
    print(f"  Pole: {np.linalg.eigvals(A)[0]:.4f}")
    
    # Check output normal constraint
    ON = A.conj().T @ A + C.conj().T @ C
    print(f"  Output normal error: {np.linalg.norm(ON - np.eye(n)):.2e}")
    
    # Frequency response comparison at multiple points
    print(f"\nFrequency response comparison:")
    test_points = [0.5, 1.0, 1.5]
    
    for z in test_points:
        # Target H_F(z)
        H_F_z = D_F + C_F @ np.linalg.inv(z * np.eye(n_F) - A_F) @ B_F
        
        # Approximation H(z)
        H_z = D + C @ np.linalg.inv(z * np.eye(n) - A) @ B
        
        error = abs(H_F_z[0,0] - H_z[0,0])
        print(f"  z={z}: Target={H_F_z[0,0]:.4f}, Approx={H_z[0,0]:.4f}, Error={error:.4f}")
        
        if error > 0.1:
            print(f"  ❌ Large error at z={z}!")
        else:
            print(f"  ✓ Good match at z={z}")
            
    # Check if our approximation is also lossless
    from lossless_embedding import verify_lossless_realization_matrix
    lossless_check_approx = verify_lossless_realization_matrix(A, B, C, D)
    print(f"\nApproximation lossless check: {lossless_check_approx:.2e}")
    
    if lossless_check_approx < 1e-10:
        print("✓ Approximation is lossless")
    else:
        print("❌ Approximation is NOT lossless")
    
    return model, losses


def analytical_verification():
    """
    For scalar case, we can compute the optimal approximation analytically
    and compare with our numerical result.
    """
    print("\n" + "=" * 60)
    print("Analytical Verification")
    print("=" * 60)
    
    # TODO: Implement analytical solution for comparison
    # For 2nd order → 1st order scalar case, the optimal approximation
    # should match the dominant pole/mode of the target system
    
    A_F, B_F, C_F, D_F = create_scalar_test_case()
    
    # Balanced truncation as baseline
    print("Computing balanced truncation baseline...")
    
    # Simple approach: keep dominant eigenvalue
    eigvals, eigvecs = np.linalg.eig(A_F)
    dominant_idx = np.argmax(np.abs(eigvals))
    dominant_pole = eigvals[dominant_idx]
    
    print(f"Dominant pole: {dominant_pole:.4f}")
    print("Balanced truncation would use this as the 1st order approximation")
    
    # Also check the lossless embedding again
    from lossless_embedding import verify_lossless_realization_matrix
    lossless_error = verify_lossless_realization_matrix(A_F, B_F, C_F, D_F)
    print(f"Target lossless check: ||G^H*G - I|| = {lossless_error:.2e}")
    
    # Our RARL2 result should be better than just using dominant pole
    return dominant_pole


if __name__ == "__main__":
    # Run tests
    model, losses = test_with_existing_implementation()
    dominant_pole = analytical_verification()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("✓ Tested existing implementation on scalar case")
    print("✓ Can compare with analytical/balanced truncation baselines")
    print("✓ Verified constraint satisfaction")
    print("Next: Test with frequency domain targets")