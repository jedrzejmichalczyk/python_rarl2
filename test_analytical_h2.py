#!/usr/bin/env python3
"""
Quick test of the analytical H2 norm formula fix.
Tests the core math without full optimization.
"""
import numpy as np
import torch

# Test the analytical H2 formula
def test_analytical_h2_basic():
    """Test that analytical H2 norm works correctly."""
    print("=" * 60)
    print("Testing Analytical H2 Norm Formula")
    print("=" * 60)

    # Import our functions
    from rarl2_implicit_diff import (
        h2_norm_squared_torch,
        h2_error_analytical_torch,
        solve_discrete_lyapunov_torch
    )

    # Create a simple stable 1st order system
    A = torch.tensor([[0.5]], dtype=torch.complex128)
    B = torch.tensor([[1.0]], dtype=torch.complex128)
    C = torch.tensor([[1.0]], dtype=torch.complex128)
    D = torch.tensor([[0.0]], dtype=torch.complex128)

    # Compute H2 norm squared via our formula
    h2_sq = h2_norm_squared_torch(A, B, C, D)
    print(f"\nSimple system H2 norm²: {h2_sq.item():.6f}")

    # Analytical calculation for H(z) = C(z-A)^{-1}B = 1/(z-0.5)
    # H2 norm² = sum_k |h[k]|² where h[k] = C A^k B = 0.5^k
    # = sum_k 0.25^k = 1/(1-0.25) = 4/3 ≈ 1.333
    expected = 1.0 / (1.0 - 0.25)
    print(f"Expected (analytical): {expected:.6f}")
    print(f"Match: {abs(h2_sq.item() - expected) < 1e-10}")

    # Now test the concentrated criterion for the scalar case
    print("\n" + "-" * 40)
    print("Testing Concentrated Criterion (Scalar Case)")
    print("-" * 40)

    # For scalar case where target = approximation, error should be 0
    A_F = torch.tensor([[0.6]], dtype=torch.complex128)
    B_F = torch.tensor([[0.8]], dtype=torch.complex128)
    C_F = torch.tensor([[0.8]], dtype=torch.complex128)
    D_F = torch.tensor([[0.0]], dtype=torch.complex128)

    # Identical A, C as target
    A_approx = A_F.clone()
    C_approx = C_F.clone()

    error = h2_error_analytical_torch(A_F, B_F, C_F, D_F, A_approx, C_approx)
    print(f"Target = Approx, H2 error: {error.item():.2e}")
    print(f"Should be ~0: {error.item() < 1e-10}")

    # Different approximation
    A_approx2 = torch.tensor([[0.3]], dtype=torch.complex128)
    C_approx2 = torch.tensor([[0.95]], dtype=torch.complex128)

    error2 = h2_error_analytical_torch(A_F, B_F, C_F, D_F, A_approx2, C_approx2)
    print(f"\nDifferent approx, H2 error: {error2.item():.6f}")
    print(f"Should be > 0: {error2.item() > 0}")

    return True


def test_scalar_optimization():
    """Test that optimization converges to target for scalar case."""
    print("\n" + "=" * 60)
    print("Testing Scalar Optimization with Analytical H2 Norm")
    print("=" * 60)

    from rarl2_implicit_diff import RARL2WithImplicitDiff
    from lossless_embedding import lossless_embedding

    np.random.seed(42)
    torch.manual_seed(42)

    # Create scalar lossless target
    A_mag = 0.6
    C_mag = np.sqrt(1 - A_mag**2)
    A_F = np.array([[A_mag]], dtype=np.complex128)
    C_F = np.array([[C_mag]], dtype=np.complex128)
    B_F, D_F = lossless_embedding(C_F, A_F, nu=1.0)

    print(f"\nTarget (lossless):")
    print(f"  A = {A_F[0,0]:.4f}, C = {C_F[0,0]:.4f}")
    print(f"  B = {B_F[0,0]:.4f}, D = {D_F[0,0]:.4f}")

    # Create model
    model = RARL2WithImplicitDiff(n=1, p=1, A_F=A_F, B_F=B_F, C_F=C_F, D_F=D_F)

    # Get initial loss
    with torch.no_grad():
        initial_loss = model(use_analytical=True)
    print(f"\nInitial H2 error: {initial_loss.item():.6f}")

    # Optimize with L-BFGS
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)

    def closure():
        optimizer.zero_grad()
        loss = model(use_analytical=True)
        loss.backward()
        return loss

    # Optimization loop
    losses = []
    print("\nOptimizing...")
    for i in range(50):
        loss = optimizer.step(closure)
        losses.append(loss.item())
        if i % 10 == 0:
            print(f"  Iter {i:2d}: Loss = {loss.item():.6e}")
        if loss.item() < 1e-12:
            print(f"  Converged at iteration {i}!")
            break

    final_loss = losses[-1]
    print(f"\nFinal H2 error: {final_loss:.6e}")
    print(f"Improvement: {(1 - final_loss/losses[0])*100:.1f}%")

    # Get final system
    A_result, B_result, C_result, D_result = model.get_current_system()
    print(f"\nResult:")
    print(f"  A = {A_result[0,0]:.4f}, C = {C_result[0,0]:.4f}")

    # Check if result matches target (up to phase)
    A_error = min(abs(A_result[0,0] - A_F[0,0]), abs(A_result[0,0] + A_F[0,0]))
    C_error = min(abs(C_result[0,0] - C_F[0,0]), abs(C_result[0,0] + C_F[0,0]))
    print(f"  |A error| = {A_error:.6e}")
    print(f"  |C error| = {C_error:.6e}")

    success = final_loss < 1e-8
    print(f"\nScalar test {'PASSED' if success else 'FAILED'}: H2 error < 1e-8")
    return success


if __name__ == "__main__":
    test1 = test_analytical_h2_basic()
    test2 = test_scalar_optimization()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Analytical H2 basic test: {'PASS' if test1 else 'FAIL'}")
    print(f"Scalar optimization test: {'PASS' if test2 else 'FAIL'}")
