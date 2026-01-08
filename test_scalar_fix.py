#!/usr/bin/env python3
"""
Fix for scalar case: Use chart center matched to target.

The key insight: The BOP chart has a limited domain. If the target is
far from the chart center, the optimization might not reach it.

Solution: Initialize the chart center from the target itself (or close to it).
"""
import numpy as np
import torch

def create_chart_center_from_target(A_target: np.ndarray, B_target: np.ndarray,
                                     C_target: np.ndarray, D_target: np.ndarray):
    """Create chart center from a target lossless realization.

    For a lossless system, [A B; C D] is unitary (up to scaling).
    We use this directly as the chart center.
    """
    n = A_target.shape[0]
    m = B_target.shape[1]
    p = C_target.shape[0]

    # Form the unitary realization matrix
    Omega = np.block([[A_target, B_target],
                      [C_target, D_target]])

    # Extract W, X, Y, Z
    W = Omega[:n, :n]        # A
    X = Omega[:n, n:n+m]     # B
    Y = Omega[n:n+p, :n]     # C
    Z = Omega[n:n+p, n:n+m]  # D

    return W, X, Y, Z


def test_scalar_with_matched_chart():
    """Test scalar case with chart center matched to target."""
    print("=" * 60)
    print("Testing Scalar Case with Matched Chart Center")
    print("=" * 60)

    from torch_chart_optimizer import ChartRARL2Torch
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

    # Create chart center FROM the target
    W, X, Y, Z = create_chart_center_from_target(A_F, B_F, C_F, D_F)
    print(f"\nChart center (from target):")
    print(f"  W = {W[0,0]:.4f}, Y = {Y[0,0]:.4f}")

    n, m = 1, 1
    chart_center = (W, X, Y, Z)
    target = (A_F, B_F, C_F, D_F)

    # Create model
    model = ChartRARL2Torch(n, m, chart_center, target)

    # Initialize V near 0 (since chart center = target, V=0 should be optimal)
    with torch.no_grad():
        model.V_real.data.fill_(0.0)
        model.V_imag.data.fill_(0.0)

    # Get initial loss - should be very close to 0!
    with torch.no_grad():
        initial_loss = model(use_analytical=True)
    print(f"\nInitial H2 error (with V=0): {initial_loss.item():.6e}")

    # This should already be very small because V=0 maps to chart center = target
    if initial_loss.item() < 1e-10:
        print("\n[SUCCESS] V=0 gives exact target!")

        # Verify the reconstruction
        V = torch.complex(model.V_real, model.V_imag)
        A_result, B_result, C_result, D_result = model.bop(V)
        print(f"\nReconstructed from V=0:")
        print(f"  A = {A_result[0,0].item():.4f}")
        print(f"  C = {C_result[0,0].item():.4f}")
        return True

    # If not zero, optimize
    print("\nOptimizing...")
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = model(use_analytical=True)
        loss.backward()
        return loss

    for i in range(50):
        loss = optimizer.step(closure)
        if i % 10 == 0:
            print(f"  Iter {i:2d}: Loss = {loss.item():.6e}")
        if loss.item() < 1e-12:
            break

    final_loss = loss.item()
    print(f"\nFinal H2 error: {final_loss:.6e}")

    success = final_loss < 1e-8
    print(f"\nScalar test {'PASSED' if success else 'FAILED'}")
    return success


def test_scalar_small_perturbation():
    """Test that optimization can recover from small perturbations."""
    print("\n" + "=" * 60)
    print("Testing Recovery from Small Perturbation")
    print("=" * 60)

    from torch_chart_optimizer import ChartRARL2Torch
    from lossless_embedding import lossless_embedding

    np.random.seed(42)
    torch.manual_seed(42)

    # Create scalar lossless target
    A_F = np.array([[0.6]], dtype=np.complex128)
    C_F = np.array([[0.8]], dtype=np.complex128)
    B_F, D_F = lossless_embedding(C_F, A_F, nu=1.0)

    # Create chart center from target
    W, X, Y, Z = create_chart_center_from_target(A_F, B_F, C_F, D_F)

    n, m = 1, 1
    chart_center = (W, X, Y, Z)
    target = (A_F, B_F, C_F, D_F)

    # Create model with small initial perturbation
    model = ChartRARL2Torch(n, m, chart_center, target)
    with torch.no_grad():
        model.V_real.data.fill_(0.1)  # Small perturbation
        model.V_imag.data.fill_(0.05)

    with torch.no_grad():
        initial_loss = model(use_analytical=True)
    print(f"\nInitial H2 error (small perturbation): {initial_loss.item():.6f}")

    # Optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    print("\nOptimizing with Adam...")
    for i in range(200):
        optimizer.zero_grad()
        loss = model(use_analytical=True)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"  Iter {i:3d}: Loss = {loss.item():.6e}")
        if loss.item() < 1e-12:
            print(f"  Converged at iteration {i}!")
            break

    final_loss = loss.item()
    print(f"\nFinal H2 error: {final_loss:.6e}")

    # Check V is back near 0
    with torch.no_grad():
        V_norm = torch.norm(torch.complex(model.V_real, model.V_imag)).item()
    print(f"|V| = {V_norm:.6f} (should be near 0)")

    success = final_loss < 1e-8
    print(f"\nPerturbation test {'PASSED' if success else 'FAILED'}")
    return success


if __name__ == "__main__":
    test1 = test_scalar_with_matched_chart()
    test2 = test_scalar_small_perturbation()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Matched chart center test: {'PASS' if test1 else 'FAIL'}")
    print(f"Perturbation recovery test: {'PASS' if test2 else 'FAIL'}")
