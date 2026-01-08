#!/usr/bin/env python3
"""
Test the chart-based optimizer with analytical H2 norm.
This uses the BOP parametrization which follows the paper more closely.
"""
import numpy as np
import torch

def test_chart_optimizer_scalar():
    """Test chart-based optimizer on scalar lossless case."""
    print("=" * 60)
    print("Testing Chart-Based Optimizer with Analytical H2 Norm")
    print("=" * 60)

    from torch_chart_optimizer import ChartRARL2Torch
    from bop import create_output_normal_chart_center
    from lossless_embedding import lossless_embedding

    np.random.seed(42)
    torch.manual_seed(42)

    # Create scalar lossless target (n=1, p=1, m=1)
    A_mag = 0.6
    C_mag = np.sqrt(1 - A_mag**2)
    A_F = np.array([[A_mag]], dtype=np.complex128)
    C_F = np.array([[C_mag]], dtype=np.complex128)
    B_F, D_F = lossless_embedding(C_F, A_F, nu=1.0)

    print(f"\nTarget (lossless):")
    print(f"  A = {A_F[0,0]:.4f}, C = {C_F[0,0]:.4f}")
    print(f"  B = {B_F[0,0]:.4f}, D = {D_F[0,0]:.4f}")

    # Create chart center
    n, m = 1, 1  # scalar case
    W, X, Y, Z = create_output_normal_chart_center(n, m)
    chart_center = (W, X, Y, Z)
    target = (A_F, B_F, C_F, D_F)

    print(f"\nChart center:")
    print(f"  W = {W[0,0]:.4f}, Y = {Y[0,0]:.4f}")

    # Create model
    model = ChartRARL2Torch(n, m, chart_center, target)

    # Get initial loss
    with torch.no_grad():
        initial_loss = model(use_analytical=True)
        initial_loss_sampled = model(use_analytical=False)
    print(f"\nInitial H2 error (analytical): {initial_loss.item():.6f}")
    print(f"Initial H2 error (sampled):    {initial_loss_sampled.item():.6f}")

    # Optimize with L-BFGS
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = model(use_analytical=True)
        loss.backward()
        return loss

    # Optimization loop
    losses = []
    print("\nOptimizing...")
    for i in range(100):
        loss = optimizer.step(closure)
        losses.append(loss.item())
        if i % 10 == 0 or i < 5:
            print(f"  Iter {i:2d}: Loss = {loss.item():.6e}")
        if loss.item() < 1e-12:
            print(f"  Converged at iteration {i}!")
            break

    final_loss = losses[-1]
    print(f"\nFinal H2 error (analytical): {final_loss:.6e}")
    print(f"Improvement: {(1 - final_loss/losses[0])*100:.1f}%")

    # Get final system via BOP
    with torch.no_grad():
        V = torch.complex(model.V_real, model.V_imag)
        A_result, B_result, C_result, D_result = model.bop(V)
        A_np = A_result.cpu().numpy()
        C_np = C_result.cpu().numpy()

    print(f"\nResult:")
    print(f"  A = {A_np[0,0]:.4f}, C = {C_np[0,0]:.4f}")

    # Check if result matches target (up to phase)
    A_error = min(abs(A_np[0,0] - A_F[0,0]), abs(A_np[0,0] + A_F[0,0]))
    C_error = min(abs(C_np[0,0] - C_F[0,0]), abs(C_np[0,0] + C_F[0,0]))
    print(f"  |A error| = {A_error:.6e}")
    print(f"  |C error| = {C_error:.6e}")

    success = final_loss < 1e-8
    print(f"\nScalar test {'PASSED' if success else 'FAILED'}: H2 error < 1e-8")
    return success


def test_chart_optimizer_mimo():
    """Test chart-based optimizer on MIMO case."""
    print("\n" + "=" * 60)
    print("Testing Chart-Based Optimizer on MIMO Case")
    print("=" * 60)

    from torch_chart_optimizer import ChartRARL2Torch
    from bop import create_output_normal_chart_center

    np.random.seed(42)
    torch.manual_seed(42)

    # Create stable MIMO target (not necessarily lossless)
    n_F, n, p, m = 4, 2, 2, 2  # Reduce from order 4 to order 2

    A_F = np.random.randn(n_F, n_F) + 1j * np.random.randn(n_F, n_F)
    A_F = A_F * 0.5 / np.max(np.abs(np.linalg.eigvals(A_F)))  # Make stable
    B_F = (np.random.randn(n_F, m) + 1j * np.random.randn(n_F, m)) * 0.5
    C_F = (np.random.randn(p, n_F) + 1j * np.random.randn(p, n_F)) * 0.5
    D_F = np.zeros((p, m), dtype=np.complex128)

    print(f"\nTarget: {n_F}x{n_F} system")
    print(f"Approximation: {n}x{n} system")

    # Create chart center for approximation order
    W, X, Y, Z = create_output_normal_chart_center(n, m)
    chart_center = (W, X, Y, Z)
    target = (A_F, B_F, C_F, D_F)

    # Create model
    model = ChartRARL2Torch(n, m, chart_center, target)

    # Get initial loss
    with torch.no_grad():
        initial_loss = model(use_analytical=True)
    print(f"\nInitial H2 error: {initial_loss.item():.6f}")

    # Optimize
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = model(use_analytical=True)
        loss.backward()
        return loss

    losses = []
    print("\nOptimizing...")
    for i in range(50):
        loss = optimizer.step(closure)
        losses.append(loss.item())
        if i % 10 == 0:
            print(f"  Iter {i:2d}: Loss = {loss.item():.6e}")

    final_loss = losses[-1]
    improvement = (1 - final_loss / losses[0]) * 100
    print(f"\nFinal H2 error: {final_loss:.6e}")
    print(f"Improvement: {improvement:.1f}%")

    success = improvement > 10  # Should get at least 10% improvement
    print(f"\nMIMO test {'PASSED' if success else 'FAILED'}: >10% improvement")
    return success


if __name__ == "__main__":
    test1 = test_chart_optimizer_scalar()
    test2 = test_chart_optimizer_mimo()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Scalar lossless test: {'PASS' if test1 else 'FAIL'}")
    print(f"MIMO reduction test:  {'PASS' if test2 else 'FAIL'}")
