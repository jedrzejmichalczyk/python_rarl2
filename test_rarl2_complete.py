#!/usr/bin/env python3
"""
Complete RARL2 Test Suite
=========================
Tests the RARL2 implementation with proper initialization and analytical H2 norm.
"""
import numpy as np
import torch

def test_scalar_lossless():
    """Test: Scalar lossless target, same-order approximation -> zero error."""
    print("=" * 60)
    print("Test 1: Scalar Lossless (n=1 → n=1)")
    print("=" * 60)

    from torch_chart_optimizer import ChartRARL2Torch
    from lossless_embedding import lossless_embedding
    from balanced_truncation import create_chart_center_from_system

    np.random.seed(42)
    torch.manual_seed(42)

    # Create scalar lossless target
    A_F = np.array([[0.6]], dtype=np.complex128)
    C_F = np.array([[0.8]], dtype=np.complex128)
    B_F, D_F = lossless_embedding(C_F, A_F, nu=1.0)

    print(f"\nTarget: A={A_F[0,0]:.4f}, C={C_F[0,0]:.4f}")

    # Initialize chart center from target (since same order)
    W, X, Y, Z = create_chart_center_from_system(A_F, B_F, C_F, D_F)

    n, m = 1, 1
    model = ChartRARL2Torch(n, m, (W, X, Y, Z), (A_F, B_F, C_F, D_F))

    # Initialize V=0 (optimal for matched chart)
    with torch.no_grad():
        model.V_real.data.fill_(0.0)
        model.V_imag.data.fill_(0.0)

    with torch.no_grad():
        initial_loss = model(use_analytical=True)
    print(f"Initial H2 error: {initial_loss.item():.2e}")

    success = initial_loss.item() < 1e-10
    print(f"Result: {'PASS' if success else 'FAIL'} (error < 1e-10)")
    return success


def test_scalar_optimization():
    """Test: Optimization recovers target from perturbation."""
    print("\n" + "=" * 60)
    print("Test 2: Scalar Optimization Recovery")
    print("=" * 60)

    from torch_chart_optimizer import ChartRARL2Torch
    from lossless_embedding import lossless_embedding
    from balanced_truncation import create_chart_center_from_system

    np.random.seed(42)
    torch.manual_seed(42)

    # Create scalar lossless target
    A_F = np.array([[0.6]], dtype=np.complex128)
    C_F = np.array([[0.8]], dtype=np.complex128)
    B_F, D_F = lossless_embedding(C_F, A_F, nu=1.0)

    # Initialize chart center from target
    W, X, Y, Z = create_chart_center_from_system(A_F, B_F, C_F, D_F)

    n, m = 1, 1
    model = ChartRARL2Torch(n, m, (W, X, Y, Z), (A_F, B_F, C_F, D_F))

    # Start with perturbation
    with torch.no_grad():
        model.V_real.data.fill_(0.2)
        model.V_imag.data.fill_(0.1)

    with torch.no_grad():
        initial_loss = model(use_analytical=True)
    print(f"\nInitial H2 error (perturbed): {initial_loss.item():.6f}")

    # Optimize with Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for i in range(200):
        optimizer.zero_grad()
        loss = model(use_analytical=True)
        loss.backward()
        optimizer.step()
        if loss.item() < 1e-12:
            print(f"Converged at iteration {i}")
            break

    final_loss = loss.item()
    print(f"Final H2 error: {final_loss:.2e}")

    success = final_loss < 1e-8
    print(f"Result: {'PASS' if success else 'FAIL'} (error < 1e-8)")
    return success


def test_model_reduction():
    """Test: Reduce order 6 lossy system to order 3."""
    print("\n" + "=" * 60)
    print("Test 3: Model Reduction (n=6 → n=3)")
    print("=" * 60)

    from torch_chart_optimizer import ChartRARL2Torch
    from balanced_truncation import get_rarl2_initialization

    np.random.seed(123)  # Different seed for variety
    torch.manual_seed(123)

    # Create stable lossy target with more structure
    n_F, n = 6, 3
    p, m = 2, 2

    # Create a more interesting target with distinct poles
    A_F = np.diag([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]).astype(np.complex128)
    # Add some off-diagonal coupling
    A_F += (np.random.randn(n_F, n_F) + 1j * np.random.randn(n_F, n_F)) * 0.05
    # Ensure stability
    max_eig = np.max(np.abs(np.linalg.eigvals(A_F)))
    if max_eig >= 1.0:
        A_F = A_F * 0.95 / max_eig
    B_F = (np.random.randn(n_F, m) + 1j * np.random.randn(n_F, m)) * 0.5
    C_F = (np.random.randn(p, n_F) + 1j * np.random.randn(p, n_F)) * 0.5
    D_F = np.zeros((p, m), dtype=np.complex128)

    print(f"\nTarget: {n_F}x{n_F} system → {n}x{n} approximation")

    # Get initialization from balanced truncation
    chart_center, bt_approx = get_rarl2_initialization(A_F, B_F, C_F, D_F, n)

    # Compute baseline BT error
    from torch_chart_optimizer import h2_error_analytical_torch
    A_bt, B_bt, C_bt, D_bt = bt_approx
    A_bt_t = torch.tensor(A_bt, dtype=torch.complex128)
    C_bt_t = torch.tensor(C_bt, dtype=torch.complex128)
    A_F_t = torch.tensor(A_F, dtype=torch.complex128)
    B_F_t = torch.tensor(B_F, dtype=torch.complex128)
    C_F_t = torch.tensor(C_F, dtype=torch.complex128)
    D_F_t = torch.tensor(D_F, dtype=torch.complex128)

    bt_error = h2_error_analytical_torch(A_F_t, B_F_t, C_F_t, D_F_t, A_bt_t, C_bt_t)
    print(f"Balanced truncation H2 error: {bt_error.item():.6f}")

    # Create RARL2 model
    model = ChartRARL2Torch(n, m, chart_center, (A_F, B_F, C_F, D_F))

    # Initialize V=0 (starts at BT solution)
    with torch.no_grad():
        model.V_real.data.fill_(0.0)
        model.V_imag.data.fill_(0.0)

    with torch.no_grad():
        initial_loss = model(use_analytical=True)
    print(f"Initial RARL2 H2 error: {initial_loss.item():.6f}")

    # Optimize
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = model(use_analytical=True)
        loss.backward()
        return loss

    for i in range(30):
        loss = optimizer.step(closure)
        if i % 10 == 0:
            print(f"  Iter {i}: Loss = {loss.item():.6e}")

    final_loss = loss.item()
    bt_err = bt_error.item()
    if bt_err > 1e-12:
        improvement = (bt_err - final_loss) / bt_err * 100
    else:
        improvement = 100.0 if final_loss < 1e-12 else 0.0

    print(f"\nFinal RARL2 H2 error: {final_loss:.6e}")
    print(f"Improvement over BT: {improvement:.1f}%")

    success = final_loss <= bt_err + 1e-10  # Should at least match BT
    print(f"Result: {'PASS' if success else 'FAIL'} (RARL2 ≤ BT)")
    return success


def test_mimo_higher_order():
    """Test: MIMO higher-order reduction."""
    print("\n" + "=" * 60)
    print("Test 4: MIMO Reduction (n=8 → n=3)")
    print("=" * 60)

    from torch_chart_optimizer import ChartRARL2Torch
    from balanced_truncation import get_rarl2_initialization

    np.random.seed(42)
    torch.manual_seed(42)

    # Create stable MIMO target
    n_F, n = 8, 3
    p, m = 3, 2

    A_F = np.random.randn(n_F, n_F) + 1j * np.random.randn(n_F, n_F)
    A_F = A_F * 0.6 / np.max(np.abs(np.linalg.eigvals(A_F)))
    B_F = (np.random.randn(n_F, m) + 1j * np.random.randn(n_F, m)) * 0.3
    C_F = (np.random.randn(p, n_F) + 1j * np.random.randn(p, n_F)) * 0.3
    D_F = np.zeros((p, m), dtype=np.complex128)

    print(f"\nTarget: {n_F}x{n_F} ({p}x{m}) system → {n}x{n} approximation")

    # Get initialization from balanced truncation
    chart_center, bt_approx = get_rarl2_initialization(A_F, B_F, C_F, D_F, n)

    # Compute baseline BT error
    from torch_chart_optimizer import h2_error_analytical_torch
    A_bt, _, C_bt, _ = bt_approx
    A_bt_t = torch.tensor(A_bt, dtype=torch.complex128)
    C_bt_t = torch.tensor(C_bt, dtype=torch.complex128)
    A_F_t = torch.tensor(A_F, dtype=torch.complex128)
    B_F_t = torch.tensor(B_F, dtype=torch.complex128)
    C_F_t = torch.tensor(C_F, dtype=torch.complex128)
    D_F_t = torch.tensor(D_F, dtype=torch.complex128)

    bt_error = h2_error_analytical_torch(A_F_t, B_F_t, C_F_t, D_F_t, A_bt_t, C_bt_t)
    print(f"Balanced truncation H2 error: {bt_error.item():.6f}")

    # Create RARL2 model
    model = ChartRARL2Torch(n, m, chart_center, (A_F, B_F, C_F, D_F))

    with torch.no_grad():
        model.V_real.data.fill_(0.0)
        model.V_imag.data.fill_(0.0)

    # Optimize
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.3, max_iter=30)

    def closure():
        optimizer.zero_grad()
        loss = model(use_analytical=True)
        loss.backward()
        return loss

    for i in range(50):
        loss = optimizer.step(closure)
        if i % 10 == 0:
            print(f"  Iter {i}: Loss = {loss.item():.6e}")

    final_loss = loss.item()
    bt_err = bt_error.item()
    if bt_err > 1e-12:
        improvement = (bt_err - final_loss) / bt_err * 100
    else:
        improvement = 100.0 if final_loss < 1e-12 else 0.0

    print(f"\nFinal RARL2 H2 error: {final_loss:.6e}")
    print(f"Improvement over BT: {improvement:.1f}%")

    success = final_loss <= bt_err + 1e-10
    print(f"Result: {'PASS' if success else 'FAIL'} (RARL2 ≤ BT)")
    return success


if __name__ == "__main__":
    results = []
    results.append(("Scalar lossless", test_scalar_lossless()))
    results.append(("Scalar optimization", test_scalar_optimization()))
    results.append(("Model reduction 4→2", test_model_reduction()))
    results.append(("MIMO reduction 8→3", test_mimo_higher_order()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed

    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
