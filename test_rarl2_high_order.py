#!/usr/bin/env python3
"""
High-Order RARL2 Tests
======================
Tests optimization for systems up to order 10.
"""
import numpy as np
import torch
import time


def create_random_stable_system(n: int, p: int, m: int, seed: int = None):
    """Create a random stable discrete-time system."""
    if seed is not None:
        np.random.seed(seed)

    # Create stable A with eigenvalues inside unit disk
    # Use diagonal + small coupling for controllable eigenvalue placement
    poles = 0.3 + 0.5 * np.random.rand(n)  # Poles between 0.3 and 0.8
    phases = 2 * np.pi * np.random.rand(n)
    eigvals = poles * np.exp(1j * phases)
    # Make some eigenvalues real
    for i in range(n // 3):
        eigvals[i] = eigvals[i].real

    A = np.diag(eigvals).astype(np.complex128)
    # Add coupling
    coupling = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) * 0.02
    A += coupling

    # Ensure stability
    max_eig = np.max(np.abs(np.linalg.eigvals(A)))
    if max_eig >= 1.0:
        A = A * 0.95 / max_eig

    B = (np.random.randn(n, m) + 1j * np.random.randn(n, m)) * 0.5
    C = (np.random.randn(p, n) + 1j * np.random.randn(p, n)) * 0.5
    D = np.zeros((p, m), dtype=np.complex128)

    return A, B, C, D


def test_reduction(n_F: int, n: int, p: int, m: int, seed: int = None):
    """Test reduction from order n_F to order n."""
    from torch_chart_optimizer import ChartRARL2Torch, h2_error_analytical_torch
    from balanced_truncation import get_rarl2_initialization

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create target
    A_F, B_F, C_F, D_F = create_random_stable_system(n_F, p, m, seed)

    # Get initialization from BT
    start_time = time.time()
    chart_center, bt_approx = get_rarl2_initialization(A_F, B_F, C_F, D_F, n)
    init_time = time.time() - start_time

    # Compute BT error
    A_bt, _, C_bt, _ = bt_approx
    A_bt_t = torch.tensor(A_bt, dtype=torch.complex128)
    C_bt_t = torch.tensor(C_bt, dtype=torch.complex128)
    A_F_t = torch.tensor(A_F, dtype=torch.complex128)
    B_F_t = torch.tensor(B_F, dtype=torch.complex128)
    C_F_t = torch.tensor(C_F, dtype=torch.complex128)
    D_F_t = torch.tensor(D_F, dtype=torch.complex128)

    bt_error = h2_error_analytical_torch(A_F_t, B_F_t, C_F_t, D_F_t, A_bt_t, C_bt_t).item()

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

    start_time = time.time()
    for i in range(30):
        loss = optimizer.step(closure)
        if loss.item() < 1e-12:
            break
    opt_time = time.time() - start_time

    final_error = loss.item()

    # Compute improvement
    if bt_error > 1e-12:
        improvement = (bt_error - final_error) / bt_error * 100
    else:
        improvement = 100.0 if final_error < 1e-12 else 0.0

    return {
        'bt_error': bt_error,
        'rarl2_error': final_error,
        'improvement': improvement,
        'init_time': init_time,
        'opt_time': opt_time,
        'total_time': init_time + opt_time,
        'success': final_error <= bt_error + 1e-10
    }


def main():
    print("=" * 70)
    print("RARL2 High-Order Test Suite")
    print("=" * 70)

    # Test configurations: (n_F, n, p, m)
    configs = [
        # Small systems
        (4, 2, 1, 1),
        (4, 2, 2, 2),
        (6, 3, 2, 2),
        # Medium systems
        (8, 4, 2, 2),
        (8, 4, 3, 2),
        (10, 5, 2, 2),
        (10, 5, 3, 3),
        # Larger systems
        (12, 6, 3, 3),
        (15, 7, 3, 3),
        (20, 10, 3, 3),
    ]

    results = []
    print(f"\n{'Config':<20} {'BT Error':>12} {'RARL2 Error':>14} {'Improv%':>10} {'Time(s)':>10} {'Status':>8}")
    print("-" * 70)

    for n_F, n, p, m in configs:
        config_str = f"{n_F}→{n} ({p}x{m})"
        try:
            result = test_reduction(n_F, n, p, m, seed=42)
            status = "PASS" if result['success'] else "FAIL"
            print(f"{config_str:<20} {result['bt_error']:>12.6f} {result['rarl2_error']:>14.2e} {result['improvement']:>10.1f} {result['total_time']:>10.2f} {status:>8}")
            results.append((config_str, result))
        except Exception as e:
            print(f"{config_str:<20} {'ERROR':>12} {str(e)[:30]}")
            results.append((config_str, {'success': False, 'error': str(e)}))

    # Summary
    passed = sum(1 for _, r in results if r.get('success', False))
    total = len(results)

    print("\n" + "=" * 70)
    print(f"Summary: {passed}/{total} tests passed")

    # Performance summary
    successful = [(c, r) for c, r in results if r.get('success', False) and 'total_time' in r]
    if successful:
        avg_time = sum(r['total_time'] for _, r in successful) / len(successful)
        max_time = max(r['total_time'] for _, r in successful)
        avg_improvement = sum(r['improvement'] for _, r in successful) / len(successful)
        print(f"Average time: {avg_time:.2f}s, Max time: {max_time:.2f}s")
        print(f"Average improvement over BT: {avg_improvement:.1f}%")

    print("=" * 70)
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
