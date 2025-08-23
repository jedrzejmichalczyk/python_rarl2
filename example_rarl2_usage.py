#!/usr/bin/env python3
"""
Example: Using Integrated RARL2 Algorithm
==========================================
Demonstrates the complete RARL2 pipeline:
1. Lossless parametrization via (C,A) pairs
2. H2 projection to compute optimal outer factor
3. Gradient-based optimization on lossless manifold
4. Conversion between lossless and lossy systems

This follows the approach from the AUTO_MOSfinal.pdf paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from rarl2_integrated import (
    RARL2Optimizer,
    douglas_shapiro_factorization,
    lossless_to_lossy,
    compute_h2_norm,
    create_output_normal_pair
)
from lossless_embedding import lossless_embedding, verify_lossless


def create_target_system(n: int = 5, p: int = 2, m: int = 2):
    """Create a stable target system to approximate."""
    np.random.seed(123)  # For reproducibility
    
    # Random stable system
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    eigvals = np.linalg.eigvals(A)
    A = A * 0.8 / np.max(np.abs(eigvals))  # Ensure stability
    
    B = np.random.randn(n, m) + 1j * np.random.randn(n, m)
    C = np.random.randn(p, n) + 1j * np.random.randn(p, n)
    D = np.random.randn(p, m) + 1j * np.random.randn(p, m) * 0.1
    
    return A, B, C, D


def main():
    """Demonstrate RARL2 algorithm."""
    
    print("=" * 60)
    print("RARL2: Rational Approximation via Lossless Parametrization")
    print("=" * 60)
    
    # Step 1: Create target system
    print("\n1. Creating target system (order 5)...")
    A_F, B_F, C_F, D_F = create_target_system(n=5, p=2, m=2)
    print(f"   Target dimensions: n={A_F.shape[0]}, m={B_F.shape[1]}, p={C_F.shape[0]}")
    
    # Step 2: Choose approximation order
    approx_order = 3
    print(f"\n2. Approximation order: {approx_order}")
    print("   (Lower order than target for model reduction)")
    
    # Step 3: Demonstrate lossless embedding
    print("\n3. Demonstrating lossless embedding...")
    C_init, A_init = create_output_normal_pair(approx_order, C_F.shape[0])
    print(f"   Created output normal pair (C,A)")
    
    # Verify output normal property
    M = A_init.conj().T @ A_init + C_init.conj().T @ C_init
    on_error = np.linalg.norm(M - np.eye(approx_order))
    print(f"   Output normal error: {on_error:.2e} (should be ~0)")
    
    # Create lossless system
    B_G, D_G = lossless_embedding(C_init, A_init)
    print(f"   Created lossless G from (C,A) pair")
    
    # Verify losslessness
    lossless_error = verify_lossless(A_init, B_G, C_init, D_G, n_points=20)
    print(f"   Max unitarity error on unit circle: {lossless_error:.2e}")
    
    # Step 4: Demonstrate Douglas-Shapiro factorization
    print("\n4. Douglas-Shapiro factorization...")
    print("   Note: This is actually inner-outer factorization")
    print("   Factorizing a test system H = C*G where G is lossless")
    
    # Create a test system
    A_test = A_init.copy()
    B_test = np.random.randn(approx_order, 2) + 1j * np.random.randn(approx_order, 2)
    C_test = C_init.copy()
    D_test = np.eye(2, dtype=np.complex128)
    
    C_factor, G_lossless = douglas_shapiro_factorization(
        A_test, B_test, C_test, D_test
    )
    print("   Factorization complete: H ≈ C_factor * G_lossless")
    
    # Step 5: H2 projection and conversion
    print("\n5. H2 projection: Converting lossless to lossy...")
    A_H, B_H, C_H, D_H = lossless_to_lossy(
        A_init, B_G, C_init, D_G,
        A_F, B_F, C_F, D_F
    )
    print(f"   Lossy approximation dimensions: n={A_H.shape[0]}")
    
    # Compute initial error
    initial_error = compute_h2_norm(A_F, B_F, C_F, D_F, A_H, B_H, C_H, D_H)
    print(f"   Initial H2 error: {np.sqrt(initial_error):.4f}")
    
    # Step 6: RARL2 Optimization
    print("\n6. RARL2 Optimization...")
    print("   Optimizing over manifold of lossless functions")
    
    optimizer = RARL2Optimizer(
        target=(A_F, B_F, C_F, D_F),
        order=approx_order
    )
    
    n_iterations = 50
    learning_rate = 0.005
    
    print(f"   Running {n_iterations} iterations...")
    print("   Iteration | H2 Error")
    print("   " + "-" * 20)
    
    for i in range(n_iterations):
        optimizer.step(learning_rate)
        
        if i % 10 == 0 or i == n_iterations - 1:
            error = optimizer.get_error()
            print(f"   {i:9d} | {np.sqrt(error):.6f}")
    
    # Get final approximation
    A_final, B_final, C_final, D_final = optimizer.get_approximation()
    final_error = compute_h2_norm(A_F, B_F, C_F, D_F, 
                                  A_final, B_final, C_final, D_final)
    
    # Step 7: Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Target system order:        {A_F.shape[0]}")
    print(f"Approximation order:        {approx_order}")
    print(f"Initial H2 error:          {np.sqrt(initial_error):.6f}")
    print(f"Final H2 error:            {np.sqrt(final_error):.6f}")
    print(f"Error reduction:           {(1 - final_error/initial_error)*100:.1f}%")
    
    # Plot convergence
    if len(optimizer.errors) > 1:
        print("\n7. Plotting convergence...")
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(np.sqrt(optimizer.errors), 'b-', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Iteration')
        plt.ylabel('H2 Error')
        plt.title('RARL2 Convergence: H2 Error vs Iteration')
        plt.savefig('rarl2_convergence.png', dpi=150, bbox_inches='tight')
        print("   Convergence plot saved as 'rarl2_convergence.png'")
    
    # Step 8: Key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("• RARL2 uses (C,A) parametrization of observable pairs")
    print("• Lossless embedding ensures stability throughout optimization")
    print("• H2 projection finds optimal outer factor C for given lossless G")
    print("• Concentrated criterion ψₙ(G) eliminates linear variable C")
    print("• Optimization on lossless manifold guarantees stable approximants")
    print("• The method can outperform balanced truncation for H2 norm")
    
    return optimizer


if __name__ == "__main__":
    optimizer = main()