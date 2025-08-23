#!/usr/bin/env python3
"""
MKL Integration for RARL2
==========================
Integrates the MKL shared library for optimized numerical routines.
"""

import numpy as np
import sys
import os
import ctypes
from typing import Tuple, Optional

# Try to import MKL bindings
try:
    # Add path to MKL bindings
    mkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            '..', 'mkl_shared_library', 'bindings', 'python')
    sys.path.insert(0, mkl_path)
    
    # Also add the build directory to LD_LIBRARY_PATH
    build_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'mkl_shared_library', 'build')
    
    # Update library path
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{build_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = build_path
    
    # Load the library directly from build path
    lib_path = os.path.join(build_path, 'libmafin_mkl.so')
    if os.path.exists(lib_path):
        # Load directly
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    
    from mafin_mkl import (
        solve_stein_discrete,
        observability_gramian,
        h2_norm_discrete,
        DiscreteTimeSystem,
        SteinOptions
    )
    
    MKL_AVAILABLE = True
    print("MKL library loaded successfully")
    
except Exception as e:
    print(f"Warning: Could not load MKL library: {e}")
    print("Falling back to scipy implementations")
    MKL_AVAILABLE = False
    
    # Fallback implementations using scipy
    from scipy import linalg
    
    def solve_stein_discrete(A: np.ndarray, Q: np.ndarray, 
                           options: Optional[dict] = None) -> Tuple[np.ndarray, float]:
        """Fallback Stein solver using scipy."""
        X = linalg.solve_discrete_lyapunov(A, Q)
        
        # Compute residual
        residual = np.linalg.norm(A @ X @ A.conj().T - X + Q, 'fro')
        return X, residual
    
    def observability_gramian(A: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fallback observability Gramian computation."""
        Q = linalg.solve_discrete_lyapunov(A.conj().T, C.conj().T @ C)
        
        # Compute residual
        residual = np.linalg.norm(A.conj().T @ Q @ A - Q + C.conj().T @ C, 'fro')
        return Q, residual
    
    def h2_norm_discrete(A: np.ndarray, B: np.ndarray, 
                        C: np.ndarray, D: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Fallback H2 norm computation."""
        # Compute observability Gramian
        Q = linalg.solve_discrete_lyapunov(A.conj().T, C.conj().T @ C)
        
        # H2 norm squared = trace(B^H * Q * B) + trace(D^H * D)
        h2_norm_sq = np.real(np.trace(B.conj().T @ Q @ B))
        if D is not None:
            h2_norm_sq += np.real(np.trace(D.conj().T @ D))
        
        h2_norm = np.sqrt(h2_norm_sq)
        return h2_norm, h2_norm_sq
    
    # Dummy class for compatibility
    class DiscreteTimeSystem:
        def __init__(self, A, B, C, D=None):
            self.A = A
            self.B = B
            self.C = C
            self.D = D if D is not None else np.zeros((C.shape[0], B.shape[1]))
    
    class SteinOptions:
        @classmethod
        def default(cls):
            return {}


def solve_stein_with_mkl(A: np.ndarray, Q: np.ndarray, 
                         use_mkl: bool = True) -> Tuple[np.ndarray, float]:
    """
    Solve discrete-time Stein equation: A*X*A^H - X = -Q
    
    Args:
        A: System matrix (must be stable)
        Q: Right-hand side (typically Hermitian)
        use_mkl: Whether to use MKL if available
        
    Returns:
        X: Solution matrix
        residual: Norm of residual
    """
    if use_mkl and MKL_AVAILABLE:
        # Use MKL implementation
        options = SteinOptions.default()
        X, residual = solve_stein_discrete(A, Q, options)
    else:
        # Use fallback
        X, residual = solve_stein_discrete(A, Q)
    
    return X, residual


def compute_observability_gramian(A: np.ndarray, C: np.ndarray,
                                 use_mkl: bool = True) -> Tuple[np.ndarray, float]:
    """
    Compute observability Gramian: A^H*Q*A - Q = -C^H*C
    
    Args:
        A: System matrix
        C: Output matrix
        use_mkl: Whether to use MKL if available
        
    Returns:
        Q: Observability Gramian
        residual: Norm of residual
    """
    if use_mkl and MKL_AVAILABLE:
        Q, residual = observability_gramian(A, C)
    else:
        # Fallback
        Q, residual = observability_gramian(A, C)
    
    return Q, residual


def compute_h2_norm(A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                   D: Optional[np.ndarray] = None,
                   use_mkl: bool = True) -> Tuple[float, float]:
    """
    Compute discrete-time H2 norm.
    
    Args:
        A, B, C, D: State-space matrices
        use_mkl: Whether to use MKL if available
        
    Returns:
        h2_norm: H2 norm
        h2_norm_squared: H2 norm squared
    """
    if use_mkl and MKL_AVAILABLE:
        h2_norm, h2_norm_sq = h2_norm_discrete(A, B, C, D)
    else:
        h2_norm, h2_norm_sq = h2_norm_discrete(A, B, C, D)
    
    return h2_norm, h2_norm_sq


# Benchmark functions
def benchmark_stein_solver():
    """Benchmark MKL vs scipy Stein solver."""
    import time
    
    sizes = [10, 20, 50, 100]
    
    print("\nBenchmarking Stein solver:")
    print("-" * 50)
    print(f"{'Size':<10} {'Scipy (ms)':<15} {'MKL (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for n in sizes:
        # Create random stable matrix
        np.random.seed(42)
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        # Make more strongly stable for better convergence
        eigvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigvals))
        A = A * 0.8 / max_eig  # Scale to have max eigenvalue 0.8
        
        Q = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        Q = Q @ Q.conj().T  # Make Hermitian positive definite
        
        # Scipy timing
        start = time.time()
        X_scipy, _ = solve_stein_with_mkl(A, Q, use_mkl=False)
        scipy_time = (time.time() - start) * 1000
        
        # MKL timing (if available)
        if MKL_AVAILABLE:
            start = time.time()
            X_mkl, _ = solve_stein_with_mkl(A, Q, use_mkl=True)
            mkl_time = (time.time() - start) * 1000
            
            speedup = scipy_time / mkl_time
            error = np.linalg.norm(X_scipy - X_mkl) / np.linalg.norm(X_scipy)
            
            print(f"{n:<10} {scipy_time:<15.2f} {mkl_time:<15.2f} {speedup:<10.2f}")
            
            if error > 1e-10:
                print(f"  Warning: Results differ by {error:.2e}")
        else:
            print(f"{n:<10} {scipy_time:<15.2f} {'N/A':<15} {'N/A':<10}")


if __name__ == '__main__':
    print(f"MKL Available: {MKL_AVAILABLE}")
    
    # Test Stein solver
    n = 5
    np.random.seed(42)
    A = 0.5 * np.eye(n) + 0.1 * np.random.randn(n, n)
    Q = np.eye(n)
    
    X, residual = solve_stein_with_mkl(A, Q)
    print(f"\nSolve Stein equation test:")
    print(f"  Matrix size: {n}x{n}")
    print(f"  Residual: {residual:.2e}")
    
    # Test observability Gramian
    C = np.random.randn(2, n)
    Q_obs, residual = compute_observability_gramian(A, C)
    print(f"\nObservability Gramian test:")
    print(f"  Residual: {residual:.2e}")
    
    # Run benchmark
    benchmark_stein_solver()