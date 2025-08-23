#!/usr/bin/env python3
"""
Lossless Embedding Implementation for RARL2
============================================
Implementation of Proposition 1 from AUTO_MOSfinal.pdf.
Creates a lossless matrix from an observable pair (C, A).
"""

import numpy as np
from scipy import linalg
from typing import Tuple

# Try to use MKL optimized routines
try:
    from mkl_integration import solve_stein_with_mkl, MKL_AVAILABLE
    USE_MKL = MKL_AVAILABLE
except ImportError:
    USE_MKL = False


def solve_discrete_lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Solve discrete-time Lyapunov equation: A^H * X * A - X = -Q
    
    Args:
        A: System matrix (must be stable)
        Q: Right-hand side matrix (typically C^H * C)
        
    Returns:
        X: Solution to the Lyapunov equation
    """
    if USE_MKL:
        # Use MKL optimized solver
        # Note: MKL solve_stein solves A*X*A^H - X = -Q
        # We need A^H*X*A - X = -Q, so we transpose
        X, _ = solve_stein_with_mkl(A.conj().T, Q)
        return X
    else:
        return linalg.solve_discrete_lyapunov(A.conj().T, Q)


def lossless_embedding(C: np.ndarray, A: np.ndarray, 
                       nu: complex = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute lossless embedding for an observable pair (C, A).
    
    According to Proposition 1 in the paper:
    Given an observable pair (C, A) with A asymptotically stable,
    the rational matrix G(z) = D + C(zI - A)^(-1)B is lossless.
    
    Args:
        C: Output matrix (p x n)
        A: System matrix (n x n), must be stable
        nu: Point where G(nu) = I, |nu| = 1
        
    Returns:
        B: Input matrix for lossless system
        D: Feedthrough matrix for lossless system
        
    Raises:
        ValueError: If A is not stable or (C, A) is not observable
    """
    n = A.shape[0]
    p = C.shape[0]
    
    # Check stability
    eigvals = np.linalg.eigvals(A)
    if np.any(np.abs(eigvals) >= 1.0):
        raise ValueError(f"A is not stable, max |eigenvalue| = {np.max(np.abs(eigvals))}")
    
    # Check observability via rank condition
    obs_matrix = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])
    rank = np.linalg.matrix_rank(obs_matrix, tol=1e-10)
    if rank < n:
        raise ValueError(f"(C, A) is not observable, rank = {rank} < {n}")
    
    # Check |nu| = 1
    if np.abs(np.abs(nu) - 1.0) > 1e-10:
        raise ValueError(f"|nu| = {np.abs(nu)} != 1")
    
    # Compute observability Gramian Q
    # Q satisfies: A^H * Q * A - Q = -C^H * C
    Q = solve_discrete_lyapunov(A, C.conj().T @ C)
    
    # Ensure Q is positive definite
    eigvals_Q = np.linalg.eigvalsh(Q)
    if np.min(eigvals_Q) < 1e-12:
        raise ValueError(f"Observability Gramian not positive definite, min eigenvalue = {np.min(eigvals_Q)}")
    
    # Compute B and D according to Proposition 1
    I_n = np.eye(n, dtype=np.complex128)
    I_p = np.eye(p, dtype=np.complex128)
    
    # Compute auxiliary matrices
    Q_inv = np.linalg.inv(Q)
    I_minus_nuAH_inv = np.linalg.inv(I_n - nu * A.conj().T)
    
    # B = -(A - νI)Q^(-1)(I - νA^H)^(-1)C^H
    B = -(A - nu * I_n) @ Q_inv @ I_minus_nuAH_inv @ C.conj().T
    
    # D = I - CQ^(-1)(I - νA^H)^(-1)C^H
    D = I_p - C @ Q_inv @ I_minus_nuAH_inv @ C.conj().T
    
    return B, D


def verify_lossless(A: np.ndarray, B: np.ndarray, 
                    C: np.ndarray, D: np.ndarray,
                    n_points: int = 100) -> float:
    """
    Verify that G(z) = D + C(zI - A)^(-1)B is lossless on unit circle.
    
    A matrix is lossless if G(z)*G(z)^H = I for all |z| = 1.
    
    Args:
        A, B, C, D: State-space matrices
        n_points: Number of points to test on unit circle
        
    Returns:
        max_error: Maximum deviation from unitarity on unit circle
    """
    n = A.shape[0]
    max_error = 0.0
    
    theta_values = np.linspace(0, 2*np.pi, n_points)
    for theta in theta_values:
        z = np.exp(1j * theta)
        
        # Check if z is not an eigenvalue of A
        zI_minus_A = z * np.eye(n) - A
        det_val = np.linalg.det(zI_minus_A)
        if np.abs(det_val) > 1e-12:
            # Compute G(z)
            G_z = D + C @ np.linalg.inv(zI_minus_A) @ B
            
            # Check unitarity: G(z) * G(z)^H = I
            GGH = G_z @ G_z.conj().T
            error = np.linalg.norm(GGH - np.eye(GGH.shape[0]))
            max_error = max(max_error, error)
    
    return max_error


def verify_lossless_realization_matrix(A: np.ndarray, B: np.ndarray, 
                                       C: np.ndarray, D: np.ndarray) -> float:
    """
    Verify losslessness using the realization matrix condition.
    
    For a lossless system, the realization matrix G = [[A, B], [C, D]]
    should satisfy: G^H * G = I (unitary condition).
    
    Args:
        A, B, C, D: State-space matrices
        
    Returns:
        error: Deviation from unitarity ||G^H * G - I||
    """
    G = np.block([[A, B], [C, D]])
    GHG = G.conj().T @ G
    I = np.eye(G.shape[0])
    error = np.linalg.norm(GHG - I)
    return error


def create_output_normal_pair(n: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an output normal pair (C, A) where A^H * A + C^H * C = I.
    
    Args:
        n: State dimension
        p: Output dimension
        
    Returns:
        C: Output matrix (p x n)
        A: Stable system matrix (n x n)
    """
    # Create random unitary matrix
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(n+p, n) + 1j * np.random.randn(n+p, n)
    U, _ = np.linalg.qr(X)
    
    # Extract A and C
    A = U[:n, :]
    C = U[n:n+p, :]
    
    # Scale A to ensure stability
    # We need all eigenvalues of A to be inside unit disk
    eigvals = np.linalg.eigvals(A)
    max_eigval = np.max(np.abs(eigvals))
    if max_eigval >= 1.0:
        A = A * 0.95 / max_eigval
    
    # Now we need to rescale to maintain output normal form
    # Compute current norm
    M = A.conj().T @ A + C.conj().T @ C
    
    # Find scaling to make M = I
    # We use iterative approach
    for _ in range(10):
        # Compute scaling factor
        eigvals_M = np.linalg.eigvalsh(M)
        scale = np.sqrt(1.0 / np.mean(eigvals_M))
        
        A = A * scale
        C = C * scale
        
        M = A.conj().T @ A + C.conj().T @ C
        error = np.linalg.norm(M - np.eye(n))
        if error < 1e-10:
            break
    
    return C, A