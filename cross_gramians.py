#!/usr/bin/env python3
"""
Cross-Gramian utilities for RARL2
=================================
Implements the cross-Gramian and dual-Gramian equations used in the
necessary conditions for optimal H2 approximation.

Key equations (discrete-time):
- Q12 solves: A_F^H · Q12 · A + C_F^H · C = Q12
- B_hat = -Q12^H · B_F
- P12 solves: A · P12 · A_F^H + B_hat · B_F^H = P12

These are Stein/Sylvester-like equations with different left/right dynamics.
We solve them by vectorizing with Kronecker products:

vec(Q12) = (A^T ⊗ A_F^H) vec(Q12) + vec(C_F^H C)
=> (I - A^T ⊗ A_F^H) vec(Q12) = vec(C_F^H C)

Similarly for P12.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def _solve_stein_two_sided(A_left: np.ndarray, A_right: np.ndarray, RHS: np.ndarray) -> np.ndarray:
    """Solve X = A_left · X · A_right + RHS for X via vectorization.

    Args:
        A_left: Left multiplier (L)
        A_right: Right multiplier (R)
        RHS: Right-hand side (same shape as X)
    Returns:
        X satisfying X - A_left X A_right = RHS
    """
    nL = A_left.shape[0]
    nR = A_right.shape[0]
    I = np.eye(nL * nR, dtype=complex)

    # Build system: (I - A_right^T ⊗ A_left) vec(X) = vec(RHS)
    # Note: vec(A_left X A_right) = (A_right^T ⊗ A_left) vec(X)
    K = I - np.kron(A_right.T, A_left)
    b = RHS.reshape(-1, order="F")

    # Solve; use least-squares for robustness
    vecX, *_ = np.linalg.lstsq(K, b, rcond=None)
    X = vecX.reshape(RHS.shape, order="F")
    return X


def compute_cross_gramian(A: np.ndarray, A_F: np.ndarray,
                          C: np.ndarray, C_F: np.ndarray) -> np.ndarray:
    """Compute Q12 solving A_F^H Q12 A + C_F^H C = Q12.

    Shapes:
        A: (n, n), A_F: (nF, nF), C: (p, n), C_F: (p, nF)
    Returns:
        Q12: (nF, n)
    """
    L = A_F.conj().T
    R = A
    RHS = C_F.conj().T @ C
    return _solve_stein_two_sided(L, R, RHS)


def compute_optimal_B(Q12: np.ndarray, B_F: np.ndarray) -> np.ndarray:
    """Compute optimal B_hat via first necessary condition: B_hat = -Q12^H B_F."""
    return -Q12.conj().T @ B_F


def compute_dual_gramian(A: np.ndarray, A_F: np.ndarray,
                         B_hat: np.ndarray, B_F: np.ndarray) -> np.ndarray:
    """Compute P12 solving A P12 A_F^H + B_hat B_F^H = P12.

    Returns:
        P12 with shape (n, nF)
    """
    L = A
    R = A_F.conj().T
    RHS = B_hat @ B_F.conj().T
    return _solve_stein_two_sided(L, R, RHS)


def compute_optimal_B_and_gramians(A: np.ndarray, A_F: np.ndarray,
                                   B_F: np.ndarray, C: np.ndarray,
                                   C_F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper to compute Q12, B_hat, and P12.

    Returns:
        (Q12, B_hat, P12)
    """
    Q12 = compute_cross_gramian(A, A_F, C, C_F)
    B_hat = compute_optimal_B(Q12, B_F)
    P12 = compute_dual_gramian(A, A_F, B_hat, B_F)
    return Q12, B_hat, P12

