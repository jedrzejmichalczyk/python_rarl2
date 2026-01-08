#!/usr/bin/env python3
"""
Balanced Truncation for Model Reduction
========================================
Provides balanced truncation as initialization for RARL2.
"""

import numpy as np
from scipy import linalg
from typing import Tuple


def solve_discrete_lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Solve A X A^H - X = -Q for X."""
    return linalg.solve_discrete_lyapunov(A, Q)


def balanced_truncation(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                        order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Balanced truncation model reduction.

    Args:
        A, B, C, D: Full-order system
        order: Target reduced order

    Returns:
        A_r, B_r, C_r, D_r: Reduced-order system
    """
    n = A.shape[0]
    if order >= n:
        return A.copy(), B.copy(), C.copy(), D.copy()

    # Solve for controllability Gramian P: A P A^H + B B^H = P
    # In scipy form: A P A^H - P = -B B^H
    P = solve_discrete_lyapunov(A, B @ B.conj().T)

    # Solve for observability Gramian Q: A^H Q A + C^H C = Q
    # In scipy form: A^H Q A - Q = -C^H C
    Q = solve_discrete_lyapunov(A.conj().T, C.conj().T @ C)

    # Ensure Gramians are Hermitian
    P = (P + P.conj().T) / 2
    Q = (Q + Q.conj().T) / 2

    # Compute balancing transformation
    # Cholesky of P: P = L L^H (with robust regularization)
    min_eig_P = np.min(np.linalg.eigvalsh(P))
    if min_eig_P < 1e-10:
        eps = max(1e-10, abs(min_eig_P)) + 1e-10
        P = P + eps * np.eye(n)

    try:
        L = linalg.cholesky(P, lower=True)
    except linalg.LinAlgError:
        # Fallback: use SVD-based square root
        eigvals_P, eigvecs_P = np.linalg.eigh(P)
        eigvals_P = np.maximum(eigvals_P, 1e-12)
        L = eigvecs_P @ np.diag(np.sqrt(eigvals_P))

    # Form L^H Q L and compute its SVD
    M = L.conj().T @ Q @ L
    M = (M + M.conj().T) / 2  # Ensure Hermitian

    eigvals, eigvecs = linalg.eigh(M)
    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Hankel singular values
    hsv = np.sqrt(np.maximum(eigvals, 0))

    # Balancing transformation
    # T = L U Σ^{-1/2}
    U = eigvecs[:, :order]
    S_inv_sqrt = np.diag(1.0 / np.sqrt(hsv[:order] + 1e-15))
    T = L @ U @ S_inv_sqrt

    # T_inv = Σ^{-1/2} U^H L^{-1}
    L_inv = linalg.inv(L)
    T_inv = S_inv_sqrt @ U.conj().T @ L_inv

    # Transform system
    A_r = T_inv @ A @ T
    B_r = T_inv @ B
    C_r = C @ T
    D_r = D.copy()

    return A_r, B_r, C_r, D_r


def balanced_truncation_output_normal(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                                       order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Balanced truncation with output-normal form.

    Returns reduced system in output-normal form (A^H A + C^H C = I).
    """
    A_r, B_r, C_r, D_r = balanced_truncation(A, B, C, D, order)
    n = A_r.shape[0]

    # Transform to output-normal form
    # Solve Q: A^H Q A + C^H C = Q
    Q = solve_discrete_lyapunov(A_r.conj().T, C_r.conj().T @ C_r)
    Q = (Q + Q.conj().T) / 2  # Ensure Hermitian

    # Cholesky: Q = L^H L (since Q is Hermitian PD)
    min_eig_Q = np.min(np.linalg.eigvalsh(Q))
    if min_eig_Q < 1e-10:
        eps = max(1e-10, abs(min_eig_Q)) + 1e-10
        Q = Q + eps * np.eye(n)

    try:
        L = linalg.cholesky(Q)  # Upper triangular: Q = L^H L
    except linalg.LinAlgError:
        # Fallback: use eigendecomposition
        eigvals_Q, eigvecs_Q = np.linalg.eigh(Q)
        eigvals_Q = np.maximum(eigvals_Q, 1e-12)
        L = eigvecs_Q @ np.diag(np.sqrt(eigvals_Q)) @ eigvecs_Q.conj().T
        # Make it upper triangular via QR
        _, L = np.linalg.qr(L.conj().T)
        L = L.conj().T

    # Transform: A' = L A L^{-1}, C' = C L^{-1}
    L_inv = linalg.inv(L)
    A_on = L @ A_r @ L_inv
    C_on = C_r @ L_inv
    B_on = L @ B_r
    D_on = D_r

    # Verify output-normal
    ON = A_on.conj().T @ A_on + C_on.conj().T @ C_on
    err = np.linalg.norm(ON - np.eye(n))
    if err > 1e-8:
        # Fallback: QR orthonormalization
        stacked = np.vstack([A_on, C_on])
        Q_orth, R = np.linalg.qr(stacked)
        A_on = Q_orth[:n, :]
        C_on = Q_orth[n:, :]

    return A_on, B_on, C_on, D_on


def create_chart_center_from_system(A: np.ndarray, B: np.ndarray,
                                     C: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Create BOP chart center from a system realization.

    For lossless systems, the realization [A B; C D] is unitary.
    For non-lossless systems, we create a nearby lossless system.
    """
    from lossless_embedding import lossless_embedding, verify_lossless_realization_matrix

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    # Check if system is already lossless
    try:
        loss_err = verify_lossless_realization_matrix(A, B, C, D)
    except Exception:
        loss_err = 1.0

    if loss_err < 1e-10:
        # System is lossless - use directly
        W, X, Y, Z = A, B, C, D
    else:
        # Create lossless system from (A, C)
        # First ensure output-normal via QR
        stacked = np.vstack([A, C])
        Q, R = np.linalg.qr(stacked)

        # FIX PHASE AMBIGUITY: Ensure R has positive real diagonal
        # This makes QR unique and prevents sign flips
        signs = np.sign(np.real(np.diag(R)))
        signs[signs == 0] = 1  # Handle zeros
        Q = Q @ np.diag(signs)
        # R = np.diag(signs) @ R  # Not needed, we only use Q

        A_on = Q[:n, :]
        C_on = Q[n:n+p, :]

        # Create lossless (B, D)
        B_loss, D_loss = lossless_embedding(C_on, A_on, nu=1.0)

        W, X, Y, Z = A_on, B_loss, C_on, D_loss

    return W.astype(np.complex128), X.astype(np.complex128), \
           Y.astype(np.complex128), Z.astype(np.complex128)


def get_rarl2_initialization(A_F: np.ndarray, B_F: np.ndarray,
                              C_F: np.ndarray, D_F: np.ndarray,
                              order: int) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """
    Get initialization for RARL2 from balanced truncation.

    Args:
        A_F, B_F, C_F, D_F: Target system
        order: Desired approximation order

    Returns:
        chart_center: (W, X, Y, Z) for BOP chart
        initial_approx: (A, B, C, D) balanced truncation result
    """
    # Get balanced truncation
    A_bt, B_bt, C_bt, D_bt = balanced_truncation_output_normal(A_F, B_F, C_F, D_F, order)

    # Create chart center from balanced truncation
    W, X, Y, Z = create_chart_center_from_system(A_bt, B_bt, C_bt, D_bt)

    return (W, X, Y, Z), (A_bt, B_bt, C_bt, D_bt)
