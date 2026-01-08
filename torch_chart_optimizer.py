#!/usr/bin/env python3
"""
Torch-autograd RARL2 with chart parametrization (no chart switching)
===================================================================
- Parameters V (real) -> complex V
- Chart deparametrization (BOP) in Torch to get (A, C)
- Optimal B_hat from cross-Gramian with target (Torch, differentiable)
- D_hat = 0 (simplification)
- Objective: H2 error via frequency sampling (Torch)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Kronecker product for torch tensors."""
    a11 = A.unsqueeze(-1).unsqueeze(-3)  # (..., m, 1, n, 1)
    b11 = B.unsqueeze(-2).unsqueeze(-4)  # (..., 1, p, 1, q)
    K = a11 * b11
    # reshape to (m*p, n*q)
    m, n = A.shape[-2], A.shape[-1]
    p, q = B.shape[-2], B.shape[-1]
    return K.reshape(m * p, n * q)


def solve_two_sided_stein(A_left: torch.Tensor, A_right: torch.Tensor, RHS: torch.Tensor) -> torch.Tensor:
    """Solve X = A_left X A_right + RHS via vectorization in Torch.
    Uses: vec(X) = (I - A_right^T ⊗ A_left)^{-1} vec(RHS)
    """
    nL = A_left.shape[0]
    nR = A_right.shape[0]
    I = torch.eye(nL * nR, dtype=A_left.dtype, device=A_left.device)
    K = I - kron(A_right.T, A_left)
    # Fortran-order vectorization: vec_F(M) stacks columns
    b = RHS.permute(1, 0).contiguous().view(-1)
    try:
        vecX = torch.linalg.solve(K, b)
    except RuntimeError:
        vecX = torch.linalg.lstsq(K, b).solution
    # reshape back with Fortran order
    X = vecX.view(nR, nL).permute(1, 0).contiguous()
    return X


def solve_discrete_lyapunov_torch(A: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Solve A^H X A - X = -Q in Torch via vectorization.
    vec(X) satisfies: (A^T ⊗ A.conj() - I) vec(X) = -vec(Q)
    """
    n = A.shape[0]
    I = torch.eye(n * n, dtype=A.dtype, device=A.device)
    M = kron(A.T, A.conj()) - I
    b = -Q.permute(1, 0).contiguous().view(-1)
    try:
        vecX = torch.linalg.solve(M, b)
    except RuntimeError:
        vecX = torch.linalg.lstsq(M, b).solution
    X = vecX.view(n, n).permute(1, 0).contiguous()
    return X


class TorchBOP(nn.Module):
    """Torch version of BOP forward map to (A, C) only (we compute B_hat via target)."""

    def __init__(self, W: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor):
        super().__init__()
        # Register chart center as buffers (constants)
        self.register_buffer("W", W)
        self.register_buffer("X", X)
        self.register_buffer("Y", Y)
        self.register_buffer("Z", Z)
        self.n = W.shape[0]
        self.m = X.shape[1]

    def _matrix_sqrt_invsqrt_ns(self, P: torch.Tensor, iters: int = 25) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (sqrt(P), invsqrt(P)) with Newton–Schulz iteration.
        Assumes P is Hermitian PSD (stabilized beforehand)."""
        # Normalize by Frobenius norm to keep spectral radius reasonable
        normF = torch.linalg.norm(P)
        c = (normF if normF > 0 else torch.tensor(1.0, dtype=P.dtype, device=P.device))
        Y = P / c
        I = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
        Z = I.clone()
        for _ in range(iters):
            T = 0.5 * (3 * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z
        # Undo scaling: sqrt(P) ≈ sqrt(c) * Y, invsqrt(P) ≈ (1/sqrt(c)) * Z
        sqrt_c = torch.sqrt(c.real).to(P.dtype)
        S = sqrt_c * Y
        Sinv = Z / sqrt_c
        return S, Sinv

    def forward(self, V: torch.Tensor, debug=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given V (p x n, complex), produce lossless (A, B, C, D).

        Args:
            V: Parameter matrix with shape (p, n) where p=Y.shape[0] (output dim)

        Steps:
          - Solve P - W^H P W = - (V^H V - Y^H Y)
          - Stabilize P via Hermitization and Gershgorin-based diagonal shift
          - Compute Λ and Λ^{-1} via Newton–Schulz (differentiable)
          - A = Λ W Λ^{-1}, C = (Y + V) Λ^{-1}
          - Complete to lossless via (B,D) = lossless_embedding(C,A)
        """
        from lossless_embedding_torch import lossless_embedding_torch, verify_lossless_torch

        # Stein for P
        RHS = V.conj().T @ V - self.Y.conj().T @ self.Y
        P = solve_two_sided_stein(self.W.conj().T, self.W, -RHS)
        # Hermitize
        P = (P + P.conj().T) * 0.5
        # Gershgorin-based PD shift: make P strictly diagonally dominant
        absP = torch.abs(P)
        row_sum_off = absP.sum(dim=1) - torch.abs(torch.diagonal(P))
        diag_real = torch.real(torch.diagonal(P))
        lower_bd = torch.min(diag_real - row_sum_off)
        shift_pd = torch.clamp(1e-6 - lower_bd, min=0.0)  # Keep as real
        if shift_pd.item() > 0:
            P = P + shift_pd * torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
        # Newton–Schulz sqrt and invsqrt (more iterations for precision)
        Lambda, Lambda_inv = self._matrix_sqrt_invsqrt_ns(P, iters=50)

        if debug:
            # Verify Lambda^H * Lambda ≈ P
            check = Lambda.conj().T @ Lambda
            print(f"  [DEBUG] P verification: ||Lambda^H*Lambda - P|| = {torch.norm(check - P).item():.6e}")

        # A and C from chart
        A = Lambda @ self.W @ Lambda_inv
        C = (self.Y + V) @ Lambda_inv

        # CRITICAL: Explicitly enforce output normal via QR
        # Stack [A; C] and orthonormalize
        stacked = torch.cat([A, C], dim=0)  # (n+p) x n
        Q, R = torch.linalg.qr(stacked)

        # FIX PHASE AMBIGUITY: Ensure R has positive diagonal
        # This makes QR decomposition unique and prevents sign flips
        signs = torch.sign(torch.real(torch.diagonal(R)))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)  # Handle zeros
        Q = Q @ torch.diag(signs.to(Q.dtype))
        R = torch.diag(signs.to(R.dtype)) @ R

        A = Q[:self.n, :]
        C = Q[self.n:, :]

        if debug:
            # Check output normal AFTER orthonormalization
            ON = A.conj().T @ A + C.conj().T @ C
            I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
            print(f"  [DEBUG] Output normal (A,C): ||A^H*A + C^H*C - I|| = {torch.norm(ON - I).item():.6e}")

        # Complete to lossless system
        B, D = lossless_embedding_torch(C, A, nu=1.0)

        if debug:
            # Verify losslessness
            loss_err = verify_lossless_torch(A, B, C, D)
            print(f"  [DEBUG] Lossless check: ||G^H*G - I||_max = {loss_err.item():.6e}")

        return A, B, C, D


def compute_optimal_B_torch(A: torch.Tensor, C: torch.Tensor,
                            A_F: torch.Tensor, B_F: torch.Tensor, C_F: torch.Tensor) -> torch.Tensor:
    """Compute B_hat = -Q12^H B_F with Q12 solving A_F^H Q12 A + C_F^H C = Q12."""
    L = A_F.conj().T
    R = A
    RHS = C_F.conj().T @ C
    Q12 = solve_two_sided_stein(L, R, RHS)
    B_hat = -(Q12.conj().T @ B_F)
    return B_hat


def frequency_response_torch(A, B, C, D, z):
    n = A.shape[0]
    if n == 0:
        return D
    zI = z * torch.eye(n, dtype=A.dtype, device=A.device)
    try:
        inv = torch.linalg.inv(zI - A)
        return D + C @ inv @ B
    except RuntimeError:
        return D


def h2_error_torch(sys1: Tuple[torch.Tensor, ...], sys2: Tuple[torch.Tensor, ...], num_samples: int = 128) -> torch.Tensor:
    """Frequency-sampled H2 error (LEGACY - use h2_error_analytical instead)."""
    A1, B1, C1, D1 = sys1
    A2, B2, C2, D2 = sys2
    device = A1.device
    omega = torch.linspace(0, 2 * np.pi, num_samples, device=device)
    acc = torch.zeros((), dtype=torch.float64, device=device)
    for w in omega:
        z = torch.exp(1j * w)
        H1 = frequency_response_torch(A1, B1, C1, D1, z)
        H2 = frequency_response_torch(A2, B2, C2, D2, z)
        E = H1 - H2
        acc = acc + torch.sum(torch.abs(E) ** 2).real
    return acc / num_samples


def h2_norm_squared_torch(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Compute ||H||²₂ for discrete-time system H(z) = D + C(zI-A)⁻¹B.

    Uses the observability Gramian Q satisfying A^H Q A - Q = -C^H C.
    Then ||H||²₂ = Tr(B^H Q B) + Tr(D^H D).
    """
    n = A.shape[0]
    if n == 0:
        # Static gain only
        return torch.sum(torch.abs(D) ** 2).real

    # Solve discrete Lyapunov: A^H Q A - Q = -C^H C
    Q = solve_discrete_lyapunov_torch(A, C.conj().T @ C)

    # H2 norm squared = Tr(B^H Q B) + Tr(D^H D)
    h2_sq = torch.trace(B.conj().T @ Q @ B).real
    if D is not None and D.numel() > 0:
        h2_sq = h2_sq + torch.sum(torch.abs(D) ** 2).real

    return h2_sq


def h2_error_analytical_torch(
    A_F: torch.Tensor, B_F: torch.Tensor, C_F: torch.Tensor, D_F: torch.Tensor,
    A: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    """Compute the EXACT H2 error using the concentrated criterion from paper eq. (9).

    J_n(C, A) = ||F||²₂ - Tr(B_F^H · Q₁₂ · Q₁₂^H · B_F)

    This gives the minimum H2 error achievable with dynamics (A, C) and optimal B̂.
    No frequency sampling - exact analytical formula.

    NOTE: This is for LOSSY approximation. For lossless-to-lossless, use
    h2_error_lossless_torch() instead.

    Args:
        A_F, B_F, C_F, D_F: Target system F
        A, C: Approximation dynamics (output normal)

    Returns:
        J_n(C, A): The optimal H2 error for this (A, C) pair
    """
    # Step 1: Compute ||F||²₂
    F_norm_sq = h2_norm_squared_torch(A_F, B_F, C_F, D_F)

    # Step 2: Compute cross-Gramian Q₁₂ solving A_F^H Q₁₂ A + C_F^H C = Q₁₂
    L = A_F.conj().T
    R = A
    RHS = C_F.conj().T @ C
    Q12 = solve_two_sided_stein(L, R, RHS)

    # Step 3: Concentrated criterion from eq. (9)
    # J_n = ||F||² - Tr(B_F^H Q₁₂ Q₁₂^H B_F)
    reduction = torch.trace(B_F.conj().T @ Q12 @ Q12.conj().T @ B_F).real

    J_n = F_norm_sq - reduction

    # Ensure non-negative (numerical errors can make it slightly negative)
    return torch.clamp(J_n, min=0.0)


def h2_error_lossless_torch(
    A_F: torch.Tensor, B_F: torch.Tensor, C_F: torch.Tensor, D_F: torch.Tensor,
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor
) -> torch.Tensor:
    """Compute EXACT H2 error ||F - G||² between two systems analytically.

    Uses: ||F - G||² = ||F||² - 2·Re·Tr(cross terms) + ||G||²

    For discrete-time systems, the cross term involves solving a Sylvester equation.

    Args:
        A_F, B_F, C_F, D_F: Target system F
        A, B, C, D: Approximation system G

    Returns:
        ||F - G||²₂
    """
    # ||F||²
    F_norm_sq = h2_norm_squared_torch(A_F, B_F, C_F, D_F)

    # ||G||²
    G_norm_sq = h2_norm_squared_torch(A, B, C, D)

    # Cross term: 2·Re·Tr(inner product)
    # <F, G> = Tr(C_F Σ C^H) + Tr(D_F^H D) where Σ solves A_F Σ A^H + B_F B^H = Σ
    # Actually for inner product: <F, G> = Tr(C_F X C^H) + Tr(D_F^H D)
    # where X solves the Sylvester equation A_F X A^H + B_F B^H = X

    # Solve X = A_F X A^H + B_F B^H
    # This is: X - A_F X A^H = B_F B^H
    n_F = A_F.shape[0]
    n = A.shape[0]

    if n_F == 0 and n == 0:
        # Both static gains
        cross = torch.trace(D_F.conj().T @ D).real
        return F_norm_sq - 2 * cross + G_norm_sq

    # Solve Sylvester equation for cross-Gramian
    # X - A_F X A^H = B_F B^H
    # Vectorized: (I - A^* ⊗ A_F) vec(X) = vec(B_F B^H)
    I = torch.eye(n_F * n, dtype=A_F.dtype, device=A_F.device)
    K = I - kron(A.conj(), A_F)
    RHS = B_F @ B.conj().T
    b = RHS.permute(1, 0).contiguous().view(-1)

    try:
        vecX = torch.linalg.solve(K, b)
    except RuntimeError:
        vecX = torch.linalg.lstsq(K, b).solution

    X = vecX.view(n, n_F).permute(1, 0).contiguous()

    # Cross term: Tr(C_F X C^H) + Tr(D_F^H D)
    cross = torch.trace(C_F @ X @ C.conj().T).real
    if D_F is not None and D is not None and D_F.numel() > 0 and D.numel() > 0:
        cross = cross + torch.trace(D_F.conj().T @ D).real

    # ||F - G||² = ||F||² - 2·Re<F,G> + ||G||²
    error_sq = F_norm_sq - 2 * cross + G_norm_sq

    return torch.clamp(error_sq, min=0.0)


class ChartRARL2Torch(nn.Module):
    """End-to-end Torch model: V -> (A,C) -> (B_hat, D_hat) -> H2 error with target."""

    def __init__(self, n: int, m: int, chart_center: Tuple[np.ndarray, ...], target: Tuple[np.ndarray, ...]):
        super().__init__()
        self.n = n
        self.m = m
        # Chart center to torch complex
        W, X, Y, Z = chart_center
        self.W = torch.tensor(W, dtype=torch.complex128)
        self.X = torch.tensor(X, dtype=torch.complex128)
        self.Y = torch.tensor(Y, dtype=torch.complex128)
        self.Z = torch.tensor(Z, dtype=torch.complex128)

        # Infer output dimension p from Y shape
        self.p = Y.shape[0]

        self.bop = TorchBOP(self.W, self.X, self.Y, self.Z)
        # Target to torch complex
        A_F, B_F, C_F, D_F = target
        self.register_buffer("A_F", torch.tensor(A_F, dtype=torch.complex128))
        self.register_buffer("B_F", torch.tensor(B_F, dtype=torch.complex128))
        self.register_buffer("C_F", torch.tensor(C_F, dtype=torch.complex128))
        self.register_buffer("D_F", torch.tensor(D_F, dtype=torch.complex128))
        # Parameters V (real-valued) - shape should be (p, n) not (m, n)!
        self.V_real = nn.Parameter(torch.randn(self.p, n, dtype=torch.float64) * 0.05)
        self.V_imag = nn.Parameter(torch.randn(self.p, n, dtype=torch.float64) * 0.05)

    def current_system(self) -> Tuple[torch.Tensor, ...]:
        """
        Get current system for optimization.

        For LOSSLESS target: Return lossless G directly
        For LOSSY target: Return lossy H = (A, B̂, C, D̂)
        """
        V = torch.complex(self.V_real, self.V_imag)
        # Get lossless G from chart
        A, B_lossless, C, D_lossless = self.bop(V)

        # Check if target is lossless (for 1D case, this is true)
        # If target is lossless, we should approximate it with lossless G!
        from lossless_embedding_torch import verify_lossless_torch
        target_lossless_err = verify_lossless_torch(self.A_F, self.B_F, self.C_F, self.D_F)

        if target_lossless_err < 1e-10:
            # Target is lossless - approximate with lossless G
            return A, B_lossless, C, D_lossless
        else:
            # Target is lossy - compute lossy H via necessary conditions
            B_hat = compute_optimal_B_torch(A, C, self.A_F, self.B_F, self.C_F)
            D_hat = torch.zeros((self.C_F.shape[0], self.B_F.shape[1]), dtype=torch.complex128, device=A.device)
            return A, B_hat, C, D_hat

    def forward(self, use_analytical: bool = True) -> torch.Tensor:
        """Compute H2 error.

        Args:
            use_analytical: If True, use exact analytical formula.
                           If False, use legacy frequency sampling (for comparison).

        For LOSSLESS targets: Uses ||F - G||² where G is the lossless approximation.
        For LOSSY targets: Uses concentrated criterion J_n(C, A).
        """
        from lossless_embedding_torch import verify_lossless_torch

        V = torch.complex(self.V_real, self.V_imag)
        A, B, C, D = self.bop(V)  # This is lossless G

        # Check if target is lossless
        target_lossless_err = verify_lossless_torch(self.A_F, self.B_F, self.C_F, self.D_F)

        if use_analytical:
            if target_lossless_err < 1e-10:
                # Target is lossless - use direct ||F - G||² between lossless systems
                loss = h2_error_lossless_torch(
                    self.A_F, self.B_F, self.C_F, self.D_F,
                    A, B, C, D
                )
            else:
                # Target is lossy - use concentrated criterion for optimal lossy approx
                loss = h2_error_analytical_torch(
                    self.A_F, self.B_F, self.C_F, self.D_F, A, C
                )
            return loss
        else:
            # Legacy frequency sampling approach
            A_sys, B_sys, C_sys, D_sys = self.current_system()
            loss = h2_error_torch(
                (self.A_F, self.B_F, self.C_F, self.D_F),
                (A_sys, B_sys, C_sys, D_sys),
                num_samples=64
            )
            return loss
