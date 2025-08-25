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

    def forward(self, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given V (m x n, complex), produce (A, C) through Λ normalization.
        Steps:
          - Solve P - W^H P W = - (V^H V - Y^H Y)
          - Stabilize P via Hermitization and Gershgorin-based diagonal shift
          - Compute Λ and Λ^{-1} via Newton–Schulz (differentiable)
          - A = Λ W Λ^{-1}, C = (Y + V) Λ^{-1}
        """
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
        shift_pd = torch.clamp(1e-6 - lower_bd, min=0.0).to(P.dtype)
        if shift_pd.item() > 0:
            P = P + shift_pd * torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
        # Newton–Schulz sqrt and invsqrt
        Lambda, Lambda_inv = self._matrix_sqrt_invsqrt_ns(P, iters=30)
        # A and C
        A = Lambda @ self.W @ Lambda_inv
        C = (self.Y + V) @ Lambda_inv
        return A, C


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
        self.bop = TorchBOP(self.W, self.X, self.Y, self.Z)
        # Target to torch complex
        A_F, B_F, C_F, D_F = target
        self.register_buffer("A_F", torch.tensor(A_F, dtype=torch.complex128))
        self.register_buffer("B_F", torch.tensor(B_F, dtype=torch.complex128))
        self.register_buffer("C_F", torch.tensor(C_F, dtype=torch.complex128))
        self.register_buffer("D_F", torch.tensor(D_F, dtype=torch.complex128))
        # Parameters V (real-valued)
        self.V_real = nn.Parameter(torch.randn(m, n, dtype=torch.float64) * 0.05)
        self.V_imag = nn.Parameter(torch.randn(m, n, dtype=torch.float64) * 0.05)

    def current_system(self) -> Tuple[torch.Tensor, ...]:
        V = torch.complex(self.V_real, self.V_imag)
        A, C = self.bop(V)
        B_hat = compute_optimal_B_torch(A, C, self.A_F, self.B_F, self.C_F)
        D_hat = torch.zeros((self.C_F.shape[0], self.B_F.shape[1]), dtype=torch.complex128, device=A.device)
        return A, B_hat, C, D_hat

    def forward(self) -> torch.Tensor:
        A, B_hat, C, D_hat = self.current_system()
        loss = h2_error_torch((self.A_F, self.B_F, self.C_F, self.D_F), (A, B_hat, C, D_hat), num_samples=64)
        return loss
