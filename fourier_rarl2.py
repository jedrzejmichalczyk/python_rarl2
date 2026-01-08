#!/usr/bin/env python3
"""
Fourier Coefficient RARL2 - Direct Markov Parameter Optimization
================================================================

This module implements RARL2 optimization directly from Markov parameters
(Fourier coefficients) WITHOUT converting to state-space representation.

The target is specified as a sequence of Markov parameters:
    F = {F_0, F_1, F_2, ..., F_K}
where F_k ∈ ℂ^{p×m} are the impulse response matrices.

For state-space systems: F_0 = D, F_k = C A^{k-1} B for k ≥ 1

Mathematical Foundation:
-----------------------
For lossless G of order n with state-space (A, B, C, D):
- G is an isometry in H2 (||G·P||² = ||P||² for any P in H2)
- G⁻¹ = G^H on the unit circle (lossless property)
- G⁻¹ is anticausal with Markov parameters:
    (G⁻¹)_0 = D^H
    (G⁻¹)_{-k} = B^H (A^H)^{k-1} C^H  for k ≥ 1

Concentrated Criterion:
    J_n(G) = ||F - G·P||²  where  P = π_+(G⁻¹·F)
           = ||F||² - ||P||²  (since G is isometry and π_- ⊥ π_+)

The causal projection P = π_+(G⁻¹·F) has Markov parameters:
    P_k = D^H F_k + B^H S_{k-1}

where S_k satisfies the backward recursion:
    S_k = C^H F_{k+1} + A^H S_{k+1}
    S_{K-1} = C^H F_K  (terminal condition)

This gives a fully differentiable criterion for optimization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional


def kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Kronecker product for torch tensors."""
    a11 = A.unsqueeze(-1).unsqueeze(-3)
    b11 = B.unsqueeze(-2).unsqueeze(-4)
    K = a11 * b11
    m, n = A.shape[-2], A.shape[-1]
    p, q = B.shape[-2], B.shape[-1]
    return K.reshape(m * p, n * q)


def solve_two_sided_stein(A_left: torch.Tensor, A_right: torch.Tensor, RHS: torch.Tensor) -> torch.Tensor:
    """Solve X = A_left X A_right + RHS via vectorization."""
    nL = A_left.shape[0]
    nR = A_right.shape[0]
    I = torch.eye(nL * nR, dtype=A_left.dtype, device=A_left.device)
    K = I - kron(A_right.T, A_left)
    b = RHS.permute(1, 0).contiguous().view(-1)
    try:
        vecX = torch.linalg.solve(K, b)
    except RuntimeError:
        vecX = torch.linalg.lstsq(K, b).solution
    X = vecX.view(nR, nL).permute(1, 0).contiguous()
    return X


class TorchBOPFourier(nn.Module):
    """BOP chart forward map for Fourier RARL2.

    Identical to TorchBOP but optimized for Fourier coefficient computation.
    """

    def __init__(self, W: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor):
        super().__init__()
        self.register_buffer("W", W)
        self.register_buffer("X", X)
        self.register_buffer("Y", Y)
        self.register_buffer("Z", Z)
        self.n = W.shape[0]
        self.m = X.shape[1]

    def _matrix_sqrt_invsqrt_ns(self, P: torch.Tensor, iters: int = 25) -> Tuple[torch.Tensor, torch.Tensor]:
        """Newton-Schulz iteration for sqrt and inverse sqrt."""
        normF = torch.linalg.norm(P)
        c = normF if normF > 0 else torch.tensor(1.0, dtype=P.dtype, device=P.device)
        Y = P / c
        I = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
        Z = I.clone()
        for _ in range(iters):
            T = 0.5 * (3 * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z
        sqrt_c = torch.sqrt(c.real).to(P.dtype)
        S = sqrt_c * Y
        Sinv = Z / sqrt_c
        return S, Sinv

    def forward(self, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map V to lossless (A, B, C, D)."""
        from lossless_embedding_torch import lossless_embedding_torch

        # Stein for P
        RHS = V.conj().T @ V - self.Y.conj().T @ self.Y
        P = solve_two_sided_stein(self.W.conj().T, self.W, -RHS)
        P = (P + P.conj().T) * 0.5

        # Gershgorin-based PD shift
        absP = torch.abs(P)
        row_sum_off = absP.sum(dim=1) - torch.abs(torch.diagonal(P))
        diag_real = torch.real(torch.diagonal(P))
        lower_bd = torch.min(diag_real - row_sum_off)
        shift_pd = torch.clamp(1e-6 - lower_bd, min=0.0)
        if shift_pd.item() > 0:
            P = P + shift_pd * torch.eye(P.shape[0], dtype=P.dtype, device=P.device)

        # Newton-Schulz
        Lambda, Lambda_inv = self._matrix_sqrt_invsqrt_ns(P, iters=50)

        # A and C from chart
        A = Lambda @ self.W @ Lambda_inv
        C = (self.Y + V) @ Lambda_inv

        # QR orthonormalization with phase fix
        stacked = torch.cat([A, C], dim=0)
        Q, R = torch.linalg.qr(stacked)
        signs = torch.sign(torch.real(torch.diagonal(R)))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q @ torch.diag(signs.to(Q.dtype))

        A = Q[:self.n, :]
        C = Q[self.n:, :]

        # Complete to lossless
        B, D = lossless_embedding_torch(C, A, nu=1.0)

        return A, B, C, D


def compute_markov_params(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                          K: int) -> List[torch.Tensor]:
    """Compute Markov parameters {H_0, H_1, ..., H_K} for state-space system.

    H_0 = D
    H_k = C A^{k-1} B  for k ≥ 1
    """
    n = A.shape[0]
    markov = [D]  # H_0 = D

    if n == 0:
        # Static system - only D term
        for _ in range(K):
            markov.append(torch.zeros_like(D))
        return markov

    A_power = torch.eye(n, dtype=A.dtype, device=A.device)
    for k in range(1, K + 1):
        H_k = C @ A_power @ B
        markov.append(H_k)
        A_power = A_power @ A

    return markov


def compute_markov_norm_squared(markov: List[torch.Tensor]) -> torch.Tensor:
    """Compute ||H||² = Σ_k ||H_k||²_F from Markov parameters."""
    norm_sq = torch.zeros((), dtype=torch.float64, device=markov[0].device)
    for H_k in markov:
        norm_sq = norm_sq + torch.sum(torch.abs(H_k) ** 2).real
    return norm_sq


def compute_causal_projection_markov(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
    F_markov: List[torch.Tensor]
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Compute Markov parameters of P = π_+(G⁻¹·F) for lossless G.

    For lossless G with state-space (A, B, C, D):
    - G⁻¹ is anticausal with (G⁻¹)_0 = D^H, (G⁻¹)_{-k} = B^H (A^H)^{k-1} C^H
    - The convolution G⁻¹ * F mixes causal and anticausal parts
    - P = π_+(G⁻¹ * F) keeps only k ≥ 0

    Derivation:
    P_k = Σ_{j=0}^∞ (G⁻¹)_{-j} F_{k+j}
        = (G⁻¹)_0 F_k + Σ_{j=1}^∞ (G⁻¹)_{-j} F_{k+j}
        = D^H F_k + Σ_{j=1}^∞ B^H (A^H)^{j-1} C^H F_{k+j}
        = D^H F_k + B^H Σ_{i=0}^∞ (A^H)^i C^H F_{k+i+1}
        = D^H F_k + B^H S_k

    where S_k = Σ_{i=0}^∞ (A^H)^i C^H F_{k+i+1} satisfies:
        S_k = C^H F_{k+1} + A^H S_{k+1}
        S_K = 0  (terminal condition, assuming F_j = 0 for j > K)

    Args:
        A, B, C, D: Lossless system state-space
        F_markov: Target Markov parameters [F_0, F_1, ..., F_K]

    Returns:
        P_markov: Causal projection Markov parameters [P_0, P_1, ..., P_K]
        P_norm_sq: ||P||² = Σ_k ||P_k||²_F
    """
    K = len(F_markov) - 1  # F_0 to F_K
    n = A.shape[0]
    device = A.device

    # Get dimensions from F_markov
    p_out, m_in = F_markov[0].shape

    # Backward recursion for S_k
    # S_k = C^H F_{k+1} + A^H S_{k+1}
    # Terminal: S_K = 0, so S_{K-1} = C^H F_K

    # We need S_0, S_1, ..., S_{K-1} (and S_K = 0)
    S = [None] * (K + 1)  # S_0, S_1, ..., S_K
    S[K] = torch.zeros((n, m_in), dtype=A.dtype, device=device)  # Terminal: S_K = 0

    A_H = A.conj().T
    C_H = C.conj().T

    # Backward sweep: S_k = C^H F_{k+1} + A^H S_{k+1}
    for k in range(K - 1, -1, -1):
        S[k] = C_H @ F_markov[k + 1] + A_H @ S[k + 1]

    # Compute P_k = D^H F_k + B^H S_k
    D_H = D.conj().T
    B_H = B.conj().T

    P_markov = []
    P_norm_sq = torch.zeros((), dtype=torch.float64, device=device)

    for k in range(K + 1):
        P_k = D_H @ F_markov[k] + B_H @ S[k]
        P_markov.append(P_k)
        P_norm_sq = P_norm_sq + torch.sum(torch.abs(P_k) ** 2).real

    return P_markov, P_norm_sq


def concentrated_criterion_fourier(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
    F_markov: List[torch.Tensor],
    F_norm_sq: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute concentrated criterion J_n(G) = ||F||² - ||P||² from Markov parameters.

    This is the exact H2 error for the optimal approximation H = G·P where G is
    lossless and P = π_+(G⁻¹·F) is the optimal stable outer factor.

    Args:
        A, B, C, D: Lossless approximation state-space
        F_markov: Target Markov parameters [F_0, F_1, ..., F_K]
        F_norm_sq: Pre-computed ||F||² (optional, computed if not provided)

    Returns:
        J_n: Concentrated criterion value (H2 error squared)
    """
    # Compute ||F||² if not provided
    if F_norm_sq is None:
        F_norm_sq = compute_markov_norm_squared(F_markov)

    # Compute ||P||²
    _, P_norm_sq = compute_causal_projection_markov(A, B, C, D, F_markov)

    # J_n = ||F||² - ||P||²
    J_n = F_norm_sq - P_norm_sq

    return torch.clamp(J_n, min=0.0)


def compute_optimal_outer_factor(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
    F_markov: List[torch.Tensor]
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Compute optimal outer factor P = π_+(G⁻¹·F) and resulting approximation H = G·P.

    Returns:
        P_markov: Outer factor Markov parameters
        H_markov: Approximation H = G·P Markov parameters (via convolution)
        error: H2 error ||F - H||²
    """
    K = len(F_markov) - 1

    # Get P
    P_markov, P_norm_sq = compute_causal_projection_markov(A, B, C, D, F_markov)

    # Compute G's Markov parameters
    G_markov = compute_markov_params(A, B, C, D, K)

    # Compute H = G * P via convolution
    # H_k = Σ_{j=0}^k G_j P_{k-j}
    H_markov = []
    for k in range(K + 1):
        H_k = torch.zeros_like(F_markov[0])
        for j in range(min(k + 1, len(G_markov))):
            if k - j < len(P_markov):
                H_k = H_k + G_markov[j] @ P_markov[k - j]
        H_markov.append(H_k)

    # Compute error ||F - H||²
    error = torch.zeros((), dtype=torch.float64, device=A.device)
    for k in range(K + 1):
        diff = F_markov[k] - H_markov[k]
        error = error + torch.sum(torch.abs(diff) ** 2).real

    return P_markov, H_markov, error


class FourierRARL2(nn.Module):
    """RARL2 optimizer for Fourier coefficient (Markov parameter) input.

    This module optimizes a lossless function G to minimize the H2 error
    ||F - G·P||² where P = π_+(G⁻¹·F) is the optimal outer factor.

    The target is specified directly as Markov parameters, without any
    state-space conversion. This preserves the exactness of the H2 criterion.

    Args:
        n: Approximation order (McMillan degree)
        chart_center: BOP chart center (W, X, Y, Z) as numpy arrays
        F_markov: Target Markov parameters [F_0, F_1, ..., F_K] as numpy arrays
    """

    def __init__(self, n: int, chart_center: Tuple[np.ndarray, ...],
                 F_markov: List[np.ndarray]):
        super().__init__()
        self.n = n

        # Chart center
        W, X, Y, Z = chart_center
        self.W = torch.tensor(W, dtype=torch.complex128)
        self.X = torch.tensor(X, dtype=torch.complex128)
        self.Y = torch.tensor(Y, dtype=torch.complex128)
        self.Z = torch.tensor(Z, dtype=torch.complex128)

        # Output dimension from Y
        self.p = Y.shape[0]
        self.m = X.shape[1]

        self.bop = TorchBOPFourier(self.W, self.X, self.Y, self.Z)

        # Target Markov parameters
        self.F_markov = [torch.tensor(F_k, dtype=torch.complex128) for F_k in F_markov]
        self.K = len(F_markov) - 1

        # Pre-compute ||F||²
        self.register_buffer("F_norm_sq", compute_markov_norm_squared(self.F_markov))

        # Learnable parameters V
        self.V_real = nn.Parameter(torch.randn(self.p, n, dtype=torch.float64) * 0.05)
        self.V_imag = nn.Parameter(torch.randn(self.p, n, dtype=torch.float64) * 0.05)

    def current_lossless_system(self) -> Tuple[torch.Tensor, ...]:
        """Get current lossless G from parameters."""
        V = torch.complex(self.V_real, self.V_imag)
        A, B, C, D = self.bop(V)
        return A, B, C, D

    def current_markov_params(self) -> List[torch.Tensor]:
        """Get Markov parameters of current lossless G."""
        A, B, C, D = self.current_lossless_system()
        return compute_markov_params(A, B, C, D, self.K)

    def forward(self) -> torch.Tensor:
        """Compute concentrated criterion J_n(G) = ||F||² - ||P||²."""
        A, B, C, D = self.current_lossless_system()
        return concentrated_criterion_fourier(A, B, C, D, self.F_markov, self.F_norm_sq)

    def get_optimal_approximation(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Get optimal approximation H = G·P and its H2 error.

        Returns:
            H_markov: Approximation Markov parameters
            error: H2 error ||F - H||²
        """
        A, B, C, D = self.current_lossless_system()
        _, H_markov, error = compute_optimal_outer_factor(A, B, C, D, self.F_markov)
        return H_markov, error


def create_fourier_chart_center(n: int, p: int, m: int) -> Tuple[np.ndarray, ...]:
    """Create a default chart center for Fourier RARL2.

    Creates an output-normal lossless system of order n with p outputs and m inputs.
    Uses random initialization with proper normalization.

    Args:
        n: System order
        p: Number of outputs
        m: Number of inputs

    Returns:
        (W, X, Y, Z): Chart center matrices
    """
    from lossless_embedding import lossless_embedding

    # Create random output-normal (A, C)
    # A^H A + C^H C = I requires proper orthonormalization
    stacked = np.random.randn(n + p, n) + 1j * np.random.randn(n + p, n)
    Q, _ = np.linalg.qr(stacked)

    A = Q[:n, :].astype(np.complex128)
    C = Q[n:n+p, :].astype(np.complex128)

    # Scale A to ensure stability (eigenvalues inside unit circle)
    eigvals = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigvals))
    if max_eig >= 0.99:
        A = A * 0.95 / max_eig
        # Re-orthonormalize
        stacked = np.vstack([A, C])
        Q, _ = np.linalg.qr(stacked)
        A = Q[:n, :]
        C = Q[n:n+p, :]

    # Complete to lossless (B, D)
    B, D = lossless_embedding(C, A, nu=1.0)

    return A, B, C, D


def statespace_to_markov(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                         K: int) -> List[np.ndarray]:
    """Convert state-space system to Markov parameters (for testing/comparison)."""
    n = A.shape[0]
    markov = [D.copy()]

    if n == 0:
        for _ in range(K):
            markov.append(np.zeros_like(D))
        return markov

    A_power = np.eye(n, dtype=A.dtype)
    for k in range(1, K + 1):
        H_k = C @ A_power @ B
        markov.append(H_k)
        A_power = A_power @ A

    return markov


def optimize_fourier_rarl2(
    F_markov: List[np.ndarray],
    n: int,
    chart_center: Optional[Tuple[np.ndarray, ...]] = None,
    max_iter: int = 100,
    lr: float = 0.3,
    tol: float = 1e-10,
    verbose: bool = True
) -> Tuple[List[np.ndarray], float, dict]:
    """Optimize RARL2 from Fourier coefficients.

    Args:
        F_markov: Target Markov parameters [F_0, F_1, ..., F_K]
        n: Approximation order
        chart_center: Optional chart center (created if not provided)
        max_iter: Maximum L-BFGS iterations
        lr: Learning rate
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        H_markov: Optimal approximation Markov parameters
        final_error: Final H2 error
        info: Dictionary with optimization info
    """
    p, m = F_markov[0].shape

    # Create chart center if not provided
    if chart_center is None:
        chart_center = create_fourier_chart_center(n, p, m)

    # Create model
    model = FourierRARL2(n, chart_center, F_markov)

    # Initialize V = 0 (start at chart center)
    with torch.no_grad():
        model.V_real.data.fill_(0.0)
        model.V_imag.data.fill_(0.0)

    # Initial loss
    with torch.no_grad():
        initial_loss = model().item()

    if verbose:
        print(f"Fourier RARL2 Optimization (order {n})")
        print(f"  Target: K={len(F_markov)-1} Markov parameters, {p}x{m}")
        print(f"  Initial loss: {initial_loss:.6e}")

    # Optimize with L-BFGS
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=30)

    losses = [initial_loss]

    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        return loss

    for i in range(max_iter // 10):  # L-BFGS does multiple inner iterations
        loss = optimizer.step(closure)
        current_loss = loss.item()
        losses.append(current_loss)

        if verbose and (i + 1) % 5 == 0:
            print(f"  Iter {(i+1)*10}: loss = {current_loss:.6e}")

        if current_loss < tol:
            if verbose:
                print(f"  Converged at iteration {(i+1)*10}")
            break

    # Get final approximation
    with torch.no_grad():
        H_markov_torch, final_error_torch = model.get_optimal_approximation()
        H_markov = [H.numpy() for H in H_markov_torch]
        final_error = final_error_torch.item()

    if verbose:
        print(f"  Final loss: {final_error:.6e}")
        improvement = (initial_loss - final_error) / initial_loss * 100 if initial_loss > 1e-12 else 0
        print(f"  Improvement: {improvement:.1f}%")

    info = {
        'initial_loss': initial_loss,
        'losses': losses,
        'n_iter': len(losses) - 1,
        'chart_center': chart_center
    }

    return H_markov, final_error, info


# Convenience function for converting between representations
def markov_to_hankel(markov: List[np.ndarray], block_rows: int, block_cols: int) -> np.ndarray:
    """Build block Hankel matrix from Markov parameters.

    Used for comparison with Loewner-based methods.
    """
    p, m = markov[0].shape
    K = len(markov) - 1

    H = np.zeros((block_rows * p, block_cols * m), dtype=markov[0].dtype)

    for i in range(block_rows):
        for j in range(block_cols):
            k = i + j + 1  # Hankel index (starts at 1, not 0)
            if k <= K:
                H[i*p:(i+1)*p, j*m:(j+1)*m] = markov[k]

    return H


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)

    # Create a test target as Markov parameters
    n_target = 4
    p, m = 2, 2
    K = 20

    # Random stable system
    A = np.diag([0.9, 0.7, 0.5, 0.3]).astype(np.complex128)
    B = (np.random.randn(n_target, m) + 1j * np.random.randn(n_target, m)) * 0.3
    C = (np.random.randn(p, n_target) + 1j * np.random.randn(p, n_target)) * 0.3
    D = np.zeros((p, m), dtype=np.complex128)

    # Convert to Markov parameters
    F_markov = statespace_to_markov(A, B, C, D, K)

    print("=" * 60)
    print("Fourier RARL2 Quick Test")
    print("=" * 60)
    print(f"Target: order {n_target}, {p}x{m}, K={K} Markov params")

    # Optimize to order 2
    n_approx = 2
    H_markov, error, info = optimize_fourier_rarl2(F_markov, n_approx, verbose=True)

    print("\nResult:")
    print(f"  Approximation order: {n_approx}")
    print(f"  H2 error: {error:.6e}")
    print(f"  ||F||²: {info['initial_loss'] + error:.6e}")  # Approximate
