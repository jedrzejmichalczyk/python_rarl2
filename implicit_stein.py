#!/usr/bin/env python3
"""
Implicit Differentiation for Stein Equations
=============================================

Provides forward solvers and backward (adjoint) passes for Stein equations
used in RARL2 optimization.

Stein equation forms:
1. BOP chart: W^H P W - P = V^H V - Y^H Y
2. Cross-Gramian: A_F^H Q12 A + C_F^H C = Q12

For implicit differentiation, given dL/dX (loss gradient w.r.t. solution),
we compute dL/d(parameters) via adjoint equations.
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SteinSolution:
    """Container for Stein equation solution with cached data for backward pass."""
    X: np.ndarray           # Solution matrix
    L: np.ndarray           # Left coefficient (A^H in A^H X B - X = C)
    R: np.ndarray           # Right coefficient (B in A^H X B - X = C)
    C: np.ndarray           # RHS matrix


def solve_stein_LXR(L: np.ndarray, R: np.ndarray, C: np.ndarray) -> SteinSolution:
    """
    Solve generalized Stein/Sylvester equation: L X R - X = C

    Handles both square and non-square X matrices.

    For X of shape (m x n):
    - L is m x m
    - R is n x n
    - C is m x n

    Vectorization: (R^T ⊗ L - I_{mn}) vec(X) = vec(C)

    Args:
        L: Left coefficient matrix (m x m)
        R: Right coefficient matrix (n x n)
        C: Right-hand side matrix (m x n)

    Returns:
        SteinSolution with solution X and cached coefficients
    """
    m, n = C.shape

    # Verify dimensions
    assert L.shape == (m, m), f"L shape {L.shape} doesn't match C rows {m}"
    assert R.shape == (n, n), f"R shape {R.shape} doesn't match C cols {n}"

    # Method: Vectorization (R^T ⊗ L - I) vec(X) = vec(C)
    # For L X R - X = C: vec(L X R) = (R^T ⊗ L) vec(X)
    # So (R^T ⊗ L - I) vec(X) = vec(C)

    I_mn = np.eye(m * n, dtype=np.complex128)
    K = np.kron(R.T, L) - I_mn

    vec_C = C.reshape(-1, order='F')
    vec_X = np.linalg.solve(K, vec_C)
    X = vec_X.reshape((m, n), order='F')

    return SteinSolution(X=X, L=L, R=R, C=C)


def solve_stein_adjoint(sol: SteinSolution, grad_X: np.ndarray) -> np.ndarray:
    """
    Solve adjoint Stein equation for backward pass.

    Forward: L X R - X = C  where X is m x n
    Adjoint: L^H Ξ R^H - Ξ = grad_X  where Ξ is m x n

    For the adjoint, we transpose the operator:
    - Forward operator T: X -> L X R - X
    - Adjoint operator T*: Ξ -> L^H Ξ R^H - Ξ

    Args:
        sol: Solution from forward pass
        grad_X: Gradient of loss w.r.t. X (dL/dX), same shape as X

    Returns:
        Ξ: Adjoint variable (same shape as X)
    """
    L_adj = sol.L.conj().T  # L^H (m x m)
    R_adj = sol.R.conj().T  # R^H (n x n)

    # Solve: L^H Ξ R^H - Ξ = grad_X
    # This has same structure: L' Ξ R' - Ξ = C' where L'=L^H, R'=R^H, C'=grad_X
    sol_adj = solve_stein_LXR(L_adj, R_adj, grad_X)
    return sol_adj.X


def stein_backward(sol: SteinSolution, grad_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradients w.r.t. Stein equation parameters.

    Forward: L X R - X = C
    Given: grad_X = dLoss/dX
    Compute: dLoss/dL, dLoss/dR, dLoss/dC

    Derivation using implicit differentiation:
    Total differential: dL @ X @ R + L @ dX @ R + L @ X @ dR - dX = dC

    Rearranging for dX:
    (L @ · @ R - I) dX = dC - dL @ X @ R - L @ X @ dR

    For loss gradient, chain rule gives:
    dLoss = <grad_X, dX>

    Solving adjoint equation: (L^H @ · @ R^H - I) Ξ = grad_X

    Then using the identity <Ξ, (L·R - I)^{-1} Y> = <(L^H·R^H - I)^{-1} Ξ, Y>:
    dLoss = <Ξ, dC - dL @ X @ R - L @ X @ dR>
          = Tr(Ξ^H dC) - Tr(Ξ^H dL X R) - Tr(Ξ^H L X dR)

    So (using Tr cycling and conjugation rules):
    - dLoss/dC = Ξ
    - dLoss/dL = -R @ X^H @ Ξ^H  (negative sign from the "-" in front)
    - dLoss/dR = -X^H @ L^H @ Ξ  (negative sign from the "-" in front)

    Note: Conjugate gradient convention: dLoss = Re(Tr(grad^H @ dM)) for real loss.

    Args:
        sol: Solution from forward pass
        grad_X: Gradient of loss w.r.t. X

    Returns:
        (grad_L, grad_R, grad_C): Gradients w.r.t. L, R, C
    """
    Xi = solve_stein_adjoint(sol, grad_X)

    # dLoss/dC = Ξ (direct dependence, positive sign)
    grad_C = Xi

    # dLoss/dL: From -Tr(Ξ^H dL X R)
    # = -Tr((X R Ξ^H)^T dL^T) cycle and transpose
    # For complex: grad_L such that dLoss = Re(Tr(grad_L^H dL))
    # grad_L = -(Ξ R^H X^H)^* = -Ξ.conj() @ R.conj().T @ X.conj().T
    # But for real-valued loss with Wirtinger: grad_L = -Xi @ R.conj().T @ X.conj().T
    grad_L = -Xi @ sol.R.conj().T @ sol.X.conj().T

    # dLoss/dR: From -Tr(Ξ^H L X dR)
    # grad_R = -(L X)^H Ξ = -X^H L^H Ξ
    grad_R = -sol.X.conj().T @ sol.L.conj().T @ Xi

    return grad_L, grad_R, grad_C


class SteinDifferentiable:
    """
    Differentiable Stein equation solver for the BOP chart.

    BOP Stein equation: W^H P W - P = V^H V - Y^H Y

    Here L = W^H, R = W, C = V^H V - Y^H Y
    """

    def __init__(self, W: np.ndarray, Y: np.ndarray):
        """
        Initialize with chart center parameters.

        Args:
            W: Chart center W matrix (n x n)
            Y: Chart center Y matrix (m x n)
        """
        self.W = W.astype(np.complex128)
        self.Y = Y.astype(np.complex128)
        self.n = W.shape[0]
        self.m = Y.shape[0]

        # Precompute Y^H Y
        self.YHY = Y.conj().T @ Y

    def forward(self, V: np.ndarray) -> Tuple[np.ndarray, SteinSolution]:
        """
        Solve BOP Stein equation: W^H P W - P = V^H V - Y^H Y

        Args:
            V: Chart parameter (m x n)

        Returns:
            (P, sol): Solution P and SteinSolution for backward pass
        """
        VHV = V.conj().T @ V
        C = VHV - self.YHY

        L = self.W.conj().T
        R = self.W

        sol = solve_stein_LXR(L, R, C)

        # Symmetrize P (it should be Hermitian)
        P = (sol.X + sol.X.conj().T) / 2
        sol.X = P

        return P, sol

    def backward(self, sol: SteinSolution, grad_P: np.ndarray,
                 V: np.ndarray) -> np.ndarray:
        """
        Compute gradient w.r.t. V through BOP Stein equation.

        Forward: W^H P W - P = V^H V - Y^H Y

        dL/dV comes from the RHS: C = V^H V - Y^H Y
        dC/dV = V (in the sense that dC = dV^H V + V^H dV)

        Args:
            sol: Solution from forward pass
            grad_P: Gradient of loss w.r.t. P (dL/dP)
            V: The V parameter used in forward pass

        Returns:
            grad_V: Gradient w.r.t. V
        """
        # Symmetrize grad_P (P is Hermitian, so grad should respect that)
        grad_P_sym = (grad_P + grad_P.conj().T) / 2

        # Get gradient w.r.t. C (the RHS)
        grad_L, grad_R, grad_C = stein_backward(sol, grad_P_sym)

        # Now compute dL/dV from dL/dC
        # C = V^H V - Y^H Y
        # dC = dV^H V + V^H dV
        #
        # For complex Wirtinger derivative:
        # dL/dV* = grad_C @ V  (gradient w.r.t. conjugate)
        # dL/dV = V @ grad_C^H (gradient w.r.t. V itself)
        #
        # For optimization with real parameters, we need:
        # grad_V = 2 * Re(grad_C @ V) for V^H V contribution

        # Using: d/dV[Tr(grad_C^H (V^H V))] = grad_C @ V (Wirtinger)
        # The full gradient for real optimization: 2 * V @ grad_C (since C is Hermitian)
        grad_V = 2 * (V @ grad_C)

        return grad_V


class CrossGramianDifferentiable:
    """
    Differentiable cross-Gramian computation.

    Cross-Gramian equation: A_F^H Q12 A + C_F^H C = Q12

    This is a Stein equation with L = A_F^H, R = A, C = C_F^H @ C_approx
    """

    def __init__(self, A_F: np.ndarray, C_F: np.ndarray, B_F: np.ndarray):
        """
        Initialize with target system parameters.

        Args:
            A_F: Target system A matrix
            C_F: Target system C matrix
            B_F: Target system B matrix (for criterion)
        """
        self.A_F = A_F.astype(np.complex128)
        self.C_F = C_F.astype(np.complex128)
        self.B_F = B_F.astype(np.complex128)
        self.n_F = A_F.shape[0]

    def forward(self, A: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, SteinSolution]:
        """
        Compute cross-Gramian Q12.

        Equation: A_F^H Q12 A + C_F^H C = Q12
        Rewrite: A_F^H Q12 A - Q12 = -C_F^H C

        Args:
            A: Approximant system matrix (n x n)
            C: Approximant output matrix (p x n)

        Returns:
            (Q12, sol): Cross-Gramian and SteinSolution for backward
        """
        L = self.A_F.conj().T
        R = A
        C_rhs = -self.C_F.conj().T @ C

        sol = solve_stein_LXR(L, R, C_rhs)
        return sol.X, sol

    def backward(self, sol: SteinSolution, grad_Q12: np.ndarray,
                 C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients w.r.t. A and C through cross-Gramian equation.

        Forward: A_F^H Q12 A - Q12 = -C_F^H C

        Args:
            sol: Solution from forward pass
            grad_Q12: Gradient of loss w.r.t. Q12
            C: Approximant C matrix

        Returns:
            (grad_A, grad_C): Gradients w.r.t. A and C
        """
        grad_L, grad_R, grad_C_rhs = stein_backward(sol, grad_Q12)

        # grad_R is dL/dA (since R = A in the Stein equation)
        grad_A = grad_R

        # grad_C_rhs is dL/d(-C_F^H C)
        # C_rhs = -C_F^H @ C
        # dL/dC = -C_F @ grad_C_rhs
        grad_C = -self.C_F @ grad_C_rhs

        return grad_A, grad_C


class DualGramianDifferentiable:
    """
    Differentiable dual-Gramian (P12) computation.

    Dual-Gramian equation: A P12 A_F^H + B_hat B_F^H = P12
    Rewrite: A P12 A_F^H - P12 = -B_hat B_F^H
    """

    def __init__(self, A_F: np.ndarray, B_F: np.ndarray):
        """
        Initialize with target system parameters.
        """
        self.A_F = A_F.astype(np.complex128)
        self.B_F = B_F.astype(np.complex128)

    def forward(self, A: np.ndarray, B_hat: np.ndarray) -> Tuple[np.ndarray, SteinSolution]:
        """
        Compute dual-Gramian P12.

        Equation: A P12 A_F^H - P12 = -B_hat B_F^H

        Args:
            A: Approximant system matrix
            B_hat: Optimal B from necessary condition

        Returns:
            (P12, sol): Dual-Gramian and SteinSolution
        """
        L = A
        R = self.A_F.conj().T
        C_rhs = -B_hat @ self.B_F.conj().T

        sol = solve_stein_LXR(L, R, C_rhs)
        return sol.X, sol

    def backward(self, sol: SteinSolution, grad_P12: np.ndarray,
                 B_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients w.r.t. A and B_hat.

        Args:
            sol: Solution from forward pass
            grad_P12: Gradient w.r.t. P12
            B_hat: Optimal B matrix

        Returns:
            (grad_A, grad_B_hat): Gradients
        """
        grad_L, grad_R, grad_C_rhs = stein_backward(sol, grad_P12)

        # grad_L is dL/dA
        grad_A = grad_L

        # C_rhs = -B_hat @ B_F^H
        # dL/dB_hat = -grad_C_rhs @ B_F
        grad_B_hat = -grad_C_rhs @ self.B_F

        return grad_A, grad_B_hat


def test_stein_gradient():
    """Test Stein equation gradient via finite differences."""
    print("Testing Stein equation implicit differentiation")
    print("=" * 60)

    np.random.seed(42)
    n = 3

    # Create stable L and R (spectral radius < 1)
    L = 0.5 * np.random.randn(n, n).astype(np.complex128)
    L = L / (np.max(np.abs(np.linalg.eigvals(L))) + 0.1)

    R = 0.5 * np.random.randn(n, n).astype(np.complex128)
    R = R / (np.max(np.abs(np.linalg.eigvals(R))) + 0.1)

    C = np.random.randn(n, n).astype(np.complex128)

    # Forward solve
    sol = solve_stein_LXR(L, R, C)
    X = sol.X

    # Verify solution
    residual = L @ X @ R - X - C
    print(f"Forward solve residual: {np.linalg.norm(residual):.2e}")

    # Test gradient w.r.t. C via finite differences
    def loss_fn(C_test):
        sol_test = solve_stein_LXR(L, R, C_test)
        return np.sum(np.abs(sol_test.X)**2)  # ||X||_F^2

    # Analytical gradient
    grad_X = 2 * X.conj()  # d/dX ||X||^2 = 2 X*
    grad_L, grad_R, grad_C = stein_backward(sol, grad_X)

    # Finite difference check for grad_C
    eps = 1e-7
    grad_C_fd = np.zeros_like(C)
    base_loss = loss_fn(C)

    for i in range(n):
        for j in range(n):
            C_pert = C.copy()
            C_pert[i, j] += eps
            grad_C_fd[i, j] = (loss_fn(C_pert) - base_loss) / eps

    rel_error = np.linalg.norm(grad_C - grad_C_fd) / (np.linalg.norm(grad_C_fd) + 1e-10)
    print(f"Gradient w.r.t. C - relative error: {rel_error:.2e}")

    if rel_error < 1e-5:
        print("✓ Stein gradient test PASSED")
    else:
        print("✗ Stein gradient test FAILED")
        print(f"  Analytical:\n{grad_C}")
        print(f"  Finite diff:\n{grad_C_fd}")

    print()
    return rel_error < 1e-5


def test_bop_stein_gradient():
    """Test BOP Stein equation gradient."""
    print("Testing BOP Stein gradient")
    print("=" * 60)

    np.random.seed(42)
    n = 2
    m = 1

    # Create stable W (spectral radius < 1)
    W = np.diag([0.5, 0.6]).astype(np.complex128)
    Y = np.random.randn(m, n).astype(np.complex128)

    # V parameter
    V = 0.1 * np.random.randn(m, n).astype(np.complex128)

    stein = SteinDifferentiable(W, Y)

    # Forward
    P, sol = stein.forward(V)
    print(f"P eigenvalues: {np.linalg.eigvalsh(P)}")

    # Loss = Tr(P)
    def loss_fn(V_test):
        P_test, _ = stein.forward(V_test)
        return np.real(np.trace(P_test))

    # Analytical gradient
    grad_P = np.eye(n, dtype=np.complex128)  # d/dP Tr(P) = I
    grad_V = stein.backward(sol, grad_P, V)

    # Finite difference
    eps = 1e-7
    grad_V_fd = np.zeros_like(V)
    base_loss = loss_fn(V)

    for i in range(m):
        for j in range(n):
            V_pert = V.copy()
            V_pert[i, j] += eps
            grad_V_fd[i, j] = (loss_fn(V_pert) - base_loss) / eps

    # Compare real parts (for real optimization)
    grad_V_real = np.real(grad_V)
    grad_V_fd_real = np.real(grad_V_fd)

    rel_error = np.linalg.norm(grad_V_real - grad_V_fd_real) / (np.linalg.norm(grad_V_fd_real) + 1e-10)
    print(f"Gradient w.r.t. V - relative error: {rel_error:.2e}")

    if rel_error < 1e-4:
        print("✓ BOP Stein gradient test PASSED")
    else:
        print("✗ BOP Stein gradient test FAILED")
        print(f"  Analytical: {grad_V_real}")
        print(f"  Finite diff: {grad_V_fd_real}")

    print()
    return rel_error < 1e-4


if __name__ == "__main__":
    test_stein_gradient()
    test_bop_stein_gradient()
